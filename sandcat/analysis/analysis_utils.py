from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import align


# mdpocket output filenames
MDPOCKET_PDB_FILENAME = 'mdpout_mdpocket.pdb'
MDPOCKET_DESCRIPTOR_FILENAME = 'mdpout_descriptors.txt'

PROTEIN_HEAVY_SELECTION = 'protein and not name H*'


def load_universe(*args, **kwargs) -> mda.Universe:
    """Load an MDAnalysis Universe, stripping CONECT records from any PDB topology first.

    OpenMM writes CONECT records with hybrid-36 (hex) atom IDs for systems
    with more than 99999 atoms, which MDAnalysis cannot parse. This wrapper
    writes a cleaned copy of the topology to a tempfile before loading.

    Args:
        *args: Positional arguments passed to mda.Universe (topology, [trajectory, ...]).
        **kwargs: Keyword arguments passed to mda.Universe.

    Returns:
        Loaded MDAnalysis Universe.
    """
    if not args:
        return mda.Universe(**kwargs)
    topology = args[0]
    if isinstance(topology, str) and topology.endswith('.pdb'):
        with open(topology) as fh:
            lines = [ln for ln in fh if not ln.startswith('CONECT')]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp.writelines(lines)
            tmp_path = tmp.name
        return mda.Universe(tmp_path, *args[1:], **kwargs)
    return mda.Universe(*args, **kwargs)


def atoms_within_cutoff(
        protein_atoms,
        ref_positions: np.ndarray,
        cutoff: float,
) -> set[tuple[int, str]]:
    """Return (resnum, atom_name) pairs for protein heavy atoms within cutoff of any ref position.

    Args:
        protein_atoms: MDAnalysis AtomGroup of protein heavy atoms.
        ref_positions: Array of reference positions in Å, shape (N, 3).
        cutoff: Distance cutoff in Å.

    Returns:
        Set of (resnum, atom_name) tuples for atoms within cutoff of any reference position.
    """
    result: set[tuple[int, str]] = set()
    for atom in protein_atoms:
        dists = np.linalg.norm(ref_positions - atom.position, axis=1)
        if dists.min() <= cutoff:
            result.add((atom.resnum, atom.name))
    return result


def get_holo_pocket_volume(holo_pdb: str, pocket_ref_pdb: str) -> float:
    """Runs mdpocket on the holo protein structure to measure the cryptic site
    pocket volume using the same --selected_pocket reference as the simulation.

    Protein-only ATOM records are used (HETATM removed) so the ligand does not
    block alpha sphere placement. The resulting volume is the void space defined
    by the protein conformation in the holo state, which serves as the 100%
    reference for volume-based pocket exposure.

    Args:
        holo_pdb: Holo structure PDB
        pocket_ref_pdb: Pocket reference PDB for --selected_pocket

    Returns:
        Holo pocket volume in Å³.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        protein_pdb = Path(tmpdir) / 'holo_protein.pdb'
        with open(holo_pdb) as src, open(protein_pdb, 'w') as dst:
            for line in src:
                if line.startswith('ATOM'):
                    dst.write(line)
            dst.write('END\n')

        pdb_list = Path(tmpdir) / 'pdb_list.txt'
        pdb_list.write_text('holo_protein.pdb\n')

        local_ref = Path(tmpdir) / 'pocket_ref.pdb'
        shutil.copy(pocket_ref_pdb, local_ref)

        subprocess.run(
            ['mdpocket', '--pdb_list', 'pdb_list.txt', '--selected_pocket', 'pocket_ref.pdb'],
            cwd=tmpdir,
            check=True,
            capture_output=True,
        )

        descriptor_file = Path(tmpdir) / MDPOCKET_DESCRIPTOR_FILENAME
        if not descriptor_file.exists():
            raise FileNotFoundError(f'mdpocket did not produce {MDPOCKET_DESCRIPTOR_FILENAME}')

        df = pd.read_csv(str(descriptor_file), sep=r'\s+')
        return float(df['pock_volume'].iloc[0])


def parse_mdpocket_all_pockets(mdpocket_pdb: str) -> list[dict[int, np.ndarray]]:
    """Parses mdpout_mdpocket.pdb and returns per-frame pocket alpha sphere positions.

    Each frame ends with ENDMDL. Within a frame, STP atoms with different residue
    numbers belong to different pockets.

    Args:
        mdpocket_pdb: Path to mdpout_mdpocket.pdb

    Returns:
        List of dicts, one per frame. Each dict maps pocket_id -> ndarray of
        shape (N, 3) with alpha sphere coordinates in Angstroms.
    """
    per_frame: list[dict[int, np.ndarray]] = []
    current: dict[int, list[list[float]]] = defaultdict(list)

    with open(mdpocket_pdb) as fh:
        for line in fh:
            if line.startswith('ENDMDL'):
                per_frame.append({pid: np.array(coords) for pid, coords in current.items()})
                current = defaultdict(list)
            elif line.startswith('ATOM'):
                pocket_id = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                current[pocket_id].append([x, y, z])

    return per_frame


def compute_per_frame_lining_atoms(
        mdpocket_pdb: str,
        sim_topology: str,
        sim_aligned_traj: str,
        ref_position: np.ndarray,
        pocket_selection_cutoff: float = 8.0,
        lining_cutoff: float = 5.0,
) -> list[set[tuple[int, str]]]:
    """Computes per-frame pocket lining atom sets from mdpocket alpha spheres.

    For each frame, finds the pocket whose alpha spheres are nearest to
    ref_position (the cryptic site ligand centroid). If the nearest alpha sphere
    is within pocket_selection_cutoff, protein heavy atoms within lining_cutoff of
    any alpha sphere in that pocket are the lining atoms for that frame.

    Args:
        mdpocket_pdb: Path to mdpout_mdpocket.pdb (alpha spheres, all pockets)
        sim_topology: Simulation topology PDB (ref.pdb)
        sim_aligned_traj: Pre-aligned trajectory DCD (state_N_aligned.dcd)
        ref_position: Reference position (Å) for pocket selection, e.g. the
            pocket reference centroid
        pocket_selection_cutoff: Max distance (Å) from the nearest pocket to
            ref_position. Frames where the nearest pocket exceeds this are closed.
        lining_cutoff: Max distance (Å) from a protein atom to any alpha sphere
            in the selected pocket for the atom to count as a lining atom

    Returns:
        List of per-frame lining atom sets. Each set contains (resid, atom_name)
        tuples. Empty sets represent closed-pocket frames.
    """
    all_pockets = parse_mdpocket_all_pockets(mdpocket_pdb)

    sim = load_universe(sim_topology, sim_aligned_traj)
    protein = sim.select_atoms(PROTEIN_HEAVY_SELECTION)

    per_frame_lining: list[set[tuple[int, str]]] = []
    for frame_idx, pockets in enumerate(all_pockets):
        if not pockets:
            per_frame_lining.append(set())
            continue

        best_spheres: np.ndarray | None = None
        best_min_dist = float('inf')
        for sphere_coords in pockets.values():
            dists = np.linalg.norm(sphere_coords - ref_position, axis=1)
            min_dist = float(dists.min())
            if min_dist < best_min_dist:
                best_min_dist = min_dist
                best_spheres = sphere_coords

        if best_min_dist > pocket_selection_cutoff:
            per_frame_lining.append(set())
            continue

        sim.trajectory[frame_idx]
        per_frame_lining.append(atoms_within_cutoff(protein, best_spheres, lining_cutoff))

    return per_frame_lining


def compute_pla_exposure(
        per_frame_lining: list[set[tuple[int, str]]],
        pla_ligand: set[tuple[int, str]],
) -> np.ndarray:
    """Computes per-frame pocket exposure (P.E.) as defined in the original SWISH paper:

        P.E. / 100% = |PLAf_pocket ∩ PLAligand| / |PLAligand|

    Args:
        per_frame_lining: Per-frame pocket lining atom sets
        pla_ligand: Fixed set of (resid, atom_name) defining the cryptic site shell

    Returns:
        Array of P.E. values in percent, one per frame
    """
    n_ligand = len(pla_ligand)
    if n_ligand == 0:
        raise ValueError('PLAligand is empty — check ligand_cutoff and pocket_ref_pdb')

    return np.array([
        len(frame_lining & pla_ligand) / n_ligand * 100.0
        for frame_lining in per_frame_lining
    ])


def compute_volume_based_exposure(
        descriptor_file: str,
        holo_volume: float,
) -> np.ndarray:
    """Computes per-frame pocket exposure as defined in the SWISH-X paper:

        P.E. = V_pocket(t) / V_holo × 100%

    Values are capped at 100%.

    Args:
        descriptor_file: Path to mdpout_descriptors.txt for a replica
        holo_volume: Holo pocket volume in Å³ from get_holo_pocket_volume

    Returns:
        Array of P.E. values in percent, one per frame
    """
    df = pd.read_csv(descriptor_file, sep=r'\s+')
    volumes = df['pock_volume'].to_numpy()
    return np.clip(volumes / holo_volume * 100.0, 0.0, 100.0)


def _plot_single_violin(
        replica_exposures: list[np.ndarray],
        title: str,
        lambda_values: list[float] | None,
        output_path: str,
) -> None:
    """Plot per-replica pocket exposure as a violin plot and save to disk.

    Args:
        replica_exposures: List of P.E. arrays (percent), one per replica.
        title: Plot title.
        lambda_values: Lambda value for each replica (x-axis labels). If None,
            replicas are labelled by index.
        output_path: Path to save the figure.
    """
    n_reps = len(replica_exposures)
    positions = list(range(1, n_reps + 1))
    labels = [f'λ {lv}' for lv in lambda_values] if lambda_values else [f'λ {i}' for i in range(n_reps)]

    fig, ax = plt.subplots(figsize=(max(8, n_reps * 1.2), 5))
    parts = ax.violinplot(replica_exposures, positions=positions, showmedians=False, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor('#808080')
        body.set_edgecolor('#404040')
        body.set_alpha(0.8)

    medians = [float(np.median(arr)) for arr in replica_exposures]
    ax.scatter(positions, medians, color='white', s=20, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('%Pocket Exposure')
    ax.set_ylim(0, 100)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()


def plot_pla_exposure_violin(
        replica_exposures: list[np.ndarray],
        lambda_values: list[float] | None = None,
        output_path: str | None = None,
) -> None:
    """Violin plot of PLA-based pocket exposure

    Args:
        replica_exposures: Per-replica PLA-based P.E. arrays (percent)
        lambda_values: Lambda value for each replica (x-axis labels)
        output_path: Path to save the figure. If None, display interactively
    """
    _plot_single_violin(
        replica_exposures,
        'PLA-based pocket exposure',
        lambda_values,
        output_path,
    )


def plot_volume_exposure_violin(
        replica_exposures: list[np.ndarray],
        lambda_values: list[float] | None = None,
        output_path: str | None = None,
) -> None:
    """Violin plot of volume-based pocket exposure

    Args:
        replica_exposures: Per-replica volume-based P.E. arrays (percent)
        lambda_values: Lambda value for each replica (x-axis labels)
        output_path: Path to save the figure. If None, display interactively
    """
    _plot_single_violin(
        replica_exposures,
        'Volume-based pocket exposure',
        lambda_values,
        output_path,
    )


PROBE_RESNAME = 'PRB'


def extract_protein_heavy_pdb(ref_pdb: str, output_pdb: str, keep_probes: bool = False) -> str:
    """Extract heavy atoms from an OpenMM state reference PDB.

    By default strips everything except protein heavy atoms. When keep_probes
    is True, probe heavy atoms (resname PRB) are retained alongside protein heavy
    atoms so the atom count matches probe-containing aligned trajectories.

    Args:
        ref_pdb: Path to a state reference PDB from the OpenMM run
        output_pdb: Path to write the output PDB
        keep_probes: If True, retain PRB probe heavy atoms. Default False.

    Returns:
        Path to the output PDB
    """
    universe = load_universe(ref_pdb)
    if keep_probes:
        kept = universe.select_atoms(f'(protein and not name H*) or (resname {PROBE_RESNAME} and not name H*)')
    else:
        kept = universe.select_atoms('protein and not (name H*)')
    kept.write(output_pdb)
    return output_pdb


def align_trajectory(
        trajectory_file: str,
        topology_pdb: str,
        align_ref_pdb: str,
        output_file: str,
        fit_selection: str = 'backbone',
        stride: int = 1,
        keep_probes: bool = False,
) -> str:
    """Align a DCD trajectory to a reference PDB and write a stripped DCD.

    topology_pdb provides the atom names/residues for the trajectory (state_stateN_ref.pdb)
    align_ref_pdb is the structure to align to — this must be pre-aligned to the holo
    structure used to generate the pocket reference.

    By default the output contains only protein heavy atoms (for mdpocket). When
    keep_probes is True, PRB probe heavy atoms are also retained so probe positions
    are available for density analysis.

    Args:
        trajectory_file: Path to the input trajectory (DCD)
        topology_pdb: Path to the state reference PDB (state_stateN_ref.pdb), used as topology
        align_ref_pdb: Path to the alignment reference PDB, pre-aligned to the holo structure
        output_file: Path to write the aligned trajectory (DCD)
        fit_selection: MDAnalysis atom selection string for fitting. Default 'backbone'
        stride: Process every Nth frame. Default 1 (every frame)
        keep_probes: If True, retain PRB probe heavy atoms in the output. Default False.

    Returns:
        Path to the aligned trajectory file
    """
    mobile = load_universe(topology_pdb, trajectory_file)
    reference = load_universe(align_ref_pdb)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_all = str(Path(tmpdir) / 'all_atoms.dcd')
        aligner = align.AlignTraj(mobile, reference, select=fit_selection, filename=tmp_all)
        aligner.run(step=stride)

        aligned = load_universe(topology_pdb, tmp_all)
        if keep_probes:
            keep = aligned.select_atoms(
                f'(protein and not name H*) or (resname {PROBE_RESNAME} and not name H*)'
            )
        else:
            keep = aligned.select_atoms('protein and not (name H*)')
        n_written = n_skipped = 0
        with mda.Writer(output_file, keep.n_atoms) as writer:
            for _ in aligned.trajectory:
                if not np.isfinite(keep.positions).all():
                    n_skipped += 1
                    continue
                writer.write(keep)
                n_written += 1

    if n_skipped:
        logging.warning(
            f'{Path(trajectory_file).name}: dropped {n_skipped} of '
            f'{n_written + n_skipped} frames containing non-finite coordinates. '
            f'These come from the trajectory itself, not from alignment, and are '
            f'a sign of a corrupt write or a blown-up frame in the simulation.',
        )
    return output_file


def run_mdpocket(
        trajectory_file: str,
        ref_pdb: str,
        pocket_ref_pdb: str,
        output_dir: str,
        trajectory_format: str = 'dcd',
) -> None:
    """Run mdpocket on a DCD trajectory to monitor a selected pocket over time

    Args:
        trajectory_file: Path to the trajectory file (DCD)
        ref_pdb: Path to the protein-heavy-atom reference PDB
        pocket_ref_pdb: Path to the pocket reference PDB with dummy atoms
        output_dir: Directory to run mdpocket in and write output files
        trajectory_format: Trajectory format string for mdpocket. Default 'dcd'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # mdpocket has a short internal path buffer — copy input files into the
    # working directory and use short relative paths to avoid truncation crashes.
    local_traj = output_path / Path(trajectory_file).name
    local_ref = output_path / 'ref.pdb'
    local_pocket = output_path / 'pocket_ref.pdb'
    if local_traj.resolve() != Path(trajectory_file).resolve():
        shutil.copy(trajectory_file, local_traj)
    shutil.copy(ref_pdb, local_ref)
    shutil.copy(pocket_ref_pdb, local_pocket)

    cmd = [
        'mdpocket',
        '--trajectory_file', local_traj.name,
        '--trajectory_format', trajectory_format,
        '-f', local_ref.name,
        '--selected_pocket', local_pocket.name,
    ]
    subprocess.run(cmd, cwd=str(output_path), check=True)

    descriptor_file = output_path / MDPOCKET_DESCRIPTOR_FILENAME
    if not descriptor_file.exists():
        raise FileNotFoundError(
            f'mdpocket did not produce expected descriptor file: {descriptor_file}'
        )


def compute_pla_ligand(
        holo_pdb: str,
        ligand_resnum: int,
        ligand_cutoff: float = 4.5,
) -> set[tuple[int, str]]:
    """Returns protein heavy atoms within ligand_cutoff of the holo cryptic-site ligand.

    Implements PLAligand as defined in the original SWISH paper: heavy receptor atoms
    within 0.45 nm of the ligand part bound to the cryptic site in the holo structure.
    Both protein and ligand positions come from holo_pdb, so PLAligand reflects the
    holo crystal geometry rather than any simulation frame.

    The holo_pdb must already be in the same coordinate frame as the aligned trajectories
    and must contain the ligand residue. Prepare this file externally by aligning the holo
    structure (with ligands intact) to the same reference used for trajectory alignment.

    Args:
        holo_pdb: Holo structure PDB containing the cryptic-site ligand, pre-aligned to
            the same reference frame as the trajectories.
        ligand_resnum: Residue number of the cryptic-site ligand in holo_pdb.
        ligand_cutoff: Distance cutoff in Angstroms. Paper uses 0.45 nm = 4.5 Å.

    Returns:
        Set of (resid, atom_name) tuples for protein heavy atoms lining the cryptic site.
    """
    holo = load_universe(holo_pdb)
    ligand = holo.select_atoms(f'resnum {ligand_resnum} and not name H*')
    if len(ligand) == 0:
        raise ValueError(f'No heavy atoms found for resnum {ligand_resnum} in {holo_pdb}')
    ligand_positions = ligand.positions

    protein = holo.select_atoms(PROTEIN_HEAVY_SELECTION)
    return atoms_within_cutoff(protein, ligand_positions, ligand_cutoff)


def compute_pla_ligand_from_ref(
        pocket_ref_pdb: str,
        sim_topology: str,
        sim_aligned_traj: str,
        ligand_cutoff: float = 4.5,
) -> set[tuple[int, str]]:
    """Returns protein heavy atoms within ligand_cutoff of any pocket_ref_pdb atom.

    Used to compute PLAligand when no holo ligand is available in the
    simulation coordinate frame. The pocket reference dummy atoms define the
    cryptic site location and protein atoms near them correspond to the binding shell.

    Args:
        pocket_ref_pdb: Pocket reference PDB with dummy atoms
        sim_topology: Protein-heavy-atom reference PDB (sim_protein_ref.pdb)
        sim_aligned_traj: Aligned trajectory DCD. Frame 0 is used for protein positions.
        ligand_cutoff: Distance cutoff in Angstroms. Default 4.5 Å.

    Returns:
        Set of (resid, atom_name) tuples for protein atoms lining the cryptic site.
    """
    ref_u = load_universe(pocket_ref_pdb)
    ref_positions = ref_u.atoms.positions

    sim = load_universe(sim_topology, sim_aligned_traj)
    protein = sim.select_atoms(PROTEIN_HEAVY_SELECTION)
    sim.trajectory[0]
    return atoms_within_cutoff(protein, ref_positions, ligand_cutoff)
