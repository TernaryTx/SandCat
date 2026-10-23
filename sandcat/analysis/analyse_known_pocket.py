from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from sandcat.analysis.analysis_utils import (
    MDPOCKET_DESCRIPTOR_FILENAME,
    MDPOCKET_PDB_FILENAME,
    align_trajectory,
    compute_per_frame_lining_atoms,
    compute_pla_exposure,
    compute_pla_ligand,
    compute_pla_ligand_from_ref,
    compute_volume_based_exposure,
    extract_protein_heavy_pdb,
    get_holo_pocket_volume,
    load_universe,
    plot_pla_exposure_violin,
    plot_volume_exposure_violin,
    run_mdpocket,
)


def align_states_and_run_mdpocket(
        swish_path: Path,
        output_path: Path,
        traj_prefix: str,
        n_replicas: int,
        ref_pdb: str,
        fit_selection: str,
        stride: int,
        pocket_ref_pdb: str,
) -> None:
    """Extract a protein heavy-atom reference PDB and run alignment + mdpocket for each state.

    The protein reference is extracted from state 0's ref PDB, stripping benzene probes
    and hydrogens so the atom count matches the aligned trajectory.

    Args:
        swish_path: Directory containing SWISH trajectory and reference PDB files.
        output_path: Directory to write aligned trajectories and mdpocket outputs.
        traj_prefix: Filename prefix for trajectory and reference PDB files.
        n_replicas: Number of SWISH replicas.
        ref_pdb: Alignment reference PDB, pre-aligned to the holo structure.
        fit_selection: MDAnalysis selection string for alignment fitting.
        stride: Process every Nth frame during alignment.
        pocket_ref_pdb: Pocket reference PDB with dummy atoms for mdpocket --selected_pocket.
    """
    state0_ref = str(swish_path / f'{traj_prefix}0_ref.pdb')
    sim_ref_pdb = str(output_path / 'sim_protein_ref.pdb')
    logging.info('Extracting protein reference PDB from state 0 reference')
    extract_protein_heavy_pdb(state0_ref, sim_ref_pdb)

    for state_idx in range(n_replicas):
        traj_file = str(swish_path / f'{traj_prefix}{state_idx}.dcd')
        state_ref = str(swish_path / f'{traj_prefix}{state_idx}_ref.pdb')
        state_output_dir = output_path / f'state_{state_idx}'
        state_output_dir.mkdir(parents=True, exist_ok=True)

        aligned_traj = str(state_output_dir / f'state_{state_idx}_aligned.dcd')
        logging.info(f'Aligning state {state_idx}: {traj_file}')
        align_trajectory(
            trajectory_file=traj_file,
            topology_pdb=state_ref,
            align_ref_pdb=ref_pdb,
            output_file=aligned_traj,
            fit_selection=fit_selection,
            stride=stride,
        )

        logging.info(f'Running mdpocket for state {state_idx}')
        run_mdpocket(
            trajectory_file=aligned_traj,
            ref_pdb=sim_ref_pdb,
            pocket_ref_pdb=pocket_ref_pdb,
            output_dir=str(state_output_dir),
            trajectory_format='dcd',
        )


def resolve_pla_ligand(
        holo_pdb: str | None,
        ligand_resnum: int | None,
        ligand_cutoff: float,
        pocket_ref_pdb: str,
        sim_ref_pdb: str,
        state0_aligned: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Determine PLAligand atom indices and the pocket centroid used for per-frame lining atoms.

    If holo_pdb and ligand_resnum are both set, uses the holo cryptic-site ligand as per
    the SWISH paper. Otherwise falls back to the pocket reference dummy atoms.

    Args:
        holo_pdb: Holo structure PDB containing the cryptic-site ligand, or None.
        ligand_resnum: Residue number of the cryptic-site ligand in holo_pdb, or None.
        ligand_cutoff: Distance cutoff in Å for PLAligand computation.
        pocket_ref_pdb: Pocket reference PDB with dummy atoms defining the cryptic site.
        sim_ref_pdb: Protein-heavy-atom reference PDB from the simulation.
        state0_aligned: Aligned trajectory DCD for state 0, used when falling back to
            pocket reference atoms.

    Returns:
        Tuple of (pla_ligand, ref_position) where pla_ligand is a set of (resid, atom_name)
        tuples and ref_position is the centroid in Å used for pocket selection.
    """
    if holo_pdb is not None and ligand_resnum is not None:
        # Paper method: PLAligand from actual holo cryptic-site ligand, centroid from ligand
        holo_u = load_universe(holo_pdb)
        ligand_atoms = holo_u.select_atoms(f'resnum {ligand_resnum} and not name H*')
        ref_position = ligand_atoms.positions.mean(axis=0)
        logging.info(
            f'Using holo ligand resnum {ligand_resnum} as PLAligand source '
            f'(centroid: {ref_position[0]:.2f}, {ref_position[1]:.2f}, {ref_position[2]:.2f} Å)'
        )
        logging.info(f'Computing PLAligand: protein heavy atoms within {ligand_cutoff:.1f} Å of holo ligand resnum {ligand_resnum}')
        pla_ligand = compute_pla_ligand(holo_pdb, ligand_resnum, ligand_cutoff)
    else:
        # Fallback: PLAligand from pocket reference dummy atoms, centroid from pocket ref
        pocket_ref_u = load_universe(pocket_ref_pdb)
        ref_position = pocket_ref_u.atoms.positions.mean(axis=0)
        logging.info(
            f'Using pocket_ref_pdb dummy atoms as PLAligand source '
            f'(centroid: {ref_position[0]:.2f}, {ref_position[1]:.2f}, {ref_position[2]:.2f} Å)'
        )
        logging.info(f'Computing PLAligand: protein heavy atoms within {ligand_cutoff:.1f} Å of pocket reference atoms')
        pla_ligand = compute_pla_ligand_from_ref(pocket_ref_pdb, sim_ref_pdb, state0_aligned, ligand_cutoff)
    logging.info(f'PLAligand contains {len(pla_ligand)} atoms')
    return pla_ligand, ref_position


def resolve_holo_volume(
        holo_volume: float | None,
        holo_pdb: str | None,
        ref_pdb: str,
        pocket_ref_pdb: str,
) -> float:
    """Return the holo pocket volume, computing it via mdpocket if not pre-supplied.

    holo_pdb must be in the same coordinate frame as pocket_ref_pdb. Falls back to
    ref_pdb if holo_pdb is not provided.

    Args:
        holo_volume: Pre-computed holo pocket volume in Å³. Returned directly if not None.
        holo_pdb: Holo structure PDB used to compute the pocket volume, or None.
        ref_pdb: Alignment reference PDB, used as fallback if holo_pdb is None.
        pocket_ref_pdb: Pocket reference PDB with dummy atoms for mdpocket --selected_pocket.

    Returns:
        Holo pocket volume in Å³.
    """
    if holo_volume is not None:
        logging.info(f'Using pre-computed holo pocket volume (V_holo): {holo_volume:.1f} Å³')
        return holo_volume

    logging.info('Computing holo pocket volume (V_holo)')
    volume_ref_pdb = holo_pdb if holo_pdb is not None else ref_pdb
    holo_volume_computed = get_holo_pocket_volume(volume_ref_pdb, pocket_ref_pdb)
    logging.info(f'V_holo: {holo_volume_computed:.1f} Å³')
    return holo_volume_computed


def collect_replica_exposures(
        output_path: Path,
        n_replicas: int,
        sim_ref_pdb: str,
        ref_position: np.ndarray,
        pla_ligand: np.ndarray,
        holo_volume_computed: float,
        lambda_values: list[float] | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute PLA-based and volume-based pocket exposure for each replica.

    Args:
        output_path: Directory containing per-replica state_N/ subdirectories.
        n_replicas: Number of SWISH replicas.
        sim_ref_pdb: Protein-heavy-atom simulation reference PDB.
        ref_position: Pocket centroid in Å used for mdpocket pocket selection.
        pla_ligand: Set of (resid, atom_name) tuples defining the cryptic site shell.
        holo_volume_computed: Holo pocket volume in Å³ used as the 100% reference.
        lambda_values: Lambda values per replica, used for logging only.

    Returns:
        Tuple of (pla_exposures, vol_exposures), each a list of per-replica P.E. arrays.
    """
    pla_exposures: list[np.ndarray] = []
    vol_exposures: list[np.ndarray] = []

    for state_idx in range(n_replicas):
        state_dir = output_path / f'state_{state_idx}'
        aligned_dcd = str(state_dir / f'state_{state_idx}_aligned.dcd')
        mdpocket_pdb = str(state_dir / MDPOCKET_PDB_FILENAME)
        descriptor_file = str(state_dir / MDPOCKET_DESCRIPTOR_FILENAME)

        logging.info(f'Computing exposure for state_{state_idx}...')
        per_frame = compute_per_frame_lining_atoms(
            mdpocket_pdb, sim_ref_pdb, aligned_dcd, ref_position,
        )
        pla_exp = compute_pla_exposure(per_frame, pla_ligand)
        pla_exposures.append(pla_exp)

        vol_exp = compute_volume_based_exposure(descriptor_file, holo_volume_computed)
        vol_exposures.append(vol_exp)

        lv = lambda_values[state_idx] if lambda_values else state_idx
        n_open_pla = int(np.sum(pla_exp > 0))
        n_open_vol = int(np.sum(vol_exp > 0))
        logging.info(f'  λ={lv} PLA: {len(per_frame)} frames  median={np.median(pla_exp):.1f}%  open={n_open_pla}/{len(per_frame)}')
        logging.info(f'  λ={lv} Vol: {len(vol_exp)} frames  median={np.median(vol_exp):.1f}%  open={n_open_vol}/{len(vol_exp)}')

    return pla_exposures, vol_exposures


def main(
        swish_dir: str,
        ref_pdb: str,
        pocket_ref_pdb: str,
        output_dir: str,
        traj_prefix: str,
        n_replicas: int,
        fit_selection: str = 'backbone',
        stride: int = 1,
        lambda_values: list[float] | None = None,
        holo_pdb: str | None = None,
        holo_volume: float | None = None,
        ligand_resnum: int | None = None,
        ligand_cutoff: float = 4.5,
        exposure_pla_output: str | None = None,
        exposure_vol_output: str | None = None,
        plot_only: bool = False,
) -> None:
    """Align OpenMM SWISH replica trajectories, run mdpocket, and plot pocket exposure.

    Expects trajectory files named {traj_prefix}N.dcd and reference PDBs named
    {traj_prefix}N_ref.pdb in swish_dir.

    Args:
        swish_dir: Directory containing {traj_prefix}N.dcd and {traj_prefix}N_ref.pdb files
        ref_pdb: Path to the alignment reference PDB, pre-aligned to the holo structure
            used to generate pocket_ref_pdb
        pocket_ref_pdb: Path to the pocket reference PDB with dummy atoms
        output_dir: Directory to write aligned trajectories and mdpocket output files
        n_replicas: Number of thermodynamic states to read, which equals the
            number of SWISH replicas run. No default: the ladder length differs
            per target and reading past it fails on a missing trajectory.
        traj_prefix: Filename prefix for trajectory and reference PDB files.
            Files are expected as {traj_prefix}N.dcd and {traj_prefix}N_ref.pdb.
        fit_selection: MDAnalysis selection string for alignment fitting. Default 'backbone'.
        stride: Process every Nth frame during alignment. Default 1.
        lambda_values: Lambda scaling values for each replica (for violin plot x-axis labels).
        holo_pdb: Holo structure PDB, must be pre-aligned to the same coordinate frame as
            the trajectories (i.e. aligned to ref_pdb) and must contain the cryptic-site
            ligand if ligand_resnum is set. Used for PLAligand and V_holo computation.
            Falls back to ref_pdb for V_holo if None.
        holo_volume: Pre-computed holo pocket volume in Å³. If provided, skips get_holo_pocket_volume.
        ligand_resnum: Residue number of the cryptic-site ligand in holo_pdb (e.g. 301 for
            CBT at the cryptic site in 1PZO). If provided, PLAligand is computed from the actual
            holo ligand as per the SWISH paper. If None, pocket_ref_pdb dummy atoms are used instead.
        ligand_cutoff: Distance cutoff in Å for PLAligand computation. Default 4.5 Å.
        exposure_pla_output: Path to save the PLA-based exposure violin plot.
            Defaults to output_dir/exposure_pla.png.
        exposure_vol_output: Path to save the volume-based exposure violin plot.
            Defaults to output_dir/exposure_vol.png.
        plot_only: If True, skip alignment and mdpocket and read existing files
            from output_dir/state_N/ directly.
    """
    output_path = Path(output_dir)
    swish_path = Path(swish_dir)

    if not plot_only:
        output_path.mkdir(parents=True, exist_ok=True)
        align_states_and_run_mdpocket(
            swish_path=swish_path,
            output_path=output_path,
            traj_prefix=traj_prefix,
            n_replicas=n_replicas,
            ref_pdb=ref_pdb,
            fit_selection=fit_selection,
            stride=stride,
            pocket_ref_pdb=pocket_ref_pdb,
        )

    sim_ref_pdb = str(output_path / 'sim_protein_ref.pdb')
    state0_aligned = str(output_path / 'state_0' / 'state_0_aligned.dcd')

    pla_ligand, ref_position = resolve_pla_ligand(
        holo_pdb=holo_pdb,
        ligand_resnum=ligand_resnum,
        ligand_cutoff=ligand_cutoff,
        pocket_ref_pdb=pocket_ref_pdb,
        sim_ref_pdb=sim_ref_pdb,
        state0_aligned=state0_aligned,
    )
    holo_volume_computed = resolve_holo_volume(
        holo_volume=holo_volume,
        holo_pdb=holo_pdb,
        ref_pdb=ref_pdb,
        pocket_ref_pdb=pocket_ref_pdb,
    )
    pla_exposures, vol_exposures = collect_replica_exposures(
        output_path=output_path,
        n_replicas=n_replicas,
        sim_ref_pdb=sim_ref_pdb,
        ref_position=ref_position,
        pla_ligand=pla_ligand,
        holo_volume_computed=holo_volume_computed,
        lambda_values=lambda_values,
    )

    np.save(str(output_path / 'exposure_pla.npy'), np.array(pla_exposures, dtype=object))
    np.save(str(output_path / 'exposure_vol.npy'), np.array(vol_exposures, dtype=object))

    pla_out = exposure_pla_output or str(output_path / 'exposure_pla.png')
    vol_out = exposure_vol_output or str(output_path / 'exposure_vol.png')
    plot_pla_exposure_violin(pla_exposures, lambda_values=lambda_values, output_path=pla_out)
    plot_volume_exposure_violin(vol_exposures, lambda_values=lambda_values, output_path=vol_out)


def parser():
    """Parse command-line arguments for OpenMM SWISH pocket analysis."""
    arg_parser = ArgumentParser(
        description='Align OpenMM SWISH replica trajectories, run mdpocket, and compute '
                    'PLA-based and volume-based pocket exposure',
    )
    arg_parser.add_argument(
        '-d', '--swish_dir',
        type=str,
        required=True,
        help='Directory containing {traj_prefix}N.dcd and {traj_prefix}N_ref.pdb files',
    )
    arg_parser.add_argument(
        '-ref', '--ref_pdb',
        type=str,
        required=True,
        help='Alignment reference PDB, pre-aligned to the holo structure used for --pocket_ref_pdb',
    )
    arg_parser.add_argument(
        '-pocket', '--pocket_ref_pdb',
        type=str,
        required=True,
        help='Pocket reference PDB with dummy atoms',
    )
    arg_parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Directory to write mdpocket output files',
    )
    arg_parser.add_argument(
        '--n_replicas',
        type=int,
        required=True,
        help='Number of SWISH replicas, which is the number of demultiplexed '
             'states to read. Required: the ladder length differs per target',
    )
    arg_parser.add_argument(
        '--traj_prefix',
        type=str,
        required=True,
        help='Filename prefix for trajectory and ref PDB files',
    )
    arg_parser.add_argument(
        '--fit_selection',
        type=str,
        default='backbone',
        help='MDAnalysis selection string for alignment fitting. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Process every Nth frame during alignment. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--lambda_values',
        type=float,
        nargs='+',
        default=None,
        help='Lambda scaling values for each replica, used as violin plot x-axis labels',
    )
    arg_parser.add_argument(
        '--holo_pdb',
        type=str,
        default=None,
        help='Holo structure PDB, pre-aligned to the same coordinate frame as the trajectories '
             '(i.e. aligned to ref_pdb). Must contain the cryptic-site ligand if --ligand_resnum '
             'is set. Used for PLAligand and V_holo computation. Falls back to ref_pdb if not set.',
    )
    arg_parser.add_argument(
        '--holo_volume',
        type=float,
        default=None,
        help='Pre-computed holo pocket volume in Å³. If set, skips get_holo_pocket_volume. '
             'Use when holo_pdb and pocket_ref_pdb are in different coordinate frames.',
    )
    arg_parser.add_argument(
        '--ligand_resnum',
        type=int,
        default=None,
        help='Residue number of the cryptic-site ligand in holo_pdb (e.g. 301 for CBT at the '
             'cryptic site in 1PZO). If set, PLAligand is computed from the actual holo ligand '
             'as per the original SWISH paper. If not set, pocket_ref_pdb dummy atoms are used instead.',
    )
    arg_parser.add_argument(
        '--ligand_cutoff',
        type=float,
        default=4.5,
        help='Distance cutoff in Å for PLAligand computation. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--exposure_pla_output',
        type=str,
        default=None,
        help='Path to save the PLA-based exposure violin plot. Defaults to output_dir/exposure_pla.png.',
    )
    arg_parser.add_argument(
        '--exposure_vol_output',
        type=str,
        default=None,
        help='Path to save the volume-based exposure violin plot. Defaults to output_dir/exposure_vol.png.',
    )
    arg_parser.add_argument(
        '--plot_only',
        action='store_true',
        help='Skip alignment and mdpocket; compute exposure from existing files in output_dir/state_N/.',
    )
    return arg_parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    args = parser()
    main(
        swish_dir=args.swish_dir,
        ref_pdb=args.ref_pdb,
        pocket_ref_pdb=args.pocket_ref_pdb,
        output_dir=args.output_dir,
        n_replicas=args.n_replicas,
        traj_prefix=args.traj_prefix,
        fit_selection=args.fit_selection,
        stride=args.stride,
        lambda_values=args.lambda_values,
        holo_pdb=args.holo_pdb,
        holo_volume=args.holo_volume,
        ligand_resnum=args.ligand_resnum,
        ligand_cutoff=args.ligand_cutoff,
        exposure_pla_output=args.exposure_pla_output,
        exposure_vol_output=args.exposure_vol_output,
        plot_only=args.plot_only,
    )
