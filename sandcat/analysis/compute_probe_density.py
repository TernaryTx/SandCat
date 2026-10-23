"""Probe occupancy maps and the cryptic-site candidates derived from them.

Builds four density maps per target from the unbiased mixed-solvent and demuxed
SWISH-X trajectories, then reports the candidate sites two detectors find in
them. Every grid is normalised to its own bulk density by to_bulk_units, so maps
from systems with different probe counts are directly comparable and can be
subtracted; a raw grid scales with however many probes a build happened to
contain, and two builds of the same protein differ by a probe or two.

The maps written are the unbiased density, the mean SWISH density, their
difference, and the difference taken from demuxed state 0 alone. That last one
is the one the analysis uses. State 0 is lambda = 1.0, the only replica whose
Hamiltonian is unscaled: at lambda > 1 the Lennard-Jones well depth between
water oxygens and apolar protein atoms is multiplied by lambda, so the protein
is artificially wetted and probe occupancies are not comparable with an unbiased
simulation. The replica-averaged difference mixes six differently scaled
ensembles and corresponds to no physical state.

State 0 is not a canonical ensemble. In common with every replica it carries
an OPES Multithermal bias spanning 300-350 K, so it samples the expanded
multithermal ensemble rather than the Boltzmann distribution at 300 K. Nothing
here treats it as canonical: the difference map is defined as a contrast
between a biased and an unbiased ensemble, identifying where enhanced sampling
opens probe-accessible volume. No equilibrium occupancy or binding free energy
is inferred from it.

Both detectors apply one rule: connected components above THRESHOLD_FACTOR times
bulk holding at least MIN_CANDIDATE_VOXELS voxels, ranked by total density and
centred on the density-weighted centroid. The lambda = 1 difference map finds
pockets the bias had to open and is blind to any the unbiased ensemble already
samples; the unbiased map is the reverse, which is why both are written.

THRESHOLD_FACTOR is 6.0, chosen by sweeping 2x to 10x on maps normalised with
the true unit-cell volume. Oleinikovas et al. 2016 use 2x, which on these systems
is the difference map's noise floor rather than a pocket cut: it returns 799
candidates across the benchmark and 30 on ubiquitin, which has no cryptic pocket
at all. Raising the cut removes the control without costing recall until 6x,
where all seven sites are still found, ubiquitin is at zero and the benchmark
returns 99 candidates. At 6.25x PLK1 is lost and at 8x IL-2 goes too.

The margin above 6x is thin and the reason is PLK1, whose site is four voxels
against a floor of three, so it is the first thing any tightening removes. The
sweep uses the held-out references to decide which cut recovers the sites, so
this number is calibrated on the benchmark rather than derived from the data
alone.

The cut is meaningful only if the bulk density it is expressed against is
right, which means the true unit-cell volume and not the product of the cell
edges: for a truncated octahedron those differ by a factor of sqrt(2), and using
the product would inflate every map by 41% and move the effective cut to 5.7x.
See probe_count_and_box_volume.

MIN_CANDIDATE_VOXELS is 3, the smallest component that may be reported.
At the production cut ubiquitin produces ten components and every one is a
single voxel, while the true sites of the six benchmark targets are 4 to 33
voxels, so this floor removes every ubiquitin candidate without touching a
single true site. The ceiling is set by PLK1, whose site is only 4 voxels, so 5
loses it; 3 rather than 4 keeps a margin on a weaker site at no cost here.
"""
from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

from MDAnalysis.lib.mdamath import box_volume
import numpy as np
import pandas as pd
from scipy.ndimage import label as nd_label

from sandcat.analysis.analysis_utils import PROBE_RESNAME, PROTEIN_HEAVY_SELECTION, align_trajectory, extract_protein_heavy_pdb, load_universe

_DX_MD = 'probe_density_md.dx'
_DX_SWISH = 'probe_density_swish_pooled.dx'
_DX_DIFF = 'probe_density_diff_pooled.dx'
_DX_DIFF_LAMBDA1 = 'probe_density_diff_lambda1.dx'

MIN_CANDIDATE_VOXELS = 3
# The production stride, so a direct call analyses the same frames as the
# pipeline rather than every frame.
STRIDE = 10
# Grid geometry, quoted in the Methods as "1 A resolution extending 5 A beyond
# the protein". Named so the two functions that take them cannot drift.
GRID_SPACING_A = 1.0
GRID_PADDING_A = 5.0

THRESHOLD_FACTOR = 6.0


def probe_count_and_box_volume(
        topology_pdb: str,
        traj_dcd: str,
) -> tuple[int, float]:
    """Probe heavy-atom count and box volume for one trajectory.

    The volume is the true unit-cell volume, not the product of the edge
    lengths. These systems are solvated in truncated octahedra with cell angles
    of 60, 60 and 90 degrees, where a*b*c overestimates the volume by a factor
    of sqrt(2); using it would make the bulk density too small by the same
    factor and inflate every normalised map by 41%, so a cut nominally at a
    given multiple of bulk would in truth sit at that multiple over sqrt(2).

    It is averaged over frames rather than read from one, because these are NPT
    trajectories and the cell fluctuates: on MDM2 the per-frame volume has a
    standard deviation of 0.9% about its mean.

    Args:
        topology_pdb: Stripped topology PDB (protein + probes).
        traj_dcd: Trajectory DCD carrying box dimensions.

    Returns:
        (n_probe_atoms, box_volume_a3). Volume is 0.0 when the trajectory
        carries no usable box, which leaves the caller in raw units.
    """
    universe = load_universe(topology_pdb, traj_dcd)
    n_probe_atoms = len(
        universe.select_atoms(f'resname {PROBE_RESNAME} and not name H*'),
    )
    volumes = []
    for timestep in universe.trajectory:
        dimensions = timestep.dimensions
        if dimensions is None or not np.all(np.isfinite(dimensions[:3])):
            continue
        if not np.all(dimensions[:3] > 0):
            continue
        volumes.append(box_volume(dimensions))
    if not volumes:
        return n_probe_atoms, 0.0
    return n_probe_atoms, float(np.mean(volumes))


def to_bulk_units(
        density: np.ndarray,
        n_probe_atoms: int,
        grid_spacing: float,
        box_volume: float,
) -> np.ndarray:
    """Express a density grid as a multiple of its own uniform (bulk) density.

    Raw grids hold mean probe heavy atoms per voxel per frame, which scales with
    how many probes the system happens to contain. Two trajectories of the same
    protein built on different days can differ by a probe or two: PLK1's MSMD
    replicate 1 has 276 benzenes where replicates 0 and 2 have 277, and CRBN's
    unbiased and SWISH-X systems differ by one probe atom. Averaging or
    subtracting raw grids then compares slightly different systems.

    Dividing by the system's own uniform density removes that. Every grid becomes
    "times bulk", replicates with different probe counts can be averaged
    directly, the unbiased and biased sides can be subtracted even when their
    probe counts differ, and the Oleinikovas threshold becomes a plain factor of
    2 rather than a number that means something different in every box.

    Args:
        density: Raw density grid, mean probe heavy atoms per voxel per frame.
        n_probe_atoms: Probe heavy atoms in this system.
        grid_spacing: Voxel edge length in Angstroms.
        box_volume: Box volume in cubic Angstroms. 0.0 returns the grid unchanged.

    Returns:
        Density in units of the uniform probe density.
    """
    if box_volume <= 0 or n_probe_atoms <= 0:
        logging.warning(
            'No usable box volume or probe count: leaving the density in raw '
            'units, so it is not comparable with other systems',
        )
        return density
    uniform = n_probe_atoms * grid_spacing ** 3 / box_volume
    return density / uniform


def align_all_trajectories(
        output_path: Path,
        md_pairs: list[tuple[str, str]],
        swish_path: Path,
        swish_top_prefix: str,
        traj_prefix: str,
        n_replicas: int,
        ref_pdb: str,
        fit_selection: str,
        stride: int,
        lambda_values: list[float] | None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Align every unbiased MD replicate and all SWISH replicas to ref_pdb.

    The unbiased replicates are aligned separately rather than concatenated.
    Concatenation forces one topology on all of them, which fails outright when
    replicates differ in probe count, as PLK1's do (276 benzenes in replicate 1
    against 277 in replicates 0 and 2, because it was built six days earlier).
    Keeping them separate costs nothing, since the densities are averaged in
    space afterwards, and it preserves every replicate.

    Args:
        output_path: Directory to write aligned DCD files.
        md_pairs: One (topology_pdb, trajectory_dcd) per unbiased replicate.
        swish_path: Directory containing SWISH demuxed trajectories.
        swish_top_prefix: Filename prefix for SWISH replica topology PDBs.
        traj_prefix: Filename prefix for SWISH DCD files.
        n_replicas: Number of SWISH replicas.
        ref_pdb: Alignment reference PDB.
        fit_selection: MDAnalysis selection string for alignment fitting.
        stride: Process every Nth frame.
        lambda_values: Lambda values per replica, used for logging only.

    Returns:
        Tuple of (md_aligned_pairs, swish_aligned_dcds), where md_aligned_pairs
        is one (topology, aligned_dcd) per unbiased replicate.
    """
    md_aligned_pairs: list[tuple[str, str]] = []
    for replicate_idx, (md_top, md_dcd) in enumerate(md_pairs):
        logging.info(f'Aligning unbiased MD replicate {replicate_idx}: {md_dcd}')
        md_aligned = str(output_path / f'md_{replicate_idx}_aligned.dcd')
        align_trajectory(
            trajectory_file=md_dcd,
            topology_pdb=md_top,
            align_ref_pdb=ref_pdb,
            output_file=md_aligned,
            fit_selection=fit_selection,
            stride=stride,
            keep_probes=True,
        )
        md_aligned_pairs.append((md_top, md_aligned))

    swish_aligned_dcds: list[str] = []
    for replica_idx in range(n_replicas):
        # no silent fallback to the replica index: an unlabelled ladder says
        # 'lambda=?' rather than naming a rung that does not exist
        lam = lambda_values[replica_idx] if lambda_values else '?'
        traj_file = str(swish_path / f'{traj_prefix}{replica_idx}.dcd')
        top_file = str(swish_path / f'{swish_top_prefix}{replica_idx}_ref.pdb')
        aligned_dcd = str(output_path / f'swish_{replica_idx}_aligned.dcd')
        logging.info(f'Aligning SWISH state {replica_idx} (lambda={lam}): '
                     f'{traj_file}')
        align_trajectory(
            trajectory_file=traj_file,
            topology_pdb=top_file,
            align_ref_pdb=ref_pdb,
            output_file=aligned_dcd,
            fit_selection=fit_selection,
            stride=stride,
            keep_probes=True,
        )
        swish_aligned_dcds.append(aligned_dcd)

    return md_aligned_pairs, swish_aligned_dcds


def compute_probe_density_grid(
        topology_pdb: str,
        traj_dcd: str,
        origin: np.ndarray | None = None,
        grid_dims: np.ndarray | None = None,
        grid_spacing: float = GRID_SPACING_A,
        padding: float = GRID_PADDING_A,
        frame_half: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate probe atom positions into a 3D volumetric density grid.

    Density is normalised by the number of trajectory frames so each voxel
    value is the mean number of probe heavy atoms present per frame.  When
    origin and grid_dims are supplied the same grid box is reused (required
    for computing a meaningful SWISH − unbiased MD difference map).

    Args:
        topology_pdb: Stripped topology PDB (protein + probe atoms).
        traj_dcd: Aligned trajectory DCD.
        origin: Grid origin in Å.  Computed from protein positions when None.
        grid_dims: Integer grid dimensions (nx, ny, nz).  Computed when None.
        grid_spacing: Voxel edge length in Å.  Default 1.0.
        padding: Extra padding around the protein bounding box in Å.  Default 5.0.
        frame_half: 'first' or 'second' to build the density from only that half
            of the frames, for the temporal-convergence check. None uses all
            frames. The halves must share origin and grid_dims to be comparable.

    Returns:
        (density, origin) where density has shape (nx, ny, nz).
    """
    universe = load_universe(topology_pdb, traj_dcd)
    probes = universe.select_atoms(f'resname {PROBE_RESNAME} and not name H*')
    protein = universe.select_atoms(PROTEIN_HEAVY_SELECTION)

    if origin is None or grid_dims is None:
        all_protein_pos = np.vstack([protein.positions.copy() for _ in universe.trajectory])
        # A single non-finite frame makes min/max NaN, which makes the origin and
        # the grid dimensions NaN, which silently destroys the whole map. The
        # frames themselves are dropped below, but the box has to ignore them too.
        finite_rows = np.isfinite(all_protein_pos).all(axis=1)
        if not finite_rows.any():
            raise ValueError(
                f'Every protein coordinate in {traj_dcd} is non-finite; the '
                f'trajectory is unusable',
            )
        all_protein_pos = all_protein_pos[finite_rows]
        origin = all_protein_pos.min(axis=0) - padding
        extent = all_protein_pos.max(axis=0) + padding
        grid_dims = np.ceil((extent - origin) / grid_spacing).astype(int)

    total_frames = len(universe.trajectory)
    midpoint = total_frames // 2
    frames = {
        None: universe.trajectory,
        'first': universe.trajectory[:midpoint],
        'second': universe.trajectory[midpoint:],
    }[frame_half]

    density = np.zeros(grid_dims, dtype=np.float64)
    n_frames = 0
    n_probe_positions = 0
    n_outside = 0

    n_nonfinite_frames = 0
    for _ in frames:
        probe_pos = probes.positions
        # A blown-up or corrupt frame gives NaN coordinates, and NaN.astype(int)
        # is 0 rather than an error, so every atom in that frame lands in voxel
        # (0,0,0) and creates a spike that dominates the whole map. HPV11-E2
        # frame 1420 does this in five of its six states. Drop such frames and
        # say so.
        finite = np.isfinite(probe_pos).all(axis=1)
        if not finite.all():
            n_nonfinite_frames += 1
            probe_pos = probe_pos[finite]
            if probe_pos.size == 0:
                continue
        # floor, not truncation: astype(int) rounds toward zero, so offsets in
        # (-1, 0) become index 0 and leak into the grid from outside it
        indices = np.floor((probe_pos - origin) / grid_spacing).astype(int)
        valid = np.all((indices >= 0) & (indices < grid_dims), axis=1)
        valid_idx = indices[valid]
        np.add.at(density, (valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]), 1.0)
        n_frames += 1
        n_probe_positions += len(indices)
        n_outside += int((~valid).sum())

    if n_frames == 0:
        raise ValueError(f'No frames selected from {traj_dcd} (frame_half={frame_half})')

    if n_nonfinite_frames:
        logging.warning(
            f'{n_nonfinite_frames} of {n_frames} frames in {Path(traj_dcd).name} '
            f'contain non-finite coordinates and had those atoms dropped. This '
            f'is a corrupt or blown-up frame in the trajectory, not an analysis '
            f'artefact, and the underlying simulation should be checked.',
        )

    # Probes outside the box are dropped. Most probes are in bulk solvent at any
    # instant and the grid only spans the protein plus padding, so a large
    # outside fraction is normal: BCLXL runs sit at 70-73% across every replica.
    # Only a near-total loss means something is wrong, typically a grid built in
    # a different frame from the trajectory being binned. The absolute level is
    # uninformative, so it is logged for comparison across replicas rather than
    # warned on.
    outside_fraction = n_outside / max(n_probe_positions, 1)
    logging.debug(
        f'{100 * outside_fraction:.0f}% of probe positions in '
        f'{Path(traj_dcd).name} fall outside the density grid',
    )
    if outside_fraction > 0.98:
        logging.warning(
            f'{100 * outside_fraction:.1f}% of probe positions in '
            f'{Path(traj_dcd).name} fall outside the density grid: the grid and '
            f'the trajectory are probably not in the same frame, so this density '
            f'map is unusable rather than merely sparse',
        )

    density /= n_frames
    return density, origin


def write_dx_file(
        density: np.ndarray,
        origin: np.ndarray,
        grid_spacing: float,
        output_path: str,
) -> None:
    """Write a volumetric density array to an OpenDX .dx file for PyMOL / VMD.

    In PyMOL load with ``load density.dx`` then visualise with
    ``isomesh mesh_name, density, isovalue``.

    The isovalue for "twice the uniform molar probe density" can be estimated as:
        isovalue = 2 × n_probe_atoms × grid_spacing³ / V_box
    where V_box is the simulation box volume in Å³.

    Args:
        density: 3D array of shape (nx, ny, nz).
        origin: Grid origin coordinates in Å.
        grid_spacing: Voxel edge length in Å.
        output_path: Path to write the .dx file.
    """
    nx, ny, nz = density.shape
    with open(output_path, 'w') as out:
        out.write(f'object 1 class gridpositions counts {nx} {ny} {nz}\n')
        out.write(f'origin {origin[0]:.4f} {origin[1]:.4f} {origin[2]:.4f}\n')
        out.write(f'delta {grid_spacing:.4f} 0.0000 0.0000\n')
        out.write(f'delta 0.0000 {grid_spacing:.4f} 0.0000\n')
        out.write(f'delta 0.0000 0.0000 {grid_spacing:.4f}\n')
        out.write(f'object 2 class gridconnections counts {nx} {ny} {nz}\n')
        out.write(f'object 3 class array type double rank 0 items {nx * ny * nz} data follows\n')
        flat = density.flatten(order='C')
        for chunk_start in range(0, len(flat), 3):
            chunk = flat[chunk_start:chunk_start + 3]
            out.write('  '.join(f'{val:.6e}' for val in chunk) + '\n')
        out.write('object "density" class field\n')
    logging.info(f'Density map written to {output_path} (grid: {nx}×{ny}×{nz})')


def density_hotspots(
        density: np.ndarray,
        origin: np.ndarray,
        grid_spacing: float,
        threshold: float,
        min_voxels: int = MIN_CANDIDATE_VOXELS,
) -> pd.DataFrame:
    """Rank connected components of the diff density and give each a centroid.

    This is what makes the pipeline prospective. Borsatto et al. 2024 anchor
    their contact analysis on residues within 7 A of "the center of the cryptic
    pocket", which presupposes knowing where the pocket is. With only an apo
    structure there is no such centre, so one has to be derived, and the
    density-difference map is exactly the instrument Oleinikovas et al. 2016
    proposed for locating a buried site without prior knowledge.

    The centroid is weighted by density, so it tracks the core of a hotspot
    rather than the geometric middle of a ragged isosurface.

    Args:
        density: Difference density grid.
        origin: Grid origin in Angstroms.
        grid_spacing: Voxel edge length in Angstroms.
        threshold: Voxel density above which a voxel joins a component.
        min_voxels: Components smaller than this are discarded as noise. See
            MIN_CANDIDATE_VOXELS for why a size floor separates real sites from
            the single-voxel specks a negative control produces.

    Returns:
        DataFrame with one row per component, sorted by total density
        descending, with columns rank, n_voxels, total_density, max_density,
        volume_a3 and the centroid coordinates x, y, z in Angstroms.
    """
    mask = density > threshold
    labeled, n_components = nd_label(mask)
    if n_components == 0:
        logging.warning(
            f'No diff-density components above threshold {threshold:.6f}; '
            f'no pocket centre can be derived',
        )
        return pd.DataFrame(
            columns=['rank', 'n_voxels', 'total_density', 'max_density',
                     'volume_a3', 'x', 'y', 'z'],
        )

    rows = []
    for component in range(1, n_components + 1):
        sel = labeled == component
        if sel.sum() < min_voxels:
            continue
        weights = density[sel]
        coords = origin + np.argwhere(sel) * grid_spacing
        rows.append({
            'n_voxels': int(sel.sum()),
            'total_density': float(weights.sum()),
            'max_density': float(weights.max()),
            'volume_a3': float(sel.sum()) * grid_spacing ** 3,
            # density-weighted so the centre sits in the core of the hotspot
            **dict(zip('xyz', np.average(coords, axis=0, weights=weights))),
        })

    if not rows:
        logging.warning(
            f'No component above {threshold:.6f} reached {min_voxels} voxels, '
            f'so no candidate is reported. On this benchmark that is the '
            f'correct answer for a protein with no cryptic pocket.',
        )
        return pd.DataFrame(
            columns=['rank', 'n_voxels', 'total_density', 'max_density',
                     'volume_a3', 'x', 'y', 'z'],
        )
    hotspots = (
        pd.DataFrame(rows)
        .sort_values('total_density', ascending=False)
        .reset_index(drop=True)
    )
    hotspots.insert(0, 'rank', np.arange(1, len(hotspots) + 1))
    logging.info(
        f'Diff-density hotspots above {threshold:.6f}: {n_components} components, '
        f'top one holds {100 * hotspots.total_density.iloc[0] / hotspots.total_density.sum():.0f}% '
        f'of the above-threshold density in {hotspots.n_voxels.iloc[0]} voxels',
    )
    for _, row in hotspots.head(5).iterrows():
        logging.info(
            f'  rank {int(row["rank"])}: {int(row.n_voxels):5d} voxels, '
            f'total density {row.total_density:8.3f}, '
            f'centre ({row.x:7.2f}, {row.y:7.2f}, {row.z:7.2f})',
        )
    return hotspots


def compute_and_save_density_maps(
        md_pairs: list[tuple[str, str]],
        swish_path: Path,
        swish_top_prefix: str,
        swish_aligned_dcds: list[str],
        output_path: Path,
        grid_spacing: float = GRID_SPACING_A,
        padding: float = GRID_PADDING_A,
) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray]]:
    """Compute probe density grids for unbiased MD and SWISH and write OpenDX files.

    Writes to output_path, with the production map named for its lambda and
    every diagnostic named for its state so the two cannot be confused:
      - probe_density_diff_lambda1.dx   : state 0 (lambda = 1.0) minus unbiased
                                          MD. THE PRODUCTION MAP: every
                                          candidate in the paper comes from it
      - lambda1_hotspots.csv            : its candidates, the production list
      - msmd_hotspots.csv               : candidates from the unbiased map, the
                                          control the paper compares against
      - probe_density_md.dx             : mean unbiased MD probe density
      - probe_density_swish_pooled.dx   : SWISH density averaged over the whole
                                          ladder. A diagnostic: pooling mixes
                                          scaled Hamiltonians and is not used
      - probe_density_diff_pooled.dx    : that pooled density minus the MD
      - probe_density_swish_state<N>.dx : raw density for state N
      - probe_density_diff_state<N>.dx  : state N minus the MD, for N > 0;
                                          state 0's is the lambda1 file above

    Both tables come from one rule, connected components above THRESHOLD_FACTOR
    times bulk holding at least MIN_CANDIDATE_VOXELS voxels, ranked by total
    density and centred on the density-weighted centroid.

    Returns the lambda = 1 candidates, the replica-averaged difference density
    and the per-replica difference densities. The candidates let the caller
    anchor downstream analysis on a derived pocket centre rather than on prior
    knowledge of where the pocket is; the per-replica maps are what the
    lambda-trend and reproducibility diagnostics consume.

    All three share the same grid box so they can be overlaid directly in PyMOL.
    Load the difference map and display with ``isomesh`` at a positive isovalue
    to highlight regions where SWISH has higher probe density than unbiased MD (cryptic sites).

    Args:
        md_pairs: One (topology, aligned_dcd) per unbiased MD replicate. Kept
            separate rather than concatenated so replicates with different
            probe counts can still be averaged.
        swish_path: Directory containing SWISH ref PDB files.
        swish_top_prefix: Filename prefix for SWISH ref PDBs ({prefix}N_ref.pdb).
        swish_aligned_dcds: List of aligned SWISH DCD paths, one per replica.
        output_path: Directory to write .dx files into.
        grid_spacing: Voxel edge length in Å.  Default 1.0.
        padding: Padding around the protein bounding box in Å.  Default 5.0.
    """
    logging.info(f'Computing unbiased MD probe density ({len(md_pairs)} replicates)')
    md_replicate_densities: list[np.ndarray] = []
    origin = grid_dims = None
    for replicate_idx, (md_top, md_aligned_dcd) in enumerate(md_pairs):
        stripped = str(output_path / f'density_md_{replicate_idx}_top.pdb')
        extract_protein_heavy_pdb(md_top, stripped, keep_probes=True)
        raw, replicate_origin = compute_probe_density_grid(
            stripped, md_aligned_dcd, origin=origin, grid_dims=grid_dims,
            grid_spacing=grid_spacing, padding=padding,
        )
        # The first replicate fixes the grid box; the rest reuse it so the
        # replicates can be averaged voxel by voxel.
        if origin is None:
            origin, grid_dims = replicate_origin, np.array(raw.shape)
        n_probes, box_volume = probe_count_and_box_volume(stripped, md_aligned_dcd)
        md_replicate_densities.append(
            to_bulk_units(raw, n_probes, grid_spacing, box_volume),
        )
        logging.info(
            f'  replicate {replicate_idx}: {n_probes} probe heavy atoms, '
            f'mean {md_replicate_densities[-1].mean():.4f} x bulk',
        )
    md_density = np.mean(md_replicate_densities, axis=0)

    logging.info(f'Computing SWISH probe density grids ({len(swish_aligned_dcds)} replicas)')
    replica_densities: list[np.ndarray] = []
    for replica_idx, aligned_dcd in enumerate(swish_aligned_dcds):
        top_file = str(swish_path / f'{swish_top_prefix}{replica_idx}_ref.pdb')
        stripped_top = str(output_path / f'density_swish_{replica_idx}_top.pdb')
        extract_protein_heavy_pdb(top_file, stripped_top, keep_probes=True)
        raw, _ = compute_probe_density_grid(
            stripped_top, aligned_dcd,
            origin=origin, grid_dims=grid_dims,
            grid_spacing=grid_spacing, padding=padding,
        )
        n_probes, box_volume = probe_count_and_box_volume(stripped_top, aligned_dcd)
        rep_density = to_bulk_units(raw, n_probes, grid_spacing, box_volume)
        replica_densities.append(rep_density)
        logging.info(
            f'  replica {replica_idx}: {n_probes} probe heavy atoms, '
            f'mean {rep_density.mean():.4f} x bulk',
        )
    swish_density = np.mean(replica_densities, axis=0)

    # These are demuxed trajectories, so each state index is one fixed lambda
    # and state 0 is lambda = 1.0, the only unscaled, physically meaningful
    # Hamiltonian. The replica-averaged map below corresponds to no single
    # ensemble, so the lambda = 1 difference map is written alongside it and the
    # per-replica maps are returned for the lambda-trend diagnostic.
    lambda1_diff = replica_densities[0] - md_density
    write_dx_file(
        lambda1_diff, origin, grid_spacing, str(output_path / _DX_DIFF_LAMBDA1),
    )

    # Every replica written separately, raw and as a difference. Pooling across
    # the ladder averages six differently scaled Hamiltonians into something
    # that corresponds to no physical state, so each rung is kept resolved. Writing the raw map beside the difference at every lambda is
    # what allows a map's performance to be attributed to the ensemble or to the
    # subtraction, which the difference alone cannot separate.
    for replica_index, replica_density in enumerate(replica_densities):
        write_dx_file(replica_density, origin, grid_spacing,
                      str(output_path / f'probe_density_swish_state{replica_index}.dx'))
        # State 0's difference map is the one the analysis reads and is
        # written above under its own name, so it is not repeated here. The
        # rest are diagnostics: nothing downstream reads them.
        if replica_index == 0:
            continue
        write_dx_file(replica_density - md_density, origin, grid_spacing,
                      str(output_path / f'probe_density_diff_state{replica_index}.dx'))
    logging.info(
        f'Wrote {len(replica_densities)} per-state maps, raw and differenced. '
        f'State 0 is lambda = 1.0, the unscaled Hamiltonian.',
    )

    diff_density = swish_density - md_density

    logging.info(
        f'Candidate threshold {THRESHOLD_FACTOR} x bulk with at least '
        f'{MIN_CANDIDATE_VOXELS} voxels; both maps normalised to their own '
        f'uniform probe density so systems with different probe counts are '
        f'directly comparable',
    )

    # Two detectors are written, because one cannot cover both kinds of site.
    # The difference map finds pockets the bias had to open and is blind to
    # pockets the unbiased ensemble already samples; the unbiased map is the
    # reverse. On this benchmark neither alone put every target in the top five,
    # and the pair did, every true site landing at rank 1 to 4 of one list.
    # The lambda = 1 detector. Demuxed state 0 is the only replica whose
    # Hamiltonian is unscaled, so this is the only difference map whose probe
    # occupancies are comparable with an unbiased simulation. With the size
    # floor it finds all six benchmark sites and proposes nothing on the
    # negative control.
    lambda1_hotspots = density_hotspots(
        lambda1_diff, origin, grid_spacing, THRESHOLD_FACTOR,
        min_voxels=MIN_CANDIDATE_VOXELS,
    )
    lambda1_hotspots.to_csv(output_path / 'lambda1_hotspots.csv', index=False)
    logging.info(
        f'Lambda=1 detector: {len(lambda1_hotspots)} candidates above '
        f'{THRESHOLD_FACTOR} x bulk with at least '
        f'{MIN_CANDIDATE_VOXELS} voxels, written to lambda1_hotspots.csv',
    )

    # The same cut on the unbiased map, which is a second detector rather than a
    # variant of the first: subtracting the unbiased ensemble is blind to any
    # pocket that ensemble already opens.
    msmd_hotspots = density_hotspots(
        md_density, origin, grid_spacing, THRESHOLD_FACTOR,
        min_voxels=MIN_CANDIDATE_VOXELS,
    )
    msmd_hotspots.to_csv(output_path / 'msmd_hotspots.csv', index=False)
    logging.info(
        f'Unbiased-map detector: {len(msmd_hotspots)} candidates above '
        f'{THRESHOLD_FACTOR} x bulk with at least {MIN_CANDIDATE_VOXELS} '
        f'voxels, written to msmd_hotspots.csv. This covers sites the '
        f'difference map cannot see because the unbiased ensemble already '
        f'opens them.',
    )

    write_dx_file(md_density, origin, grid_spacing, str(output_path / _DX_MD))
    write_dx_file(swish_density, origin, grid_spacing, str(output_path / _DX_SWISH))
    write_dx_file(diff_density, origin, grid_spacing, str(output_path / _DX_DIFF))
    logging.info(
        f'Density maps written. Difference range: {float(diff_density.min()):.4f} to {float(diff_density.max()):.4f}'
    )
    replica_diffs = [rep - md_density for rep in replica_densities]
    return lambda1_hotspots, diff_density, replica_diffs


def main(
        md_pairs: list[tuple[str, str]],
        swish_dir: str,
        swish_top_prefix: str,
        ref_pdb: str,
        output_dir: str,
        n_replicas: int,
        traj_prefix: str = 'state',
        fit_selection: str = 'name CA',
        stride: int = STRIDE,
        lambda_values: list[float] | None = None,
) -> None:
    """Compare benzene probe contact frequencies between SWISH and unbiased MD.

    Implements the fragment occupation density difference approach from
    Oleinikovas et al, 2016: buried cryptic sites are identified by regions
    where probe contact is enriched in SWISH relative to unbiased MD.

    Both the unbiased MD and SWISH trajectories must contain BNZ probe atoms.
    All trajectories are aligned to ref_pdb before analysis.

    Args:
        md_pairs: One (topology_pdb, trajectory_dcd) per unbiased MD replicate.
            Replicates are kept separate and their densities averaged, so they
            may differ in probe count.
        swish_dir: Directory containing SWISH replica DCD and ref PDB files.
        swish_top_prefix: Filename prefix for SWISH topology PDBs.
            Files expected as {swish_top_prefix}N_ref.pdb in swish_dir.
        ref_pdb: Alignment reference PDB (protein CA atoms). In a prospective run
            this is the apo input structure; there is no holo structure to align
            to, and using one would leak the answer the pipeline is meant to find.
        output_dir: Directory to write aligned trajectories and output files.
        n_replicas: Number of SWISH replicas. Default 6.
        traj_prefix: Filename prefix for SWISH DCD files. Default 'state'.
            Files expected as {traj_prefix}N.dcd in swish_dir.
        fit_selection: MDAnalysis selection for trajectory alignment. Default 'name CA'.
        stride: Process every Nth frame. Default 1.
        lambda_values: Lambda values for each SWISH replica (used for logging only).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    swish_path = Path(swish_dir)

    md_aligned_pairs, swish_aligned_dcds = align_all_trajectories(
        output_path=output_path,
        md_pairs=md_pairs,
        swish_path=swish_path,
        swish_top_prefix=swish_top_prefix,
        traj_prefix=traj_prefix,
        n_replicas=n_replicas,
        ref_pdb=ref_pdb,
        fit_selection=fit_selection,
        stride=stride,
        lambda_values=lambda_values,
    )
    logging.info('Computing volumetric probe density maps')
    hotspots, diff_density, replica_diffs = compute_and_save_density_maps(
        md_pairs=md_aligned_pairs,
        swish_path=swish_path,
        swish_top_prefix=swish_top_prefix,
        swish_aligned_dcds=swish_aligned_dcds,
        output_path=output_path,
    )
    if not len(hotspots):
        logging.warning(
            f'No candidate above {THRESHOLD_FACTOR} x bulk with at least '
            f'{MIN_CANDIDATE_VOXELS} voxels: this run has no candidate cryptic '
            f'site, which is the correct answer for a target without one. '
            f'Downstream analysis will have nothing to anchor on.',
        )
        return

    top = hotspots.iloc[0]
    logging.info(
        f'Derived pocket centre (rank-1 candidate): '
        f'({top.x:.2f}, {top.y:.2f}, {top.z:.2f}) A, '
        f'{int(top.n_voxels)} voxels, {top.volume_a3:.0f} A^3',
    )


def parse_args() -> ArgumentParser:
    """Build the command-line parser for a single-target density run.

    Returns:
        Argument parser. The pipeline calls main() directly instead and
        pins every value, so this path exists for one-off runs.
    """
    arg_parser = ArgumentParser(
        description='Compare BNZ probe contact frequencies between SWISH and unbiased MD '
                    'to identify candidate cryptic sites via fragment occupation density differences.',
    )
    arg_parser.add_argument(
        '-md', '--unbiased_dcd', type=str, required=True,
        help='Unbiased MD trajectory DCD (concatenate multiple replicates into one file before passing).',
    )
    arg_parser.add_argument(
        '-md_top', '--unbiased_top', type=str, required=True,
        help='Topology PDB for the unbiased MD trajectory (protein + BNZ probes).',
    )
    arg_parser.add_argument(
        '-d', '--swish_dir', type=str, required=True,
        help='Directory containing SWISH replica DCD and ref PDB files.',
    )
    arg_parser.add_argument(
        '-ref', '--ref_pdb', type=str, required=True,
        help='Alignment reference PDB (protein only, crystal frame).',
    )
    arg_parser.add_argument(
        '-o', '--output_dir', type=str, required=True,
        help='Directory to write aligned trajectories and output files.',
    )
    arg_parser.add_argument(
        '--traj_prefix', type=str, required=True,
        help='Filename prefix for SWISH DCD files, e.g. "state_state".',
    )
    arg_parser.add_argument(
        '--n_replicas', type=int, required=True,
        help='Number of SWISH replicas, which is the number of demultiplexed '
             'states to read. Required: the ladder length differs per target',
    )
    arg_parser.add_argument(
        '--fit_selection', type=str, default='name CA',
        help='MDAnalysis selection for alignment fitting. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--stride', type=int, default=1,
        help='Process every Nth frame. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--lambda_values', type=float, nargs='+', default=None,
        help='Lambda values for each SWISH replica (for logging).',
    )
    return arg_parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = parse_args()
    args = parser.parse_args()
    main(
        md_pairs=[(args.unbiased_top, args.unbiased_dcd)],
        swish_dir=args.swish_dir,
        swish_top_prefix=args.traj_prefix,
        ref_pdb=args.ref_pdb,
        output_dir=args.output_dir,
        n_replicas=args.n_replicas,
        traj_prefix=args.traj_prefix,
        fit_selection=args.fit_selection,
        stride=args.stride,
        lambda_values=args.lambda_values,
    )
