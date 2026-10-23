"""Per-residue and per-region identification ranks, for the SI ranking figure.

Every named residue is ranked, not only the aromatic ones: a site can be gated
by backbone movement, as TEM-1's is, whose rank-1 hotspot has twenty residues
within 7 A of which none is aromatic, so an aromatics-only figure cannot show
such a target at all. Rank is reported rather than a count, because the count
loses the thing the figure is for, which is at what rank each named residue was
reached.

This recomputes both ranks from the candidate lists.

A residue's rank is the position of the first candidate that reaches it, where
reach is the median over the lambda = 1 trajectory of the residue's closest
heavy-atom distance to the candidate centre, within RESIDUE_A. That is the
number a user acting on a ranked list actually experiences, since they inspect
candidates in order, and it is the same test that decides a candidate's reported
environment, so the rank always names a candidate whose environment contains the
residue. A region's rank is the position of the first candidate within HIT_A of
the held-out pocket reference, which is what the benchmark already reports.

Each named residue is then annotated with how it behaves on the unscaled
ensemble, which is the other half of the claim. Three classes are distinguished
because they are different observations and a residue needs only one: aromatic
rotamer switching is a rotamer change on an aromatic side chain, displacement is
travel of the side chain without a rotamer change, and some residues do both.
The classes name what is measured rather than a mechanism, because a residue can
switch rotamer throughout the run without its rotamer state differing between
open and closed frames. Both are taken as
ranks within the residue's own protein rather than as absolute cuts, because no
absolute cut discriminates: at 251 frames of lambda = 1 sampling, 94 per cent of
residues travel at least 2 A and 75 per cent switch rotamer at least once.

The detector is the production one, a flat 6x bulk cut with a 3-voxel floor on
the lambda = 1 difference map, which is the only configuration that finds all
six benchmark sites with no ubiquitin candidates. That map, named in DENSITY
below, is the only density this script reads. The density directory also holds
the replica-averaged difference, the unbiased map and the per-replica
diagnostics; none of those is the production detector and none should be
substituted here.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import mdtraj
import numpy as np
import pandas as pd
from gridData import Grid
from scipy.ndimage import label as nd_label


from sandcat.analysis.experiments import (target_dir,
                         DIFFERENCE_MAP, STATE_TOPOLOGY, apo_residues,
                         closest_approach, distance_trace,
                         STATE_TRAJECTORY,
                         ANALYSIS, HIT_A, RESIDUE_A, SIM_APO, POCKET_REFERENCES,
                         heavy_atoms_by_resseq,
                         TARGET_RESIDUES, true_pocket)


# the production detector: SWISH-X state 0 minus MSMD, the only difference
# map built from the unscaled Hamiltonian
DENSITY = DIFFERENCE_MAP
from sandcat.analysis.residue_motion import motion_table

warnings.filterwarnings('ignore')

from sandcat.analysis.compute_probe_density import (  # noqa: E402
    MIN_CANDIDATE_VOXELS as MIN_VOXELS, THRESHOLD_FACTOR as THRESHOLD)
from sandcat.analysis.pocket_residue_dynamics import AROMATICS  # noqa: E402


def candidates(density, origin, spacing, low=THRESHOLD, min_voxels=MIN_VOXELS):
    """The production detector: flat cut, size floor, ranked by total density.
    Args:
        density: Difference density grid in bulk units.
        origin: (3,) grid origin in Angstroms.
        spacing: Voxel edge length in Angstroms.
        low: Isolevel as a multiple of bulk.
        min_voxels: Smallest component that may be reported.

    Returns:
        One row per candidate, ranked by total density, with the voxel count and
        the density-weighted centroid.
    """
    labelled, n = nd_label(density > low)
    rows = []
    for component in range(1, n + 1):
        selected = labelled == component
        if selected.sum() < min_voxels:
            continue
        weights = density[selected]
        centre = np.average(origin + np.argwhere(selected) * spacing, axis=0,
                            weights=np.clip(weights, 1e-9, None))
        rows.append({'total': float(weights.sum()), 'x': centre[0],
                     'y': centre[1], 'z': centre[2]})
    if not rows:
        return np.empty((0, 3))
    table = pd.DataFrame(rows).sort_values('total', ascending=False)
    return table[['x', 'y', 'z']].to_numpy()



def first_candidate_near_coordinates(centres: np.ndarray, atoms: np.ndarray,
                                     cutoff_a: float):
    """1-based rank of the first candidate within cutoff_a of a fixed structure.

    Measured on one set of coordinates, not over the trajectory, which is what
    matching a candidate to a held-out pocket reference needs.

    Args:
        centres: (candidates, 3) candidate centres, in rank order.
        atoms: (atoms, 3) coordinates to measure to, a pocket or a residue.
        cutoff_a: Distance in Angstroms that counts as reaching.

    Returns:
        The 1-based rank of the first candidate within cutoff_a, or None if
        none of them reaches.
    """
    for index, centre in enumerate(centres, start=1):
        if closest_approach(atoms, centre) <= cutoff_a:
            return index
    return None


def first_candidate_reaching_residue(centres: np.ndarray, xyz: np.ndarray,
                                     heavy: list[int], cutoff_a: float):
    """1-based rank of the first candidate the residue reaches, or None.

    Reach is the median over the trajectory of the residue's closest
    heavy-atom distance to the candidate centre, the same test
    plot_candidate_residues uses to decide a candidate's environment, so the
    rank always names a candidate whose reported environment holds the residue.

    Args:
        centres: (candidates, 3) candidate centres, in rank order.
        xyz: (frames, atoms, 3) trajectory coordinates in Angstroms, scaled
            once by the caller rather than per residue.
        heavy: Heavy-atom indices of the residue.
        cutoff_a: Median distance in Angstroms that counts as reaching.

    Returns:
        The 1-based rank of the first candidate reached, or None.
    """
    if not heavy:
        return None
    for index, centre in enumerate(centres, start=1):
        median = float(np.median(distance_trace(xyz, heavy, centre)))
        if median <= cutoff_a:
            return index
    return None


def motion_class(row) -> str:
    """How this residue behaves: aromatic rotamer switching, displacement, or neither.

    The labels name what is measured rather than what it is assumed to cause.
    An aromatic side chain may switch rotamer throughout the trajectory without
    its rotamer state differing between the open and closed ends of the opening
    coordinate, so membership of the first class establishes mobility and not a
    contribution to pocket opening.

    The split is at the median of the residue's own protein, so a class says
    the residue moves more than a typical residue of the same protein. A top
    quartile cut would instead say it is exceptionally mobile, which is a
    statement about the tail rather than about how each residue behaves.

    Args:
        row: One residue's row, carrying flip_rank, span_rank, n_residues and
            resname. The ranks are judged against the residue's own protein
            rather than an absolute scale.

    Returns:
        'aromatic rotamer switching' for an aromatic in the upper half by
        dwell-filtered switching, 'side-chain displacement' for anything else in
        the upper half by either measure, 'neither' otherwise.
    """
    flips = row.flip_rank <= row.n_residues / 2.0
    moves = row.span_rank <= row.n_residues / 2.0
    # a partition, not overlapping labels: the aromatic class is the more
    # specific observation so it wins where a residue satisfies both
    if flips and row.resname in AROMATICS:
        return 'aromatic rotamer switching'
    if moves or flips:
        return 'side-chain displacement'
    return 'neither'


def main() -> None:
    """Rank every target's candidates and residues, writing the two csv tables.

    Writes ranking_residues.csv, one row per named residue with the rank
    the blind ordering gave it, and ranking_regions.csv, one row per target
    per map with the rank of the candidate matching the held-out pocket.
    """
    residue_rows, region_rows = [], []
    for target in POCKET_REFERENCES:
        folder = target_dir(target) / 'density'
        if not (folder / DENSITY).exists():
            continue
        coords = apo_residues(str(SIM_APO[target]))
        motion = motion_table(folder)
        motion['n_residues'] = len(motion)
        motion = motion.set_index('resSeq')
        pocket = true_pocket(target, str(SIM_APO[target]))

        grid = Grid(str(folder / DENSITY))
        centres = candidates(grid.grid, np.array(grid.origin),
                             float(grid.delta[0]))
        region_rows.append({
            'target': target, 'source': 'diff_l1',
            'n_candidates': len(centres),
            'region_rank': (first_candidate_near_coordinates(
                                centres, pocket, HIT_A)
                            if pocket is not None and len(centres) else None)})

        trajectory = mdtraj.load(str(folder / STATE_TRAJECTORY),
                                 top=str(folder / STATE_TOPOLOGY))
        heavy_atoms = heavy_atoms_by_resseq(trajectory.topology)
        # scaled once per target, not once per residue
        xyz = trajectory.xyz * 10.0

        for resid in TARGET_RESIDUES.get(target, []):
                if resid not in coords:
                    continue
                rank = (first_candidate_reaching_residue(
                            centres, xyz, heavy_atoms.get(resid, []),
                            RESIDUE_A)
                        if len(centres) else None)
                row = motion.loc[resid] if resid in motion.index else None
                residue_rows.append({
                    'target': target, 'resSeq': resid,
                    'resname': row.resname if row is not None else '?',
                    'rank': rank, 'n_candidates': len(centres),
                    'flips': int(row.flips) if row is not None else 0,
                    'span_a': float(row.span_a) if row is not None else 0.0,
                    'motion': motion_class(row) if row is not None else 'neither'})
        print(f'  {target}: done')

    residues = pd.DataFrame(residue_rows)
    regions = pd.DataFrame(region_rows)
    residues.to_csv(ANALYSIS / 'ranking_residues.csv', index=False)
    regions.to_csv(ANALYSIS / 'ranking_regions.csv', index=False)

    found = residues['rank'].notna()
    print(f'\nnamed residues: {len(residues)}, identified {int(found.sum())}, '
          f'of which top 5 {int((residues["rank"] <= 5).sum())}')
    print(residues.motion.value_counts().to_string())
    print('\nwritten: ranking_residues.csv, ranking_regions.csv')


if __name__ == '__main__':
    main()
