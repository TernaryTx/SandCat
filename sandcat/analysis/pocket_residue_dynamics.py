"""Side-chain dynamics of residues lining a blindly-derived cryptic pocket.

This replaces two things that did not survive benchmarking.

The residue selection could be taken from ``pocket_accessibility.pdb``, keeping
residues above 7.5 per cent of the maximum per-residue enrichment. On the eight
September 2026 targets that maximum is exactly 0.000 for four of them, because
SWISH-X shows lower probe contact than unbiased MSMD at 77 to 100 per cent of
residues, so the selection returned nothing. Measured against random expectation
the surviving selection was no better than chance on six of seven targets. Here
residues are instead taken from the density-derived pocket centre, which is the
same anchor Borsatto et al. 2024 use ("within 7 A of the center of the cryptic
pocket") except that the centre is computed from the difference map rather than
supplied, so nothing about the answer is assumed.

A ring analysis could instead compare each cluster representative to the
reference and reported the angle between ring-plane normals. Two limits made
that hard to act on. It is static, so it can say a ring moved but not whether
the residue is bistable, when it switched, or whether switching accompanies the
pocket opening. And it used the absolute cosine, which correctly treats the
symmetry-degenerate 180-degree rotation of Phe and Tyr as no change, but which
therefore measures ring reorientation rather than flipping, despite the name.

Per-frame chi1 and chi2 across the demuxed trajectories give the distribution
instead of a single angle, and a residue occupying two populated rotamer states
is the thing worth reporting: a gate, rather than a residue that is merely
flexible.
"""
from __future__ import annotations

import logging

import mdtraj
import numpy as np
import pandas as pd

# Aromatics only: these are the residues whose ring position gates access to a
# buried site, and the ones the SWISH literature highlights.
AROMATICS = ('PHE', 'TYR', 'TRP', 'HIS', 'HIE', 'HID', 'HIP')

# Twofold symmetric about chi2, so chi2 and chi2 + 180 are the same structure
# and folding them onto [0, 180) stops one physical state being counted as two.
# Phe and Tyr because the ring is symmetric; Asp because chi2 rotates a
# carboxylate whose two oxygens are equivalent, exactly as glutamate's chi3
# does below. Asparagine's chi2 looks identical in atom names but turns an
# amide, which is NOT symmetric, so the two must not be lumped together.
SYMMETRIC_CHI2 = ('PHE', 'TYR', 'ASP')

# chi3 of glutamate rotates a carboxylate, whose two oxygens are equivalent, so
# it is twofold symmetric in the same way a Phe ring is. Glutamine's chi3 looks
# identical in atom names but turns an amide, which is NOT symmetric, so the two
# must not be lumped together. Arg, Lys and Met chi3 carry no symmetry.
SYMMETRIC_CHI3 = ('GLU',)

# Probes and solvent are present in these topologies and must never be treated
# as pocket residues: a benzene diffuses freely and spans 180 to 215 A over a
# trajectory, which swamps every real side chain when residues are ranked by
# how far they move.
from sandcat.analysis.experiments import (  # noqa: E402
    DWELL, NON_HEAVY_ELEMENTS, NON_PROTEIN, closest_approach,
    distance_trace)


def residues_near_point(
        structure: mdtraj.Trajectory,
        centre: np.ndarray,
        radius_a: float,
        aromatic_only: bool = False,
) -> pd.DataFrame:
    """Residues with any heavy atom within ``radius_a`` of a point.

    Named for what it measures rather than for a pocket, because a second
    function of the same name elsewhere took a point cloud and a different
    cutoff. The radius_a has no default: the caller states which shell it means.

    Args:
        structure: Single-frame trajectory holding the reference structure.
        centre: The point to measure from, in Angstroms.
        radius_a: Inclusion distance in Angstroms, stated by the caller.
        aromatic_only: Keep only residues whose rings can gate access.

    Returns:
        DataFrame with residue index, resSeq, resname and closest approach.
    """
    xyz = structure.xyz[0] * 10.0
    rows = []
    for residue in structure.topology.residues:
        if residue.name in NON_PROTEIN:
            continue
        heavy = [a.index for a in residue.atoms if a.element.symbol not in NON_HEAVY_ELEMENTS]
        if not heavy:
            continue
        distance = closest_approach(xyz[heavy], centre)
        if distance > radius_a:
            continue
        if aromatic_only and residue.name not in AROMATICS:
            continue
        rows.append({'residue_index': residue.index, 'resSeq': residue.resSeq,
                     'resname': residue.name, 'distance_a': round(distance, 2)})
    result = pd.DataFrame(rows).sort_values('distance_a').reset_index(drop=True) \
        if rows else pd.DataFrame(columns=['residue_index', 'resSeq', 'resname', 'distance_a'])
    # An infinite radius means the caller wants every residue and will narrow
    # the set itself, which is what the apo-driven path does. Naming a distance
    # there would print "within inf A", so that case says what it means.
    shell = ('Every residue in the structure' if np.isinf(radius_a) else
             f'Residues within {radius_a:g} A of '
             f'({centre[0]:.1f}, {centre[1]:.1f}, {centre[2]:.1f})')
    logging.info(
        f'{shell}: {len(result)}' + (' aromatic' if aromatic_only else ''),
    )
    return result


def chi_angles(
        trajectory: mdtraj.Trajectory,
        residue_indices: list[int],
) -> pd.DataFrame:
    """Per-frame chi1, chi2 and chi3 in degrees for the given residues.

    chi2 is folded onto [0, 180) for the residues in SYMMETRIC_CHI2, the Phe
    and Tyr rings and the Asp carboxylate, and chi3 for those in
    SYMMETRIC_CHI3, so that one physical rotamer is not split across two
    apparent states.

    Args:
        trajectory: Trajectory to measure, already loaded.
        residue_indices: Topology residue indices to report.

    Returns:
        Long-format DataFrame with frame, residue_index, resname, chi1, chi2,
        chi3. chi3 is NaN for residues whose side chain is too short to have one.
    """
    wanted = set(residue_indices)
    chi1_idx, chi1 = mdtraj.compute_chi1(trajectory)
    chi2_idx, chi2 = mdtraj.compute_chi2(trajectory)
    chi3_idx, chi3 = mdtraj.compute_chi3(trajectory)

    def residue_of(atom_indices):
        """The residue an angle's atom quadruple belongs to.

        Args:
            atom_indices: Atom indices defining one dihedral.

        Returns:
            The mdtraj residue owning the first of those atoms.
        """
        return trajectory.topology.atom(int(atom_indices[0])).residue

    columns = {}
    for idx, values, name in ((chi1_idx, chi1, 'chi1'), (chi2_idx, chi2, 'chi2'),
                              (chi3_idx, chi3, 'chi3')):
        for column, atoms in enumerate(idx):
            residue = residue_of(atoms)
            if residue.index not in wanted:
                continue
            degrees = np.degrees(values[:, column])
            if name == 'chi2' and residue.name in SYMMETRIC_CHI2:
                degrees = np.mod(degrees, 180.0)
            if name == 'chi3' and residue.name in SYMMETRIC_CHI3:
                degrees = np.mod(degrees, 180.0)
            columns[(residue.index, residue.name, name)] = degrees

    if not columns:
        return pd.DataFrame(
            columns=['frame', 'residue_index', 'resname', 'chi1', 'chi2', 'chi3'])

    n_frames = trajectory.n_frames
    records = []
    residues = {(i, n) for i, n, _ in columns}
    for residue_index, resname in sorted(residues):
        records.append(pd.DataFrame({
            'frame': np.arange(n_frames),
            'residue_index': residue_index,
            'resname': resname,
            'chi1': columns.get((residue_index, resname, 'chi1'), np.full(n_frames, np.nan)),
            'chi2': columns.get((residue_index, resname, 'chi2'), np.full(n_frames, np.nan)),
            'chi3': columns.get((residue_index, resname, 'chi3'), np.full(n_frames, np.nan)),
        }))
    return pd.concat(records, ignore_index=True)



def dwell_filtered_switches(labels: np.ndarray, dwell: int) -> int:
    """Transitions in a state series, ignoring runs shorter than ``dwell``.

    Args:
        labels: State index per frame.
        dwell: Frames a state must be held before the move counts.

    Returns:
        Number of transitions surviving the filter.
    """
    if labels.size <= 1:
        return 0
    settled, current, run = labels.copy(), labels[0], 1
    for i in range(1, labels.size):
        run = run + 1 if labels[i] == labels[i - 1] else 1
        if run >= dwell:
            current = labels[i]
        settled[i] = current
    return int((np.diff(settled) != 0).sum())


def rotamer_switches(angles: np.ndarray, period: float,
                     dwell: int = DWELL) -> int:
    """Transitions between rotamer bins, ignoring flicker shorter than dwell.

    Bins the dihedral on fixed edges and defers to dwell_filtered_switches, so
    the rule that decides what counts as a transition is written once and both
    counters in the analysis obey the same one.

    Args:
        angles: Dihedral angle per frame, in degrees, NaN where undefined.
        period: Rotational period in degrees. 360 for an asymmetric dihedral,
            180 where the terminal group is twofold symmetric.
        dwell: Frames a state must be held before the move counts. Filters the
            flicker across a basin boundary that would otherwise dominate.

    Returns:
        Number of transitions surviving the dwell filter.
    """
    if angles.size == 0 or not np.isfinite(angles).any():
        return 0
    if period == 180.0:
        folded = np.mod(angles, 180.0)
        edges = np.array([60.0, 120.0])
    else:
        folded = np.mod(angles + 180.0, 360.0) - 180.0
        edges = np.array([-120.0, 0.0, 120.0])
    raw = np.digitize(folded, edges)
    # The angle is circular, so the first and last bins meet at the wrap and
    # hold one rotamer between them: on the 360 period they are the two halves
    # of trans, and 179 and -179 are two degrees apart rather than in different
    # states. Merging them before the dwell filter is what stops a side chain
    # jittering about 180 from reporting a switch on every crossing; the filter
    # cannot remove those, since each excursion is genuinely held.
    raw[raw == len(edges)] = 0
    return dwell_filtered_switches(raw, dwell)


def rotamer_states(
        angles: np.ndarray,
        period: float = 360.0,
        n_bins: int = 36,
        min_occupancy: float = 0.1,
) -> dict:
    """Describe a dihedral time series as populated states and switches.

    A residue that gates a pocket should occupy two or more rotamer states with
    real occupancy and switch between them. One that merely wobbles occupies a
    single broad basin. Counting populated histogram peaks separates those
    without assuming how many states there should be.

    Args:
        angles: Dihedral values in degrees.
        period: 360 normally, 180 for a symmetry-folded chi2.
        n_bins: Histogram resolution.
        min_occupancy: Fraction of frames a bin needs to count as populated.

    Returns:
        Dict with n_states, occupancies, n_transitions and switch_rate.
    """
    finite = angles[np.isfinite(angles)]
    if finite.size == 0:
        return {'n_states': 0, 'occupancies': [], 'n_transitions': 0, 'switch_rate': 0.0}

    counts, edges = np.histogram(np.mod(finite, period), bins=n_bins, range=(0.0, period))
    fractions = counts / counts.sum()

    # A single basin spans several bins, so thresholding bins individually
    # reports one rotamer as three states. Group contiguous occupied bins into
    # basins first, wrapping around the period since the angle is circular, and
    # only then ask whether a basin holds enough frames to count.
    occupied = counts > 0
    basin_of_bin = np.full(n_bins, -1)
    basin = 0
    for b in range(n_bins):
        if not occupied[b]:
            continue
        if b > 0 and occupied[b - 1]:
            basin_of_bin[b] = basin_of_bin[b - 1]
        else:
            basin_of_bin[b] = basin
            basin += 1
    # join the ends: 359 degrees and 1 degree are neighbours
    if occupied[0] and occupied[-1] and basin_of_bin[0] != basin_of_bin[-1]:
        basin_of_bin[basin_of_bin == basin_of_bin[-1]] = basin_of_bin[0]

    occupancy = {b: float(fractions[basin_of_bin == b].sum())
                 for b in set(basin_of_bin[basin_of_bin >= 0])}
    kept = {b for b, f in occupancy.items() if f >= min_occupancy}

    bin_of = np.clip(np.digitize(np.mod(finite, period), edges) - 1, 0, n_bins - 1)
    state_of_bin = np.array([
        sorted(kept).index(b) if b in kept else -1 for b in basin_of_bin
    ])
    labels = state_of_bin[bin_of]
    current, cleaned = -1, []
    for label in labels:
        if label >= 0:
            current = label
        cleaned.append(current)
    cleaned = np.array(cleaned)
    valid = cleaned >= 0
    # Same dwell rule as the switch count the classification uses: a state must
    # be held for DWELL frames before the move counts, so flicker across a
    # basin boundary is not a transition. Unfiltered, this counted several
    # times as many switches as the filtered measure reported beside it.
    n_transitions = dwell_filtered_switches(cleaned[valid], DWELL)

    return {
        'n_states': len(kept),
        'occupancies': [round(occupancy[b], 3) for b in sorted(kept)],
        'n_transitions': n_transitions,
        'switch_rate': round(n_transitions / max(finite.size, 1), 4),
    }


def residue_pocket_distance(
        trajectory: mdtraj.Trajectory,
        residue_indices: list[int],
        centre: np.ndarray,
) -> pd.DataFrame:
    """Per-frame closest approach of each residue to the pocket centre.

    Ring rotamers are only one way a cryptic site opens. TEM-1's main site is
    gated by backbone movement, and its rank-1 density hotspot has sixteen
    residues within 7 A of which none is aromatic, so a ring-flip analysis has
    nothing to measure there. Distance from the residue to the pocket centre
    does not care which degree of freedom moved: a ring swinging out of the
    cavity and a loop peeling back both increase it, so one measure covers both
    mechanisms and the bistability test is the same in each case.

    The centre is fixed in the aligned frame, so this is read relative to the
    reference structure the trajectories were aligned to.

    Args:
        trajectory: Aligned trajectory.
        residue_indices: Topology residue indices to track.
        centre: Pocket centre in Angstroms.

    Returns:
        Long-format DataFrame with frame, residue_index, resname, distance_a.
    """
    wanted = {r.index: r for r in trajectory.topology.residues
              if r.index in set(residue_indices)}
    if not wanted:
        return pd.DataFrame(columns=['frame', 'residue_index', 'resname', 'distance_a'])
    xyz = trajectory.xyz * 10.0
    records = []
    for index, residue in wanted.items():
        heavy = [a.index for a in residue.atoms if a.element.symbol not in NON_HEAVY_ELEMENTS]
        if not heavy:
            continue
        distances = distance_trace(xyz, heavy, centre)
        records.append(pd.DataFrame({
            'frame': np.arange(trajectory.n_frames),
            'residue_index': index, 'resname': residue.name,
            'distance_a': distances,
        }))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def linear_states(
        values: np.ndarray,
        n_bins: int = 30,
        min_occupancy: float = 0.1,
) -> dict:
    """States and switches for a non-periodic observable such as a distance.

    Same reasoning as rotamer_states but without wrapping: contiguous occupied
    bins are merged into one basin before occupancy is tested, so a single broad
    distribution is one state rather than several.

    Args:
        values: Observable per frame.
        n_bins: Histogram resolution.
        min_occupancy: Fraction of frames a basin needs to count.

    Returns:
        Dict with n_states, occupancies, n_transitions, switch_rate and range.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {'n_states': 0, 'occupancies': [], 'n_transitions': 0,
                'switch_rate': 0.0, 'span_a': 0.0}
    counts, edges = np.histogram(finite, bins=n_bins)
    fractions = counts / counts.sum()
    occupied = counts > 0

    basin_of_bin = np.full(n_bins, -1)
    basin = 0
    for b in range(n_bins):
        if not occupied[b]:
            continue
        if b > 0 and occupied[b - 1]:
            basin_of_bin[b] = basin_of_bin[b - 1]
        else:
            basin_of_bin[b] = basin
            basin += 1

    occupancy = {b: float(fractions[basin_of_bin == b].sum())
                 for b in set(basin_of_bin[basin_of_bin >= 0])}
    kept = sorted(b for b, f in occupancy.items() if f >= min_occupancy)
    bin_of = np.clip(np.digitize(finite, edges) - 1, 0, n_bins - 1)
    state_of_bin = np.array([kept.index(b) if b in kept else -1 for b in basin_of_bin])
    labels = state_of_bin[bin_of]
    current, cleaned = -1, []
    for label in labels:
        if label >= 0:
            current = label
        cleaned.append(current)
    cleaned = np.array(cleaned)
    valid = cleaned >= 0
    # Same dwell rule as the switch count the classification uses: a state must
    # be held for DWELL frames before the move counts, so flicker across a
    # basin boundary is not a transition. Unfiltered, this counted several
    # times as many switches as the filtered measure reported beside it.
    n_transitions = dwell_filtered_switches(cleaned[valid], DWELL)
    return {
        'n_states': len(kept),
        'occupancies': [round(occupancy[b], 3) for b in kept],
        'n_transitions': n_transitions,
        'switch_rate': round(n_transitions / max(finite.size, 1), 4),
        'span_a': round(float(finite.max() - finite.min()), 2),
    }
