"""End-to-end: rank cryptic-site candidates, then explain how each one opens.

What this does, in order, and why each step is shaped the way it is. Every
choice below came out of benchmarking eight targets blind against held-out
references, so the reasoning is recorded rather than assumed.

1. Call the protein positive or negative for a buried cryptic site, from the
   volume of difference density above twice bulk. Ubiquitin, which has no
   cryptic pocket, has 227 such voxels against 740 to 1828 for seven real
   targets, a 2.2x gap. Its peak density is entirely normal, so intensity and
   candidate count cannot separate it and extent is what does.

2. Rank candidate sites. The density locates sites well but ranks them poorly,
   and no statistic computed on the density fixes that, so candidates are
   reported as a short list to inspect rather than a single answer.

3. For each candidate, describe how it opens. The gating residue is the one
   with the largest excursion at that site, not necessarily one that switches
   between two states. TEM-1 opens by LEU220 backbone movement spanning 17.6 A
   at lambda = 1 and 40.9 A under bias; CRBN's W264 sweeps without ever
   resolving into two basins, matching the paper describing it as "movement of
   W264" required to accommodate the ligand; Bcl-xL's TYR185 does switch
   discretely, 8 times at lambda = 1. So span is the primary
   ordering and bistability a secondary flag, and both are reported per state
   because discrete states appear only at particular lambda.

Residues are taken from the candidate centre in the aligned frame. The topology
PDB is not aligned: for CRBN it sits 103 A from its own density, so measuring
against it silently returns no residues at all.
"""
from __future__ import annotations

import logging
from pathlib import Path

import mdtraj
import numpy as np
import pandas as pd

from sandcat.analysis.compute_probe_density import THRESHOLD_FACTOR
from sandcat.analysis.experiments import (DIFFERENCE_MAP, GATING_RADIUS_A,
                                          bundle_into_analysis,
                                          HOTSPOTS_FILE,
                                          closest_approach,
                                          apo_residues,
                                          NON_PROTEIN,
                                          kabsch,
                                          N_CANDIDATES, analysis_relative,
                                          pml_preamble,
                                          residue_distance_traces)
from sandcat.analysis.pocket_residue_dynamics import (
    AROMATICS, SYMMETRIC_CHI2, SYMMETRIC_CHI3, chi_angles, linear_states,
    residues_near_point,
    residue_pocket_distance, rotamer_states,
)

# Percentile of the opening coordinate taken as the open state. A percentile
# rather than the maximum, so the frame stands for a slice of the trajectory
# instead of a single excursion.
OPEN_PCT = 95.0
# Residues whose displacement defines a candidate's opening coordinate. Quoted
# in the Methods as "the six with the largest range of closest approach".
N_DRIVERS = 6
# How close two candidate centres must be, after mapping one protomer onto
# another, to be called the same site on different chains.
SYMMETRY_TOLERANCE_A = 6.0
# Span on state 0 at or above which a residue is flagged as a large mover in
# the run log. A label only: the driver selection is by rank, not by this cut.
LARGE_MOVEMENT_A = 5.0


def apo_pocket_resseqs(apo_pdb: str, centre: np.ndarray,
                       radius_a: float) -> set[int]:
    """resSeq values within radius_a of the centre in the apo input structure.

    Deliberately the apo and not a trajectory median, which is the measure the
    rest of the prospective analysis uses. The residues selected here are
    ranked by how far they move, so selecting them on the trajectory would
    choose on the same quantity it then ranks by. Membership on the apo breaks
    that circularity: the pool is fixed before any motion is measured.

    It also keeps the pool tight, which matters because ranking on motion alone
    rewards mobility regardless of proximity. A wider or trajectory-based shell
    admits residues that are never near the site: HPV-11 E2 GLU39, 9.7 A away
    in the apo and 9.2 A by median, enters at 10 A and displaces GLN95 at
    7.2 A, changing the frame that is chosen.

    Membership is judged on the structure the simulation started from, not on
    the trajectory's first frame. The apo input defines the coordinate frame
    every density map is in, it does not move if the equilibration is repeated,
    and it is what a prospective user holds before any simulation is run.

    Args:
        apo_pdb: Path to the apo input structure.
        centre: (3,) candidate centre in Angstroms.
        radius_a: Inclusion distance in Angstroms.

    Returns:
        Simulation resSeq values with a heavy atom inside that radius_a.
    """
    return {resseq for resseq, xyz in apo_residues(apo_pdb).items()
            if closest_approach(xyz, centre) <= radius_a}


def describe_candidate(
        trajectories: list[tuple[int, mdtraj.Trajectory]],
        centre: np.ndarray,
        apo_resseqs: set[int],
) -> pd.DataFrame:
    """How each residue at one candidate site behaves, per state.

    Args:
        trajectories: (state index, aligned trajectory) pairs.
        centre: Candidate centre in Angstroms, from the density map.
        apo_resseqs: The residues at this candidate, decided on the apo input.
            Indices still come from the trajectory; only which residues count
            comes from the apo, which does not move between runs.

    Returns:
        One row per residue per state, with span, states and switches.
    """
    rows = []
    for state, trajectory in trajectories:
        # Membership comes from the apo; only the indices come from the
        # trajectory, so the shell is read with no radius and then narrowed to
        # the apo's answer. Measuring a frame-0 shell here as well would be
        # work thrown away, since the apo decides either way.
        residues = residues_near_point(trajectory[0], centre, radius_a=np.inf)
        residues = residues[residues.resSeq.isin(apo_resseqs)]
        if residues.empty:
            continue
        distances = residue_pocket_distance(
            trajectory, residues.residue_index.tolist(), centre,
        )
        angles = chi_angles(trajectory, residues.residue_index.tolist())
        meta = residues.set_index('residue_index')
        for residue_index, group in distances.groupby('residue_index'):
            moved = linear_states(group.distance_a.to_numpy())
            name = meta.resname[residue_index]
            chi = angles[angles.residue_index == residue_index]
            # chi1, chi2 and chi3 are all tested, each on its own period.
            # Folding chi2 for a symmetric group makes a ring reorientation
            # driven by chi1 invisible in chi2, so chi1 is always read on the
            # full 360 degree period and the three are summed.
            if len(chi):
                # chi_angles has already folded the twofold-symmetric
                # dihedrals onto [0, 180), so rotamer_states has to be told the
                # same period or it spreads a 180 degree range over a 360
                # degree histogram and halves the resolution.
                chi1 = rotamer_states(chi.chi1.to_numpy(), period=360.0)
                chi2 = rotamer_states(
                    chi.chi2.to_numpy(),
                    period=180.0 if name in SYMMETRIC_CHI2 else 360.0)
                # chi3 exists only for Arg, Lys, Met, Glu and Gln and moves
                # the terminal group of a long side chain: TEM1's ARG222
                # switches it 11 times at its own site against 4 and 2 on chi1
                # and chi2, so omitting it would call that residue static.
                chi3 = rotamer_states(
                    chi.chi3.to_numpy(),
                    period=180.0 if name in SYMMETRIC_CHI3 else 360.0)
            else:
                chi1 = chi2 = chi3 = {'n_states': 0, 'n_transitions': 0}
            rows.append({
                'state': state,
                # topology order, the only key that survives a comparison
                # between two ensembles: the MSMD and SWISH topologies of the
                # same system are numbered differently (TEM1's differ by 24),
                # so resSeq cannot be used to match a residue across them
                'residue_index': int(residue_index),
                'resSeq': int(meta.resSeq[residue_index]),
                'resname': name,
                'aromatic': name in AROMATICS,
                'apo_distance_a': float(meta.distance_a[residue_index]),
                'span_a': moved['span_a'],
                'position_states': moved['n_states'],
                'position_switches': moved['n_transitions'],
                'chi1_states': chi1['n_states'],
                'chi1_switches': chi1['n_transitions'],
                'chi2_states': chi2['n_states'],
                'chi2_switches': chi2['n_transitions'],
                'chi3_states': chi3['n_states'],
                'chi3_switches': chi3['n_transitions'],
            })
    return pd.DataFrame(rows)


def rank_gating_residues(described: pd.DataFrame,
                         top_n: int = N_DRIVERS) -> pd.DataFrame:
    """Order residues by how much they move at lambda = 1, flagging switchers.

    Ordered on state 0 alone, the unscaled Hamiltonian, and not on the
    maximum across the ladder, which would order by how far a residue moves
    when the hydrophobic term is inflated: TEM-1's LEU221 spans 32.8 A at
    lambda = 1.35 against 8.8 A at lambda = 1. The frame these residues go on to
    select is taken from state 0, so they are ordered in the same ensemble
    they are measured in. The per-state spans remain in the table.

    Span first because the two gates confirmed against the literature, TEM-1's
    LEU220 and CRBN's W264, both move several Angstroms without ever occupying
    two resolved basins. Bistability is reported alongside because when it does
    appear, as for Bcl-xL's TYR185, it is strong evidence of a discrete gate.

    Args:
        described: Output of describe_candidate.
        top_n: Rows to return.

    Returns:
        Residues ordered by their span on state 0.
    """
    if described.empty:
        return pd.DataFrame()
    unscaled = described[described.state == 0]
    span_at_lambda1 = (unscaled.groupby('resSeq').span_a.max()
                       if not unscaled.empty else pd.Series(dtype=float))
    # Everything is read on state 0, the same ensemble the ordering uses:
    # counting a basin that only appears at lambda = 1.35 would label a residue
    # from an ensemble the selection never sees.
    grouped = unscaled.groupby(['resSeq', 'resname', 'aromatic']).agg(
        max_span_a=('span_a', 'max'),
        mean_span_a=('span_a', 'mean'),
        states_bistable=('position_states', lambda s: int((s > 1).sum())),
        total_switches=('position_switches', 'sum'),
        chi1_switches=('chi1_switches', 'sum'),
        chi2_switches=('chi2_switches', 'sum'),
        chi3_switches=('chi3_switches', 'sum'),
    ).reset_index()
    grouped['span_a_lambda1'] = grouped.resSeq.map(span_at_lambda1).fillna(0.0)
    grouped['gate_flag'] = np.where(
        grouped.states_bistable > 0, 'switches',
        np.where(grouped.span_a_lambda1 >= LARGE_MOVEMENT_A,
                 'large movement', ''),
    )
    return grouped.sort_values('span_a_lambda1', ascending=False).head(top_n)



def run(
        density_dir: str,
        output_dir: str,
        n_replicas: int,
        n_candidates: int = N_CANDIDATES,
        radius_a: float = GATING_RADIUS_A,
        hotspots_file: str = HOTSPOTS_FILE,
        apo_pdb: str = '',
) -> pd.DataFrame:
    """Rank candidates and explain each one, writing a report per candidate.

    Args:
        density_dir: Directory written by compute_probe_density.
        output_dir: Where to write pocket_dynamics.csv and the summary.
        n_candidates: Candidates to characterise, strongest density first.
        radius_a: Residue inclusion radius_a in Angstroms.
        n_replicas: SWISH replicas available.
        hotspots_file: Which detector's candidates to characterise.
            lambda1_hotspots.csv is the production detector, the difference map
            from demuxed state 0; msmd_hotspots.csv is the unbiased map.
        apo_pdb: Apo input structure, required. It decides which residues sit
            at each candidate and which protomer each candidate belongs to,
            never frame 0 of state 0. Frame 0 is one conformation of a
            pocket that opens and closes, so a residue sitting just inside or
            just outside the shell at that instant flips category and the
            membership depends on where the trajectory happened to start. The
            apo does not move when the equilibration is repeated and is what a
            prospective user holds before running anything. Indices still come
            from the trajectory; only membership comes from the apo. It is also
            the closed half of every open/closed session.


    Returns:
        Combined per-residue table across candidates.

    Raises:
        ValueError: If apo_pdb is not given, since candidate membership and
            the protomer assignment must come from a structure that does not
            move between runs.
        FileNotFoundError: If the density directory holds no candidate table
            or none of the state trajectories it names.
    """
    # Checked before any trajectory is read: without it the run would load
    # every state and the candidate table before failing.
    if not apo_pdb:
        raise ValueError(
            'apo_pdb is required: candidate membership and the protomer '
            'assignment must come from a structure that does not move '
            'between runs, and frame 0 of a demultiplexed trajectory is '
            'whichever conformation held the slot at the first checkpoint')

    density_path, out = Path(density_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trajectories = []
    for state in range(n_replicas):
        top = density_path / f'density_swish_{state}_top.pdb'
        dcd = density_path / f'swish_{state}_aligned.dcd'
        if top.exists() and dcd.exists():
            trajectories.append((state, mdtraj.load(str(dcd), top=str(top))))
    if not trajectories:
        raise FileNotFoundError(f'No aligned SWISH trajectories in {density_path}')

    hotspots = pd.read_csv(density_path / hotspots_file).head(n_candidates)
    # In an oligomer the same site appears once per protomer, and independent
    # recovery at each copy is a confidence signal the state measures did not
    # provide. Monomers pass through unchanged.
    #
    # The protomer transforms come from the apo where one is given, for the same
    # reason the residue membership does: frame 0 is one conformation and the
    # chains have drifted relative to each other by then, so the superposition
    # that says "these two candidates are the same site" is measured against a
    # structure that moves between runs.
    symmetry_reference = mdtraj.load(apo_pdb)
    hotspots = group_symmetry_candidates(hotspots, symmetry_reference)

    all_rows = []
    for _, hotspot in hotspots.iterrows():
        rank = int(hotspot['rank'])
        centre = hotspot[['x', 'y', 'z']].to_numpy(float)
        apo_resseqs = apo_pocket_resseqs(apo_pdb, centre, radius_a)
        described = describe_candidate(trajectories, centre, apo_resseqs)
        if described.empty:
            continue
        described.insert(0, 'candidate_rank', rank)
        all_rows.append(described)
        gating = rank_gating_residues(described)
        if gating.empty:
            continue

        write_open_closed_pml(density_path, trajectories[0][1],
                              trajectories[0][0], gating, centre, rank, out,
                              apo_pdb=apo_pdb)

        top_gate = gating.iloc[0]
        logging.info(
            f'Candidate {rank} ({int(hotspot.n_voxels)} voxels): top mover '
            f'{top_gate.resname}{int(top_gate.resSeq)} spanning '
            f'{top_gate.max_span_a:.1f} A [{top_gate.gate_flag or "no flag"}]',
        )

    # The per-residue table is the only one written: any ranking or verdict
    # derived from it would be a second opinion on the detector's own order.
    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined.to_csv(out / 'pocket_dynamics.csv', index=False)
    logging.info(f'Written to {out}: pocket_dynamics.csv and one open frame '
                 f'per candidate')
    return combined


def usable_frames(trajectory: mdtraj.Trajectory) -> np.ndarray:
    """Boolean mask of frames whose unit cell is physically meaningful.

    A frame can carry a corrupt unit cell while its positions stay finite, so
    densities and Cartesian distances are unaffected and nothing raises. A
    structure written from such a frame carries the cell into its CRYST1
    record, and any later measurement that applies periodic boundaries, every
    dihedral among them, then returns NaN silently. Frames are screened here so
    one cannot be chosen to represent an open conformation. HPV-11 E2's
    demultiplexed state 0 holds one, [inf, 4.1e4, 1.2e16] nm at frame 1420
    of 2501.

    Args:
        trajectory: Trajectory to screen.

    Returns:
        Boolean array, True where the frame may be written out.
    """
    lengths = trajectory.unitcell_lengths
    if lengths is None:
        return np.ones(trajectory.n_frames, dtype=bool)
    return (np.isfinite(lengths).all(axis=1)
            & (lengths > 0).all(axis=1)
            & (lengths < 1e3).all(axis=1))


def opening_coordinate(traces: dict[int, np.ndarray]) -> np.ndarray:
    """Mean percentile rank across the driving residues, per frame.

    A percentile rather than the raw distance, because a residue sitting at 11 A
    and one at 4 A would otherwise contribute on completely different scales and
    the larger would decide the answer on its own.

    Args:
        traces: Per-frame distance to the site centre, per driving residue.

    Returns:
        One value per frame, the mean percentile rank across those residues.
    """
    ranks = []
    for values in traces.values():
        order = values.argsort().argsort()
        ranks.append(order / max(len(values) - 1, 1))
    return np.mean(ranks, axis=0)


def residue_traces(trajectory: mdtraj.Trajectory, centre: np.ndarray,
                   wanted: set[int]) -> dict[int, np.ndarray]:
    """Per-frame closest heavy-atom distance to the centre, per residue.
    Args:
        trajectory: Aligned trajectory to measure over.
        centre: (3,) candidate centre in Angstroms.
        wanted: Simulation resSeq values to measure.

    Returns:
        resSeq mapped to its per-frame closest heavy-atom distance to the
        centre. Solvent and probe residues are never included.
    """
    return residue_distance_traces(trajectory, centre, wanted)


def write_open_closed_pml(
        density_dir: Path,
        trajectory: mdtraj.Trajectory,
        state: int,
        gating: pd.DataFrame,
        centre: np.ndarray,
        rank: int,
        output_dir: Path,
        apo_pdb: str | None = None,
        n_drivers: int = N_DRIVERS,
) -> Path:
    """Write the open frame for one candidate, plus a session against the apo.

    The open frame is the one nearest the OPEN_PCT percentile of an opening
    coordinate averaged over the driving residues, which is the rule
    snapshot_frames uses. The driving residues are the ones the source papers
    name where the caller supplies them and the top movers at this candidate
    otherwise, so a matched candidate on a characterised target reproduces that
    target's snapshot exactly rather than disagreeing with it.

    The closed state is the apo input, loaded straight into the session. It is
    the structure the simulation started from, it does not move if the
    equilibration is repeated, and it is what a prospective user holds before
    running anything, so no closed frame is written.

    Args:
        density_dir: Density directory, for the difference map the session draws.
        trajectory: Aligned trajectory for the state the frame comes from.
        state: Thermodynamic state index, recorded in the session header.
            State 0 is lambda = 1.0.
        gating: Residues at this candidate ranked by range of closest approach,
            the top n_drivers of which drive the opening coordinate.
        centre: (3,) candidate centre in Angstroms.
        rank: Candidate rank, used in the file names and the header.
        output_dir: Where to write the frame and the session.
        apo_pdb: Apo input, loaded as the closed half of the session. Without
            it the session shows the open frame alone.
        n_drivers: How many top movers define the opening coordinate.

    Returns:
        The output directory, unchanged, or early if no residue could be
        measured at this candidate.
    """
    driving = sorted(int(r) for r in gating.resSeq.head(n_drivers))
    traces = residue_traces(trajectory, centre, set(driving))
    if not traces:
        return output_dir

    coordinate = opening_coordinate(traces)
    usable = usable_frames(trajectory)
    if not usable.any():
        raise ValueError(
            f'candidate {rank}: no frame has a usable unit cell, so no open '
            f'conformation can be written')
    # the percentile is taken over usable frames only, so a discarded frame
    # cannot shift the target value either
    candidates_idx = np.flatnonzero(usable)
    target = np.percentile(coordinate[usable], OPEN_PCT)
    opened = int(candidates_idx[
        np.abs(coordinate[usable] - target).argmin()])
    if not usable.all():
        logging.info(
            f'candidate {rank}: {int((~usable).sum())} frame(s) excluded from '
            f'the open-frame search for a non-finite unit cell')
    protein = trajectory.atom_slice(trajectory.topology.select('protein'))
    open_pdb = output_dir / f'candidate{rank}_open.pdb'
    protein[opened].save_pdb(str(open_pdb))

    resi = '+'.join(str(r) for r in driving)
    source = f'the {len(driving)} top movers at this candidate'
    pml_path = output_dir / f'candidate{rank}_open_closed.pml'
    pml = [
        '# ' + '=' * 68,
        f'#  Candidate {rank}: how the site opens',
        f'#  Open: frame {opened} of {trajectory.n_frames} of state {state},',
        f'#  at the {OPEN_PCT:g}th percentile of an opening coordinate built from',
        f'#  {source}, so it stands for about a twentieth of the trajectory',
        '#  rather than a single excursion.',
        '#  Closed: the apo input structure the simulation started from.',
        '# ' + '=' * 68,
        '',
    ]
    pml += pml_preamble(pml_path)
    pml += [f'load {analysis_relative(open_pdb)}, opened']
    if apo_pdb:
        # A caller may hold the apo anywhere; the session can only load what
        # sits under the analysis root, so it is brought in here rather than
        # left to the benchmark wrapper.
        apo_pdb = bundle_into_analysis(apo_pdb)
        pml += [f'load {analysis_relative(apo_pdb)}, apo',
                'hide everything',
                'show cartoon, apo or opened',
                'hide everything, elem H',
                'align polymer and apo, polymer and opened',
                'color grey70, apo',
                'color 0x0DBFBF, opened',
                'set cartoon_transparency, 0.55, apo']
        closed = 'apo'
    else:
        pml += ['hide everything', 'show cartoon, opened',
                'hide everything, elem H', 'color 0x0DBFBF, opened']
        closed = None
    both = f'({closed} or opened)' if closed else 'opened'
    pml += [
        '',
        '# Driving residues as sticks in both states',
        f'show sticks, {both} and resi {resi} and not elem H',
        f'color 0xF56170, opened and resi {resi}',
    ]
    if closed:
        pml.append(f'color 0x9aa0a6, {closed} and resi {resi}')
    pml += [
        '',
        '# The candidate site itself',
        f'pseudoatom site, pos=[{centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f}]',
        'show spheres, site',
        'set sphere_scale, 0.9, site',
        'color 0xFFC766, site',
        '',
        # The map is loaded by a path relative to the analysis root like
        # everything else in the session, so a density directory outside that
        # root has its map brought in rather than referenced where it lies.
        f'load {analysis_relative(bundle_into_analysis(Path(density_dir) / DIFFERENCE_MAP))}, '
        'diff_density',
        f'isosurface surf_diff, diff_density, {THRESHOLD_FACTOR:g}',
        'color 0x858CE0, surf_diff',
        'set transparency, 0.35, surf_diff',
        '',
        'bg_color white',
        f'orient opened and resi {resi}',
        'deselect',
        '',
        '# Grey is the apo, teal the open frame. Red sticks are the driving',
        '# residues in the open state, grey the same residues in the apo.',
        f'# The yellow sphere is the candidate centre and the purple surface its',
        f'# density at {THRESHOLD_FACTOR:g}x bulk.',
    ]
    pml_path.write_text('\n'.join(pml) + '\n')
    return pml_path



def protomer_transforms(structure: mdtraj.Trajectory, max_rmsd: float = 3.0):
    """Superpositions mapping the first protomer onto each of the others.

    Protomers are not always written with the same leading residues, so the
    offset between chains is found by maximising residue-name agreement rather
    than assumed to be zero: Bcl-xL's chain B carries an extra glycine, and
    matching position by position agrees on only 8 residue names out of 139.

    Args:
        structure: Single-frame structure holding all chains.
        max_rmsd: Reject a pairing that superposes worse than this.

    Returns:
        List of (rotation, translation), empty for a monomer.
    """
    chains = {}
    for residue in structure.topology.residues:
        if residue.name in NON_PROTEIN:
            continue
        chains.setdefault(residue.chain.index, []).append(residue)
    chains = {k: v for k, v in chains.items() if len(v) >= 20}
    if len(chains) < 2:
        return []

    xyz = structure.xyz[0] * 10.0
    keys = sorted(chains)
    first = chains[keys[0]]
    transforms = []
    for other_key in keys[1:]:
        other = chains[other_key]
        best_offset, best_matches = 0, -1
        for offset in range(-5, 6):
            matches = sum(
                1 for i, residue in enumerate(first)
                if 0 <= i + offset < len(other)
                and residue.name == other[i + offset].name
            )
            if matches > best_matches:
                best_offset, best_matches = offset, matches

        a, b = [], []
        for i, residue in enumerate(first):
            j = i + best_offset
            if not (0 <= j < len(other)) or residue.name != other[j].name:
                continue
            ca_a = [at.index for at in residue.atoms if at.name == 'CA']
            ca_b = [at.index for at in other[j].atoms if at.name == 'CA']
            if ca_a and ca_b:
                a.append(xyz[ca_a[0]]); b.append(xyz[ca_b[0]])
        if len(a) < 20:
            continue
        a, b = np.array(a), np.array(b)
        rotation, translation = kabsch(a, b)
        rmsd = float(np.sqrt((((a @ rotation.T + translation) - b) ** 2).sum(1).mean()))
        if rmsd > max_rmsd:
            logging.info(
                f'Chains {keys[0]}/{other_key} superpose at {rmsd:.1f} A, '
                f'not treated as copies of one another',
            )
            continue
        logging.info(
            f'Protomer {other_key} related to protomer {keys[0]} by a '
            f'{rmsd:.2f} A superposition over {len(a)} CA',
        )
        transforms.append((rotation, translation))
    return transforms


def group_symmetry_candidates(
        hotspots: pd.DataFrame,
        structure: mdtraj.Trajectory,
        tolerance_a: float = SYMMETRY_TOLERANCE_A,
) -> pd.DataFrame:
    """Mark candidates that are the same site on different protomers.

    In a homo-oligomer every cryptic site exists once per protomer, so finding
    it independently at each copy is evidence rather than redundancy: the
    protomers are separate samples of the same opening, and a site that appears
    at all of them is far less likely to be a fluctuation than one that appears
    at a single copy. Nothing else in the pipeline provides that check, and the
    state-based reproducibility measure did not discriminate when tested.

    Args:
        hotspots: Ranked candidates with x, y, z columns.
        structure: Single-frame structure in the same frame as the candidates.
        tolerance_a: How close a mapped centre must land to count as the same site.

    Returns:
        hotspots with symmetry_group and n_protomers columns added.
    """
    result = hotspots.copy()
    transforms = protomer_transforms(structure)
    n_protomers = len(transforms) + 1
    result['symmetry_group'] = range(len(result))
    result['n_protomers'] = 1
    if not transforms:
        return result

    centres = result[['x', 'y', 'z']].to_numpy(float)
    group_of = list(range(len(result)))
    for i, centre in enumerate(centres):
        for rotation, translation in transforms:
            mapped = centre @ rotation.T + translation
            distances = np.linalg.norm(centres - mapped, axis=1)
            partner = int(np.argmin(distances))
            if distances[partner] <= tolerance_a and partner != i:
                merged = min(group_of[i], group_of[partner])
                old = max(group_of[i], group_of[partner])
                group_of = [merged if g == old else g for g in group_of]

    result['symmetry_group'] = group_of
    counts = pd.Series(group_of).value_counts()
    result['n_protomers'] = [counts[g] for g in group_of]
    found = result[result.n_protomers > 1]
    if len(found):
        for group, rows in found.groupby('symmetry_group'):
            ranks = ', '.join(str(int(r)) for r in rows['rank'])
            logging.info(
                f'Candidates {ranks} are the same site on '
                f'{len(rows)} of {n_protomers} protomers',
            )
    else:
        logging.info(
            f'No candidate was found at more than one of the {n_protomers} '
            f'protomers, which is weak evidence for all of them',
        )
    return result
