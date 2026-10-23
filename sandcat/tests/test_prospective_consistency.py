"""Invariants the prospective analysis must hold across its stages.

The stages run as separate scripts over a shared tree, so nothing in the code
forces them to agree on what they measure. These tests supply that force. They
are synthetic and cheap: each checks a definition rather than a result, so none
needs a real trajectory.

Two definitions have to be identical everywhere:

  - One measurement. The residue-to-candidate distance decides which residues
    a candidate reports, which drive its opening coordinate and which cutoff
    applies, so every stage must obtain it from the same function.
  - One residue set. Solvent, ions and the ACE/NME caps carry a resSeq, so a
    stage that does not exclude them reports a capping group as a residue and
    ranks it among the real ones.
"""
from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import mdtraj
import numpy as np
import pandas as pd
import pytest

from sandcat.analysis import experiments, pocket_residue_dynamics, residue_ranking
from sandcat.analysis.analyse_pocket_dynamics import (
    N_DRIVERS, OPEN_PCT, opening_coordinate, rank_gating_residues, run,
)
from sandcat.analysis.build_density_maps import LAMBDA_LADDER, SIMULATION_INPUTS
from sandcat.analysis.compute_probe_density import (
    GRID_PADDING_A, GRID_SPACING_A, MIN_CANDIDATE_VOXELS, STRIDE,
    THRESHOLD_FACTOR, density_hotspots,
)
from sandcat.analysis.experiments import (
    DWELL, NON_HEAVY_ELEMENTS, NON_PROTEIN, apo_residues,
    heavy_atoms_by_resseq, residue_distance_traces,
)
from sandcat.analysis.literature_numbering import label
from sandcat.analysis.plot_site_residue_figure import PURPLE, SITE_COLOUR, TEAL
from sandcat.analysis.pocket_residue_dynamics import (
    AROMATICS, rotamer_states, rotamer_switches,
)
from sandcat.analysis.residue_motion import BACKBONE
from sandcat.analysis.residue_ranking import first_candidate_reaching_residue, motion_class

ANALYSIS_SOURCES = Path(experiments.__file__).parent


ATOM_NAMES = (' CA ', ' CB ', ' CG ')


def write_pdb(path: Path, residues: list[tuple[str, int, tuple[float, float, float] | list[tuple[float, float, float]]]]) -> Path:
    """A minimal PDB, one line per atom, enough to exercise the selectors.

    Each entry is (resname, resSeq, position), or (resname, resSeq, [positions])
    to give a residue several atoms. Multi-atom residues matter wherever a test
    has to tell the nearest heavy atom from the farthest: with one atom each the
    two are the same number and the distinction cannot be tested.
    """
    lines = []
    serial = 0
    for resname, resseq, where in residues:
        points = where if isinstance(where, list) else [where]
        for index, (x, y, z) in enumerate(points):
            serial += 1
            lines.append(
                f'ATOM  {serial:5d} {ATOM_NAMES[index % len(ATOM_NAMES)]} '
                f'{resname:>3s} A{resseq:4d}    '
                f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C')
    lines.append('END')
    path.write_text('\n'.join(lines) + '\n')
    return path


# Caps, solvent and ions are never reported as residues, on any path.

def test_caps_are_excluded_from_the_trajectory_path(tmp_path: Path) -> None:
    pdb = write_pdb(tmp_path / 'capped.pdb', [
        ('ACE', 0, (0.0, 0.0, 0.0)),
        ('ALA', 1, (1.0, 0.0, 0.0)),
        ('NME', 2, (2.0, 0.0, 0.0)),
        ('HOH', 3, (3.0, 0.0, 0.0)),
    ])
    traj = mdtraj.load(str(pdb))
    traces = residue_distance_traces(traj, np.zeros(3))
    assert set(traces) == {1}, (
        'only the real residue may be measured; a capping group carries '
        'a resSeq but is not a residue of the protein')


def test_caps_are_excluded_from_the_apo_path(tmp_path: Path) -> None:
    """The apo selector must exclude what the trajectory selector does.

    The apo decides which residues sit at a candidate and which are
    eligible to drive its opening coordinate, so a cap admitted here would
    be ranked against real side chains.
    """
    pdb = write_pdb(tmp_path / 'capped.pdb', [
        ('ACE', 0, (0.0, 0.0, 0.0)),
        ('ALA', 1, (1.0, 0.0, 0.0)),
        ('NME', 2, (2.0, 0.0, 0.0)),
    ])
    assert set(apo_residues(str(pdb))) == {1}


def test_both_paths_exclude_the_same_names(tmp_path: Path) -> None:
    names = ('ALA', 'GLY', 'HOH', 'ACE', 'NME', 'BNZ', 'PRB')
    assert set(names) - {'ALA', 'GLY'} <= set(NON_PROTEIN)
    residues = [(name, i + 1, (float(i), 0.0, 0.0))
                for i, name in enumerate(names)]
    pdb = write_pdb(tmp_path / 'everything.pdb', residues)
    traj = mdtraj.load(str(pdb))
    from_trajectory = set(residue_distance_traces(traj, np.zeros(3)))
    from_apo = set(apo_residues(str(pdb)))
    assert from_trajectory == from_apo, (
        'the apo and trajectory selectors must agree on what a residue is')




# The measurement the whole analysis rests on is defined exactly once.

def test_trace_matches_an_independent_computation(tmp_path: Path) -> None:
    pdb = write_pdb(tmp_path / 'two.pdb', [
        ('ALA', 1, (3.0, 4.0, 0.0)),
        ('GLY', 2, (0.0, 0.0, 12.0)),
    ])
    traj = mdtraj.load(str(pdb))
    traces = residue_distance_traces(traj, np.zeros(3))
    assert traces[1][0] == pytest.approx(5.0)
    assert traces[2][0] == pytest.approx(12.0)


def test_the_trace_is_the_nearest_heavy_atom_not_the_farthest(tmp_path: Path) -> None:
    """The measurement is a closest approach, which needs two atoms to pin.

    Every cutoff in the analysis is applied to this number, so reading the
    farthest atom instead would widen every environment without failing
    anything that only has one atom per residue to look at.
    """
    pdb = write_pdb(tmp_path / 'spread.pdb', [
        ('ALA', 1, [(3.0, 0.0, 0.0), (9.0, 0.0, 0.0), (6.0, 0.0, 0.0)]),
    ])
    traj = mdtraj.load(str(pdb))
    trace = residue_distance_traces(traj, np.zeros(3))[1]
    assert trace[0] == pytest.approx(3.0), (
        'the trace must be the closest heavy atom, not the farthest (9.0) '
        'or the centroid (6.0)')


def test_closest_approach_agrees_with_the_trace(tmp_path: Path) -> None:
    """The one-structure reader and the trajectory reader measure the same."""
    pdb = write_pdb(tmp_path / 'spread.pdb', [
        ('ALA', 1, [(3.0, 0.0, 0.0), (9.0, 0.0, 0.0)]),
    ])
    traj = mdtraj.load(str(pdb))
    centre = np.zeros(3)
    coords = traj.xyz[0] * 10.0
    assert experiments.closest_approach(coords, centre) == pytest.approx(
        float(residue_distance_traces(traj, centre)[1][0]))


def test_wanted_restricts_without_changing_values(tmp_path: Path) -> None:
    pdb = write_pdb(tmp_path / 'two.pdb', [
        ('ALA', 1, (3.0, 4.0, 0.0)),
        ('GLY', 2, (0.0, 0.0, 12.0)),
    ])
    traj = mdtraj.load(str(pdb))
    everything = residue_distance_traces(traj, np.zeros(3))
    restricted = residue_distance_traces(traj, np.zeros(3), wanted={1})
    assert set(restricted) == {1}
    assert restricted[1] == pytest.approx(everything[1])


def test_no_stage_reimplements_the_measurement() -> None:
    """Only experiments.py may spell out either distance kernel.

    Matched on the numpy shapes rather than on one exact spelling, so a
    stage cannot evade the check by splitting the expression over two
    lines or renaming its variables.
    """
    trace = re.compile(r'axis\s*=\s*2\s*\)\s*\.min\(\s*axis\s*=\s*1')
    single = re.compile(r'axis\s*=\s*1\s*\)\s*\.min\(\s*\)')
    offenders = {}
    for path in sorted(ANALYSIS_SOURCES.glob('*.py')):
        if path.name == 'experiments.py':
            continue
        source = path.read_text()
        hits = [name for name, pattern in (('distance_trace', trace),
                                           ('closest_approach', single))
                if pattern.search(source)]
        if hits:
            offenders[path.name] = hits
    assert offenders == {}, (
        f'{offenders} spell out a distance kernel; call '
        f'experiments.distance_trace or experiments.closest_approach')


def test_the_kernels_have_one_caller_each_in_experiments() -> None:
    """The shared readers are built on the kernels, not beside them."""
    source = inspect.getsource(experiments.residue_distance_traces)
    assert 'distance_trace(' in source




# Both state counters suppress flicker by the same rule.

def test_a_single_frame_excursion_is_not_a_transition() -> None:
    angles = np.array([-60.0] * 20 + [180.0] + [-60.0] * 20)
    assert rotamer_switches(angles, 360.0) == 0
    assert rotamer_states(angles)['n_transitions'] == 0, (
        'a single-frame excursion is flicker across a bin boundary, not '
        'a rotamer transition, and both counters must reject it')


def test_a_held_change_is_a_transition() -> None:
    angles = np.array([-60.0] * 20 + [60.0] * 20)
    assert rotamer_switches(angles, 360.0) == 1


def test_the_dwell_is_shared() -> None:
    """One rule, one implementation: the binner defers to the filter."""
    assert pocket_residue_dynamics.DWELL is DWELL
    source = inspect.getsource(pocket_residue_dynamics.rotamer_switches)
    assert 'dwell_filtered_switches' in source, (
        'rotamer_switches must defer to dwell_filtered_switches rather '
        'than carry a second copy of the dwell loop')




# The names one stage writes and the next reads are declared once.

def test_the_state_filenames_follow_the_production_state() -> None:
    n = experiments.PRODUCTION_STATE
    assert experiments.STATE_TRAJECTORY == f'swish_{n}_aligned.dcd'
    assert experiments.STATE_TOPOLOGY == f'density_swish_{n}_top.pdb'


def test_the_difference_map_is_named_for_its_lambda_not_its_index() -> None:
    """State 0 is lambda = 1.0, so the map is named for the lambda."""
    assert 'lambda1' in experiments.DIFFERENCE_MAP
    assert 'lambda0' not in experiments.DIFFERENCE_MAP


def test_the_dynamics_stage_defaults_to_the_production_detector() -> None:
    default = inspect.signature(run).parameters['hotspots_file'].default
    assert default == experiments.HOTSPOTS_FILE, (
        'the default must name the lambda = 1 difference map, the only '
        'detector whose Hamiltonian is unscaled')


def test_the_state_count_has_one_authority() -> None:
    for target, (_, _, _, states) in SIMULATION_INPUTS.items():
        assert states in LAMBDA_LADDER, f'{target} has no ladder'
        ladder = LAMBDA_LADDER[states]
        assert len(ladder) == states
        assert ladder[0] == 1.0, 'state 0 is the unscaled Hamiltonian'




# Every distance the analysis applies is a named constant, in Angstroms.

def test_the_radii_are_distinct_and_ordered() -> None:
    assert experiments.HIT_A < experiments.GATING_RADIUS_A < experiments.RESIDUE_A


def test_no_heavy_atom_test_is_spelled_out_by_hand() -> None:
    """One element exclusion, so every stage agrees what is heavy."""
    pattern = re.compile(r"symbol\s*(?:!=\s*'H'|not in \('H')")
    offenders = [path.name for path in sorted(ANALYSIS_SOURCES.glob('*.py'))
                 if pattern.search(path.read_text())]
    assert offenders == [], (
        f'{offenders} spell out the heavy-atom test; use '
        f'NON_HEAVY_ELEMENTS so every stage agrees')


def test_non_heavy_elements_covers_hydrogen_and_virtual_sites() -> None:
    assert 'H' in NON_HEAVY_ELEMENTS
    assert 'VS' in NON_HEAVY_ELEMENTS




# Each parameter the Methods quotes, pinned to the value it quotes.
#
#     These are calibrated choices, not implementation details: the density cut
#     was set by a sweep, the voxel floor by the smallest benchmark site, the
#     stride by the frame spacing the counts are interpreted against. Changing
#     one silently makes the published Methods wrong while every other test
#     still passes, so each is asserted here against the sentence that states it.
#

def test_the_density_cut_and_voxel_floor() -> None:
    assert THRESHOLD_FACTOR == 6.0, (
        'Methods: candidate sites are the regions "exceeding 6 x bulk '
        'occupancy"')
    assert MIN_CANDIDATE_VOXELS == 3, (
        'Methods: regions "containing at least three voxels"')


def test_the_grid_geometry_and_stride() -> None:
    assert GRID_SPACING_A == 1.0 and GRID_PADDING_A == 5.0, (
        'Methods: "1 A resolution extending 5 A beyond the protein"')
    assert STRIDE == 10, 'Methods: "Every 10th frame was analysed"'


def test_the_production_state_is_the_unscaled_hamiltonian() -> None:
    assert experiments.PRODUCTION_STATE == 0
    for states, ladder in LAMBDA_LADDER.items():
        assert ladder[experiments.PRODUCTION_STATE] == 1.0, (
            f'the production state must be lambda = 1.0 on the {states}-'
            f'state ladder, which the Methods call "the lambda = 1.0 '
            f'replica"')


def test_the_three_radii() -> None:
    assert experiments.RESIDUE_A == 10.0, (
        'Methods: a residue belongs to the environment when its median '
        'distance is "within 10 A"')
    assert experiments.GATING_RADIUS_A == 8.0, (
        'Methods: drivers are residues "within 8 A of the candidate '
        'centre in the apo input structure"')
    assert experiments.HIT_A == 2.0, (
        'Methods: a candidate matches a reference when its centre is '
        '"within 2 A of the nearest point of that pocket"')


def test_the_open_frame_rule() -> None:
    assert OPEN_PCT == 95.0, (
        'Methods: "The frame nearest the 95th percentile of that '
        'coordinate was taken as the open representative"')
    assert N_DRIVERS == 6, (
        'Methods: "The six with the largest range of closest approach"')


def test_the_dwell_matches_the_methods() -> None:
    assert DWELL == 3, (
        'Methods: "A transition was then counted only where a bin '
        'persists for at least three consecutive frames"')


def test_the_candidate_depth_covers_the_deepest_matched_site() -> None:
    """PLK1's site is candidate 6, so the stage must characterise at least six."""
    assert experiments.N_CANDIDATES >= 6




# What each rule actually decides, pinned on synthetic input.
#
#     The rest of this file fixes names, structure and parameter values. These
#     fix direction and precedence: a rule can keep every name and constant and
#     still choose the opposite frame, the opposite residues or the opposite
#     class.
#

def test_the_open_frame_is_the_one_where_drivers_are_furthest() -> None:
    """Open means the gate has moved away from the site, not towards it."""
    closed_then_open = {1: np.array([2.0, 3.0, 4.0, 9.0]),
                        2: np.array([1.0, 2.0, 3.0, 8.0])}
    coordinate = opening_coordinate(closed_then_open)
    assert int(np.argmax(coordinate)) == 3, (
        'the opening coordinate must peak where the driving residues sit '
        'furthest from the centre, not nearest')
    assert int(np.argmin(coordinate)) == 0


def test_drivers_are_the_most_mobile_residues() -> None:
    """The opening coordinate is built from movers, not from still residues."""
    described = pd.DataFrame([
        {'state': 0, 'resSeq': 10, 'resname': 'LEU', 'aromatic': False,
         'span_a': 12.0, 'position_states': 1, 'position_switches': 0,
         'chi1_switches': 0, 'chi2_switches': 0, 'chi3_switches': 0},
        {'state': 0, 'resSeq': 20, 'resname': 'ALA', 'aromatic': False,
         'span_a': 0.5, 'position_states': 1, 'position_switches': 0,
         'chi1_switches': 0, 'chi2_switches': 0, 'chi3_switches': 0},
    ])
    ranked = rank_gating_residues(described, top_n=1)
    assert list(ranked.resSeq) == [10], (
        'the residue spanning 12 A must outrank the one spanning 0.5 A')


def test_the_classes_split_at_the_median_and_aromatics_win() -> None:
    upper = pd.Series({'flip_rank': 1.0, 'span_rank': 1.0,
                       'n_residues': 100, 'resname': 'PHE'})
    assert motion_class(upper) == 'aromatic rotamer switching', (
        'an aromatic in the upper half by switching takes the more '
        'specific class even when it is also in the upper half by span')
    non_aromatic = upper.copy()
    non_aromatic['resname'] = 'LEU'
    assert motion_class(non_aromatic) == 'side-chain displacement'
    # rank 60 of 100 is below the median split on both measures
    lower = pd.Series({'flip_rank': 60.0, 'span_rank': 60.0,
                       'n_residues': 100, 'resname': 'PHE'})
    assert motion_class(lower) == 'neither'
    # and the split is the median, not a quartile: rank 40 is still upper
    mid = pd.Series({'flip_rank': 40.0, 'span_rank': 60.0,
                     'n_residues': 100, 'resname': 'TRP'})
    assert motion_class(mid) == 'aromatic rotamer switching'


def test_excursions_across_the_wrap_are_not_transitions() -> None:
    """179 and -179 are two degrees apart, so the bins either side merge.

    Held either side of the wrap on purpose: a single-frame flicker is
    removed by the dwell filter whether or not the bins are merged, so
    only a held excursion tests the merge.
    """
    angles = np.array([179.0] * 5 + [-179.0] * 5 + [179.0] * 5)
    assert rotamer_switches(angles, 360.0) == 0, (
        'a side chain sitting either side of the 180 degree wrap holds '
        'one rotamer and must not register a switch on every crossing')
    # a real change of basin, held just as long, still counts
    moved = np.array([-60.0] * 5 + [60.0] * 5)
    assert rotamer_switches(moved, 360.0) == 1


def test_the_density_cut_excludes_voxels_exactly_at_the_threshold() -> None:
    """The Methods say "exceeding" the cut, so equality is not a candidate."""
    grid = np.zeros((6, 6, 6))
    grid[1:4, 1, 1] = 6.0          # exactly at the cut
    found = density_hotspots(grid, np.zeros(3), 1.0, threshold=6.0,
                             min_voxels=3)
    assert found.empty
    grid[1:4, 1, 1] = 6.5          # above it
    assert len(density_hotspots(grid, np.zeros(3), 1.0, threshold=6.0,
                                min_voxels=3)) == 1


def test_a_residue_exactly_at_the_reach_counts_as_reached() -> None:
    """RESIDUE_A is inclusive, so a median of exactly 10 A is inside."""
    xyz = np.full((3, 1, 3), 0.0)
    xyz[:, 0, 0] = experiments.RESIDUE_A
    centres = np.zeros((1, 3))
    assert first_candidate_reaching_residue(
        centres, xyz, [0], experiments.RESIDUE_A) == 1




# The residue names the paper prints, pinned to the offsets that make them.
#
#     Every residue the Results name is identified by its literature number, not
#     its simulation number, so an offset is as much a published quantity as a
#     cutoff. A wrong one relabels every bar in Figures 4, S3, S4 and S5 while
#     the geometry behind them stays correct, which nothing else here would
#     notice.
#

def test_the_offsets_reproduce_the_residues_the_results_name() -> None:
    # (target, simulation resSeq, residue, the label the paper prints)
    published = [
        ('TEM1', 220, 'LEU', 'LEU220'),      # Results: L220
        ('TEM1', 244, 'ARG', 'ARG244'),      # Results: R244
        ('MDM2', 96, 'HIS', 'HIS96'),        # Results: H96
        ('MDM2', 100, 'TYR', 'TYR100'),      # Results: Y100
        ('HPV11-E2', 19, 'TYR', 'TYR19'),    # Results: Y19
        ('IL2', 37, 'PHE', 'PHE42'),         # Results: F42, offset 5
        ('IL2', 33, 'ARG', 'ARG38'),         # Results: R38
        ('PLK1', 481, 'TYR', 'TYR481'),      # Results: Y481
        ('CRBN', 164, 'TRP', 'TRP264'),      # Results: W264, offset 100
        ('CRBN', 173, 'LEU', 'LEU273'),      # Results: L273
    ]
    for target, resseq, resname, want in published:
        assert label(target, resname, resseq) == want, (
            f'{target} resSeq {resseq} must print as {want}')


def test_the_homodimer_keeps_its_protomers_apart() -> None:
    """Bcl-xL's two copies share literature numbers, so the chain is kept."""
    assert label('BCLXL', 'TYR', 45, 'A') == 'TYR45 A'
    assert label('BCLXL', 'TYR', 185, 'B') == 'TYR45 B', (
        'Results: "Bcl-xL Y45 chain B"; simulation 185 is the chain B copy '
        'of literature residue 45')
    assert label('BCLXL', 'PHE', 49, 'A') == 'PHE49 A'
    assert label('BCLXL', 'PHE', 230, 'B') == 'PHE90 B'




# Atom and residue sets the published classes depend on.

def test_the_span_is_a_side_chain_measure() -> None:
    for atom in ('N', 'CA', 'C', 'O'):
        assert atom in BACKBONE, (
            f'{atom} is backbone; counting it in the span would make the '
            f'Methods\' "how far their side chain moves" untrue and shift '
            f'the median split for every residue')


def test_histidine_counts_as_aromatic() -> None:
    """Results: MDM2's H96 shows aromatic rotamer switching."""
    for name in ('HIS', 'HIE', 'HID', 'HIP'):
        assert name in AROMATICS, (
            f'{name} must be aromatic, or MDM2 H96 is reclassified')
    assert 'TRP' in AROMATICS and 'PHE' in AROMATICS and 'TYR' in AROMATICS


def test_the_production_detector_is_the_difference_map() -> None:
    """Named, not just linked: the value itself must be the lambda = 1 map."""
    assert experiments.HOTSPOTS_FILE == 'lambda1_hotspots.csv', (
        'the production detector is the lambda = 1 difference map; '
        'msmd_hotspots.csv is the unbiased control')




# What the captions and the released folder layout promise.

def test_the_ranking_reads_the_difference_map() -> None:
    """Figure 4 and the recovery percentages are the lambda = 1 contrast.

    msmd_hotspots.csv and the unbiased map are the control; ranking against
    them would still produce a complete, plausible table.
    """
    assert residue_ranking.DENSITY == experiments.DIFFERENCE_MAP


def test_teal_marks_the_matched_candidate_and_purple_another() -> None:
    """Figure 4 caption: "Teal denotes a residue that the candidate site
    matching the cryptic site reaches within 10 A, and purple a residue
    reached only by another candidate site on the same protein."
    """
    assert TEAL == '#0DBFBF' and PURPLE == '#858ce0', (
        'the panels use the teal and purple')
    assert SITE_COLOUR['correct'] == TEAL, (
        'teal is the residue the matching candidate reaches')
    assert SITE_COLOUR['other'] == PURPLE, (
        'purple is the residue only another candidate reaches')


def test_the_bundled_apo_sits_where_the_sessions_look_for_it() -> None:
    """The published sessions load inputs/<apo>.pdb relative to the root."""
    assert experiments.APO_DIR == 'inputs'




# The reusable half of the package must not require the benchmark setup.
#
#     experiments holds both: cutoffs, residue sets and the distance kernels,
#     which read no file, alongside the directories the paper pipeline runs in.
#     Resolving the directories at import time made every importer of a constant
#     set two environment variables for a calculation that needs neither.
#
#     Run in a subprocess with the variables removed, because conftest sets them
#     for the rest of the suite and an import inside this process has already
#     happened.
#

def run_without_env(statement: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items()
           if k not in ('SANDCAT_PAPER_DIR', 'SANDCAT_ANALYSIS_DIR')}
    env['PYTHONPATH'] = str(Path(experiments.__file__).parents[2])
    return subprocess.run([sys.executable, '-c', statement],
                          capture_output=True, text=True, env=env)


def test_the_kernels_and_constants_import_with_no_environment() -> None:
    done = run_without_env(
        'from sandcat.analysis.experiments import ('
        '    DWELL, NON_PROTEIN, closest_approach, distance_trace)\n'
        'from sandcat.analysis.pocket_residue_dynamics import ('
        '    rotamer_states, rotamer_switches)\n'
        'print("ok")')
    assert done.returncode == 0, (
        f'importing a constant or a kernel must not need the benchmark '
        f'directories:\n{done.stderr}')
    assert 'ok' in done.stdout


def test_the_benchmark_directories_still_refuse_when_unset() -> None:
    """Lazy, not silently defaulted: a wrong tree would be read in silence."""
    done = run_without_env(
        'from sandcat.analysis.experiments import ANALYSIS')
    assert done.returncode != 0
    assert 'SANDCAT_ANALYSIS_DIR is not set' in done.stderr
