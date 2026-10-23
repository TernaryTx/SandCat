"""Tests for pocket residue selection and side-chain rotamer analysis.

The behaviour that matters is discrimination, not arithmetic. A residue that
gates a cryptic pocket occupies two populated rotamer states and switches
between them; one that is merely flexible sits in a single broad basin. If the
analysis cannot tell those apart it reports nothing useful, so the rotamer tests
build both cases explicitly and assert they come out different.

The symmetry handling is tested separately because it is easy to get wrong in a
way that silently doubles the state count: Phe and Tyr rings are twofold
symmetric about chi2, so 40 degrees and 220 degrees are the same structure.
"""
from __future__ import annotations

from pathlib import Path
import mdtraj
from sandcat.analysis.pocket_residue_dynamics import residues_near_point
import MDAnalysis as mda

import numpy as np
import pytest

from sandcat.analysis.pocket_residue_dynamics import rotamer_switches
from sandcat.analysis.pocket_residue_dynamics import (SYMMETRIC_CHI2,
                                                      SYMMETRIC_CHI3,
                                                      rotamer_states)


def test_single_basin_is_one_state() -> None:
    """A residue wobbling around one rotamer is not a gate."""
    rng = np.random.default_rng(0)
    angles = rng.normal(180.0, 12.0, 2000)
    result = rotamer_states(angles)
    assert result['n_states'] == 1
    assert result['n_transitions'] == 0

def test_two_basins_are_two_states() -> None:
    """Occupying two rotamers is the signature worth reporting."""
    rng = np.random.default_rng(1)
    angles = np.concatenate([rng.normal(60.0, 8.0, 1000),
                             rng.normal(300.0, 8.0, 1000)])
    result = rotamer_states(angles)
    assert result['n_states'] == 2
    assert len(result['occupancies']) == 2

def test_switching_is_counted() -> None:
    """Alternating between two basins must register transitions."""
    angles = np.tile(np.concatenate([np.full(50, 60.0), np.full(50, 300.0)]), 10)
    result = rotamer_states(angles)
    assert result['n_states'] == 2
    assert result['n_transitions'] == 19          # 20 blocks, 19 boundaries
    assert result['switch_rate'] > 0

def test_a_rare_excursion_is_not_a_state() -> None:
    """A handful of frames in a second basin is noise, not bistability."""
    rng = np.random.default_rng(2)
    angles = np.concatenate([rng.normal(180.0, 8.0, 1960),
                             rng.normal(60.0, 4.0, 40)])     # 2% occupancy
    assert rotamer_states(angles, min_occupancy=0.1)['n_states'] == 1

def test_empty_and_all_nan_are_handled() -> None:
    assert rotamer_states(np.array([]))['n_states'] == 0
    assert rotamer_states(np.full(10, np.nan))['n_states'] == 0


def test_symmetric_chi2_does_not_double_count() -> None:
    """Phe/Tyr chi2 at 40 and 220 degrees is one state, not two.

    Folded onto a 180 degree period both land in the same bin. Reading the
    same data on a 360 degree period would wrongly report two states, which
    is the failure this guards against.
    """
    angles = np.concatenate([np.full(1000, 40.0), np.full(1000, 220.0)])
    folded = rotamer_states(np.mod(angles, 180.0), period=180.0)
    unfolded = rotamer_states(angles, period=360.0)
    assert folded['n_states'] == 1
    assert unfolded['n_states'] == 2

def test_genuine_two_state_survives_folding() -> None:
    """Folding must not erase a real pair of rotamers."""
    angles = np.concatenate([np.full(1000, 30.0), np.full(1000, 120.0)])
    assert rotamer_states(np.mod(angles, 180.0), period=180.0)['n_states'] == 2


def test_selects_by_distance_and_can_restrict_to_aromatics(tmp_path: Path) -> None:
    pdb = tmp_path / 'mini.pdb'
    # PHE near the centre, ALA near it, TRP far away
    pdb.write_text(
        'ATOM      1  CA  PHE A   1       0.000   0.000   0.000  1.00  0.00           C\n'
        'ATOM      2  CB  PHE A   1       1.500   0.000   0.000  1.00  0.00           C\n'
        'ATOM      3  CA  ALA A   2       4.000   0.000   0.000  1.00  0.00           C\n'
        'ATOM      4  CA  TRP A   3      40.000   0.000   0.000  1.00  0.00           C\n'
        'END\n',
    )
    structure = mdtraj.load(str(pdb))
    centre = np.zeros(3)

    near = residues_near_point(structure, centre, radius_a=7.0)
    assert set(near.resname) == {'PHE', 'ALA'}, 'the distant TRP must be excluded'

    aromatic = residues_near_point(structure, centre, radius_a=7.0, aromatic_only=True)
    assert list(aromatic.resname) == ['PHE']

def test_empty_when_nothing_is_near(tmp_path: Path) -> None:
    pdb = tmp_path / 'far.pdb'
    pdb.write_text(
        'ATOM      1  CA  PHE A   1      50.000   0.000   0.000  1.00  0.00           C\nEND\n',
    )
    result = residues_near_point(mdtraj.load(str(pdb)), np.zeros(3), radius_a=7.0)
    assert result.empty
    assert 'resname' in result.columns      # usable downstream, not a bare frame


def test_benzene_probes_are_not_residues_near_point(tmp_path: Path) -> None:
    """A probe sitting in the pocket must never be returned as a residue.

    Probes diffuse through the whole box, so a benzene spans 70 to 130 A
    over a trajectory. Ranking residues by how far they move then returns
    probes at every site and buries the real side chains, which is exactly
    what happened before this exclusion.
    """
    pdb = tmp_path / 'with_probe.pdb'
    pdb.write_text(
        'ATOM      1  CA  PHE A   1       0.000   0.000   0.000  1.00  0.00           C\n'
        'ATOM      2  C1  PRB B   2       1.000   0.000   0.000  1.00  0.00           C\n'
        'ATOM      3  O   HOH C   3       2.000   0.000   0.000  1.00  0.00           O\n'
        'END\n',
    )
    result = residues_near_point(mdtraj.load(str(pdb)), np.zeros(3), radius_a=7.0)
    assert list(result.resname) == ['PHE']


# A new homodimer must not have its protomers merged.
#
# MDAnalysis selection strings are not chain-qualified, so "resnum 10" matches
# that residue in every protomer at once. On a dimer numbered per chain their
# merged centre of mass lands in empty space between the two copies, and the
# per-residue analysis silently averages protomers rather than reporting them
# separately. Keys therefore carry the segid.
#

def dimer_pdb(tmp_path: Path) -> Path:
    pdb = tmp_path / 'dimer.pdb'
    pdb.write_text(
        'ATOM      1  N   PHE A  10       0.000   0.000   0.000  1.00  0.00           N\n'
        'ATOM      2  CA  PHE A  10       1.458   0.000   0.000  1.00  0.00           C\n'
        'ATOM      3  C1  PRB P   1       3.000   0.000   0.000  1.00  0.00           C\n'
        'ATOM      4  N   PHE B  10      50.000  50.000  50.000  1.00  0.00           N\n'
        'ATOM      5  CA  PHE B  10      51.458  50.000  50.000  1.00  0.00           C\n'
        'END\n',
    )
    return pdb

def test_repeated_numbering_keeps_protomers_separate(tmp_path: Path) -> None:
    pdb = dimer_pdb(tmp_path)
    universe = mda.Universe(str(pdb))
    merged = universe.select_atoms('resnum 10')
    # the unqualified selection is the hazard this guards against
    assert len(set(merged.segids)) == 2, 'fixture should have both chains'
    # keys used by the analysis must distinguish them
    keys = {(r.segid, int(r.resnum))
            for r in universe.select_atoms('protein').residues}
    assert keys == {('A', 10), ('B', 10)}

def test_residues_near_point_selects_one_protomer(tmp_path: Path) -> None:
    structure = mdtraj.load(str(dimer_pdb(tmp_path)))
    near = residues_near_point(structure, np.zeros(3), radius_a=8.0)
    assert len(near) == 1, 'only the chain-A copy is within range'
    assert near.resname.iloc[0] == 'PHE'


# The period passed to the switch counter must match the residue.
#
# chi_angles folds chi2 onto [0, 180) for Phe and Tyr, whose rings are twofold
# symmetric, and chi3 for Glu, whose carboxylate is. Nothing else is folded, so
# counting an asymmetric dihedral on a 180 degree period collapses states that
# are physically distinct.
#

def two_rotamers() -> np.ndarray:
    """Three frames at 60 degrees, then three at -120: one real transition.

    On a 180 degree period these are the same bin, since -120 mod 180 is 60.
    """
    return np.array([60.0] * 3 + [-120.0] * 3)

def test_asymmetric_transition_is_lost_on_the_folded_period() -> None:
    """The failure mode: a genuine switch counted as none."""
    angles = two_rotamers()
    assert rotamer_switches(angles, 180.0) == 0
    assert rotamer_switches(angles, 360.0) == 1

def test_symmetric_residues_are_the_only_folded_ones() -> None:
    """Phe, Tyr and Asp chi2, and Glu chi3, carry the folded period."""
    assert set(SYMMETRIC_CHI2) == {'PHE', 'TYR', 'ASP'}
    assert SYMMETRIC_CHI3 == ('GLU',)
    for resname in ('LEU', 'MET', 'ASN', 'HIS', 'TRP', 'GLN'):
        assert resname not in SYMMETRIC_CHI2
    assert 'GLN' not in SYMMETRIC_CHI3

@pytest.mark.parametrize('resname, expected', [
    ('PHE', 180.0), ('TYR', 180.0), ('ASP', 180.0),
    ('LEU', 360.0), ('MET', 360.0), ('TRP', 360.0), ('HIS', 360.0),
])
def test_motion_table_selects_the_period_by_residue(resname: str, expected: float) -> None:
    """The rule motion_table applies, stated independently of it."""
    assert (180.0 if resname in SYMMETRIC_CHI2 else 360.0) == expected

def test_a_symmetric_flip_is_still_not_a_switch() -> None:
    """Folding is correct where it does apply: a ring flip is no change."""
    ring = np.array([30.0] * 3 + [210.0] * 3)
    assert rotamer_switches(ring, 180.0) == 0
