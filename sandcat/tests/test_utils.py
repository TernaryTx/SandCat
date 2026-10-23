"""Tests for sandcat.utils — requires openmmforcefields for phosaa14SB templates."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip('openmmforcefields')

import openmm.app as app  # noqa: E402

from sandcat.utils import _patch_nonstandard_residue_bonds, load_protein_pdb  # noqa: E402

DATA = Path(__file__).parent / 'data'


def test_load_protein_pdb_patches_sep_internal_bonds() -> None:
    """SEP residue without CONECT records must have all internal bonds after load.

    Regression guard for the private ForceField._templates API path in
    _patch_nonstandard_residue_bonds: if OpenMM renames or removes the
    attribute this test will catch it before production jobs silently fall
    back to OpenFF 2.2.0.
    """
    pdb = load_protein_pdb(str(DATA / 'ala_sep.pdb'))
    topology = pdb.topology

    bonded_atoms = {a.index for b in topology.bonds() for a in (b.atom1, b.atom2)}

    sep_residues = [r for r in topology.residues() if r.name == 'SEP']
    assert sep_residues, 'No SEP residue found in fixture'

    for res in sep_residues:
        unbonded = [a.name for a in res.atoms() if a.index not in bonded_atoms]
        assert not unbonded, (
            f'SEP atoms still unbonded after patching: {unbonded!r}'
        )


def test_patch_nonstandard_residue_bonds_is_noop_for_standard_residues() -> None:
    """Calling the patcher on an all-standard PDB must not raise or change bonds."""
    pdb = app.PDBFile(str(DATA / 'mini_peptide.pdb'))
    bonds_before = list(pdb.topology.bonds())
    _patch_nonstandard_residue_bonds(pdb.topology)
    bonds_after = list(pdb.topology.bonds())
    assert len(bonds_before) == len(bonds_after)
