"""Tests for the native OpenMM ForceField XML port of DES-Amber-SF1.0 (+ TIP4P-D + benzene).

The XML was validated per-term against GROMACS when it was generated. These tests guard:
  * the three XML files load and cover all biologically relevant rtp residue templates,
  * a protein solvates with TIP4P-D natively (no gmx) with one M-site virtual site per water,
  * desamber routes through the ffxml branch and parametrises benzene from the PRB template.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('openmm')
from openmm import app, unit  # noqa: E402

from sandcat.variables import (  # noqa: E402
    ALL_FORCEFIELDS, DESAMBER_FORCEFIELDS, FORCEFIELD_OPTIONS, uses_template_probe,
)
from sandcat.utils import create_probe_system_generator  # noqa: E402

MINI_PDB = Path(__file__).parent / 'data' / 'mini_peptide.pdb'


def test_registered_and_paths_exist():
    assert 'desamber' in ALL_FORCEFIELDS
    xmls, water = FORCEFIELD_OPTIONS['desamber']
    assert water == 'tip4pew'
    assert len(xmls) == 3 and all(os.path.exists(p) for p in xmls)
    assert uses_template_probe('desamber')
    assert not uses_template_probe('ff19')


def test_forcefield_loads_with_water_and_probe_templates():
    ff = app.ForceField(*DESAMBER_FORCEFIELDS)
    names = set(ff._templates)
    # protein + variants + water + benzene probe templates all present
    assert {'ALA', 'HID', 'HIE', 'HIP', 'CYX', 'HOH', 'PRB'} <= names
    # HOH carries the TIP4P-D M-site virtual particle
    hoh = ff._templates['HOH']
    assert any(vs is not None for vs in hoh.virtualSites)


def test_native_solvation_no_gmx():
    """End-to-end native build: prep -> addHydrogens -> TIP4P-D addSolvent -> createSystem."""
    pdbfixer = pytest.importorskip('pdbfixer')
    fixer = pdbfixer.PDBFixer(filename=str(MINI_PDB))
    fixer.findMissingResidues(); fixer.findMissingAtoms(); fixer.addMissingAtoms()
    ff = app.ForceField(*DESAMBER_FORCEFIELDS)
    mod = app.Modeller(fixer.topology, fixer.positions)
    mod.addHydrogens(ff)
    mod.addSolvent(ff, model='tip4pew', padding=0.8 * unit.nanometer, neutralize=True)
    n_wat = sum(1 for r in mod.topology.residues() if r.name == 'HOH')
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds,
                             rigidWater=True, hydrogenMass=4 * unit.amu)
    n_vsite = sum(system.isVirtualSite(i) for i in range(system.getNumParticles()))
    assert n_wat > 0 and n_vsite == n_wat


def test_desamber_template_is_only_used_for_builtin_benzene() -> None:
    assert uses_template_probe('desamber', 'benzene')
    assert not uses_template_probe('desamber', '/tmp/custom.sdf')
    assert not uses_template_probe(
        'desamber',
        'benzene',
        '/tmp/custom_benzene.offxml',
    )


def test_desamber_accepts_registered_smirnoff_probe() -> None:
    probe_mol, system_generator = create_probe_system_generator(
        'desamber',
        'pyrimidine',
    )

    system = system_generator.forcefield.createSystem(
        probe_mol.to_topology().to_openmm(),
        nonbondedMethod=app.NoCutoff,
    )

    assert system.getNumParticles() == probe_mol.n_atoms
