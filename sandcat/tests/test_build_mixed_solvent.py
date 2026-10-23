"""Tests for the force-field system builder (sandcat.utils.build_mixed_solvent_system).

Checks that the ffxml path builds and returns a SystemGenerator. Skips when
openmmforcefields (the ffxml dependency that utils.py imports at module load) is unavailable.
"""
from __future__ import annotations

import random
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip('openmmforcefields')
from openff.toolkit import Molecule  # noqa: E402
from openmm import Vec3, app, unit  # noqa: E402
from sandcat.msmd_options import normalise_msmd_probe_options  # noqa: E402
from sandcat.utils import (  # noqa: E402
    _add_probe_copies,
    _add_solvent_with_seed,
    _align_small_molecule_14_scales,
    _build_periodic_tree,
    _validate_unique_probes,
    add_mixed_solvent,
    build_multi_ligand_solvent_system,
    build_mixed_solvent_system,
    create_multi_probe_system_generator,
    multi_ligand_probe_resnames,
    resolve_probe_sdf_path,
    uses_historical_probe_14_correction,
)
from sandcat.variables import DESAMBER_BENZENE_FORCEFIELD  # noqa: E402


def test_add_probe_copies_accepts_atoms_without_formal_charge() -> None:
    """OpenMM releases before Atom.formalCharge must remain supported."""
    source = app.Topology()
    chain = source.addChain()
    residue = source.addResidue('PRB', chain)
    source.addAtom('C1', app.Element.getBySymbol('C'), residue)
    model = app.Modeller(app.Topology(), [])

    _add_probe_copies(model, source, [np.zeros((1, 3))])

    assert model.topology.getNumAtoms() == 1


@patch('sandcat.utils.add_custom_metal_bonds')
@patch('sandcat.utils.add_mixed_solvent')
@patch('sandcat.utils.load_probe')
@patch('sandcat.utils.SystemGenerator')
@patch('sandcat.utils.app')
def test_ffxml_branch_returns_system_generator(
        mock_app, mock_sysgen_cls, mock_load_probe, mock_add_mixed, mock_metal_bonds):
    out_system, out_model, system_generator = build_mixed_solvent_system(
        'ff19', '/tmp/protein.pdb', 'benzene', 1.0,
    )
    # ffxml path builds a SystemGenerator and returns it (needed for checkpoint restart)
    mock_sysgen_cls.assert_called_once()
    assert system_generator is mock_sysgen_cls.return_value
    forcefield = mock_sysgen_cls.return_value.forcefield
    forcefield.createSystem.assert_called_once()
    assert out_system is forcefield.createSystem.return_value
    assert (
        mock_add_mixed.call_args.kwargs['preserve_single_water_replacement']
        is True
    )


@pytest.mark.parametrize('probe_type', ['/tmp/custom_probe.sdf', 'benzene'])
@patch('sandcat.utils.add_custom_metal_bonds')
@patch('sandcat.utils.add_mixed_solvent')
@patch('sandcat.utils.load_probe')
@patch('sandcat.utils.SystemGenerator')
@patch('sandcat.utils.app')
def test_custom_probe_overrides_desamber_benzene_template(
        mock_app,
        mock_sysgen_cls,
        mock_load_probe,
        mock_add_mixed,
        mock_metal_bonds,
        probe_type: str,
) -> None:
    probe_ff_path = '/tmp/custom_probe.offxml'

    build_mixed_solvent_system(
        'desamber',
        '/tmp/protein.pdb',
        probe_type,
        1.0,
        probe_ff_path=probe_ff_path,
    )

    mock_sysgen_cls.assert_called_once()
    call_kwargs = mock_sysgen_cls.call_args.kwargs
    assert call_kwargs['small_molecule_forcefield'] == probe_ff_path
    assert call_kwargs['molecules'] == [mock_load_probe.return_value]
    assert DESAMBER_BENZENE_FORCEFIELD not in call_kwargs['forcefields']
    assert (
        mock_add_mixed.call_args.kwargs['preserve_single_water_replacement']
        is (probe_type == 'benzene')
    )


def test_multi_ligand_options_require_paired_forcefields_without_concentration() -> None:
    probe_types, probe_ff_paths, concentration = normalise_msmd_probe_options(
        probe_types=['ligand_a.sdf', 'ligand_b.sdf'],
        probe_ff_paths=['ligand_a.offxml', 'ligand_b.offxml'],
        conc_mol_per_litre=None,
    )

    assert probe_types == ['ligand_a.sdf', 'ligand_b.sdf']
    assert probe_ff_paths == ['ligand_a.offxml', 'ligand_b.offxml']
    assert concentration is None


@pytest.mark.parametrize(
    'probe_ff_paths,concentration,error_match',
    [
        (['ligand_a.offxml', 'ligand_b.offxml'], 0.25, 'concentration'),
        (['ligand_a.offxml'], None, 'one --probe_ff_path per --probe'),
        (None, None, 'one --probe_ff_path per --probe'),
    ],
)
def test_multi_ligand_options_reject_invalid_pairing(
        probe_ff_paths: list[str] | None,
        concentration: float | None,
        error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        normalise_msmd_probe_options(
            probe_types=['ligand_a.sdf', 'ligand_b.sdf'],
            probe_ff_paths=probe_ff_paths,
            conc_mol_per_litre=concentration,
        )


def test_single_ligand_options_retain_one_molar_default() -> None:
    probe_types, probe_ff_paths, concentration = normalise_msmd_probe_options(
        probe_types=['ligand.sdf'],
        probe_ff_paths=['ligand.offxml'],
        conc_mol_per_litre=None,
    )

    assert probe_types == ['ligand.sdf']
    assert probe_ff_paths == ['ligand.offxml']
    assert concentration == pytest.approx(1.0)


@patch('sandcat.utils._align_template_14_scales')
@patch('sandcat.utils.SMIRNOFFTemplateGenerator')
@patch('sandcat.utils.load_probe')
@patch('sandcat.utils.SystemGenerator')
def test_multi_probe_generator_registers_each_forcefield(
        mock_system_generator: MagicMock,
        mock_load_probe: MagicMock,
        mock_template_generator: MagicMock,
        mock_align_scales: MagicMock,
) -> None:
    first_molecule = MagicMock()
    second_molecule = MagicMock()
    mock_load_probe.side_effect = [first_molecule, second_molecule]
    second_template = MagicMock()
    mock_template_generator.return_value = second_template

    molecules, system_generator = create_multi_probe_system_generator(
        forcefield='desamber',
        probe_types=['ligand_a.sdf', 'ligand_b.sdf'],
        probe_ff_paths=['ligand_a.offxml', 'ligand_b.offxml'],
    )

    assert molecules == [first_molecule, second_molecule]
    assert system_generator is mock_system_generator.return_value
    assert (
        mock_system_generator.call_args.kwargs['small_molecule_forcefield']
        == 'ligand_a.offxml'
    )
    assert mock_system_generator.call_args.kwargs['molecules'] == [first_molecule]
    assert (
        DESAMBER_BENZENE_FORCEFIELD
        not in mock_system_generator.call_args.kwargs['forcefields']
    )
    assert mock_template_generator.call_args.kwargs == {
        'molecules': [second_molecule],
        'forcefield': 'ligand_b.offxml',
    }
    forcefield = mock_system_generator.return_value.forcefield
    assert mock_align_scales.call_count == 2
    assert forcefield.registerTemplateGenerator.call_args.args == (
        second_template.generator,
    )
    forcefield.registerTemplateMatcher.assert_called_once()


def test_graph_identical_stereoisomers_are_supported() -> None:
    first_molecule = Molecule.from_smiles('F[C@H](Cl)Br')
    second_molecule = Molecule.from_smiles('F[C@@H](Cl)Br')

    _validate_unique_probes(
        [first_molecule, second_molecule],
        ['first.sdf', 'second.sdf'],
    )


def test_chemically_identical_ligands_are_rejected() -> None:
    first_molecule = Molecule.from_smiles('CCO')
    second_molecule = Molecule.from_smiles('CCO')

    with pytest.raises(ValueError, match='chemically identical ligands'):
        _validate_unique_probes(
            [first_molecule, second_molecule],
            ['first.sdf', 'second.sdf'],
        )


def test_multi_ligand_resnames_are_stable_and_pdb_compatible() -> None:
    assert multi_ligand_probe_resnames(3) == ['P00', 'P01', 'P02']

    with pytest.raises(ValueError, match='supports 1-100 ligands'):
        multi_ligand_probe_resnames(101)


def test_real_multi_probe_generators_create_one_openmm_system() -> None:
    probe_types = [
        resolve_probe_sdf_path('benzene'),
        resolve_probe_sdf_path('pyrimidine'),
    ]
    molecules, system_generator = create_multi_probe_system_generator(
        forcefield='ff14',
        probe_types=probe_types,
        probe_ff_paths=['openff-2.2.0.offxml', 'openff-2.2.1.offxml'],
    )
    model = app.Modeller(app.Topology(), [])
    for probe_index, molecule in enumerate(molecules):
        topology = molecule.to_topology().to_openmm()
        for residue in topology.residues():
            residue.name = f'P{probe_index:02d}'
        positions = molecule.conformers[0].magnitude * 0.1 * unit.nanometer
        model.add(topology, positions)

    system = system_generator.forcefield.createSystem(
        model.topology,
        nonbondedMethod=app.NoCutoff,
    )

    assert len(system_generator.forcefield._templateGenerators) == 2
    assert system.getNumParticles() == sum(molecule.n_atoms for molecule in molecules)


def test_stereoisomers_use_paired_templates_after_pdb_round_trip(
        tmp_path: Path,
) -> None:
    probe_types = []
    for probe_index, smiles in enumerate(
            ('F[C@H](Cl)Br', 'F[C@@H](Cl)Br')):
        molecule = Molecule.from_smiles(smiles)
        molecule.generate_conformers(n_conformers=1)
        probe_path = tmp_path / f'stereoisomer_{probe_index}.sdf'
        molecule.to_file(probe_path, file_format='sdf')
        probe_types.append(str(probe_path))

    molecules, system_generator = create_multi_probe_system_generator(
        forcefield='ff14',
        probe_types=probe_types,
        probe_ff_paths=['openff-2.2.0.offxml', 'openff-2.2.1.offxml'],
    )
    model = app.Modeller(app.Topology(), [])
    for probe_resname, molecule in zip(
            ('P00', 'P01'), molecules, strict=True):
        topology = molecule.to_topology().to_openmm()
        for residue in topology.residues():
            residue.name = probe_resname
        model.add(topology, molecule.conformers[0].to_openmm())

    pdb_text = StringIO()
    app.PDBFile.writeFile(model.topology, model.positions, pdb_text)
    reloaded_pdb = app.PDBFile(StringIO(pdb_text.getvalue()))
    restart_model = app.Modeller(reloaded_pdb.topology, reloaded_pdb.positions)
    restart_model.addExtraParticles(system_generator.forcefield)
    system = system_generator.forcefield.createSystem(
        restart_model.topology,
        nonbondedMethod=app.NoCutoff,
    )
    matched_templates = system_generator.forcefield.getMatchingTemplates(
        restart_model.topology,
    )

    assert [template.name for template in matched_templates] == [
        molecule.to_smiles() for molecule in molecules
    ]
    assert matched_templates[0] is not matched_templates[1]
    assert system.getNumParticles() == sum(molecule.n_atoms for molecule in molecules)


@patch('sandcat.utils.add_custom_metal_bonds')
@patch('sandcat.utils.add_mixed_solvent')
@patch('sandcat.utils._add_solvent_with_seed')
@patch('sandcat.utils.create_multi_probe_system_generator')
@patch('sandcat.utils.app.PDBFile')
def test_multi_ligand_system_adds_one_copy_of_each_ligand(
        mock_pdb_file: MagicMock,
        mock_create_generator: MagicMock,
        mock_add_solvent: MagicMock,
        mock_add_mixed_solvent: MagicMock,
        mock_add_metal_bonds: MagicMock,
) -> None:
    first_molecule = MagicMock()
    second_molecule = MagicMock()
    system_generator = MagicMock()
    mock_create_generator.return_value = (
        [first_molecule, second_molecule],
        system_generator,
    )

    build_multi_ligand_solvent_system(
        forcefield='ff19',
        protein_pdb_path='protein.pdb',
        probe_types=['ligand_a.sdf', 'ligand_b.sdf'],
        probe_ff_paths=['ligand_a.offxml', 'ligand_b.offxml'],
        seed=20,
    )

    assert mock_add_solvent.call_count == 1
    assert mock_add_mixed_solvent.call_count == 2
    assert mock_add_mixed_solvent.call_args_list[0].kwargs == {
        'probe_mol': first_molecule,
        'probe_resname': 'P00',
        'seed': 20,
        'probe_count': 1,
    }
    assert mock_add_mixed_solvent.call_args_list[1].kwargs == {
        'probe_mol': second_molecule,
        'probe_resname': 'P01',
        'seed': 21,
        'probe_count': 1,
    }
    system_generator.forcefield.createSystem.assert_called_once()

def test_equivalent_custom_probe_14_scales_are_aligned_exactly() -> None:
    template_generator = SimpleNamespace(
        _coulomb14scale='0.8333333333',
        _lj14scale='0.5',
    )
    protein_generator = SimpleNamespace(
        coulomb14scale=0.8333,
        lj14scale=0.5,
    )
    system_generator = SimpleNamespace(
        template_generator=template_generator,
        forcefield=SimpleNamespace(
            getGenerators=lambda: [protein_generator],
        ),
    )

    _align_small_molecule_14_scales(system_generator)

    assert template_generator._coulomb14scale == str(
        protein_generator.coulomb14scale,
    )
    assert template_generator._lj14scale == str(protein_generator.lj14scale)


def test_non_equivalent_custom_probe_14_scales_are_rejected() -> None:
    template_generator = SimpleNamespace(
        _coulomb14scale='1.0',
        _lj14scale='1.0',
    )
    protein_generator = SimpleNamespace(
        coulomb14scale=0.8333,
        lj14scale=0.5,
    )
    system_generator = SimpleNamespace(
        template_generator=template_generator,
        forcefield=SimpleNamespace(
            getGenerators=lambda: [protein_generator],
        ),
    )

    with pytest.raises(ValueError, match='unsupported 1-4 scaling factors'):
        _align_small_molecule_14_scales(system_generator)


@pytest.mark.parametrize(
    'probe_type,probe_ff_path,expected',
    [
        ('benzene', None, True),
        ('benzene', 'custom.offxml', False),
        ('custom.sdf', None, False),
    ],
)
def test_historical_probe_14_correction_requires_unmodified_builtin(
        probe_type: str,
        probe_ff_path: str | None,
        expected: bool,
) -> None:
    assert uses_historical_probe_14_correction(
        probe_type,
        probe_ff_path,
    ) is expected


def test_resolve_probe_sdf_path_accepts_custom_sdf(tmp_path) -> None:
    probe_path = tmp_path / 'custom.sdf'
    probe_path.write_text('custom probe')

    assert resolve_probe_sdf_path(str(probe_path)) == str(probe_path)


def test_resolve_probe_sdf_path_rejects_non_sdf_path(tmp_path) -> None:
    probe_path = tmp_path / 'custom.mol2'
    probe_path.write_text('custom probe')

    with pytest.raises(ValueError, match='SDF path'):
        resolve_probe_sdf_path(str(probe_path))


def test_add_solvent_with_seed_is_reproducible_and_restores_random_state() -> None:
    random_draws = []
    test_start_state = random.getstate()

    class RecordingModeller:
        def addSolvent(self, forcefield: object, **kwargs: object) -> None:
            random_draws.append(random.random())

    try:
        random.seed(2026)
        original_state = random.getstate()
        modeller = RecordingModeller()
        for _ in range(2):
            _add_solvent_with_seed(
                omm_model=modeller,
                forcefield=object(),
                water_model='tip4pew',
                ionic_strength_molar=0.15,
                box_kwargs={},
                seed=42,
            )
            assert random.getstate() == original_state

        assert random_draws[0] == random_draws[1]
    finally:
        random.setstate(test_start_state)


def test_periodic_tree_finds_neighbours_across_box_boundary() -> None:
    tree, source_indices = _build_periodic_tree(
        np.array([[2.95, 1.0, 1.0]]),
        np.diag([3.0, 3.0, 3.0]),
    )

    assert tree is not None
    neighbours = tree.query_ball_point(
        np.array([[0.05, 1.0, 1.0]]),
        0.11,
    )
    assert {
        int(source_indices[tree_index])
        for atom_neighbours in neighbours
        for tree_index in atom_neighbours
    } == {0}


@patch('sandcat.utils._n_probes_for_concentration', return_value=2)
def test_ligand_probe_removes_all_overlapping_waters(
        mock_n_probes: MagicMock,
) -> None:
    topology = app.Topology()
    chain = topology.addChain()
    positions = []
    for x_coord in np.arange(0.5, 2.6, 0.2):
        for y_coord in np.arange(0.5, 2.6, 0.2):
            residue = topology.addResidue('HOH', chain)
            topology.addAtom('O', app.element.oxygen, residue)
            positions.append(Vec3(x_coord, y_coord, 1.5))
    topology.setPeriodicBoxVectors((
        Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3),
    ) * unit.nanometer)
    model = app.Modeller(topology, positions * unit.nanometer)

    probe = Molecule.from_smiles('CCCC')
    probe.generate_conformers(n_conformers=1)
    add_mixed_solvent(
        model,
        probe,
        seed=7,
        clash_distance_nm=0.15,
        solvent_clearance_nm=0.25,
    )

    final_positions = np.asarray(
        model.positions.value_in_unit(unit.nanometer),
    )
    probe_positions = np.array([
        final_positions[atom.index]
        for atom in model.topology.atoms()
        if atom.residue.name == 'PRB' and atom.element.symbol != 'H'
    ])
    water_positions = np.array([
        final_positions[atom.index]
        for atom in model.topology.atoms()
        if atom.residue.name == 'HOH' and atom.element.symbol == 'O'
    ])
    distances = np.linalg.norm(
        probe_positions[:, np.newaxis, :] - water_positions[np.newaxis, :, :],
        axis=2,
    )

    assert sum(residue.name == 'PRB' for residue in model.topology.residues()) == 2
    assert distances.min() >= 0.25


def test_sequential_distinct_ligands_avoid_each_other() -> None:
    topology = app.Topology()
    chain = topology.addChain()
    positions = []
    for x_coord in np.arange(0.5, 2.6, 0.2):
        for y_coord in np.arange(0.5, 2.6, 0.2):
            residue = topology.addResidue('HOH', chain)
            topology.addAtom('O', app.element.oxygen, residue)
            positions.append(Vec3(x_coord, y_coord, 1.5))
    topology.setPeriodicBoxVectors((
        Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3),
    ) * unit.nanometer)
    model = app.Modeller(topology, positions * unit.nanometer)
    first_probe = Molecule.from_smiles('CCCC')
    second_probe = Molecule.from_smiles('CCO')
    first_probe.generate_conformers(n_conformers=1)
    second_probe.generate_conformers(n_conformers=1)

    add_mixed_solvent(
        model,
        first_probe,
        probe_resname='P00',
        probe_count=1,
        seed=7,
    )
    add_mixed_solvent(
        model,
        second_probe,
        probe_resname='P01',
        probe_count=1,
        seed=8,
    )

    final_positions = np.asarray(
        model.positions.value_in_unit(unit.nanometer),
    )
    heavy_positions_by_resname = {
        probe_resname: np.array([
            final_positions[atom.index]
            for atom in model.topology.atoms()
            if atom.residue.name == probe_resname
            and atom.element is not None
            and atom.element.symbol != 'H'
        ])
        for probe_resname in ('P00', 'P01')
    }
    distances = np.linalg.norm(
        heavy_positions_by_resname['P00'][:, np.newaxis, :]
        - heavy_positions_by_resname['P01'][np.newaxis, :, :],
        axis=2,
    )

    assert distances.min() >= 0.2


@patch('sandcat.utils._n_probes_for_concentration', return_value=2)
def test_bundled_probe_replaces_exactly_one_water_per_probe(
        mock_n_probes: MagicMock,
) -> None:
    topology = app.Topology()
    chain = topology.addChain()
    positions = []
    for x_coord in np.arange(0.5, 2.6, 0.2):
        residue = topology.addResidue('HOH', chain)
        topology.addAtom('O', app.element.oxygen, residue)
        positions.append(Vec3(x_coord, 1.5, 1.5))
    model = app.Modeller(topology, positions * unit.nanometer)
    initial_water_count = len(positions)

    probe = Molecule.from_smiles('c1ccccc1')
    probe.generate_conformers(n_conformers=1)
    add_mixed_solvent(
        model,
        probe,
        seed=7,
        preserve_single_water_replacement=True,
    )

    final_water_count = sum(
        residue.name == 'HOH' for residue in model.topology.residues()
    )
    final_probe_count = sum(
        residue.name == 'PRB' for residue in model.topology.residues()
    )
    assert final_water_count == initial_water_count - 2
    assert final_probe_count == 2
