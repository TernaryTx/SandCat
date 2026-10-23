from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openmm import unit

from sandcat.pipeline.run_unbiased_md import (
    add_reporters,
    build_fresh_simulation,
    can_restart_md,
    main,
    restart_simulation,
    validate_msmd_restart_parameters,
)
from sandcat.structure import PdbFile


@pytest.fixture()
def mock_protein(tmp_path):
    pdb = tmp_path / 'protein.pdb'
    pdb.write_text('ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n')
    return PdbFile(str(pdb))


def _make_simulation_mock(topology_mock):
    sim = MagicMock()
    sim.topology = topology_mock
    return sim


@patch('sandcat.pipeline.run_unbiased_md.wrap_align_trajectory')
@patch('sandcat.pipeline.run_unbiased_md.mdt.Topology.from_openmm')
@patch('sandcat.pipeline.run_unbiased_md.add_reporters')
@patch('sandcat.pipeline.run_unbiased_md.build_fresh_simulation')
@patch('sandcat.pipeline.run_unbiased_md.can_restart_md', return_value=False)
def test_postprocessing_called_with_correct_args(
        mock_can_restart,
        mock_build,
        mock_add_reporters,
        mock_from_openmm,
        mock_wrap_align,
        mock_protein,
        tmp_path,
):
    """wrap_align_trajectory is called with the right paths and remove_solvent flag."""
    md_topology = MagicMock()
    mock_from_openmm.return_value = md_topology

    md_traj = MagicMock()
    mock_wrap_align.return_value = md_traj

    simulation = MagicMock()
    mock_build.return_value = (
        simulation,
        100,
        {
            'revision': 'probe-intermolecular-repulsion-v3',
            'modified_probe_14_exceptions': 0,
        },
    )

    output_prefix = str(tmp_path / 'md_output')
    main(mock_protein, output_prefix=output_prefix, remove_solvent=True)

    dcd_path = f'{output_prefix}.dcd'
    mock_from_openmm.assert_called_once_with(simulation.topology)
    mock_wrap_align.assert_called_once_with(
        dcd_path, md_topology, mock_protein.path, True,
    )
    assert mock_add_reporters.call_args.args[2] == f'{output_prefix}.xml'
    simulation.saveState.assert_called_once_with(f'{output_prefix}.xml')
    md_traj.save.assert_called_once_with(f'aligned_{output_prefix}.dcd')
    md_traj.__getitem__.return_value.save.assert_called_once_with(f'{output_prefix}_last_frame.pdb')
    provenance_path = Path(f'{output_prefix}_provenance.json')
    assert provenance_path.is_file()
    assert '"workflow": "msmd"' in provenance_path.read_text()


@patch('sandcat.pipeline.run_unbiased_md.wrap_align_trajectory')
@patch('sandcat.pipeline.run_unbiased_md.mdt.Topology.from_openmm')
@patch('sandcat.pipeline.run_unbiased_md.add_reporters')
@patch('sandcat.pipeline.run_unbiased_md.build_fresh_simulation')
@patch('sandcat.pipeline.run_unbiased_md.can_restart_md', return_value=False)
def test_keep_solvent_passes_false_to_wrap_align(
        mock_can_restart,
        mock_build,
        mock_add_reporters,
        mock_from_openmm,
        mock_wrap_align,
        mock_protein,
        tmp_path,
):
    """remove_solvent=False is forwarded when keep_solvent is requested."""
    mock_from_openmm.return_value = MagicMock()
    mock_wrap_align.return_value = MagicMock()
    mock_build.return_value = (
        MagicMock(),
        100,
        {
            'revision': 'probe-intermolecular-repulsion-v3',
            'modified_probe_14_exceptions': 0,
        },
    )

    output_prefix = str(tmp_path / 'md_output')
    main(mock_protein, output_prefix=output_prefix, remove_solvent=False)

    _, _, _, remove_solvent_arg = mock_wrap_align.call_args.args
    assert remove_solvent_arg is False


@patch('sandcat.pipeline.run_unbiased_md.LangevinMiddleIntegrator')
@patch('sandcat.pipeline.run_unbiased_md.Platform.getPlatformByName')
@patch('sandcat.pipeline.run_unbiased_md.app.Simulation')
@patch('sandcat.pipeline.run_unbiased_md.app.PDBFile.writeFile')
@patch('sandcat.pipeline.run_unbiased_md.minimise_equilibrate_npt_sandcat')
@patch('sandcat.pipeline.run_unbiased_md.add_probe_repulsion')
@patch('sandcat.pipeline.run_unbiased_md.build_multi_ligand_solvent_system')
def test_build_fresh_multi_ligand_simulation_uses_one_copy_system_path(
        mock_build_multi: MagicMock,
        mock_add_repulsion: MagicMock,
        mock_equilibrate: MagicMock,
        mock_write_pdb: MagicMock,
        mock_simulation: MagicMock,
        mock_platform: MagicMock,
        mock_integrator: MagicMock,
        mock_protein: PdbFile,
        tmp_path: Path,
) -> None:
    system = MagicMock()
    omm_model = MagicMock()
    mock_build_multi.return_value = (system, omm_model, MagicMock())
    mock_equilibrate.return_value = (MagicMock(), MagicMock())

    build_fresh_simulation(
        input_protein=mock_protein,
        forcefield='ff19',
        probe_types=['ligand_a.sdf', 'ligand_b.sdf'],
        probe_ff_paths=['ligand_a.offxml', 'ligand_b.offxml'],
        conc_mol_per_litre=None,
        add_hydrogens=False,
        timestep=4 * unit.femtosecond,
        seed=42,
        ref_pdb_path=str(tmp_path / 'md_output_ref.pdb'),
        total_steps=100,
        ensemble='NVT',
    )

    mock_build_multi.assert_called_once_with(
        forcefield='ff19',
        protein_pdb_path=mock_protein.path,
        probe_types=['ligand_a.sdf', 'ligand_b.sdf'],
        probe_ff_paths=['ligand_a.offxml', 'ligand_b.offxml'],
        box_info_json_path=None,
        seed=42,
    )
    assert (
        mock_add_repulsion.call_args.kwargs['correct_probe_14_exceptions']
        is False
    )
    assert mock_add_repulsion.call_args.kwargs['probe_resname'] == ['P00', 'P01']
    assert mock_equilibrate.call_args.kwargs['additional_solvent_resnames'] == [
        'P00',
        'P01',
    ]


@patch('sandcat.pipeline.run_unbiased_md.wrap_align_trajectory')
@patch('sandcat.pipeline.run_unbiased_md.mdt.Topology.from_openmm')
@patch('sandcat.pipeline.run_unbiased_md.add_reporters')
@patch('sandcat.pipeline.run_unbiased_md.restart_simulation')
@patch('sandcat.pipeline.run_unbiased_md.create_probe_system_generator')
@patch('sandcat.pipeline.run_unbiased_md.can_restart_md', return_value=True)
def test_restart_without_simulation_provenance_backfills_file(
        mock_can_restart: MagicMock,
        mock_create_generator: MagicMock,
        mock_restart: MagicMock,
        mock_add_reporters: MagicMock,
        mock_from_openmm: MagicMock,
        mock_wrap_align: MagicMock,
        mock_protein: PdbFile,
        tmp_path: Path,
) -> None:
    """A legacy MSMD restart gains simulation provenance when it has none."""
    simulation = MagicMock()
    mock_restart.return_value = (
        simulation,
        0,
        {
            'revision': 'probe-intermolecular-repulsion-v3',
            'modified_probe_14_exceptions': 0,
        },
    )
    mock_create_generator.return_value = (MagicMock(), MagicMock())
    mock_from_openmm.return_value = MagicMock()
    mock_wrap_align.return_value = MagicMock()
    output_prefix = str(tmp_path / 'md_output')

    with patch('sandcat.pipeline.run_unbiased_md.reconcile_msmd_restart'):
        main(mock_protein, output_prefix=output_prefix)

    provenance_text = Path(f'{output_prefix}_provenance.json').read_text()
    assert '"workflow": "msmd"' in provenance_text
    assert '"provenance_reconstructed_on_restart": true' in provenance_text


@pytest.mark.parametrize(
    'parameter,requested_value',
    [
        ('forcefield', 'ff19'),
        ('probe_type', 'pyrimidine'),
        ('probe_ff_path', 'different.offxml'),
        ('ensemble', 'NVT'),
        ('time_step_fs', 2),
    ],
)
def test_msmd_restart_rejects_changed_hamiltonian_parameters(
        parameter: str,
        requested_value: object,
        tmp_path: Path,
) -> None:
    output_prefix = str(tmp_path / 'md_output')
    recorded_parameters = {
        'forcefield': 'desamber',
        'probe_type': 'benzene',
        'probe_ff_path': None,
        'ensemble': 'NPT',
        'time_step_fs': 4,
    }
    Path(f'{output_prefix}_provenance.json').write_text(json.dumps({
        'revision': 'probe-intermolecular-repulsion-v3',
        'workflow': 'msmd',
        'parameters': recorded_parameters,
    }))
    requested_parameters = dict(recorded_parameters)
    requested_parameters[parameter] = requested_value

    with pytest.raises(RuntimeError, match=parameter):
        validate_msmd_restart_parameters(
            output_prefix,
            requested_parameters,
        )


def test_msmd_restart_rejects_legacy_multi_ligand_residue_mapping(
        tmp_path: Path,
) -> None:
    output_prefix = str(tmp_path / 'md_output')
    Path(f'{output_prefix}_provenance.json').write_text(json.dumps({
        'revision': 'probe-intermolecular-repulsion-v3',
        'workflow': 'msmd',
        'parameters': {
            'forcefield': 'ff19',
            'probe_type': ['ligand_a.sdf', 'ligand_b.sdf'],
            'probe_ff_path': ['ligand_a.offxml', 'ligand_b.offxml'],
            'ensemble': 'NPT',
            'time_step_fs': 4,
        },
    }))

    with pytest.raises(RuntimeError, match='probe_resname_mapping'):
        validate_msmd_restart_parameters(
            output_prefix,
            {
                'forcefield': 'ff19',
                'probe_type': ['ligand_a.sdf', 'ligand_b.sdf'],
                'probe_ff_path': ['ligand_a.offxml', 'ligand_b.offxml'],
                'probe_resname_mapping': [
                    {'probe_type': 'ligand_a.sdf', 'resname': 'P00'},
                    {'probe_type': 'ligand_b.sdf', 'resname': 'P01'},
                ],
                'ensemble': 'NPT',
                'time_step_fs': 4,
            },
        )


@patch('sandcat.pipeline.run_unbiased_md.create_probe_system_generator')
@patch('sandcat.pipeline.run_unbiased_md.can_restart_md', return_value=True)
def test_msmd_restart_validates_parameters_before_rebuilding_system(
        mock_can_restart: MagicMock,
        mock_create_generator: MagicMock,
        mock_protein: PdbFile,
        tmp_path: Path,
) -> None:
    output_prefix = str(tmp_path / 'md_output')
    Path(f'{output_prefix}_provenance.json').write_text(json.dumps({
        'revision': 'probe-intermolecular-repulsion-v3',
        'workflow': 'msmd',
        'parameters': {
            'forcefield': 'desamber',
            'probe_type': 'benzene',
            'probe_ff_path': None,
            'ensemble': 'NPT',
            'time_step_fs': 4,
        },
    }))

    with pytest.raises(RuntimeError, match='forcefield'):
        main(
            mock_protein,
            output_prefix=output_prefix,
            forcefield='ff19',
        )

    mock_create_generator.assert_not_called()


def test_can_restart_md_requires_xml_state(tmp_path: Path) -> None:
    """Only an XML state with its DCD and reference PDB enables restart."""
    output_prefix = tmp_path / 'md_output'
    Path(f'{output_prefix}.dcd').touch()
    Path(f'{output_prefix}_ref.pdb').touch()
    Path(f'{output_prefix}.chk').touch()

    assert can_restart_md(str(output_prefix)) is False

    Path(f'{output_prefix}.xml').touch()

    assert can_restart_md(str(output_prefix)) is True


@patch('sandcat.pipeline.run_unbiased_md.app.CheckpointReporter')
@patch('sandcat.pipeline.run_unbiased_md.app.DCDReporter')
@patch('sandcat.pipeline.run_unbiased_md.get_n_steps', return_value=25)
def test_add_reporters_writes_xml_state(
        mock_get_n_steps: MagicMock,
        mock_dcd_reporter: MagicMock,
        mock_checkpoint_reporter: MagicMock,
        tmp_path: Path,
) -> None:
    """The OpenMM checkpoint reporter writes a serialized XML State."""
    simulation = MagicMock()
    simulation.reporters = []
    dcd_path = str(tmp_path / 'md_output.dcd')
    state_xml_path = str(tmp_path / 'md_output.xml')

    add_reporters(
        simulation,
        dcd_path,
        state_xml_path,
        timestep=MagicMock(),
        verbose=False,
    )

    mock_dcd_reporter.assert_called_once_with(dcd_path, 25, append=False)
    mock_checkpoint_reporter.assert_called_once_with(
        state_xml_path,
        25,
        writeState=True,
    )
    assert simulation.reporters == [
        mock_dcd_reporter.return_value,
        mock_checkpoint_reporter.return_value,
    ]


@patch('sandcat.pipeline.run_unbiased_md.add_probe_repulsion')
@patch('sandcat.pipeline.run_unbiased_md.add_custom_metal_bonds')
@patch('sandcat.pipeline.run_unbiased_md.Platform.getPlatformByName')
@patch('sandcat.pipeline.run_unbiased_md.app.Simulation')
@patch('sandcat.pipeline.run_unbiased_md.app.Modeller')
@patch('sandcat.pipeline.run_unbiased_md.app.PDBFile')
@patch('sandcat.pipeline.run_unbiased_md.LangevinMiddleIntegrator')
def test_restart_simulation_loads_xml_state(
        mock_integrator: MagicMock,
        mock_pdb_file: MagicMock,
        mock_modeller: MagicMock,
        mock_simulation_class: MagicMock,
        mock_get_platform: MagicMock,
        mock_add_custom_metal_bonds: MagicMock,
        mock_add_probe_repulsion: MagicMock,
        tmp_path: Path,
) -> None:
    """Restart restores the portable XML state instead of a binary checkpoint."""
    ref_pdb = mock_pdb_file.return_value
    omm_model = mock_modeller.return_value
    system_generator = MagicMock()
    simulation = mock_simulation_class.return_value
    simulation.currentStep = 40
    state_xml_path = str(tmp_path / 'md_output.xml')

    _, steps_remaining, probe_provenance = restart_simulation(
        state_xml_path=state_xml_path,
        ref_pdb_path=str(tmp_path / 'md_output_ref.pdb'),
        timestep=4 * unit.femtosecond,
        total_steps=100,
        ensemble='NVT',
        system_generator=system_generator,
    )

    simulation.loadState.assert_called_once_with(state_xml_path)
    simulation.loadCheckpoint.assert_not_called()
    assert steps_remaining == 60
    assert probe_provenance == mock_add_probe_repulsion.return_value
