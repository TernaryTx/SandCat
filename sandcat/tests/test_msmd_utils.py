from __future__ import annotations

import struct
from pathlib import Path

import mdtraj as mdt
import numpy as np
import pytest
from openmm import Context, System, VerletIntegrator, XmlSerializer, unit

from sandcat.msmd_utils import backup_msmd_restart, reconcile_msmd_restart


def _write_restart_xml(path: Path, step: int, n_atoms: int = 1) -> None:
    """Write a minimal portable OpenMM State for restart tests."""
    system = System()
    for _ in range(n_atoms):
        system.addParticle(12.0)
    integrator = VerletIntegrator(1.0 * unit.femtosecond)
    context = Context(system, integrator)
    context.setPositions(np.zeros((n_atoms, 3)) * unit.nanometer)
    context.setVelocities(
        np.zeros((n_atoms, 3)) * unit.nanometer / unit.picosecond,
    )
    context.setStepCount(step)
    state = context.getState(getPositions=True, getVelocities=True)
    path.write_text(XmlSerializer.serialize(state))


def _write_restart_dcd(
        path: Path,
        n_frames: int,
        first_step: int = 10,
        interval: int = 10,
        n_atoms: int = 1,
) -> None:
    """Write a small DCD with OpenMM-style step metadata."""
    coordinates = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    cell_lengths = np.ones((n_frames, 3), dtype=np.float32)
    cell_angles = np.full((n_frames, 3), 90.0, dtype=np.float32)
    with mdt.formats.DCDTrajectoryFile(str(path), 'w') as dcd_file:
        dcd_file.write(coordinates, cell_lengths, cell_angles)
    last_step = first_step + (n_frames - 1) * interval
    with path.open('r+b') as dcd_file:
        dcd_file.seek(12)
        dcd_file.write(struct.pack('<i', first_step))
        dcd_file.seek(16)
        dcd_file.write(struct.pack('<i', interval))
        dcd_file.seek(20)
        dcd_file.write(struct.pack('<i', last_step))


def test_backup_msmd_restart_promotes_only_matching_complete_pair(
        tmp_path: Path,
) -> None:
    """Only a complete XML/DCD pair should replace the durable snapshot."""
    output_prefix = tmp_path / 'md_output'
    state_xml_path = Path(f'{output_prefix}.xml')
    dcd_path = Path(f'{output_prefix}.dcd')
    _write_restart_xml(state_xml_path, step=30)
    _write_restart_dcd(dcd_path, n_frames=3)

    assert backup_msmd_restart(str(output_prefix)) is True
    xml_backup = Path(f'{state_xml_path}.bak')
    dcd_backup = Path(f'{dcd_path}.bak')
    original_xml_backup = xml_backup.read_bytes()
    original_dcd_backup = dcd_backup.read_bytes()

    state_xml_path.write_text('<State>incomplete')
    assert backup_msmd_restart(str(output_prefix)) is False
    assert xml_backup.read_bytes() == original_xml_backup
    assert dcd_backup.read_bytes() == original_dcd_backup

    _write_restart_xml(state_xml_path, step=40)
    assert backup_msmd_restart(str(output_prefix)) is False
    assert xml_backup.read_bytes() == original_xml_backup
    assert dcd_backup.read_bytes() == original_dcd_backup


def test_backup_msmd_restart_trims_dcd_copied_ahead_of_xml(
        tmp_path: Path,
) -> None:
    """Frames appended after the XML snapshot should be discarded."""
    output_prefix = tmp_path / 'md_output'
    state_xml_path = Path(f'{output_prefix}.xml')
    dcd_path = Path(f'{output_prefix}.dcd')
    _write_restart_xml(state_xml_path, step=20)
    _write_restart_dcd(dcd_path, n_frames=3)

    assert backup_msmd_restart(str(output_prefix)) is True

    xml_backup = Path(f'{state_xml_path}.bak')
    dcd_backup = Path(f'{dcd_path}.bak')
    backup_state = XmlSerializer.deserialize(xml_backup.read_text())
    assert backup_state.getStepCount() == 20
    with dcd_backup.open('rb') as dcd_file:
        dcd_file.seek(8)
        assert struct.unpack('<i', dcd_file.read(4))[0] == 2
        dcd_file.seek(20)
        assert struct.unpack('<i', dcd_file.read(4))[0] == 20


def test_reconcile_msmd_restart_trims_dcd_ahead_of_xml(tmp_path: Path) -> None:
    """An interrupted DCD-first upload should roll back to the XML step."""
    output_prefix = tmp_path / 'md_output'
    state_xml_path = Path(f'{output_prefix}.xml')
    dcd_path = Path(f'{output_prefix}.dcd')
    _write_restart_xml(state_xml_path, step=20)
    _write_restart_dcd(dcd_path, n_frames=3)
    original_size = dcd_path.stat().st_size

    reconcile_msmd_restart(str(output_prefix))

    assert dcd_path.stat().st_size < original_size
    with dcd_path.open('rb') as dcd_file:
        dcd_file.seek(8)
        assert struct.unpack('<i', dcd_file.read(4))[0] == 2
        dcd_file.seek(20)
        assert struct.unpack('<i', dcd_file.read(4))[0] == 20
    assert backup_msmd_restart(str(output_prefix)) is True


def test_reconcile_msmd_restart_trims_complete_frames_beyond_stale_header(
        tmp_path: Path,
) -> None:
    """Complete appended frames should be detected even if the header is stale."""
    output_prefix = tmp_path / 'md_output'
    state_xml_path = Path(f'{output_prefix}.xml')
    dcd_path = Path(f'{output_prefix}.dcd')
    _write_restart_xml(state_xml_path, step=20)
    _write_restart_dcd(dcd_path, n_frames=4)
    with dcd_path.open('r+b') as dcd_file:
        dcd_file.seek(8)
        dcd_file.write(struct.pack('<i', 2))
        dcd_file.seek(20)
        dcd_file.write(struct.pack('<i', 20))
    original_size = dcd_path.stat().st_size

    reconcile_msmd_restart(str(output_prefix))

    assert dcd_path.stat().st_size < original_size
    with dcd_path.open('rb') as dcd_file:
        dcd_file.seek(8)
        assert struct.unpack('<i', dcd_file.read(4))[0] == 2
        dcd_file.seek(20)
        assert struct.unpack('<i', dcd_file.read(4))[0] == 20
    assert backup_msmd_restart(str(output_prefix)) is True


def test_reconcile_msmd_restart_rejects_dcd_behind_xml(tmp_path: Path) -> None:
    """A DCD missing the XML frame cannot be repaired without losing state."""
    output_prefix = tmp_path / 'md_output'
    _write_restart_xml(Path(f'{output_prefix}.xml'), step=30)
    _write_restart_dcd(Path(f'{output_prefix}.dcd'), n_frames=2)

    with pytest.raises(RuntimeError, match='DCD ends before XML step 30'):
        reconcile_msmd_restart(str(output_prefix))
