from __future__ import annotations

import logging
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from openmm import OpenMMException, XmlSerializer

from sandcat.variables import CHECKPOINT_BACKUP_SUFFIX, MD_OUTPUT_PREFIX


def _read_xml_restart_info(state_xml_path: str) -> tuple[int, int] | None:
    """Return the step and particle count from a complete OpenMM XML State.

    Args:
        state_xml_path: Path to the serialized OpenMM State.

    Returns:
        The state step and particle count, or ``None`` when the state is not a
        complete, loadable restart.
    """
    state_path = Path(state_xml_path)
    if not state_path.is_file() or state_path.stat().st_size == 0:
        return None
    try:
        state = XmlSerializer.deserialize(state_path.read_text())
        positions = state.getPositions()
        velocities = state.getVelocities()
    except (OSError, UnicodeDecodeError, ValueError, OpenMMException):
        return None
    if len(positions) == 0 or len(positions) != len(velocities):
        return None
    return state.getStepCount(), len(positions)


@dataclass(frozen=True)
class _DcdRestartInfo:
    """Validated metadata needed to reconcile an OpenMM DCD restart."""

    byte_order: str
    n_frames: int
    n_atoms: int
    first_step: int
    interval: int
    header_size: int
    frame_size: int
    has_unit_cell: bool

    @property
    def last_step(self) -> int:
        """Return the simulation step represented by the final DCD frame."""
        return self.first_step + (self.n_frames - 1) * self.interval


def _read_dcd_record(dcd_file: BinaryIO, byte_order: str) -> bytes:
    """Read and validate one Fortran-style DCD record."""
    record_size = struct.unpack(f'{byte_order}i', dcd_file.read(4))[0]
    payload = dcd_file.read(record_size)
    closing_size = struct.unpack(f'{byte_order}i', dcd_file.read(4))[0]
    if len(payload) != record_size or closing_size != record_size:
        raise ValueError('Incomplete DCD record.')
    return payload


def _read_dcd_header(
        path: Path,
) -> tuple[str, tuple[int, ...], int, int] | None:
    """Return byte order, control words, atom count, and header size."""
    try:
        with path.open('rb') as dcd_file:
            record_marker = dcd_file.read(4)
            byte_order = next(
                order for order in ('<', '>')
                if struct.unpack(f'{order}i', record_marker)[0] == 84
            )
            dcd_file.seek(0)
            control_record = _read_dcd_record(dcd_file, byte_order)
            if len(control_record) != 84 or control_record[:4] != b'CORD':
                return None
            controls = struct.unpack(f'{byte_order}20i', control_record[4:])

            _read_dcd_record(dcd_file, byte_order)  # trajectory title
            atom_record = _read_dcd_record(dcd_file, byte_order)
            if len(atom_record) != 4:
                return None
            n_atoms = struct.unpack(f'{byte_order}i', atom_record)[0]
            return byte_order, controls, n_atoms, dcd_file.tell()
    except (OSError, StopIteration, ValueError, struct.error):
        return None


def _has_complete_dcd_last_frame(
        path: Path,
        restart_info: _DcdRestartInfo,
) -> bool:
    """Return whether the final DCD frame has complete Fortran records."""
    coordinate_record_size = restart_info.n_atoms * 4
    record_sizes = [48] if restart_info.has_unit_cell else []
    record_sizes.extend([coordinate_record_size] * 3)
    last_frame_offset = (
        restart_info.header_size
        + (restart_info.n_frames - 1) * restart_info.frame_size
    )
    try:
        with path.open('rb') as dcd_file:
            dcd_file.seek(last_frame_offset)
            for record_size in record_sizes:
                record = _read_dcd_record(dcd_file, restart_info.byte_order)
                if len(record) != record_size:
                    return False
    except (OSError, ValueError, struct.error):
        return False
    return True


def _read_dcd_restart_info(dcd_path: str) -> _DcdRestartInfo | None:
    """Read and structurally validate an OpenMM DCD restart trajectory.

    Args:
        dcd_path: Path to the DCD trajectory.

    Returns:
        DCD header and frame-layout information, or ``None`` when the file is
        incomplete or unsupported.
    """
    path = Path(dcd_path)
    if not path.is_file() or path.stat().st_size < 112:
        return None
    header = _read_dcd_header(path)
    if header is None:
        return None
    byte_order, controls, n_atoms, header_size = header
    header_n_frames, first_step, interval = controls[:3]
    if (
        header_n_frames <= 0
        or n_atoms <= 0
        or interval <= 0
        or controls[8] != 0
    ):
        return None

    coordinate_record_size = n_atoms * 4
    frame_size = 3 * (coordinate_record_size + 8)
    has_unit_cell = controls[10] != 0
    if has_unit_cell:
        frame_size += 56
    trajectory_size = path.stat().st_size - header_size
    n_frames, partial_frame_size = divmod(trajectory_size, frame_size)
    if n_frames <= 0 or partial_frame_size != 0:
        return None
    if n_frames != header_n_frames:
        logging.warning(
            'DCD header reports %d frames but the file contains %d complete '
            'frames; using the complete-frame count.',
            header_n_frames,
            n_frames,
        )

    restart_info = _DcdRestartInfo(
        byte_order=byte_order,
        n_frames=n_frames,
        n_atoms=n_atoms,
        first_step=first_step,
        interval=interval,
        header_size=header_size,
        frame_size=frame_size,
        has_unit_cell=has_unit_cell,
    )
    if not _has_complete_dcd_last_frame(path, restart_info):
        return None
    return restart_info


def _patch_dcd_header(
        dcd_path: str,
        n_frames: int,
        last_step: int,
        byte_order: str,
) -> None:
    """Update the DCD frame and final-step counters after reconciliation.

    Args:
        dcd_path: Path to the DCD trajectory.
        n_frames: Number of complete frames retained in the file.
        last_step: Simulation step represented by the last frame.
        byte_order: DCD integer byte order, ``<`` or ``>``.
    """
    with open(dcd_path, 'r+b') as dcd_file:
        dcd_file.seek(8)
        dcd_file.write(struct.pack(f'{byte_order}i', n_frames))
        dcd_file.seek(20)
        dcd_file.write(struct.pack(f'{byte_order}i', last_step))


def _validate_msmd_restart_pair(
        state_xml_path: str,
        dcd_path: str,
) -> tuple[int, _DcdRestartInfo] | None:
    """Return restart metadata when XML and DCD describe the same step.

    Args:
        state_xml_path: Path to the OpenMM XML State.
        dcd_path: Path to the DCD trajectory.

    Returns:
        XML step and DCD metadata, or ``None`` when either file is invalid or
        the pair is inconsistent.
    """
    xml_info = _read_xml_restart_info(state_xml_path)
    dcd_info = _read_dcd_restart_info(dcd_path)
    if xml_info is None or dcd_info is None:
        return None
    xml_step, xml_particle_count = xml_info
    if (
        dcd_info.n_atoms != xml_particle_count
        or dcd_info.last_step != xml_step
    ):
        return None
    return xml_step, dcd_info


def backup_msmd_restart(output_prefix: str = MD_OUTPUT_PREFIX) -> bool:
    """Create an immutable, validated XML/DCD snapshot for S3 upload.

    The live files are copied to temporary paths, validated as a matching
    restart pair, and atomically promoted to ``.bak`` siblings. A failed copy
    never replaces the previous known-good snapshot.

    Args:
        output_prefix: Prefix shared by the MSMD XML and DCD files.

    Returns:
        ``True`` when a new valid snapshot was promoted, otherwise ``False``.
    """
    state_xml_path = f'{output_prefix}.xml'
    dcd_path = f'{output_prefix}.dcd'
    source_paths = (state_xml_path, dcd_path)
    if any(not Path(path).is_file() for path in source_paths):
        return False

    temporary_paths = tuple(
        f'{path}{CHECKPOINT_BACKUP_SUFFIX}.tmp'
        for path in source_paths
    )
    backup_paths = tuple(
        f'{path}{CHECKPOINT_BACKUP_SUFFIX}'
        for path in source_paths
    )
    try:
        for source_path, temporary_path in zip(source_paths, temporary_paths):
            shutil.copy2(source_path, temporary_path)
        # The live DCD may gain frames while a large trajectory is being
        # copied. The XML was copied first and is therefore the durable
        # checkpoint boundary; discard any later frames from the private DCD
        # copy before validating and publishing the pair.
        try:
            _reconcile_msmd_restart_pair(*temporary_paths)
        except RuntimeError:
            return False
        restart_info = _validate_msmd_restart_pair(*temporary_paths)
        if restart_info is None:
            return False
        xml_step, dcd_info = restart_info
        _patch_dcd_header(
            temporary_paths[1],
            dcd_info.n_frames,
            xml_step,
            dcd_info.byte_order,
        )
        for temporary_path, backup_path in zip(temporary_paths, backup_paths):
            os.replace(temporary_path, backup_path)
    finally:
        for temporary_path in temporary_paths:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass
    return True


def _reconcile_msmd_restart_pair(state_xml_path: str, dcd_path: str) -> None:
    """Make a DCD consistent with its durable XML restart state.

    Periodic uploads publish the immutable DCD before the XML. If an instance
    is interrupted between those uploads, S3 can therefore contain a valid DCD
    that is newer than the still-valid XML. This function safely truncates that
    DCD to the XML step before OpenMM appends to it.

    Args:
        state_xml_path: Path to the durable XML checkpoint.
        dcd_path: Path to the DCD trajectory to reconcile.

    Raises:
        RuntimeError: If either file is invalid, the DCD is behind the XML, or
            the XML step is not represented by a DCD frame.
    """
    xml_info = _read_xml_restart_info(state_xml_path)
    dcd_info = _read_dcd_restart_info(dcd_path)
    if xml_info is None:
        raise RuntimeError(f'Invalid MSMD XML restart state: {state_xml_path}')
    if dcd_info is None:
        raise RuntimeError(f'Invalid MSMD DCD restart trajectory: {dcd_path}')

    xml_step, xml_particle_count = xml_info
    if dcd_info.n_atoms != xml_particle_count:
        raise RuntimeError(
            'MSMD XML and DCD restart files have different particle counts.'
        )
    step_offset = xml_step - dcd_info.first_step
    if step_offset < 0 or step_offset % dcd_info.interval != 0:
        raise RuntimeError(
            f'MSMD XML step {xml_step} is not represented by a DCD frame.'
        )
    expected_frames = step_offset // dcd_info.interval + 1
    if dcd_info.n_frames < expected_frames:
        raise RuntimeError(
            f'MSMD DCD ends before XML step {xml_step}: '
            f'{dcd_info.n_frames} frames available, {expected_frames} required.'
        )
    if dcd_info.n_frames > expected_frames:
        trimmed_size = (
            dcd_info.header_size
            + expected_frames * dcd_info.frame_size
        )
        with open(dcd_path, 'r+b') as dcd_file:
            dcd_file.truncate(trimmed_size)
        logging.warning(
            'Trimming MSMD DCD from %d to %d frames to match XML step %d.',
            dcd_info.n_frames,
            expected_frames,
            xml_step,
        )
    _patch_dcd_header(
        dcd_path,
        expected_frames,
        xml_step,
        dcd_info.byte_order,
    )
    if _validate_msmd_restart_pair(state_xml_path, dcd_path) is None:
        raise RuntimeError(
            'MSMD restart files remain inconsistent after reconciliation.'
        )


def reconcile_msmd_restart(output_prefix: str = MD_OUTPUT_PREFIX) -> None:
    """Make a downloaded DCD consistent with its durable XML restart state.

    Periodic uploads publish the immutable DCD before the XML. If an instance
    is interrupted between those uploads, S3 can therefore contain a valid DCD
    that is newer than the still-valid XML. This function safely truncates that
    DCD to the XML step before OpenMM appends to it.

    Args:
        output_prefix: Prefix shared by the MSMD XML and DCD files.

    Raises:
        RuntimeError: If either file is invalid, the DCD is behind the XML, or
            the XML step is not represented by a DCD frame.
    """
    _reconcile_msmd_restart_pair(
        f'{output_prefix}.xml',
        f'{output_prefix}.dcd',
    )
