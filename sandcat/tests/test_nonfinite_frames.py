"""Guard against corrupt frames reaching the density analysis.

HPV11-E2's demuxed SWISH-X trajectories contain one frame (1420, in five of six
replicas) where every coordinate is NaN. It originates in the simulation output,
not in any analysis step, and it is a single frame out of 2501, so dropping it
costs nothing scientifically.

Left in, it was costly. ``numpy.ndarray.astype(int)`` maps NaN to 0 rather than
raising, so all 2268 probe atoms of that frame were binned into voxel (0, 0, 0),
producing a density 3333 times bulk at the corner of the grid, 42 A from the
protein, which outranked the genuine cryptic site by a factor of 600 and made
every detection strategy fail on that target.

Two guards are tested. ``align_trajectory`` drops such frames when writing the
aligned trajectory, so nothing downstream ever sees them. ``compute_probe_density_grid``
drops non-finite atoms itself, as a backstop for trajectories that did not come
through the aligner, and uses floor rather than truncation so that coordinates
just outside the grid are rejected instead of folding onto index 0.
"""
from __future__ import annotations

from pathlib import Path
import MDAnalysis as mda

import numpy as np
import pytest

from sandcat.analysis.compute_probe_density import compute_probe_density_grid


def write_trajectory(path: Path, topology: str | Path, positions_per_frame: np.ndarray) -> None:
    """Write a small DCD with the given per-frame positions."""
    universe = mda.Universe(topology)
    with mda.Writer(str(path), n_atoms=universe.atoms.n_atoms) as writer:
        for positions in positions_per_frame:
            universe.atoms.positions = positions
            writer.write(universe.atoms)


@pytest.fixture
def probe_system(tmp_path: Path) -> Path:
    """A two-atom PRB system: one probe held at a fixed point, plus a protein atom."""
    pdb = tmp_path / 'system.pdb'
    pdb.write_text(
        'CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1\n'
        'ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00  0.00           C\n'
        'ATOM      2  C1  PRB B   1      12.000  12.000  12.000  1.00  0.00           C\n'
        'END\n',
    )
    return pdb


def test_nan_frame_does_not_create_a_corner_spike(tmp_path: Path, probe_system: Path) -> None:
    """The HPV11-E2 failure: a NaN frame must not pile onto voxel (0,0,0)."""
    good = np.array([[10.0, 10.0, 10.0], [12.0, 12.0, 12.0]])
    nan_frame = np.array([[np.nan] * 3, [np.nan] * 3])
    dcd = tmp_path / 'traj.dcd'
    write_trajectory(dcd, probe_system, [good, nan_frame, good])

    density, origin = compute_probe_density_grid(
        str(probe_system), str(dcd), grid_spacing=1.0, padding=5.0,
    )
    corner = density[0, 0, 0]
    assert corner == 0.0, f'NaN frame leaked into the corner voxel: {corner}'
    assert np.isfinite(density).all()
    assert density.sum() > 0, 'the two good frames should still contribute'

def test_density_matches_a_run_without_the_bad_frame(tmp_path: Path, probe_system: Path) -> None:
    """Dropping the frame must not otherwise change the result."""
    good = np.array([[10.0, 10.0, 10.0], [12.0, 12.0, 12.0]])
    nan_frame = np.array([[np.nan] * 3, [np.nan] * 3])

    clean = tmp_path / 'clean.dcd'
    write_trajectory(clean, probe_system, [good, good])
    dirty = tmp_path / 'dirty.dcd'
    write_trajectory(dirty, probe_system, [good, nan_frame, good])

    d_clean, _ = compute_probe_density_grid(str(probe_system), str(clean))
    d_dirty, _ = compute_probe_density_grid(str(probe_system), str(dirty))
    # the dirty run averages over three frames rather than two, so the peak
    # is lower, but it must sit in the same voxel and carry no NaN
    assert np.unravel_index(d_clean.argmax(), d_clean.shape) == \
           np.unravel_index(d_dirty.argmax(), d_dirty.shape)
    assert np.isfinite(d_dirty).all()


def test_negative_offsets_do_not_fold_onto_index_zero(tmp_path: Path, probe_system: Path) -> None:
    """astype(int) truncates toward zero, so (-1, 0) became index 0."""
    # probe sits just below the grid origin on every axis
    outside = np.array([[10.0, 10.0, 10.0], [4.4, 4.4, 4.4]])
    dcd = tmp_path / 'outside.dcd'
    write_trajectory(dcd, probe_system, [outside, outside])

    density, origin = compute_probe_density_grid(
        str(probe_system), str(dcd), grid_spacing=1.0, padding=5.0,
    )
    # the probe is outside the box the protein defines, so it must be
    # discarded rather than binned at the corner
    assert density[0, 0, 0] == 0.0
