from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from sandcat import get_examples_path
from sandcat.tests import get_plumed_path

# Skip the cloned PLUMED2 source tree's own pytest suite — it expects the
# PLUMED build environment, fixtures we don't have, and would slow the
# sandcat test run from 1 s to 2 minutes.
collect_ignore_glob = ['plumed2/*', 'plumed2']

EXAMPLES_DIR = Path(get_examples_path())

# sandcat.analysis.experiments requires these and exits at import if either is
# unset. That is right for a real run, where a wrong default would silently
# analyse the wrong tree, but it stops pytest collecting at all. Point them at a
# scratch directory unless the caller has already chosen one, so the suite is
# self-sufficient; tests that need the real trees set their own.
os.environ.setdefault('SANDCAT_PAPER_DIR', str(EXAMPLES_DIR))
os.environ.setdefault(
    'SANDCAT_ANALYSIS_DIR',
    tempfile.mkdtemp(prefix='sandcat-test-analysis-'))

# .dylib on macOS, .so on Linux
_kernel_stem = Path(os.path.join(get_plumed_path(), 'src/lib/libplumedKernel'))
PLUMED_KERNEL = next(
    (p for p in (_kernel_stem.with_suffix('.dylib'), _kernel_stem.with_suffix('.so')) if p.exists()),
    _kernel_stem.with_suffix('.so'),  # fallback for the skip message
)

# Atom indices for the diala dihedrals (0-based, see run_diala_opes.py).
PHI_ATOMS = (4, 6, 8, 14)
PSI_ATOMS = (6, 8, 14, 16)


def _have_plumed_kernel() -> bool:
    return PLUMED_KERNEL.exists()


@pytest.fixture(scope='session')
def diala_paths() -> tuple[str, str]:
    """Return (prmtop_path, inpcrd_path) for the alanine dipeptide example."""
    prmtop = EXAMPLES_DIR / 'diala.prmtop'
    inpcrd = EXAMPLES_DIR / 'diala.inpcrd'
    if not prmtop.exists() or not inpcrd.exists():
        pytest.skip(f'alanine dipeptide example files missing under {EXAMPLES_DIR}')
    return str(prmtop), str(inpcrd)


@pytest.fixture(scope='session')
def diala_system_factory(diala_paths):
    """Return a fresh (system, topology, positions) per call — Forces cannot be shared across Contexts."""

    prmtop_path, inpcrd_path = diala_paths

    def _build():
        prmtop = app.AmberPrmtopFile(prmtop_path)
        inpcrd = app.AmberInpcrdFile(inpcrd_path)
        system = prmtop.createSystem(
            nonbondedMethod=app.NoCutoff, constraints=app.HBonds,
        )
        return system, prmtop.topology, inpcrd.positions

    return _build


@pytest.fixture(scope='session')
def diala_unbiased_trajectory(diala_system_factory):
    """Short unbiased Langevin trajectory of diala in vacuum.

    Returns (positions_nm, cvs) arrays of shape (n_frames, n_atoms, 3) and (n_frames, 2).
    """
    system, topology, positions = diala_system_factory()
    integrator = mm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(20260505)
    platform = mm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 20260505)
    simulation.minimizeEnergy()

    n_frames = 600
    pos_traj = np.zeros((n_frames, system.getNumParticles(), 3), dtype=np.float64)
    cvs = np.zeros((n_frames, 2), dtype=np.float64)

    phi_force = mm.CustomTorsionForce('theta')
    phi_force.addTorsion(*PHI_ATOMS, [])
    psi_force = mm.CustomTorsionForce('theta')
    psi_force.addTorsion(*PSI_ATOMS, [])
    cv_force = mm.CustomCVForce('phi_cv + psi_cv')
    cv_force.addCollectiveVariable('phi_cv', phi_force)
    cv_force.addCollectiveVariable('psi_cv', psi_force)
    free_group = max(set(range(32)) - {f.getForceGroup() for f in system.getForces()})
    cv_force.setForceGroup(free_group)
    system.addForce(cv_force)
    simulation.context.reinitialize(preserveState=True)

    for k in range(n_frames):
        simulation.step(1)
        state = simulation.context.getState(getPositions=True)
        pos_traj[k] = np.array(state.getPositions().value_in_unit(unit.nanometer))
        cv_vals = cv_force.getCollectiveVariableValues(simulation.context)
        cvs[k] = np.array(cv_vals)
    return pos_traj, cvs


@pytest.fixture(scope='session')
def plumed_kernel_path():
    """Skip tests that depend on PLUMED if the libplumedKernel build is absent."""
    if not _have_plumed_kernel():
        pytest.skip(f'PLUMED kernel not built at {PLUMED_KERNEL}')
    os.environ['PLUMED_KERNEL'] = str(PLUMED_KERNEL)
    return str(PLUMED_KERNEL)
