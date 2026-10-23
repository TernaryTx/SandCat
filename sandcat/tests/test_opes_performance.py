from __future__ import annotations

import time

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
import plumed
from .conftest import PLUMED_KERNEL, _have_plumed_kernel
import os

from sandcat.opes import METAD, OPES, OpesVariable
from sandcat.opes_expanded import (
    KB_KJ_PER_MOL_K,
    OpesExpanded,
    add_opes_expanded_bias,
    build_opes_expanded_integrator,
)

PHI_ATOMS_0BASED = (4, 6, 8, 14)
PSI_ATOMS_0BASED = (6, 8, 14, 16)

N_WARMUP = 400
N_TIMED = 1500
PACE = 200


def _torsion_force(atoms: tuple[int, int, int, int]) -> mm.CustomTorsionForce:
    f = mm.CustomTorsionForce('theta')
    f.addTorsion(*atoms, [])
    return f


def _build_unbiased(diala_system_factory):
    system, topology, positions = diala_system_factory()
    integrator = mm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(42)
    sim = app.Simulation(
        topology, system, integrator, mm.Platform.getPlatformByName('Reference'),
    )
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
    sim.minimizeEnergy()
    return sim, system


def _build_opes(diala_system_factory):
    system, topology, positions = diala_system_factory()
    variables = [
        OpesVariable(force=_torsion_force(PHI_ATOMS_0BASED),
                     min_value=-np.pi, max_value=np.pi,
                     periodic=True, grid_width=80),
        OpesVariable(force=_torsion_force(PSI_ATOMS_0BASED),
                     min_value=-np.pi, max_value=np.pi,
                     periodic=True, grid_width=80),
    ]
    opes = OPES(
        system=system, variables=variables,
        temperature=300.0 * unit.kelvin,
        barrier=30.0 * unit.kilojoule_per_mole,
        pace=PACE, sigma=[0.20, 0.30], mode=METAD,
    )
    integrator = mm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(42)
    sim = app.Simulation(
        topology, system, integrator, mm.Platform.getPlatformByName('Reference'),
    )
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
    sim.minimizeEnergy()
    return sim, opes


def _build_opes_expanded(diala_system_factory):
    system, topology, positions = diala_system_factory()
    add_opes_expanded_bias(system, n_temps=4)
    integrator = build_opes_expanded_integrator(
        timestep=1.0 * unit.femtosecond,
        collision_rate=1.0 / unit.picosecond,
        temperature=300.0 * unit.kelvin,
        n_temps=4,
    )
    integrator.setRandomNumberSeed(42)
    sim = app.Simulation(
        topology, system, integrator, mm.Platform.getPlatformByName('Reference'),
    )
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
    sim.minimizeEnergy()
    opes = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=300.0 * unit.kelvin,
        t_max=450.0 * unit.kelvin,
        n_temps=4, pace=PACE, observation_steps=5,
        geom_spacing=False,
    )
    opes.apply_to_context(sim.context)
    return sim, opes


def _make_plumed(n_atoms: int, log_path: str):
    """Build a fresh plumed.Plumed instance with shared boilerplate."""
    p = plumed.Plumed()
    p.cmd('setNatoms', n_atoms)
    p.cmd('setMDEngine', 'sandcat-perf')
    p.cmd('setTimestep', 0.001)
    p.cmd('setKbT', KB_KJ_PER_MOL_K * 300.0)
    p.cmd('setLogFile', log_path)
    p.cmd('init')
    return p


def _build_plumed_metad_observer(plumed_kernel_path, n_atoms: int, tmp_path):
    """PLUMED OPES_METAD passive observer — positions fed each step, forces discarded."""
    p = _make_plumed(n_atoms, str(tmp_path / 'plumed_metad.log'))
    phi_atoms = ','.join(str(i + 1) for i in PHI_ATOMS_0BASED)
    psi_atoms = ','.join(str(i + 1) for i in PSI_ATOMS_0BASED)
    for line in (
        f'phi: TORSION ATOMS={phi_atoms}',
        f'psi: TORSION ATOMS={psi_atoms}',
        f'opes: OPES_METAD ARG=phi,psi PACE={PACE} TEMP=300 BARRIER=30 SIGMA=0.20,0.30 '
        f'STATE_WFILE={str(tmp_path / "KERNELS")} STATE_WSTRIDE={PACE}',
    ):
        p.cmd('readInputLine', line)
    return p


def _build_plumed_expanded_observer(plumed_kernel_path, n_atoms: int, tmp_path):
    """PLUMED OPES_EXPANDED passive observer — needs PE pushed each step via setEnergy."""
    p = _make_plumed(n_atoms, str(tmp_path / 'plumed_expanded.log'))
    for line in (
        'ene: ENERGY',
        'ecv: ECV_MULTITHERMAL ARG=ene TEMP=300 TEMP_MIN=300 TEMP_MAX=450 '
        'TEMP_STEPS=4 NO_GEOM_SPACING',
        f'opes: OPES_EXPANDED ARG=ecv.* PACE={PACE} OBSERVATION_STEPS=5 '
        f'FILE={str(tmp_path / "DELTAFS")}',
    ):
        p.cmd('readInputLine', line)
    return p


@pytest.fixture(scope='session')
def _perf_results(diala_system_factory, tmp_path_factory):
    """Time each driver configuration once; PLUMED timings are optional and skipped per-test."""
    if _have_plumed_kernel():
        os.environ['PLUMED_KERNEL'] = str(PLUMED_KERNEL)
        plumed_kernel_path = str(PLUMED_KERNEL)
    else:
        plumed_kernel_path = None

    results: dict[str, float] = {}

    # 1) Unbiased Langevin baseline.
    sim, _ = _build_unbiased(diala_system_factory)
    sim.step(N_WARMUP)
    t0 = time.perf_counter()
    sim.step(N_TIMED)
    results['unbiased'] = time.perf_counter() - t0

    # 2) Sandcat OPES_METAD.
    sim, opes = _build_opes(diala_system_factory)
    opes.step(sim, N_WARMUP)  # warm up: ~2 deposits inside this window
    t0 = time.perf_counter()
    opes.step(sim, N_TIMED)
    results['opes_metad'] = time.perf_counter() - t0

    # 3) Sandcat OPES_EXPANDED.
    sim, opes_e = _build_opes_expanded(diala_system_factory)
    opes_e.step(sim, N_WARMUP)
    t0 = time.perf_counter()
    opes_e.step(sim, N_TIMED)
    results['opes_expanded'] = time.perf_counter() - t0

    # 4) Optional: unbiased MD coupled to PLUMED OPES_METAD / OPES_EXPANDED.
    if plumed_kernel_path is not None:
        tmp_path = tmp_path_factory.mktemp('plumed_perf')
        for label, factory, push_energy in (
            ('plumed_metad_observer', _build_plumed_metad_observer, False),
            ('plumed_expanded_observer', _build_plumed_expanded_observer, True),
        ):
            sim, system = _build_unbiased(diala_system_factory)
            n_atoms = system.getNumParticles()
            masses = np.array([
                system.getParticleMass(i).value_in_unit(unit.amu)
                for i in range(n_atoms)
            ])
            plumed_obj = factory(plumed_kernel_path, n_atoms, tmp_path)
            box = np.eye(3) * 100.0
            ff_mask = 0xFFFFFFFF & ~(1 << 1) if push_energy else None

            def _push(step_idx, pos, u_kj=None,
                      _plumed=plumed_obj, _box=box, _masses=masses):
                forces = np.zeros_like(pos)
                virial = np.zeros((3, 3))
                _plumed.cmd('setStep', step_idx)
                _plumed.cmd('setBox', _box)
                _plumed.cmd('setMasses', _masses)
                _plumed.cmd('setPositions', pos)
                _plumed.cmd('setForces', forces)
                _plumed.cmd('setVirial', virial)
                if u_kj is not None:
                    _plumed.cmd('setEnergy', float(u_kj))
                _plumed.cmd('calc')

            def _step_and_push(step_idx):
                sim.step(1)
                pos = np.array(sim.context.getState(getPositions=True)
                               .getPositions().value_in_unit(unit.nanometer))
                if push_energy:
                    u = sim.context.getState(getEnergy=True).getPotentialEnergy()
                    _push(step_idx, pos, u.value_in_unit(unit.kilojoule_per_mole))
                else:
                    _push(step_idx, pos)

            for k in range(N_WARMUP):
                _step_and_push(k + 1)
            t0 = time.perf_counter()
            for k in range(N_TIMED):
                _step_and_push(N_WARMUP + k + 1)
            results[label] = time.perf_counter() - t0
            del plumed_obj

    return results


def test_perf_summary(_perf_results, capsys):
    """Print a steps/sec table; always passes — diagnostic only."""
    base = _perf_results['unbiased']
    lines = [
        '',
        f'Performance summary ({N_TIMED} steps after {N_WARMUP}-step warmup, '
        'alanine dipeptide vacuum, Reference platform):',
    ]
    for key, t in _perf_results.items():
        steps_per_sec = N_TIMED / t
        ratio = t / base
        lines.append(
            f'  {key:18s}  {t:6.3f}s  '
            f'{steps_per_sec:7.0f} steps/s  '
            f'{ratio:5.2f}x baseline'
        )
    with capsys.disabled():
        print('\n'.join(lines))


def test_opes_metad_overhead_within_bounds(_perf_results):
    """OPES_METAD overhead must be <15x baseline; regression indicates grid rebuilt every step."""
    ratio = _perf_results['opes_metad'] / _perf_results['unbiased']
    assert ratio < 15.0, (
        f'OPES_METAD overhead = {ratio:.2f}x baseline — expected <15x. '
        'A regression here usually means the bias grid is being rebuilt '
        'every step instead of every PACE.'
    )


def test_opes_expanded_overhead_within_bounds(_perf_results):
    """OPES_EXPANDED overhead must be <15x baseline; regression indicates ΔF recomputed every step."""
    ratio = _perf_results['opes_expanded'] / _perf_results['unbiased']
    assert ratio < 15.0, (
        f'OPES_EXPANDED overhead = {ratio:.2f}x baseline — expected <15x. '
        'Likely cause: ΔF being recomputed every step rather than every PACE.'
    )


def test_plumed_metad_observer_overhead_within_bounds(_perf_results):
    """PLUMED OPES_METAD observer must be <25x baseline (skipped without libplumedKernel)."""
    if 'plumed_metad_observer' not in _perf_results:
        pytest.skip('PLUMED kernel not available — observer timing skipped')
    ratio = _perf_results['plumed_metad_observer'] / _perf_results['unbiased']
    assert ratio < 25.0, (
        f'PLUMED OPES_METAD observer overhead = {ratio:.2f}x baseline'
    )


def test_plumed_expanded_observer_overhead_within_bounds(_perf_results):
    """PLUMED OPES_EXPANDED observer must be <30x baseline (skipped without libplumedKernel)."""
    if 'plumed_expanded_observer' not in _perf_results:
        pytest.skip('PLUMED kernel not available — observer timing skipped')
    ratio = _perf_results['plumed_expanded_observer'] / _perf_results['unbiased']
    assert ratio < 30.0, (
        f'PLUMED OPES_EXPANDED observer overhead = {ratio:.2f}x baseline'
    )
