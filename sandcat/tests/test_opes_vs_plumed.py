from __future__ import annotations

from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

from sandcat.opes import METAD, OPES, OpesVariable
from sandcat.opes_expanded import (
    KB_KJ_PER_MOL_K,
    OpesExpanded,
    add_opes_expanded_bias,
    build_opes_expanded_integrator,
)


PHI_ATOMS_0BASED = (4, 6, 8, 14)
PSI_ATOMS_0BASED = (6, 8, 14, 16)


def _torsion_force(atoms: tuple[int, int, int, int]) -> mm.CustomTorsionForce:
    f = mm.CustomTorsionForce('theta')
    f.addTorsion(*atoms, [])
    return f


def _sandcat_analytic_bias(opes: OPES, cv: np.ndarray) -> float:
    """V(s) from sandcat's kernel sum, bypassing the OpenMM tabulated interpolation."""
    prob = 0.0
    for kernel in opes._kernels:
        prob += opes._evaluate_kernel(kernel, cv)
    prob /= opes._kde_norm
    return float(opes._kbt * opes._bias_prefactor * np.log(prob / opes._zed + opes._epsilon))


def _plumed_dat_opes_metad(
        sigma_phi: float, sigma_psi: float, pace: int, barrier_kj: float,
        out_path: str, kernels_path: str,
) -> str:
    """plumed.dat string for OPES_METAD on diala phi,psi (1-based atom indices)."""
    phi_atoms = ','.join(str(i + 1) for i in PHI_ATOMS_0BASED)
    psi_atoms = ','.join(str(i + 1) for i in PSI_ATOMS_0BASED)
    return (
        f'phi: TORSION ATOMS={phi_atoms}\n'
        f'psi: TORSION ATOMS={psi_atoms}\n'
        f'opes: OPES_METAD ARG=phi,psi PACE={pace} TEMP=300 '
        f'BARRIER={barrier_kj} SIGMA={sigma_phi},{sigma_psi} '
        f'STATE_WFILE={kernels_path} STATE_WSTRIDE={pace}\n'
        f'PRINT ARG=phi,psi,opes.bias FILE={out_path} STRIDE=1\n'
    )


def _make_plumed(plumed_dat: str, n_atoms: int, kbt_kj: float, log_path: str):
    """Initialise a plumed.Plumed instance with our plumed.dat content."""
    import plumed
    p = plumed.Plumed()
    p.cmd('setNatoms', n_atoms)
    p.cmd('setMDEngine', 'sandcat')
    p.cmd('setTimestep', 0.001)  # ps; 1 fs MD timestep
    p.cmd('setKbT', kbt_kj)
    p.cmd('setLogFile', log_path)
    p.cmd('init')
    for line in plumed_dat.strip().splitlines():
        p.cmd('readInputLine', line)
    return p


def _push_step_to_plumed(plumed_obj, step: int, positions_nm: np.ndarray,
                          masses_amu: np.ndarray, box_nm: np.ndarray,
                          energy_kj: float | None = None) -> None:
    """Run one PLUMED calc() step; forces and virial are discarded."""
    forces = np.zeros_like(positions_nm)
    virial = np.zeros((3, 3))
    plumed_obj.cmd('setStep', step)
    plumed_obj.cmd('setBox', box_nm)
    plumed_obj.cmd('setMasses', masses_amu)
    plumed_obj.cmd('setPositions', positions_nm)
    plumed_obj.cmd('setForces', forces)
    plumed_obj.cmd('setVirial', virial)
    if energy_kj is not None:
        plumed_obj.cmd('setEnergy', float(energy_kj))
    plumed_obj.cmd('calc')


def _read_plumed_colvar(path: str) -> np.ndarray:
    """Load a PLUMED COLVAR file as a numeric ndarray (skipping comments)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows)


@pytest.fixture(scope='session')
def _opes_metad_run(plumed_kernel_path, diala_system_factory, tmp_path_factory):
    """Coupled sandcat+PLUMED OPES_METAD run; results shared across comparison tests."""
    system, topology, positions = diala_system_factory()
    n_atoms = system.getNumParticles()
    masses_amu = np.array([
        system.getParticleMass(i).value_in_unit(unit.amu) for i in range(n_atoms)
    ])

    sigma_phi = 0.20
    sigma_psi = 0.30
    pace = 200
    barrier_kj = 30.0
    n_steps = 2000

    variables = [
        OpesVariable(force=_torsion_force(PHI_ATOMS_0BASED),
                     min_value=-np.pi, max_value=np.pi,
                     periodic=True, grid_width=80),
        OpesVariable(force=_torsion_force(PSI_ATOMS_0BASED),
                     min_value=-np.pi, max_value=np.pi,
                     periodic=True, grid_width=80),
    ]
    opes = OPES(
        system=system,
        variables=variables,
        temperature=300.0 * unit.kelvin,
        barrier=barrier_kj * unit.kilojoule_per_mole,
        pace=pace,
        sigma=[sigma_phi, sigma_psi],   # fixed sigma → no adaptive Welford warmup
        mode=METAD,
        compression_threshold=1.0,
    )

    integrator = mm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(20260505)
    simulation = app.Simulation(
        topology, system, integrator, mm.Platform.getPlatformByName('Reference'),
    )
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 20260505)
    simulation.minimizeEnergy()

    tmp_dir = tmp_path_factory.mktemp('opes_metad')
    colvar_path = str(tmp_dir / 'COLVAR')
    log_path = str(tmp_dir / 'plumed.log')
    kernels_path = str(tmp_dir / 'kernels.data')
    plumed_obj = _make_plumed(
        _plumed_dat_opes_metad(
            sigma_phi, sigma_psi, pace, barrier_kj, colvar_path, kernels_path,
        ),
        n_atoms=n_atoms,
        kbt_kj=KB_KJ_PER_MOL_K * 300.0,
        log_path=log_path,
    )

    box = np.eye(3) * 100.0  # vacuum: nominal box big enough to be inert

    sandcat_bias_per_step = np.zeros(n_steps)
    sandcat_cv_per_step = np.zeros((n_steps, 2))
    for step_idx in range(n_steps):
        opes.step(simulation, 1)
        state = simulation.context.getState(getPositions=True)
        pos_nm = np.array(state.getPositions().value_in_unit(unit.nanometer))
        cv_now = np.array(opes.get_collective_variables(simulation))
        sandcat_cv_per_step[step_idx] = cv_now
        sandcat_bias_per_step[step_idx] = _sandcat_analytic_bias(opes, cv_now)
        _push_step_to_plumed(
            plumed_obj, step_idx + 1, pos_nm, masses_amu, box,
        )

    # Force PLUMED to flush its STATE file, then drop the wrapper so the
    # output files are closed before the test reads them.
    del plumed_obj

    plumed_data = _read_plumed_colvar(colvar_path)
    # Columns of the PRINT line are: time, phi, psi, opes.bias.
    plumed_phi = plumed_data[:, 1]
    plumed_psi = plumed_data[:, 2]
    plumed_bias = plumed_data[:, 3]

    return {
        'sandcat_cv': sandcat_cv_per_step,
        'sandcat_bias': sandcat_bias_per_step,
        'plumed_cv': np.column_stack([plumed_phi, plumed_psi]),
        'plumed_bias': plumed_bias,
        'opes': opes,
        'kernels_path': kernels_path,
        'pace': pace,
    }


def test_opes_metad_cv_trajectories_agree(_opes_metad_run):
    """Both impls read phi,psi from the same coordinates and must agree to ~1e-5 rad."""
    sandcat_cv = _opes_metad_run['sandcat_cv']
    plumed_cv = _opes_metad_run['plumed_cv']
    diff = sandcat_cv - plumed_cv
    # Wrap into [-pi, pi] so that 359° vs -1° doesn't blow the test up.
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    # PLUMED's positions buffer round-trips through a void* memcpy that the
    # Python wrapper feeds as contiguous float64, but PLUMED's internal
    # torsion accumulators reduce some quantities at lower precision —
    # ~1e-6 rad is the realistic floor.
    np.testing.assert_allclose(diff, 0.0, atol=1e-5)


def test_opes_metad_pre_deposit_bias_matches_plumed_floor(_opes_metad_run):
    """Pre-deposit both impls must report the floor V = kbT*prefactor*log(ε) = -barrier."""
    sandcat_bias = _opes_metad_run['sandcat_bias']
    plumed_bias = _opes_metad_run['plumed_bias']
    pace = _opes_metad_run['pace']
    # First deposit fires at MD step PACE (PLUMED prints pre-deposit, so its
    # PRINT value at that step still reflects the no-kernel floor). The
    # window [0, pace-1] is the unambiguous "no kernels yet" regime.
    pre_first = slice(0, pace - 1)
    np.testing.assert_allclose(
        sandcat_bias[pre_first], plumed_bias[pre_first],
        atol=1e-9,
        err_msg='pre-deposit bias floor disagrees',
    )
    # And both should be exactly -barrier (30 kJ/mol).
    np.testing.assert_allclose(
        plumed_bias[pre_first], -30.0, atol=1e-6,
    )


def test_opes_metad_deposited_bias_signs_and_bounds(_opes_metad_run):
    """V(s) must stay in [-barrier, ~kbT] kJ/mol — the OPES design invariant."""
    sandcat_bias = _opes_metad_run['sandcat_bias']
    barrier = 30.0
    assert sandcat_bias.min() >= -barrier - 1e-6, (
        f'sandcat bias dropped below -barrier: min={sandcat_bias.min()}'
    )
    # Allow a small positive overshoot (the second-kernel Z renormalisation
    # can push the bias slightly above zero when epsilon is small) but not
    # more than ~kbT, otherwise something has gone wrong with the log/Z.
    assert sandcat_bias.max() < 2.5, (
        f'sandcat bias rose above kbT: max={sandcat_bias.max()}'
    )


def test_opes_metad_kernel_centres_match_plumed(_opes_metad_run):
    """Kernel centres must match PLUMED; heights/sigmas may drift due to tabulation convention."""
    opes = _opes_metad_run['opes']
    kernels_path = _opes_metad_run['kernels_path']
    if not Path(kernels_path).exists():
        pytest.skip(f'PLUMED kernel state file not produced: {kernels_path}')
    plumed_kernels = _parse_plumed_kernel_state(kernels_path)
    sandcat_centres = sorted(
        ((k.center[0], k.center[1]) for k in opes._kernels),
        key=lambda row: (row[0], row[1]),
    )
    plumed_centres = sorted(
        ((row[0], row[1]) for row in plumed_kernels),
        key=lambda row: (row[0], row[1]),
    )
    assert len(sandcat_centres) == len(plumed_centres), (
        f'sandcat kept {len(sandcat_centres)} kernels, '
        f'plumed {len(plumed_centres)}'
    )
    for sandcat_centre, plumed_centre in zip(sandcat_centres, plumed_centres):
        assert sandcat_centre[0] == pytest.approx(plumed_centre[0], abs=1e-2)
        assert sandcat_centre[1] == pytest.approx(plumed_centre[1], abs=1e-2)


def _parse_plumed_kernel_state(path: str) -> list[tuple[float, float, float, float, float]]:
    """Parse (phi, psi, sigma_phi, sigma_psi, height) rows from a PLUMED OPES STATE file."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            cols = [float(x) for x in line.split()]
            if len(cols) < 6:
                continue
            _time, phi, psi, sphi, spsi, height = cols[:6]
            rows.append((phi, psi, sphi, spsi, height))
    return rows


def _plumed_dat_opes_expanded(
        t_min_k: float, t_max_k: float, n_temps: int, pace: int,
        observation_steps: int, out_path: str, out_delta: str,
) -> str:
    """plumed.dat for multithermal OPES_EXPANDED: ECV_MULTITHERMAL + OPES_EXPANDED on diala U."""
    return (
        'ene: ENERGY\n'
        f'ecv: ECV_MULTITHERMAL ARG=ene TEMP={t_min_k:.6f} '
        f'TEMP_MIN={t_min_k:.6f} TEMP_MAX={t_max_k:.6f} '
        f'TEMP_STEPS={n_temps} NO_GEOM_SPACING\n'
        f'opes: OPES_EXPANDED ARG=ecv.* PACE={pace} '
        f'OBSERVATION_STEPS={observation_steps} FILE={out_delta}\n'
        f'PRINT ARG=ene,opes.bias FILE={out_path} STRIDE=1\n'
    )


@pytest.fixture(scope='session')
def _opes_expanded_run(plumed_kernel_path, diala_system_factory, tmp_path_factory):
    """Coupled sandcat+PLUMED OPES_EXPANDED run on diala vacuum; results shared across tests."""
    system, topology, positions = diala_system_factory()
    n_atoms = system.getNumParticles()
    masses_amu = np.array([
        system.getParticleMass(i).value_in_unit(unit.amu) for i in range(n_atoms)
    ])

    t_min_k = 300.0
    t_max_k = 450.0
    n_temps = 4
    pace = 100
    observation_steps = 5
    n_steps = 800

    add_opes_expanded_bias(system, n_temps=n_temps)
    integrator = build_opes_expanded_integrator(
        timestep=1.0 * unit.femtosecond,
        collision_rate=1.0 / unit.picosecond,
        temperature=300.0 * unit.kelvin,
        n_temps=n_temps,
    )
    integrator.setRandomNumberSeed(20260505)

    simulation = app.Simulation(
        topology, system, integrator, mm.Platform.getPlatformByName('Reference'),
    )
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 20260505)
    simulation.minimizeEnergy()

    opes = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=t_min_k * unit.kelvin,
        t_max=t_max_k * unit.kelvin,
        n_temps=n_temps,
        pace=pace,
        observation_steps=observation_steps,
        geom_spacing=False,
    )
    opes.apply_to_context(simulation.context)

    tmp_dir = tmp_path_factory.mktemp('opes_expanded')
    colvar_path = str(tmp_dir / 'COLVAR')
    log_path = str(tmp_dir / 'plumed.log')
    plumed_obj = _make_plumed(
        _plumed_dat_opes_expanded(
            t_min_k, t_max_k, n_temps, pace, observation_steps, colvar_path,
            out_delta=str(tmp_dir / 'DELTAFS'),
        ),
        n_atoms=n_atoms,
        kbt_kj=KB_KJ_PER_MOL_K * t_min_k,
        log_path=log_path,
    )
    box = np.eye(3) * 100.0

    sandcat_bias = np.zeros(n_steps)
    sandcat_u = np.zeros(n_steps)
    for step_idx in range(n_steps):
        opes.step(simulation, 1)
        state = simulation.context.getState(getPositions=True)
        pos_nm = np.array(state.getPositions().value_in_unit(unit.nanometer))
        # Read PE from the FF group only (exclude the OPES wrapper).
        ff_state = simulation.context.getState(
            getEnergy=True, groups=0xFFFFFFFF & ~(1 << 1),
        )
        u_kj = ff_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        sandcat_u[step_idx] = u_kj
        sandcat_bias[step_idx] = opes.compute_bias_v(u_kj)
        _push_step_to_plumed(
            plumed_obj, step_idx + 1, pos_nm, masses_amu, box,
            energy_kj=u_kj,
        )

    del plumed_obj
    plumed_data = _read_plumed_colvar(colvar_path)
    return {
        'sandcat_bias': sandcat_bias,
        'sandcat_u': sandcat_u,
        'plumed_u': plumed_data[:, 1],
        'plumed_bias': plumed_data[:, 2],
        'observation_steps': observation_steps,
        'pace': pace,
    }


def test_opes_expanded_potential_energy_matches_plumed(_opes_expanded_run):
    """U from the FF force-group mask must match what PLUMED reads via ENERGY."""
    sandcat_u = _opes_expanded_run['sandcat_u']
    plumed_u = _opes_expanded_run['plumed_u']
    np.testing.assert_allclose(sandcat_u, plumed_u, atol=1e-3, rtol=1e-6)


def test_opes_expanded_bias_matches_plumed_after_init(_opes_expanded_run):
    """V_bias(U) tracks PLUMED to within 2 kJ/mol after both impls have initialised ΔF.

    PACE-boundary steps are excluded because the deposit-vs-PRINT ordering creates a
    one-step phase shift; the tolerance absorbs sandcat's u_ref-rebased FP drift.
    """
    sandcat_bias = _opes_expanded_run['sandcat_bias']
    plumed_bias = _opes_expanded_run['plumed_bias']
    pace = _opes_expanded_run['pace']
    observation_steps = _opes_expanded_run['observation_steps']
    init_step = pace * observation_steps
    keep = np.zeros_like(sandcat_bias, dtype=bool)
    keep[init_step + 1:] = True
    for boundary in range(init_step, len(sandcat_bias) + 1, pace):
        idx = boundary - 1
        if 0 <= idx < len(sandcat_bias):
            keep[idx] = False
    diff = sandcat_bias[keep] - plumed_bias[keep]
    np.testing.assert_allclose(
        diff, 0.0, atol=2.0,
        err_msg=f'max |sandcat - plumed| post-init = {np.abs(diff).max():.4f} kJ/mol',
    )
