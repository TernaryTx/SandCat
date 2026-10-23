from __future__ import annotations

import math

import numpy as np
import openmm as mm
import openmm.unit as unit
import pytest

from sandcat.opes_expanded import (
    KB_KJ_PER_MOL_K,
    OPES_BIAS_FORCE_GROUP,
    OPES_WRAP_MARKER_PARAM,
    OpesExpanded,
    _build_alpha_step_expression,
    _build_bias_formula,
    _temp_grid,
    add_opes_expanded_bias,
    build_opes_expanded_integrator,
)


def test_temp_grid_geometric_endpoints():
    """Geometric grid spans [t_min, t_max] with constant ratio between adjacent temperatures."""
    grid = _temp_grid(300.0, 600.0, 5, geom_spacing=True)
    assert grid[0] == pytest.approx(300.0)
    assert grid[-1] == pytest.approx(600.0)
    # Geometric ratios are constant.
    ratios = grid[1:] / grid[:-1]
    np.testing.assert_allclose(ratios, ratios[0])


def test_temp_grid_linear_in_beta_endpoints():
    """Linear-in-beta grid spans [t_min, t_max] with equal spacing in 1/T."""
    grid = _temp_grid(300.0, 600.0, 5, geom_spacing=False)
    assert grid[0] == pytest.approx(300.0)
    assert grid[-1] == pytest.approx(600.0)
    # 1/T should be linear, NOT T itself.
    inv_t = 1.0 / grid
    np.testing.assert_allclose(np.diff(inv_t), np.diff(inv_t)[0], rtol=1e-12)


def test_temp_grid_too_few_points_rejected():
    """Requesting fewer than 2 temperature points raises ValueError."""
    with pytest.raises(ValueError, match='n_temps'):
        _temp_grid(300.0, 600.0, 1)


def test_build_bias_formula_uses_marker_and_lists_components():
    """Bias formula contains the wrap marker, the kT/u_ref globals, the per-grid
    args, the U=sum(e_k) line, and the LSE max-shift structure."""
    n_temps = 4
    formula = _build_bias_formula(n_cvs=3, n_temps=n_temps)
    assert OPES_WRAP_MARKER_PARAM in formula
    assert 'opes_kt' in formula
    assert 'opes_u_ref' in formula
    for j in range(n_temps):
        assert f'opes_dbeta_{j}' in formula
        assert f'opes_df_shift_{j}' in formula
        assert f'arg_{j}' in formula
    assert 'max_arg' in formula
    assert 'e_0 + e_1 + e_2' in formula


def _toy_system_with_two_forces() -> mm.System:
    system = mm.System()
    system.addParticle(1.0)
    bond = mm.HarmonicBondForce()
    nb = mm.NonbondedForce()
    nb.addParticle(0.0, 0.1, 0.1)
    system.addForce(bond)
    system.addForce(nb)
    return system


def test_add_opes_expanded_bias_adds_wrapper_with_required_globals():
    """add_opes_expanded_bias appends one CustomCVForce with all required global parameters."""
    system = _toy_system_with_two_forces()
    n_forces_before = system.getNumForces()
    n_temps = 4
    wrapper = add_opes_expanded_bias(system, n_temps=n_temps)
    assert system.getNumForces() == n_forces_before + 1
    assert wrapper.getForceGroup() == OPES_BIAS_FORCE_GROUP
    names = {wrapper.getGlobalParameterName(i)
             for i in range(wrapper.getNumGlobalParameters())}
    for required in (OPES_WRAP_MARKER_PARAM, 'opes_u_ref',
                     'opes_alpha', 'opes_kt'):
        assert required in names
    for j in range(n_temps):
        assert f'opes_dbeta_{j}' in names
        assert f'opes_df_shift_{j}' in names
    # Two child forces (HarmonicBond + NonbondedForce) → two CVs.
    assert wrapper.getNumCollectiveVariables() == 2


def test_add_opes_expanded_bias_skips_barostat_and_cmm():
    """Barostat and CMMotionRemover are excluded from the CV wrapper force count."""
    system = _toy_system_with_two_forces()
    system.addForce(mm.MonteCarloBarostat(1.0, 300.0, 25))
    system.addForce(mm.CMMotionRemover())
    wrapper = add_opes_expanded_bias(system, n_temps=2)
    # Same two child forces; barostat and CMMotionRemover are skipped.
    assert wrapper.getNumCollectiveVariables() == 2


def _opes_expanded(t_min=300.0, t_max=400.0, n_temps=4) -> OpesExpanded:
    return OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=t_min * unit.kelvin,
        t_max=t_max * unit.kelvin,
        n_temps=n_temps,
        pace=100,
        observation_steps=10,
    )


def test_opes_expanded_dbeta_is_zero_at_t0():
    """dbeta is zero at T0 and strictly negative for all higher temperatures."""
    op = _opes_expanded(t_min=300.0, t_max=600.0, n_temps=4)
    # 300 K is the first grid point → dβ[0] should be 0 (no temperature offset).
    assert op.dbeta[0] == pytest.approx(0.0)
    # Higher T means lower β, so dβ < 0 for T > T0.
    assert (op.dbeta[1:] < 0).all()


def test_opes_expanded_compute_bias_v_returns_zero_pre_init():
    """compute_bias_v returns 0.0 before init_from_observations has been called."""
    op = _opes_expanded()
    assert op.compute_bias_v(123.456) == 0.0


def test_opes_expanded_compute_bias_v_matches_hand_calc():
    """LSE definition: V(U) = -kT * log( (1/N) sum_j exp(-Δβ_j ΔU + ΔF_j/kT - Δβ_j U_ref) )
    Here we set ΔF arbitrarily and compare against a direct sum.
    """
    op = _opes_expanded(t_min=300.0, t_max=400.0, n_temps=3)
    op._u_ref_kj = -50.0
    op._deltaf_kj = np.array([0.5, -0.3, 0.7])
    op._initialized = True

    u_kj = -42.0
    delta_u = u_kj - op._u_ref_kj
    df_shift = op._deltaf_kj - op._kt0_kj * op._dbeta * op._u_ref_kj
    args = -op._dbeta * delta_u + df_shift / op._kt0_kj
    expected = -op._kt0_kj * (
        math.log(np.exp(args).mean()) if args.max() < 700 else
        args.max() + math.log(np.exp(args - args.max()).sum() / args.size)
    )
    assert op.compute_bias_v(u_kj) == pytest.approx(expected, rel=1e-12)


def test_opes_expanded_compute_alpha_at_t0_only_is_zero():
    """Alpha is zero when all grid temperatures equal T0 (dbeta = 0 everywhere)."""
    op = _opes_expanded(t_min=300.0, t_max=300.0, n_temps=2)
    op._u_ref_kj = -42.0
    op._deltaf_kj = np.zeros(2)
    op._initialized = True
    # All temperatures equal T0 → dβ = 0 everywhere → alpha = 0.
    assert op._compute_alpha(-42.0) == pytest.approx(0.0)


def test_opes_expanded_init_from_observations_zero_for_t0_only():
    """ΔF initialises to zero and u_ref to the mean energy when all temperatures equal T0."""
    op = _opes_expanded(t_min=300.0, t_max=300.0, n_temps=3)
    op._obs_energies = [-10.0, -12.0, -14.0, -11.5]
    op.init_from_observations()
    # All temperatures equal T0 → dβ=0, log of sum should make ΔF = 0.
    np.testing.assert_allclose(op.deltaf_kj, 0.0, atol=1e-12)
    assert op.initialized
    assert op._u_ref_kj == pytest.approx(np.mean([-10.0, -12.0, -14.0, -11.5]))


def test_opes_expanded_init_from_observations_against_direct_lse():
    """init_from_observations should produce the same ΔF as a direct
    log-sum-exp over all observations, evaluated through compute_bias_v
    (which shares the exact internal arg layout). The bias at any test
    energy must agree between the running update and a fresh impl that
    rebuilds ΔF from scratch by running the same loop on the same data.
    """
    op = _opes_expanded(t_min=300.0, t_max=500.0, n_temps=4)
    rng = np.random.default_rng(7)
    obs = rng.normal(-50.0, 5.0, size=12)
    op._obs_energies = list(obs)
    op.init_from_observations()

    # Direct: ΔF_j = -kT log( (1/N) sum_t exp(-Δβ_j (U_t - U_ref)) )
    # in the canonical (un-shifted) sense, then compute_bias_v applies the
    # shift internally. Build ΔF by direct vectorised LSE and compare.
    kt = op._kt0_kj
    u_ref = op._u_ref_kj
    args = -op._dbeta[:, None] * (np.array(obs) - u_ref)  # (n_temps, n_obs)
    log_mean = (args.max(axis=1)
                + np.log(np.exp(args - args.max(axis=1)[:, None]).sum(axis=1))
                - math.log(len(obs)))
    canonical_df = -kt * log_mean
    # Convert to the shifted basis used internally:
    expected_df_shifted = canonical_df + kt * op._dbeta * u_ref
    np.testing.assert_allclose(
        op.deltaf_kj, expected_df_shifted, rtol=1e-12, atol=1e-12,
    )


def test_opes_expanded_update_deltaf_keeps_finite_and_increments_counter():
    """update_deltaf increments the step counter and keeps ΔF and rct finite."""
    op = _opes_expanded(t_min=300.0, t_max=500.0, n_temps=4)
    op._deltaf_kj = np.array([0.1, 0.2, 0.3, 0.4])
    op._u_ref_kj = -50.0
    op._counter = 5
    op._rct_kj = 0.12
    op._initialized = True

    op.update_deltaf(bias_v_kj=0.5, u_kj=-48.0)
    assert op._counter == 6
    assert np.isfinite(op._deltaf_kj).all()
    assert math.isfinite(op._rct_kj)


def test_opes_expanded_reweighted_log_weight_simple_form():
    """Reweighted log weight equals -(beta_target - beta0) * U at the chosen target temperature."""
    op = _opes_expanded(t_min=300.0, t_max=500.0, n_temps=3)
    u_kj = -40.0
    bias_v = 1.5
    op._u_ref_kj = -42.0
    op._deltaf_kj = np.array([0.1, 0.2, 0.3])
    op._initialized = True

    # Direct: -(β_target - β_0) U at the chosen target temperature.
    target_t = 450.0
    beta_target = 1.0 / (KB_KJ_PER_MOL_K * target_t)
    expected = -(beta_target - 1.0 / op._kt0_kj) * u_kj
    assert op.reweighted_log_weight(u_kj, target_t) == pytest.approx(expected)


def test_opes_expanded_apply_to_context_pushes_globals_after_init():
    """apply_to_context sets opes_initialized=1, kT, u_ref, alpha, and the per-grid LSE globals."""
    system = _toy_system_with_two_forces()
    n_temps = 3
    add_opes_expanded_bias(system, n_temps=n_temps)
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions([(0.0, 0.0, 0.0)])

    op = _opes_expanded(t_min=300.0, t_max=400.0, n_temps=n_temps)
    op._u_ref_kj = -10.0
    op._deltaf_kj = np.array([0.1, 0.2, 0.3])
    op._initialized = True
    op.apply_to_context(context)
    assert context.getParameter('opes_initialized') == pytest.approx(1.0)
    assert context.getParameter('opes_u_ref') == pytest.approx(-10.0)
    assert context.getParameter('opes_kt') == pytest.approx(op._kt0_kj)
    assert math.isfinite(context.getParameter('opes_alpha'))
    expected_df_shift = op.deltaf_kj - op._kt0_kj * op.dbeta * op._u_ref_kj
    for j in range(n_temps):
        assert context.getParameter(f'opes_dbeta_{j}') == pytest.approx(op.dbeta[j])
        assert context.getParameter(f'opes_df_shift_{j}') == pytest.approx(expected_df_shift[j])


def test_opes_expanded_state_roundtrip_preserves_arrays():
    """Saving and loading state restores deltaf, u_ref, counter, and initialized flag."""
    op = _opes_expanded(t_min=300.0, t_max=500.0, n_temps=4)
    op._deltaf_kj = np.array([0.1, 0.2, 0.3, 0.4])
    op._u_ref_kj = -42.0
    op._counter = 7
    op._rct_kj = 0.5
    op._initialized = True
    state = op.get_state()

    fresh = _opes_expanded(t_min=300.0, t_max=500.0, n_temps=4)
    fresh.load_state(state)
    np.testing.assert_allclose(fresh.deltaf_kj, op.deltaf_kj)
    assert fresh._u_ref_kj == op._u_ref_kj
    assert fresh._counter == op._counter
    assert fresh.initialized


def _bias_only_state_for(system, opes, position=(0.0, 0.0, 0.0)):
    """Eval the wrapper energy alone via getState(groups=1<<OPES_BIAS_FORCE_GROUP).

    Returns the wrapper energy in kJ/mol. Uses the Reference platform so we
    measure the math, not the GPU.
    """
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(
        system, integrator, mm.Platform.getPlatformByName('Reference'),
    )
    context.setPositions([position] * system.getNumParticles())
    opes.apply_to_context(context)
    state = context.getState(
        getEnergy=True, groups=1 << OPES_BIAS_FORCE_GROUP,
    )
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _system_with_constant_u(u_kj: float) -> mm.System:
    """Single-particle system whose only Force is a constant offset of u_kj kJ/mol.

    A `CustomExternalForce(<constant>)` evaluates to its value; combined
    with the OPES wrapper we can dial U_FF arbitrarily and check the LSE
    energy against Python's compute_bias_v at the same U.
    """
    system = mm.System()
    system.addParticle(1.0)
    constant = mm.CustomExternalForce(f'{u_kj}')
    constant.addParticle(0)
    system.addForce(constant)
    return system


def test_wrapper_lse_matches_python_at_u_ref():
    """At U == u_ref the on-device LSE must match compute_bias_v(u_ref) exactly."""
    u_ref = -640_000.0
    system = _system_with_constant_u(u_ref)
    n_temps = 8
    add_opes_expanded_bias(system, n_temps=n_temps)

    op = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=300.0 * unit.kelvin,
        t_max=350.0 * unit.kelvin,
        n_temps=n_temps,
        pace=100, observation_steps=10,
        geom_spacing=False,
    )
    rng = np.random.default_rng(20260507)
    op._deltaf_kj = rng.uniform(-50_000.0, 50_000.0, size=n_temps)
    op._deltaf_kj[0] = 0.0
    op._u_ref_kj = u_ref
    op._initialized = True

    on_device = _bias_only_state_for(system, op)
    expected = op.compute_bias_v(u_ref)
    assert on_device == pytest.approx(expected, rel=1e-4, abs=1e-3)


def test_wrapper_lse_matches_python_off_diagonal():
    """At U != u_ref the LSE saturates while affine would diverge — the
    wrapper must follow the LSE.

    This is the regression that broke 1jwp swap acceptance: the
    pre-LSE affine wrapper agreed at u_ref but drifted by hundreds of
    reduced units once |U − u_ref| ≳ a few × σ_U.
    """
    u_ref = -640_000.0
    n_temps = 30  # production grid size
    rng = np.random.default_rng(20260507)
    deltaf = rng.uniform(0.0, 90_000.0, size=n_temps)
    deltaf.sort()
    deltaf[0] = 0.0

    op = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=300.0 * unit.kelvin,
        t_max=350.0 * unit.kelvin,
        n_temps=n_temps, pace=100, observation_steps=10,
        geom_spacing=False,
    )
    op._deltaf_kj = deltaf
    op._u_ref_kj = u_ref
    op._initialized = True

    # Sweep U from u_ref by displacements that span the multithermal range.
    for delta_u in (-30_000.0, -10_000.0, -3_000.0, -1_000.0,
                    1_000.0, 3_000.0, 10_000.0, 30_000.0):
        u_actual = u_ref + delta_u
        system = _system_with_constant_u(u_actual)
        add_opes_expanded_bias(system, n_temps=n_temps)
        on_device = _bias_only_state_for(system, op)
        expected = op.compute_bias_v(u_actual)
        # Per-grid args are computed in float32 on Reference too; expect a
        # few kJ/mol absolute slack for the largest displacement, which is
        # still microscopic vs the ~700 kJ/mol affine error at the same Δ.
        assert on_device == pytest.approx(expected, rel=1e-3, abs=2.0), (
            f'on-device {on_device:.3f} vs python {expected:.3f} at ΔU={delta_u}'
        )


def test_wrapper_returns_zero_before_init():
    """opes_initialized=0 must zero out the wrapper energy regardless of ΔF."""
    u_ref = -1_000.0
    system = _system_with_constant_u(u_ref + 50.0)
    n_temps = 4
    add_opes_expanded_bias(system, n_temps=n_temps)

    op = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=300.0 * unit.kelvin,
        t_max=350.0 * unit.kelvin,
        n_temps=n_temps, pace=100, observation_steps=10,
        geom_spacing=False,
    )
    op._deltaf_kj = np.array([0.0, 1_000.0, 2_000.0, 3_000.0])
    op._u_ref_kj = u_ref
    op._initialized = False  # gate off

    on_device = _bias_only_state_for(system, op)
    assert on_device == pytest.approx(0.0, abs=1e-9)


def test_wrapper_lse_is_t0_at_t0_only_grid():
    """All-T0 grid (Δβ=0 everywhere) means the LSE collapses to a constant,
    and compute_bias_v returns the same constant by construction. The
    wrapper energy must equal that constant (independent of u_ref / ΔF).
    """
    u_actual = -42.0
    system = _system_with_constant_u(u_actual)
    n_temps = 5
    add_opes_expanded_bias(system, n_temps=n_temps)

    op = _opes_expanded(t_min=300.0, t_max=300.0, n_temps=n_temps)
    op._deltaf_kj = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # uniform
    op._u_ref_kj = -100.0  # arbitrarily far
    op._initialized = True

    expected = op.compute_bias_v(u_actual)
    on_device = _bias_only_state_for(system, op)
    assert on_device == pytest.approx(expected, abs=1e-6)


def test_wrapper_lse_finite_at_production_argument_magnitudes():
    """Stress test: ΔF and u_ref at production scale (ΔF up to ~100k kJ/mol,
    u_ref ~ -640k kJ/mol). The on-device LSE must remain finite — pre-shift
    args span ~37k, but the max-shifted exp() inputs stay in [-large, 0],
    so the sum and the energy are well-defined.
    """
    u_ref = -640_000.0
    n_temps = 30
    deltaf = np.linspace(0.0, 95_000.0, n_temps)
    op = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=300.0 * unit.kelvin,
        t_max=350.0 * unit.kelvin,
        n_temps=n_temps, pace=100, observation_steps=10,
        geom_spacing=False,
    )
    op._deltaf_kj = deltaf
    op._u_ref_kj = u_ref
    op._initialized = True

    for delta_u in (-25_000.0, 0.0, 25_000.0):
        u_actual = u_ref + delta_u
        system = _system_with_constant_u(u_actual)
        add_opes_expanded_bias(system, n_temps=n_temps)
        on_device = _bias_only_state_for(system, op)
        assert math.isfinite(on_device), (
            f'LSE blew up at ΔU={delta_u}: got {on_device}'
        )


def _alpha_step_after_one_step(system, opes, platform_name='Reference'):
    """Build the on-device integrator, attach it, run 1 step, return opes_alpha_step."""
    integrator = build_opes_expanded_integrator(
        timestep=1.0 * unit.femtosecond,
        collision_rate=1.0 / unit.picosecond,
        temperature=300.0 * unit.kelvin,
        n_temps=opes.n_temps,
    )
    context = mm.Context(
        system, integrator, mm.Platform.getPlatformByName(platform_name),
    )
    context.setPositions([(0.0, 0.0, 0.0)] * system.getNumParticles())
    context.setVelocitiesToTemperature(300.0 * unit.kelvin)
    opes.apply_to_context(context)
    integrator.step(1)
    return integrator.getGlobalVariableByName('opes_alpha_step')


def test_integrator_alpha_step_matches_python_at_u_ref():
    """On-device opes_alpha_step at energy=u_ref equals OpesExpanded._compute_alpha(u_ref)."""
    n_temps = 30
    rng = np.random.default_rng(20260518)
    df = rng.uniform(0.0, 70_000.0, size=n_temps); df[0] = 0.0
    u_ref = -440_000.0

    op = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=300.0 * unit.kelvin, t_max=350.0 * unit.kelvin,
        n_temps=n_temps, pace=100, observation_steps=10,
    )
    op._deltaf_kj = df
    op._u_ref_kj = u_ref
    op._initialized = True

    system = _system_with_constant_u(u_ref)
    add_opes_expanded_bias(system, n_temps=n_temps)
    actual = _alpha_step_after_one_step(system, op)
    expected = op._compute_alpha(u_ref)
    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-8)


def test_integrator_alpha_step_off_diagonal_matches_python_lse():
    """Sweep U away from u_ref; on-device α tracks the exact softmax-weighted ⟨Δβ⟩.

    This is the regression-guard for the 2026-05-18 per-step bias drift: the
    pre-fix integrator used α frozen at u_ref for the entire PACE; this test
    fixes that by checking α at multiple U values against the exact Python LSE.
    """
    n_temps = 30
    rng = np.random.default_rng(2026_05_18)
    df = rng.uniform(0.0, 70_000.0, size=n_temps); df[0] = 0.0
    u_ref = -440_000.0

    op = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=300.0 * unit.kelvin, t_max=350.0 * unit.kelvin,
        n_temps=n_temps, pace=100, observation_steps=10,
    )
    op._deltaf_kj = df
    op._u_ref_kj = u_ref
    op._initialized = True

    for delta_u in (-8000.0, -1500.0, -100.0, 0.0, 100.0, 1500.0, 8000.0):
        u_actual = u_ref + delta_u
        system = _system_with_constant_u(u_actual)
        add_opes_expanded_bias(system, n_temps=n_temps)
        actual = _alpha_step_after_one_step(system, op)
        expected = op._compute_alpha(u_actual)
        assert actual == pytest.approx(expected, rel=1e-4, abs=1e-7), (
            f'ΔU={delta_u}: on-device α={actual}, Python α={expected}'
        )


def test_integrator_alpha_step_zero_before_init():
    """Before init, opes_initialized=0 gates α=0 regardless of grid globals."""
    n_temps = 6
    op = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=300.0 * unit.kelvin, t_max=350.0 * unit.kelvin,
        n_temps=n_temps, pace=100, observation_steps=10,
    )
    assert not op.initialized

    system = _system_with_constant_u(-100_000.0)
    add_opes_expanded_bias(system, n_temps=n_temps)
    actual = _alpha_step_after_one_step(system, op)
    assert actual == pytest.approx(0.0, abs=1e-12)


def test_integrator_alpha_step_finite_at_production_argument_magnitudes():
    """Stress test: u_ref=-640k, ΔF up to 95k, |ΔU| up to 25k. No NaN/inf even
    though raw arg_j magnitudes exceed 30k. Float32 safety check on the
    integrator path (mirrors test_wrapper_lse_finite_at_production_argument_magnitudes
    but for the per-step force-scaling code path)."""
    n_temps = 30
    rng = np.random.default_rng(20260518)
    df = rng.uniform(0.0, 95_000.0, size=n_temps); df[0] = 0.0
    u_ref = -640_000.0

    op = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=300.0 * unit.kelvin, t_max=350.0 * unit.kelvin,
        n_temps=n_temps, pace=100, observation_steps=10,
    )
    op._deltaf_kj = df
    op._u_ref_kj = u_ref
    op._initialized = True

    for delta_u in (-25_000.0, -5_000.0, 0.0, 5_000.0, 25_000.0):
        u_actual = u_ref + delta_u
        system = _system_with_constant_u(u_actual)
        add_opes_expanded_bias(system, n_temps=n_temps)
        actual = _alpha_step_after_one_step(system, op)
        assert math.isfinite(actual), (
            f'opes_alpha_step blew up at ΔU={delta_u}: got {actual}'
        )


def test_build_alpha_step_expression_rejects_too_few_temps():
    with pytest.raises(ValueError, match='n_temps must be >= 2'):
        _build_alpha_step_expression(1)


def test_wrapper_swap_constants_cancel_in_pairwise_delta():
    """For two states with different bias parameters but the same U (single particle),
    the difference in wrapper energy is exactly V_LSE_a(U) − V_LSE_b(U) — which
    is what enters the swap criterion.

    The earlier affine wrapper would also satisfy this; the test guards
    against regressions where state-specific constants leak into the
    pairwise delta.
    """
    u_actual = -640_000.0
    n_temps = 6
    rng = np.random.default_rng(2026)
    df_a = rng.uniform(0.0, 80_000.0, size=n_temps); df_a[0] = 0.0
    df_b = rng.uniform(0.0, 80_000.0, size=n_temps); df_b[0] = 0.0

    op_a = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=300.0 * unit.kelvin, t_max=350.0 * unit.kelvin,
        n_temps=n_temps, pace=100, observation_steps=10,
        geom_spacing=False,
    )
    op_a._deltaf_kj = df_a
    op_a._u_ref_kj = u_actual - 5_000.0
    op_a._initialized = True

    op_b = OpesExpanded(
        temperature=300.0 * unit.kelvin,
        t_min=300.0 * unit.kelvin, t_max=350.0 * unit.kelvin,
        n_temps=n_temps, pace=100, observation_steps=10,
        geom_spacing=False,
    )
    op_b._deltaf_kj = df_b
    op_b._u_ref_kj = u_actual + 7_000.0
    op_b._initialized = True

    system_a = _system_with_constant_u(u_actual)
    add_opes_expanded_bias(system_a, n_temps=n_temps)
    e_a_on_device = _bias_only_state_for(system_a, op_a)

    system_b = _system_with_constant_u(u_actual)
    add_opes_expanded_bias(system_b, n_temps=n_temps)
    e_b_on_device = _bias_only_state_for(system_b, op_b)

    expected = op_a.compute_bias_v(u_actual) - op_b.compute_bias_v(u_actual)
    actual = e_a_on_device - e_b_on_device
    assert actual == pytest.approx(expected, rel=1e-3, abs=1e-1)
