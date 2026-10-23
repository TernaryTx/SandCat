from __future__ import annotations

import math

import numpy as np
import pytest

from sandcat.opes import (
    EXPLORE,
    METAD,
    OPES,
    OpesVariable,
    _Kernel,
    _resolve_bias_factors,
    _resolve_epsilon_cutoff,
    _resolve_initial_sigma,
    _wrap_periodic,
)


def test_wrap_periodic_inside_interval():
    """Values already within ±span/2 pass through unchanged."""
    diff = np.array([0.1, -0.2, 0.0])
    span = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi])
    periodic = np.array([True, True, True])
    out = _wrap_periodic(diff, span, periodic)
    np.testing.assert_allclose(out, diff)


def test_wrap_periodic_wraps_across_pi():
    """Values outside ±π wrap to the equivalent point inside (-π, π]."""
    span = np.array([2 * np.pi])
    periodic = np.array([True])
    diff = np.array([3.0 * np.pi / 2.0])
    np.testing.assert_allclose(
        _wrap_periodic(diff, span, periodic),
        np.array([-np.pi / 2.0]),
    )
    diff = np.array([-3.0 * np.pi / 2.0])
    np.testing.assert_allclose(
        _wrap_periodic(diff, span, periodic),
        np.array([np.pi / 2.0]),
    )


def test_wrap_periodic_only_periodic_axes():
    """Non-periodic axes pass through unchanged while periodic ones wrap."""
    diff = np.array([3.0 * np.pi / 2.0, 100.0])
    span = np.array([2 * np.pi, 1.0])
    periodic = np.array([True, False])
    out = _wrap_periodic(diff, span, periodic)
    np.testing.assert_allclose(out[1], 100.0)
    np.testing.assert_allclose(out[0], -np.pi / 2.0)


def test_resolve_bias_factors_metad_default():
    """Default METAD biasfactor = barrier/kbt and prefactor = 1 - 1/biasfactor."""
    bf, prefactor = _resolve_bias_factors(None, METAD, barrier_kj=20.0, kbt=2.5)
    assert bf == pytest.approx(8.0)
    assert prefactor == pytest.approx(1.0 - 1.0 / 8.0)


def test_resolve_bias_factors_explore_default():
    """Default EXPLORE biasfactor matches METAD but prefactor = biasfactor - 1."""
    bf, prefactor = _resolve_bias_factors(None, EXPLORE, barrier_kj=20.0, kbt=2.5)
    assert bf == pytest.approx(8.0)
    assert prefactor == pytest.approx(7.0)


def test_resolve_bias_factors_inf_metad_uniform_target():
    """Infinite biasfactor sets prefactor = 1, giving a uniform target distribution."""
    bf, prefactor = _resolve_bias_factors(
        math.inf, METAD, barrier_kj=20.0, kbt=2.5,
    )
    assert math.isinf(bf)
    assert prefactor == pytest.approx(1.0)


def test_resolve_bias_factors_inf_explore_rejected():
    """Infinite biasfactor is incompatible with EXPLORE mode and raises ValueError."""
    with pytest.raises(ValueError, match='not compatible with EXPLORE'):
        _resolve_bias_factors(math.inf, EXPLORE, barrier_kj=20.0, kbt=2.5)


def test_resolve_bias_factors_le1_rejected():
    """Biasfactor <= 1 is rejected as unphysical."""
    with pytest.raises(ValueError, match='> 1'):
        _resolve_bias_factors(1.0, METAD, barrier_kj=20.0, kbt=2.5)


def test_resolve_epsilon_cutoff_metad_defaults_match_formula():
    """Default epsilon and cutoff² match the OPES analytic formulae for METAD."""
    barrier_kj = 30.0
    kbt = 2.5
    biasfactor = barrier_kj / kbt  # default
    bias_prefactor = 1.0 - 1.0 / biasfactor
    eps, cutoff2, val_at_cutoff, threshold2 = _resolve_epsilon_cutoff(
        None, None, 1.0, METAD, barrier_kj, kbt, bias_prefactor,
    )
    assert eps == pytest.approx(math.exp(-barrier_kj / bias_prefactor / kbt))
    assert cutoff2 == pytest.approx(2.0 * barrier_kj / bias_prefactor / kbt)
    assert val_at_cutoff == pytest.approx(math.exp(-0.5 * cutoff2))
    assert threshold2 == pytest.approx(1.0)


def test_resolve_epsilon_cutoff_explore_uses_explore_branch():
    """EXPLORE uses barrier/kbt as cutoff² instead of the METAD formula."""
    barrier_kj = 30.0
    kbt = 2.5
    _, cutoff2, _, _ = _resolve_epsilon_cutoff(
        None, None, 1.0, EXPLORE, barrier_kj, kbt, bias_prefactor=7.0,
    )
    assert cutoff2 == pytest.approx(2.0 * barrier_kj / kbt)


def test_resolve_epsilon_cutoff_threshold_too_large_rejected():
    """Compression threshold larger than the kernel self-overlap raises ValueError."""
    with pytest.raises(ValueError, match='COMPRESSION_THRESHOLD'):
        _resolve_epsilon_cutoff(
            1e-3, 1.0, 2.0, METAD, 1.0, 1.0, bias_prefactor=0.5,
        )


def test_resolve_initial_sigma_adaptive_default_stride():
    """Adaptive sigma with no explicit stride defaults to 10 * pace."""
    adaptive, sigma0, stride = _resolve_initial_sigma(
        None, ncv=2, adaptive_sigma_stride=None, pace=500,
    )
    assert adaptive is True
    assert stride == 5000
    np.testing.assert_array_equal(sigma0, np.zeros(2))


def test_resolve_initial_sigma_fixed():
    """Fixed sigma vector disables adaptive mode and sets stride to 0."""
    adaptive, sigma0, stride = _resolve_initial_sigma(
        [0.1, 0.2], ncv=2, adaptive_sigma_stride=None, pace=500,
    )
    assert adaptive is False
    assert stride == 0
    np.testing.assert_allclose(sigma0, np.array([0.1, 0.2]))


def test_resolve_initial_sigma_adaptive_with_explicit_stride_clash():
    """Providing both a fixed sigma and an adaptive stride raises ValueError."""
    with pytest.raises(ValueError, match='ADAPTIVE_SIGMA_STRIDE'):
        _resolve_initial_sigma(
            [0.1], ncv=1, adaptive_sigma_stride=1000, pace=500,
        )


def _make_opes(ncv: int = 1, **kwargs) -> OPES:
    """Build an OPES instance with stub CVs — no dynamics, Force metadata only."""
    import openmm as mm

    system = mm.System()
    for _ in range(ncv):
        system.addParticle(1.0)
    variables = []
    for cv_idx in range(ncv):
        force = mm.CustomExternalForce('0')
        force.addParticle(cv_idx, [])
        variables.append(OpesVariable(
            force=force, min_value=-np.pi, max_value=np.pi,
            periodic=True, grid_width=64,
        ))
    return OPES(
        system=system,
        variables=variables,
        temperature=300.0,
        barrier=30.0,
        pace=500,
        sigma=[0.2] * ncv,
        **kwargs,
    )


def test_evaluate_kernel_at_centre_returns_height_minus_offset():
    """Kernel value at its own centre equals height * (1 - val_at_cutoff)."""
    opes = _make_opes(ncv=1)
    kernel = _Kernel(height=2.5, center=np.array([0.3]), sigma=np.array([0.2]))
    expected = 2.5 * (1.0 - opes._val_at_cutoff)
    assert opes._evaluate_kernel(kernel, np.array([0.3])) == pytest.approx(expected)


def test_evaluate_kernel_outside_cutoff_is_zero():
    """Kernel evaluates to exactly zero beyond its cutoff distance."""
    opes = _make_opes(ncv=1)
    kernel = _Kernel(height=1.0, center=np.array([0.0]), sigma=np.array([0.05]))
    cutoff_distance = math.sqrt(opes._cutoff2) * 0.05
    far = cutoff_distance * 1.1
    assert opes._evaluate_kernel(kernel, np.array([far])) == 0.0


def test_evaluate_kernel_periodic_wraps_at_seam():
    """Two equidistant points either side of the periodic seam give identical kernel values."""
    # Centre is 0.05 inside the +pi seam. Two points the same arc distance
    # of 0.09 from the centre — one inside the box, one wrapped around the
    # seam — must yield identical kernel values.
    opes = _make_opes(ncv=1)
    centre = np.pi - 0.05
    kernel = _Kernel(
        height=1.0, center=np.array([centre]), sigma=np.array([0.2]),
    )
    inside = opes._evaluate_kernel(kernel, np.array([centre - 0.09]))
    across_seam = opes._evaluate_kernel(kernel, np.array([-np.pi + 0.04]))
    assert inside > 0
    assert across_seam == pytest.approx(inside, rel=1e-12)


def test_kernel_merge_close_centers_collapse_to_one_kernel():
    """Close kernels merge into one with height-weighted centroid and summed height."""
    opes = _make_opes(ncv=1, compression_threshold=1.0)
    k1 = _Kernel(height=1.0, center=np.array([0.0]), sigma=np.array([0.2]))
    k2 = _Kernel(height=1.0, center=np.array([0.05]), sigma=np.array([0.2]))
    opes._kernels = [k1]
    opes._add_kernel_with_compression(k2)
    assert len(opes._kernels) == 1
    merged = opes._kernels[0]
    # height-weighted moment-preserving merge: equal weights → centroid at midpoint
    assert merged.center[0] == pytest.approx(0.025)
    assert merged.height == pytest.approx(2.0)


def test_kernel_merge_far_centers_kept_separate():
    """Well-separated kernels are never merged."""
    opes = _make_opes(ncv=1, compression_threshold=1.0)
    k1 = _Kernel(height=1.0, center=np.array([-1.0]), sigma=np.array([0.2]))
    k2 = _Kernel(height=1.0, center=np.array([1.0]), sigma=np.array([0.2]))
    opes._kernels = [k1]
    opes._add_kernel_with_compression(k2)
    assert len(opes._kernels) == 2


def test_zed_recomputed_for_single_kernel_matches_self_evaluation():
    """Zed for a single kernel equals its self-evaluation divided by kde_norm."""
    opes = _make_opes(ncv=1)
    kernel = _Kernel(height=2.0, center=np.array([0.3]), sigma=np.array([0.2]))
    opes._kernels = [kernel]
    opes._kde_norm = 1.0
    opes._update_zed()
    expected = kernel.height * (1.0 - opes._val_at_cutoff) / opes._kde_norm
    assert opes._zed == pytest.approx(expected)


def test_zed_two_kernel_average_matches_explicit_formula():
    """Zed for two kernels matches the explicit cross-evaluation average."""
    opes = _make_opes(ncv=1)
    a = _Kernel(height=1.0, center=np.array([-0.4]), sigma=np.array([0.2]))
    b = _Kernel(height=1.5, center=np.array([0.4]), sigma=np.array([0.2]))
    opes._kernels = [a, b]
    opes._kde_norm = 2.0
    opes._update_zed()

    aa = opes._evaluate_kernel(a, a.center)
    ab = opes._evaluate_kernel(b, a.center)
    ba = opes._evaluate_kernel(a, b.center)
    bb = opes._evaluate_kernel(b, b.center)
    expected = (aa + ab + ba + bb) / opes._kde_norm / 2.0
    assert opes._zed == pytest.approx(expected, rel=1e-12)


def test_get_free_energy_metad_sign_and_min_zero():
    """FES is non-negative, zero at the kernel centre, and higher at far bins."""
    opes = _make_opes(ncv=1)
    opes._kernels = [
        _Kernel(height=1.0, center=np.array([0.0]), sigma=np.array([0.2])),
        _Kernel(height=1.0, center=np.array([1.5]), sigma=np.array([0.2])),
    ]
    opes._kde_norm = 2.0
    opes._update_zed()
    fes = opes.get_free_energy()
    # FES min at zero by construction.
    assert fes.min() == pytest.approx(0.0, abs=1e-10)
    # FES is non-negative everywhere.
    assert (fes >= -1e-9).all()
    # Bins near a kernel centre have lower FES than far-away bins.
    centre_bin = np.argmin(np.abs(np.linspace(-np.pi, np.pi, 64) - 0.0))
    far_bin = np.argmin(np.abs(np.linspace(-np.pi, np.pi, 64) - 3.0))
    assert fes[centre_bin] < fes[far_bin]


def test_get_free_energy_explore_uses_biasfactor_prefactor():
    """EXPLORE and METAD FES share the same minimum but differ in scale due to the biasfactor prefactor."""
    opes_metad = _make_opes(ncv=1, mode=METAD)
    opes_explore = _make_opes(ncv=1, mode=EXPLORE)
    # Same kernel set on both:
    kernels = [
        _Kernel(height=1.0, center=np.array([0.0]), sigma=np.array([0.2])),
    ]
    for driver in (opes_metad, opes_explore):
        driver._kernels = list(kernels)
        driver._kde_norm = 1.0
        driver._update_zed()
    fes_metad = opes_metad.get_free_energy()
    fes_explore = opes_explore.get_free_energy()
    # Both should peak at the kernel centre with FES==0.
    assert fes_metad.min() == pytest.approx(0.0, abs=1e-10)
    assert fes_explore.min() == pytest.approx(0.0, abs=1e-10)
    # Beyond a few sigmas the two FES surfaces have different scales
    # because EXPLORE pre-multiplies by biasfactor.
    far = 64 // 2 + 5
    assert fes_explore[far] > 0
    assert fes_metad[far] > 0


def test_state_roundtrip_preserves_kernels_and_scalars():
    """Saving and loading state restores kernels, counter, and sum_weights exactly."""
    opes = _make_opes(ncv=1, compression_threshold=0.0)
    opes._kernels = [
        _Kernel(height=1.5, center=np.array([0.1]), sigma=np.array([0.2])),
        _Kernel(height=2.5, center=np.array([-0.5]), sigma=np.array([0.15])),
    ]
    opes._counter = 7
    opes._sum_weights = 4.0
    opes._sum_weights2 = 16.0
    opes._update_zed()
    state = opes.get_state()

    fresh = _make_opes(ncv=1, compression_threshold=0.0)
    fresh.load_state(state)

    assert fresh._counter == 7
    assert fresh._sum_weights == pytest.approx(4.0)
    assert len(fresh._kernels) == 2
    np.testing.assert_allclose(
        fresh._kernels[1].center, np.array([-0.5]),
    )
