from __future__ import annotations

import numpy as np
import pytest
from openmm import unit

from sandcat.analysis.swish_health import (
    acceptance_rates,
    parse_args,
    sigma_u_per_state,
    swap_energy_gaps,
    window_edges,
    window_means,
)
from sandcat.variables import DEFAULT_TEMPERATURE_K, SWISH_ITER_LENGTH


def test_window_edges_include_partial_final_window() -> None:
    np.testing.assert_array_equal(
        window_edges(2501, window_ns=10, iteration_ps=10),
        [0, 1000, 2000, 2501],
    )


def test_cli_defaults_reuse_simulation_temperature_and_iteration_length() -> None:
    parser = parse_args()
    assert parser.get_default('temperature_k') == pytest.approx(
        DEFAULT_TEMPERATURE_K,
    )
    assert parser.get_default('iteration_ps') == pytest.approx(
        SWISH_ITER_LENGTH.value_in_unit(unit.picosecond),
    )


@pytest.mark.parametrize(
    'n_iterations,window_ns,iteration_ps',
    [(0, 10.0, 10.0), (10, 0.0, 10.0), (10, 10.0, 0.0)],
)
def test_window_edges_reject_nonpositive_inputs(
        n_iterations: int,
        window_ns: float,
        iteration_ps: float,
) -> None:
    with pytest.raises(ValueError):
        window_edges(n_iterations, window_ns, iteration_ps)


def test_sigma_u_tracks_occupied_thermodynamic_state() -> None:
    energies = np.array([
        [[1.0, 20.0], [10.0, 2.0]],
        [[10.0, 4.0], [3.0, 20.0]],
    ])
    states = np.array([[0, 1], [1, 0]])
    result = sigma_u_per_state(energies, states, np.array([0, 2]), thermal_energy_kj=1.0)
    np.testing.assert_allclose(result, [[1.0, 1.0]])


def test_acceptance_rates_are_windowed_per_neighbour_pair() -> None:
    accepted = np.zeros((2, 3, 3), dtype=int)
    proposed = np.zeros((2, 3, 3), dtype=int)
    accepted[:, 0, 1] = [1, 0]
    proposed[:, 0, 1] = [1, 1]
    accepted[:, 1, 2] = [1, 1]
    proposed[:, 1, 2] = [1, 1]
    result = acceptance_rates(accepted, proposed, np.array([0, 2]))
    np.testing.assert_allclose(result, [[0.5, 1.0]])


def test_swap_energy_gap_uses_cross_evaluated_energies() -> None:
    energies = np.array([[[1.0, 4.0], [3.0, 2.0]]])
    states = np.array([[0, 1]])
    result = swap_energy_gaps(energies, states, thermal_energy_kj=2.0)
    np.testing.assert_allclose(result, [[8.0]])


def test_window_means_average_each_series() -> None:
    values = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    result = window_means(values, np.array([0, 2, 3]))
    np.testing.assert_allclose(result, [[2.0, 4.0], [5.0, 10.0]])
