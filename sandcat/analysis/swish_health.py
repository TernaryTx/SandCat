from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from openmm import unit

from sandcat.variables import (
    DEFAULT_TEMPERATURE_K,
    KB_KJ_PER_MOL_K,
    SWISH_ITER_LENGTH,
)


def window_edges(
        n_iterations: int,
        window_ns: float = 10.0,
        iteration_ps: float = SWISH_ITER_LENGTH.value_in_unit(unit.picosecond),
) -> np.ndarray:
    """Return iteration indices delimiting fixed simulation-time windows.

    Args:
        n_iterations: Number of recorded replica-exchange iterations.
        window_ns: Width of each analysis window in nanoseconds.
        iteration_ps: Simulation time represented by one iteration in picoseconds.

    Returns:
        Array of inclusive lower and exclusive upper window boundaries.

    Raises:
        ValueError: If an input value is not positive.
    """
    if n_iterations <= 0:
        raise ValueError('n_iterations must be positive.')
    if window_ns <= 0:
        raise ValueError('window_ns must be positive.')
    if iteration_ps <= 0:
        raise ValueError('iteration_ps must be positive.')
    iterations_per_window = max(1, round(window_ns * 1000.0 / iteration_ps))
    edges = np.arange(0, n_iterations + 1, iterations_per_window, dtype=int)
    if edges[-1] != n_iterations:
        edges = np.append(edges, n_iterations)
    return edges


def window_centres_ns(edges: np.ndarray, iteration_ps: float) -> np.ndarray:
    """Return the simulation time at the centre of each window.

    Args:
        edges: Iteration-index window boundaries.
        iteration_ps: Simulation time represented by one iteration in picoseconds.

    Returns:
        Window-centre times in nanoseconds.
    """
    return 0.5 * (edges[:-1] + edges[1:]) * iteration_ps / 1000.0


def sigma_u_per_state(
        energies: np.ndarray,
        states: np.ndarray,
        edges: np.ndarray,
        thermal_energy_kj: float,
) -> np.ndarray:
    """Calculate windowed potential-energy fluctuations per thermodynamic state.

    Args:
        energies: Reduced potentials with shape ``(iteration, replica, state)``.
        states: State occupied by each replica at each iteration.
        edges: Iteration-index window boundaries.
        thermal_energy_kj: Thermal energy used to convert reduced potentials to kJ/mol.

    Returns:
        Sigma(U) values with shape ``(window, state)`` in kJ/mol.
    """
    n_states = energies.shape[2]
    sigma_u = np.full((len(edges) - 1, n_states), np.nan)
    for window_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if upper - lower < 2:
            continue
        window_states = states[lower:upper]
        window_energies = energies[lower:upper]
        for state_index in range(n_states):
            state_mask = window_states == state_index
            values = window_energies[..., state_index][state_mask] * thermal_energy_kj
            if values.size > 1:
                sigma_u[window_index, state_index] = values.std()
    return sigma_u


def acceptance_rates(
        accepted: np.ndarray,
        proposed: np.ndarray,
        edges: np.ndarray,
) -> np.ndarray:
    """Calculate windowed acceptance rates for neighbouring state pairs.

    Args:
        accepted: Per-iteration accepted-swap count matrices.
        proposed: Per-iteration proposed-swap count matrices.
        edges: Iteration-index window boundaries.

    Returns:
        Acceptance fractions with shape ``(window, neighbouring pair)``.
    """
    n_states = accepted.shape[1]
    rates = np.full((len(edges) - 1, n_states - 1), np.nan)
    for window_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        window_accepted = accepted[lower:upper].sum(axis=0)
        window_proposed = proposed[lower:upper].sum(axis=0)
        for state_index in range(n_states - 1):
            n_proposed = window_proposed[state_index, state_index + 1]
            if n_proposed > 0:
                rates[window_index, state_index] = (
                    window_accepted[state_index, state_index + 1] / n_proposed
                )
    return rates


def state_occupancy_shares(states: np.ndarray, state_index: int = 0) -> np.ndarray:
    """Fraction of the run each replica spent in one thermodynamic state.

    Acceptance rates and round-trip counts both describe transport along the
    ladder as a whole and can look healthy while one replica still monopolises a
    state. That matters most at state 0, because the lambda = 1 ensemble is what
    the prospective density analysis is built from: if a single replica held the
    slot for most of the run, that ensemble is close to one continuous
    trajectory rather than an exchange-enhanced one, and it carries
    correspondingly less independent sampling.

    An even ladder gives roughly ``1 / n_replicas`` per replica. A share far
    above that, with one or more replicas never reaching the state at all,
    indicates that the ensemble carries less independent sampling than the
    replica count suggests.

    Args:
        states: Per-iteration replica-to-state assignments, shape
            ``(iteration, replica)``, as stored in the reporter's ``states``
            variable.
        state_index: Thermodynamic state to report on. 0 is lambda = 1.

    Returns:
        Fraction of iterations each replica spent in that state, indexed by
        replica.

    Raises:
        ValueError: If ``state_index`` is not present in ``states``.
    """
    if state_index not in np.unique(states):
        raise ValueError(f'state {state_index} does not appear in the assignments')
    return (states == state_index).mean(axis=0)


def swap_energy_gaps(
        energies: np.ndarray,
        states: np.ndarray,
        thermal_energy_kj: float,
) -> np.ndarray:
    """Calculate the symmetrized swap-energy gap for neighbouring states.

    Args:
        energies: Reduced potentials with shape ``(iteration, replica, state)``.
        states: State occupied by each replica at each iteration.
        thermal_energy_kj: Thermal energy used to convert reduced potentials to kJ/mol.

    Returns:
        Swap-energy gaps with shape ``(iteration, neighbouring pair)`` in kJ/mol.
    """
    n_iterations, _, n_states = energies.shape
    replica_at_state = np.argsort(states, axis=1)
    iteration_indices = np.arange(n_iterations)
    gaps = np.full((n_iterations, n_states - 1), np.nan)
    for state_index in range(n_states - 1):
        lower_replica = replica_at_state[:, state_index]
        upper_replica = replica_at_state[:, state_index + 1]
        gaps[:, state_index] = (
            energies[iteration_indices, upper_replica, state_index]
            + energies[iteration_indices, lower_replica, state_index + 1]
            - energies[iteration_indices, lower_replica, state_index]
            - energies[iteration_indices, upper_replica, state_index + 1]
        )
    return gaps * thermal_energy_kj


def window_means(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Average a two-dimensional time series within analysis windows.

    Args:
        values: Values with shape ``(iteration, series)``.
        edges: Iteration-index window boundaries.

    Returns:
        Windowed means with shape ``(window, series)``.
    """
    means = np.full((len(edges) - 1, values.shape[1]), np.nan)
    for window_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if upper > lower:
            means[window_index] = np.nanmean(values[lower:upper], axis=0)
    return means


def load_rct_series(
        nc_path: Path,
        n_replicas: int,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """Load OPES reweighting cutoffs from SWISH-X DeltaFs sidecars.

    Args:
        nc_path: Main or backup MultiStateReporter path.
        n_replicas: Number of replicas expected in the run.

    Returns:
        Common time axis in nanoseconds and ``(replica, rct)`` series pairs.
    """
    prefix = nc_path.name.removesuffix('.bak').removesuffix('.nc')
    times = []
    series = []
    for replica_index in range(n_replicas):
        deltaf_path = nc_path.parent / f'{prefix}_DeltaFs.{replica_index}.data'
        if not deltaf_path.exists():
            continue
        with deltaf_path.open() as deltaf_file:
            fields = deltaf_file.readline().split()[2:]
        data = np.atleast_2d(np.loadtxt(deltaf_path, comments='#'))
        times.append(data[:, fields.index('time')])
        series.append((replica_index, data[:, fields.index('rct')]))
    if not times:
        return np.array([]), []
    common_length = min(len(time_values) for time_values in times)
    return (
        times[0][:common_length] / 1000.0,
        [(replica, values[:common_length]) for replica, values in series],
    )


def plot_swish_health(
        nc_path: Path,
        output_path: Path,
        window_ns: float = 10.0,
        temperature_k: float = DEFAULT_TEMPERATURE_K,
        iteration_ps: float = SWISH_ITER_LENGTH.value_in_unit(unit.picosecond),
) -> None:
    """Create a four-panel health report for a SWISH or SWISH-X run.

    Args:
        nc_path: Main or validated backup MultiStateReporter NetCDF file.
        output_path: Destination PNG path.
        window_ns: Simulation-time window for reported metrics.
        temperature_k: Reference temperature in kelvin.
        iteration_ps: Simulation time represented by one exchange iteration.

    Raises:
        FileNotFoundError: If the NetCDF file does not exist.
    """
    if not nc_path.exists():
        raise FileNotFoundError(f'{nc_path} does not exist.')
    with netCDF4.Dataset(nc_path) as dataset:
        n_iterations = int(dataset.variables['last_iteration'][0]) + 1
        energies = np.asarray(dataset.variables['energies'][:n_iterations])
        states = np.asarray(dataset.variables['states'][:n_iterations])
        accepted = np.asarray(dataset.variables['accepted'][:n_iterations])
        proposed = np.asarray(dataset.variables['proposed'][:n_iterations])

    n_replicas = energies.shape[1]
    n_states = energies.shape[2]
    edges = window_edges(n_iterations, window_ns, iteration_ps)
    time_ns = window_centres_ns(edges, iteration_ps)
    thermal_energy_kj = KB_KJ_PER_MOL_K * temperature_k
    sigma_u = sigma_u_per_state(energies, states, edges, thermal_energy_kj)
    rates = acceptance_rates(accepted, proposed, edges)
    gaps = window_means(swap_energy_gaps(energies, states, thermal_energy_kj), edges)
    rct_time_ns, rct_series = load_rct_series(nc_path, n_replicas)

    # Reported rather than plotted: a skewed lambda = 1 slot is invisible in the
    # acceptance rates and round-trip counts, and it is the ensemble the
    # prospective density analysis is built from.
    shares = state_occupancy_shares(states, state_index=0)
    logging.info(
        'state 0 occupancy by replica (%%): %s',
        ', '.join(f'{share * 100:.1f}' for share in shares),
    )
    if shares.max() > 3.0 / n_replicas:
        logging.warning(
            'replica %d held state 0 for %.0f%% of the run against an even '
            'share of %.0f%%; the lambda = 1 ensemble carries less independent '
            'sampling than the replica count suggests',
            int(shares.argmax()), shares.max() * 100, 100.0 / n_replicas,
        )
    if rct_series:
        snapshot_time_ns = n_iterations * iteration_ps / 1000.0
        rct_length = int(np.searchsorted(rct_time_ns, snapshot_time_ns, side='right'))
        rct_time_ns = rct_time_ns[:rct_length]
        rct_series = [
            (replica_index, values[:rct_length])
            for replica_index, values in rct_series
        ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    state_colors = plt.cm.viridis(np.linspace(0, 0.9, n_states))
    pair_colors = plt.cm.plasma(np.linspace(0, 0.85, n_states - 1))

    if rct_series:
        for replica_index, values in rct_series:
            axes[0, 0].plot(
                rct_time_ns, values, color=state_colors[replica_index],
                linewidth=1.0, label=f'replica {replica_index}',
            )
    else:
        axes[0, 0].text(
            0.5, 0.5, 'RCT is available for SWISH-X runs',
            transform=axes[0, 0].transAxes, ha='center', va='center',
        )
    axes[0, 0].set_ylabel('rct (kJ/mol)')
    axes[0, 0].set_xlim(-10, 260)
    if rct_series:
        axes[0, 0].legend(ncol=3, fontsize=8, loc='lower right', frameon=False)

    for pair_index in range(n_states - 1):
        axes[0, 1].plot(
            time_ns, gaps[:, pair_index], color=pair_colors[pair_index],
            marker='o', markersize=3, label=f'{pair_index}↔{pair_index + 1}',
        )
    axes[0, 1].axhline(0.0, color='grey', linestyle=':', linewidth=0.8)
    axes[0, 1].set_ylabel(
        r'$\langle \Delta\Delta U \rangle$ per $\lambda$-pair (kJ/mol)',
    )
    axes[0, 1].set_ylim(-5, 40)
    axes[0, 1].legend(
        ncol=n_states - 1, fontsize=8, loc='lower right', frameon=False,
    )

    for state_index in range(n_states):
        axes[1, 0].plot(
            time_ns, sigma_u[:, state_index], color=state_colors[state_index],
            marker='s', markersize=3, label=f'state {state_index}',
        )
    axes[1, 0].set_ylabel(r'$\sigma(U)$ per $\lambda$-state (kJ/mol)')
    axes[1, 0].set_xlabel('time (ns)')
    axes[1, 0].margins(y=0.3)
    axes[1, 0].autoscale_view(scalex=False, scaley=True)
    _, sigma_u_top = axes[1, 0].get_ylim()
    axes[1, 0].set_ylim(0.0, sigma_u_top)
    axes[1, 0].legend(ncol=3, fontsize=8, loc='upper right', frameon=False)

    for pair_index in range(n_states - 1):
        axes[1, 1].plot(
            time_ns, rates[:, pair_index], color=pair_colors[pair_index],
            marker='^', markersize=3, label=f'{pair_index}↔{pair_index + 1}',
        )
    axes[1, 1].axhline(0.2, color='grey', linestyle=':', linewidth=0.8)
    axes[1, 1].set_ylabel('neighbour acceptance')
    axes[1, 1].set_xlabel('time (ns)')
    axes[1, 1].set_ylim(-0.02, 1.15)
    axes[1, 1].legend(
        ncol=n_states - 1, fontsize=8, loc='upper right', frameon=False,
    )

    figure.suptitle(f'SWISH health check ({window_ns:g} ns windows)')
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    logging.info('Wrote SWISH health report to %s', output_path)


def main() -> None:
    """Read a SWISH reporter snapshot and write a time-resolved health-check PNG.

    The report contains OPES RCT values when SWISH-X DeltaFs sidecars are present,
    neighbouring-replica swap-energy gaps, per-state potential-energy fluctuations,
    and neighbouring-pair acceptance rates. Metrics are aggregated into fixed windows
    of simulated time. Reading a validated ``.nc.bak`` snapshot is recommended while
    a simulation is still running so the live HDF5 writer is not disturbed.
    """
    parser = parse_args()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    plot_swish_health(
        nc_path=args.nc_path,
        output_path=args.output,
        window_ns=args.window_ns,
        temperature_k=args.temperature_k,
        iteration_ps=args.iteration_ps,
    )


def parse_args() -> argparse.ArgumentParser:
    """Create the command-line parser for SWISH health plotting.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description='Plot time-resolved health metrics for a SWISH or SWISH-X run.',
    )
    parser.add_argument('--nc-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window-ns', type=float, default=10.0)
    parser.add_argument(
        '--temperature-k', type=float,
        default=DEFAULT_TEMPERATURE_K,
    )
    parser.add_argument(
        '--iteration-ps', type=float,
        default=SWISH_ITER_LENGTH.value_in_unit(unit.picosecond),
    )
    return parser


if __name__ == '__main__':
    main()
