from __future__ import annotations

import csv
import logging
import random
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from tempfile import mkdtemp

try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
    MPI_RANK = _COMM.Get_rank()
    MPI_SIZE = _COMM.Get_size()
except ImportError:
    _COMM = None
    MPI_RANK = 0
    MPI_SIZE = 1

import numpy as np
import openmm as mm
from openmm import app, unit
from openmmtools import mcmc, multistate, states

from sandcat.md_variables import FRICTION, TEMPERATURE, get_n_steps
from sandcat.md_utils import (
    OpesReplicaExchangeSampler,
    SwishLambdaState,
    add_contact_map_upper_wall,
    add_probe_repulsion,
    create_static_swish_systems,
    create_swish_system,
)
from sandcat.opes_expanded import (
    OpesExpanded,
    OpesExpandedDynamicsMove,
    add_opes_expanded_bias,
    build_opes_expanded_integrator,
)
from sandcat.config import add_config_arg, parse_args_with_config
from sandcat.utils import add_mixed_solvent, create_probe_system_generator, load_protein_pdb, solvent_box_kwargs
from sandcat.variables import (
    ALL_FORCEFIELDS,
    FORCEFIELD_OPTIONS,
    PROBE_CONCENTRATION_HELP,
    PROBE_SDF_MAP,
    SWISH_ITER_LENGTH,
)

# Genuinely shared across helpers — every timed config uses the same CUDA platform.
PLATFORM_NAME = 'CUDA'
PLATFORM_PROPERTIES = {'Precision': 'mixed'}


def _platform() -> mm.Platform:
    return mm.Platform.getPlatformByName(PLATFORM_NAME)


def _build_system(
        pdb_path: str,
        probe_type: str,
        conc_mol_per_litre: float,
        forcefield: str,
        cmap_at: float = 30.0,
        box_info_json_path: str | None = None,
        probe_ff_path: str | None = None,
) -> tuple[app.System, app.Modeller, app.ForceField]:
    """Build the SWISH-style mixed-solvent system from a PDB.

    Mirrors the setup in ``run_swish.main`` up through the contact-map upper
    wall, so all four configurations time the same molecular topology.
    Deterministic given ``add_mixed_solvent``'s default seed, so independent
    construction on each MPI rank lands at identical coordinates.
    """
    protein = load_protein_pdb(pdb_path)
    omm_model = app.Modeller(protein.topology, protein.positions)
    probe_mol, system_generator = create_probe_system_generator(
        forcefield=forcefield,
        probe_type=probe_type,
        probe_ff_path=probe_ff_path,
    )
    _, water_model = FORCEFIELD_OPTIONS[forcefield]
    random.seed(42)
    omm_model.addSolvent(
        system_generator.forcefield, model=water_model,
        ionicStrength=0.15 * unit.molar,
        **solvent_box_kwargs(box_info_json_path),
    )
    add_mixed_solvent(
        omm_model,
        probe_mol=probe_mol,
        conc_mol_per_litre=conc_mol_per_litre,
        preserve_single_water_replacement=probe_type in PROBE_SDF_MAP,
    )
    system = system_generator.forcefield.createSystem(
        omm_model.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometer,
        constraints=app.HBonds,
        hydrogenMass=4 * unit.amu,
    )
    add_probe_repulsion(system, omm_model)
    add_contact_map_upper_wall(system, omm_model, at=cmap_at)
    return system, omm_model, system_generator.forcefield


def _quick_warmup(system, topology, positions, timestep_qty, warmup_steps: int = 250) -> tuple:
    """Minimise + 1 ps unbiased Langevin → stable starting positions/box.

    Production NVT below uses these directly. We deliberately skip the
    multi-ns NPT equilibration that ``run_swish`` does in real production,
    since this script measures per-step throughput, not box convergence.
    """
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, timestep_qty)
    simulation = app.Simulation(
        topology, system, integrator, _platform(), PLATFORM_PROPERTIES,
    )
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    simulation.step(warmup_steps)
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    return state.getPositions(), state.getPeriodicBoxVectors()


def _time_unbiased(
        system, topology, positions, box, n_steps, timestep_qty,
        warmup_steps: int = 250,
) -> float:
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, timestep_qty)
    simulation = app.Simulation(
        topology, system, integrator, _platform(), PLATFORM_PROPERTIES,
    )
    simulation.context.setPositions(positions)
    simulation.context.setPeriodicBoxVectors(*box)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    simulation.step(warmup_steps)
    t0 = time.perf_counter()
    simulation.step(n_steps)
    return time.perf_counter() - t0


def _time_opes_expanded(
        system, topology, positions, box, n_steps, timestep_qty,
        opes_t_max: float = 350.0,
        opes_n_temps: int = 30,
        opes_pace: int = 2500,
        opes_observation_steps: int = 100,
        warmup_steps: int = 250,
) -> float:
    """Add OPES_EXPANDED on potential energy and time NVT propagation."""
    add_opes_expanded_bias(system, n_temps=opes_n_temps)
    integrator = build_opes_expanded_integrator(
        timestep=timestep_qty, collision_rate=FRICTION, temperature=TEMPERATURE,
        n_temps=opes_n_temps,
    )
    simulation = app.Simulation(
        topology, system, integrator, _platform(), PLATFORM_PROPERTIES,
    )
    simulation.context.setPositions(positions)
    simulation.context.setPeriodicBoxVectors(*box)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    opes = OpesExpanded(
        temperature=TEMPERATURE,
        t_min=TEMPERATURE,
        t_max=opes_t_max * unit.kelvin,
        n_temps=opes_n_temps,
        pace=opes_pace,
        observation_steps=opes_observation_steps,
    )
    opes.apply_to_context(simulation.context)
    opes.step(simulation, warmup_steps)
    t0 = time.perf_counter()
    opes.step(simulation, n_steps)
    return time.perf_counter() - t0


def _time_swish(
        system, omm_model, positions, box,
        n_replicas, n_steps_per_replica, timestep_qty, tmpdir,
        forcefield_name, forcefield,
        scale_max: float = 1.35,
        warmup_iterations: int = 1,
) -> tuple[float, int]:
    """Plain SWISH HREMD via openmmtools' ReplicaExchangeSampler.
    """
    swish_system, scale_factors = create_swish_system(
        system, omm_model, n_replicas=n_replicas, scale_max=scale_max,
        forcefield_name=forcefield_name, forcefield=forcefield,
    )
    steps_per_iter = get_n_steps(SWISH_ITER_LENGTH, timestep_qty)
    n_iters = max(1, n_steps_per_replica // steps_per_iter)

    move = mcmc.LangevinDynamicsMove(
        timestep=timestep_qty, collision_rate=FRICTION, n_steps=steps_per_iter,
    )
    base_thermo = states.ThermodynamicState(system=swish_system, temperature=TEMPERATURE)
    thermo_states = [
        states.CompoundThermodynamicState(base_thermo, [SwishLambdaState(lam=lam)])
        for lam in scale_factors
    ]
    n_sys = swish_system.getNumParticles()
    n_pos = len(positions)
    if n_sys != n_pos:
        raise RuntimeError(
            f'[rank {MPI_RANK}] SWISH particle mismatch: '
            f'swish_system={n_sys}, positions={n_pos}, '
            f'base system={system.getNumParticles()}, '
            f'topology atoms={omm_model.topology.getNumAtoms()}'
        )
    sampler_states = [
        states.SamplerState(positions=positions, box_vectors=box)
        for _ in scale_factors
    ]
    reporter = multistate.MultiStateReporter(
        f'{tmpdir}/swish_bench.nc', checkpoint_interval=10**9,
    )
    simulation = multistate.ReplicaExchangeSampler(
        mcmc_moves=move, number_of_iterations=n_iters + warmup_iterations,
    )
    simulation.create(
        thermodynamic_states=thermo_states,
        sampler_states=sampler_states,
        storage=reporter,
    )
    if warmup_iterations > 0:
        simulation.run(n_iterations=warmup_iterations)
    if _COMM is not None:
        _COMM.Barrier()
    t0 = time.perf_counter()
    simulation.run()
    if _COMM is not None:
        _COMM.Barrier()
    wall = time.perf_counter() - t0
    return wall, n_iters * steps_per_iter


def _time_swish_x(
        system, omm_model, positions, box,
        n_replicas, n_steps_per_replica, timestep_qty, tmpdir,
        forcefield_name, forcefield,
        scale_max: float = 1.35,
        opes_t_max: float = 350.0,
        opes_n_temps: int = 30,
        opes_pace: int = 2500,
        opes_observation_steps: int = 100,
        warmup_iterations: int = 1,
) -> tuple[float, int]:
    """SWISH × OPES_EXPANDED via OpesReplicaExchangeSampler — one OpesExpanded
    composable state per replica, force scaling on the device.

    Runs ``warmup_iterations`` untimed first to absorb the per-step LSE
    integrator's CUDA JIT (~3 min for n_temps=30) plus openmmtools
    ContextCache build; only steady-state iterations contribute to wall.
    """
    swish_system, scale_factors = create_swish_system(
        system, omm_model, n_replicas=n_replicas, scale_max=scale_max,
        forcefield_name=forcefield_name, forcefield=forcefield,
    )
    add_opes_expanded_bias(swish_system, n_temps=opes_n_temps)

    steps_per_iter = get_n_steps(SWISH_ITER_LENGTH, timestep_qty)
    n_iters = max(1, n_steps_per_replica // steps_per_iter)

    base_thermo = states.ThermodynamicState(system=swish_system, temperature=TEMPERATURE)
    thermo_states = []
    for lam in scale_factors:
        opes = OpesExpanded(
            temperature=TEMPERATURE,
            t_min=TEMPERATURE,
            t_max=opes_t_max * unit.kelvin,
            n_temps=opes_n_temps,
            pace=opes_pace,
            observation_steps=opes_observation_steps,
        )
        thermo_states.append(states.CompoundThermodynamicState(
            base_thermo, [SwishLambdaState(lam=lam), opes],
        ))

    moves = [
        OpesExpandedDynamicsMove(
            timestep=timestep_qty,
            collision_rate=FRICTION,
            n_steps_per_iteration=steps_per_iter,
            pace=opes_pace,
            n_temps=opes_n_temps,
        )
        for _ in range(n_replicas)
    ]
    sampler_states = [
        states.SamplerState(positions=positions, box_vectors=box)
        for _ in scale_factors
    ]
    reporter = multistate.MultiStateReporter(
        f'{tmpdir}/swish_x_bench.nc', checkpoint_interval=10**9,
    )
    simulation = OpesReplicaExchangeSampler(
        mcmc_moves=moves, number_of_iterations=n_iters + warmup_iterations,
    )
    simulation.create(
        thermodynamic_states=thermo_states,
        sampler_states=sampler_states,
        storage=reporter,
    )
    if warmup_iterations > 0:
        simulation.run(n_iterations=warmup_iterations)
    if _COMM is not None:
        _COMM.Barrier()
    t0 = time.perf_counter()
    simulation.run()
    if _COMM is not None:
        _COMM.Barrier()
    wall = time.perf_counter() - t0
    return wall, n_iters * steps_per_iter


def _make_result(label, n_replicas, n_atoms, n_steps, timestep_fs, wall_s):
    sim_ns_per_replica = n_steps * timestep_fs * 1e-6  # fs → ns
    if wall_s <= 0:
        steps_per_s = ns_per_day = aggregate = float('nan')
    else:
        steps_per_s = n_steps / wall_s
        ns_per_day = sim_ns_per_replica / wall_s * 86400.0
        aggregate = ns_per_day * n_replicas
    return {
        'config': label,
        'n_replicas': n_replicas,
        'n_atoms': n_atoms,
        'n_steps_per_replica': n_steps,
        'timestep_fs': timestep_fs,
        'wall_s': round(wall_s, 3),
        'steps_per_s': round(steps_per_s, 1),
        'ns_per_day_per_replica': round(ns_per_day, 3),
        'aggregate_ns_per_day': round(aggregate, 3),
    }


def _print_table(results: list[dict]) -> None:
    cols = [
        ('config', '<'), ('n_replicas', '>'), ('n_atoms', '>'),
        ('n_steps_per_replica', '>'), ('wall_s', '>'),
        ('steps_per_s', '>'), ('ns_per_day_per_replica', '>'),
        ('aggregate_ns_per_day', '>'),
    ]
    headers = [col[0] for col in cols]
    rows = [[str(r[col[0]]) for col in cols] for r in results]
    widths = [
        max(len(h), *(len(row[i]) for row in rows))
        for i, h in enumerate(headers)
    ]
    fmt_row = '  '.join(
        f'{{:{cols[i][1]}{widths[i]}}}' for i in range(len(cols))
    )
    print(fmt_row.format(*headers))
    print(fmt_row.format(*['-' * w for w in widths]))
    for row in rows:
        print(fmt_row.format(*row))


def _write_csv(results: list[dict], path: str) -> None:
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def main(
        protein: str,
        probe_type: str = 'benzene',
        conc_mol_per_litre: float = 1.0,
        n_replicas: int = 6,
        simtime_ps: float = 100.0,
        time_step_fs: int = 4,
        forcefield: str = 'desamber',
        output: str = 'benchmark_perf.csv',
        verbose: bool = False,
        box_info_json_path: str | None = None,
        probe_ff_path: str | None = None,
) -> None:
    """Benchmark the four sandcat MD configurations on a PDB.

    Runs unbiased Langevin, OPES_EXPANDED, SWISH HREMD and SWISH-X HREMD on
    the same SWISH-style mixed-solvent system, then writes per-config
    ns/day, steps/s and aggregate ns/day to ``output`` and stdout.

    Args:
        protein: input protein PDB filename (path on disk).
        probe_type: built-in probe name ('benzene', 'pyrimidine') or path to
            a custom probe SDF file.
        conc_mol_per_litre: target probe molarity, mol/L.
        n_replicas: HREMD replicas for the SWISH / SWISH-X configs (must be even).
        simtime_ps: per-replica production length, ps.
        time_step_fs: MD timestep, fs.
        forcefield: force field / water model combination ('desamber', 'ff14', or 'ff19').
        output: output CSV path for the timing table.
        verbose: enable INFO-level logging on rank 0.
        box_info_json_path: optional JSON file with box parameters; same ``-bi``
            format used by the SandCat MD pipelines. Defaults to 1.0 nm cubic padding.
        probe_ff_path: path to a bespoke offxml force field for the probe.
    """
    if verbose and MPI_RANK == 0:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    timestep_qty = time_step_fs * unit.femtosecond
    n_steps = int(simtime_ps * 1000.0 / time_step_fs)

    if MPI_RANK == 0:
        logging.info(
            f'Configs: unbiased, opes_expanded, swish (n={n_replicas}), '
            f'swish_x (n={n_replicas}); MPI_SIZE={MPI_SIZE}, '
            f'simtime={simtime_ps} ps, dt={time_step_fs} fs, '
            f'n_steps_per_replica={n_steps}'
        )

    # Each rank builds the system independently — add_mixed_solvent is seeded
    # so the resulting topologies and coordinates are bit-identical.
    system, omm_model, openmm_forcefield = _build_system(
        protein, probe_type, conc_mol_per_litre, forcefield,
        box_info_json_path=box_info_json_path,
        probe_ff_path=probe_ff_path,
    )
    n_atoms = system.getNumParticles()

    # Minimise + 1 ps warmup on rank 0; broadcast positions/box to others.
    if MPI_RANK == 0:
        positions, box = _quick_warmup(
            create_static_swish_systems(
                system, omm_model, n_replicas=n_replicas,
                forcefield_name=forcefield, forcefield=openmm_forcefield,
            )[0],
            omm_model.topology, omm_model.positions, timestep_qty,
        )
        pos_array = np.array(positions.value_in_unit(unit.nanometer))
        box_array = np.array([
            v.value_in_unit(unit.nanometer) for v in box
        ])
    else:
        pos_array = box_array = None

    if _COMM is not None:
        pos_array = _COMM.bcast(pos_array, root=0)
        box_array = _COMM.bcast(box_array, root=0)
    positions = pos_array * unit.nanometer
    box = tuple(mm.Vec3(*row) * unit.nanometer for row in box_array)

    # Shared tmp dir for the openmmtools NC reporter — created on rank 0,
    # broadcast to others so they all point at the same path.
    if MPI_RANK == 0:
        tmpdir = mkdtemp(prefix='sandcat_bench_')
    else:
        tmpdir = None
    if _COMM is not None:
        tmpdir = _COMM.bcast(tmpdir, root=0)

    results: list[dict] = []

    if MPI_RANK == 0:
        replica = create_static_swish_systems(
            system, omm_model, n_replicas=n_replicas,
            forcefield_name=forcefield, forcefield=openmm_forcefield,
        )[0]
        wall = _time_unbiased(
            replica, omm_model.topology, positions, box, n_steps, timestep_qty,
        )
        results.append(_make_result(
            'unbiased', 1, n_atoms, n_steps, time_step_fs, wall,
        ))
        logging.info(f'unbiased: {wall:.2f} s')
    if _COMM is not None:
        _COMM.Barrier()

    if MPI_RANK == 0:
        replica = create_static_swish_systems(
            system, omm_model, n_replicas=n_replicas,
            forcefield_name=forcefield, forcefield=openmm_forcefield,
        )[0]
        wall = _time_opes_expanded(
            replica, omm_model.topology, positions, box, n_steps, timestep_qty,
        )
        results.append(_make_result(
            'opes_expanded', 1, n_atoms, n_steps, time_step_fs, wall,
        ))
        logging.info(f'opes_expanded: {wall:.2f} s')
    if _COMM is not None:
        _COMM.Barrier()

    wall, ran_steps = _time_swish(
        system, omm_model, positions, box,
        n_replicas, n_steps, timestep_qty, tmpdir,
        forcefield, openmm_forcefield,
    )
    if MPI_RANK == 0:
        results.append(_make_result(
            'swish', n_replicas, n_atoms, ran_steps, time_step_fs, wall,
        ))
        logging.info(f'swish: {wall:.2f} s ({ran_steps} steps/replica)')
    if _COMM is not None:
        _COMM.Barrier()

    wall, ran_steps = _time_swish_x(
        system, omm_model, positions, box,
        n_replicas, n_steps, timestep_qty, tmpdir,
        forcefield, openmm_forcefield,
    )
    if MPI_RANK == 0:
        results.append(_make_result(
            'swish_x', n_replicas, n_atoms, ran_steps, time_step_fs, wall,
        ))
        logging.info(f'swish_x: {wall:.2f} s ({ran_steps} steps/replica)')
    if _COMM is not None:
        _COMM.Barrier()

    if MPI_RANK == 0:
        print()
        _print_table(results)
        _write_csv(results, output)
        print(f'\nWrote {output}')


def parse_args() -> ArgumentParser:
    """Parse command-line arguments for the benchmark script."""
    arg_parser = ArgumentParser(
        description=(
            'Benchmark sandcat MD configurations on a PDB.\n'
            '\n'
            'Times the four production paths over a fixed wall-clock window so they can\n'
            'be compared directly:\n'
            '  1. Unbiased Langevin (1 replica, 1 GPU)\n'
            '  2. OPES_EXPANDED (single-replica multithermal, 1 GPU)\n'
            '  3. SWISH HREMD       (N replicas, N MPI ranks, 1 GPU via CUDA MPS)\n'
            '  4. SWISH-X HREMD     (N replicas, N MPI ranks, 1 GPU via CUDA MPS)\n'
            '\n'
            'Each configuration runs the same molecular system: the SWISH-style\n'
            'mixed-solvent setup (probes, inter-probe repulsion, contact-map upper wall)\n'
            'that ``run_swish.py`` produces, deterministically built from the PDB on\n'
            'every rank. Production is NVT — the script does a brief minimisation +\n'
            '1 ps unbiased warmup, no NPT equilibration, since the goal is per-step\n'
            'throughput rather than equilibration quality.\n'
            '\n'
            'For N number of replicas, run:\n'
            '    mpirun -np N python benchmark_perf.py -p protein.pdb --n_replicas N\n'
            '\n'
            'Outputs ns/day and steps/s to stdout and ``benchmark_perf.csv``.\n'
            'Single-replica configs (1, 2) execute on rank 0 while ranks 1..N wait at\n'
            'an MPI barrier; multi-rank configs (3, 4) use all ranks via openmmtools\'\n'
            'internal mpiplus distribution. CUDA MPS is required for ranks to share\n'
            'one GPU efficiently — without it, replica propagation is serialised.'
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument(
        '-p', '--protein', type=str, required=True,
        help='input protein PDB filename',
    )
    arg_parser.add_argument(
        '--probe', type=str,
        help='built-in probe name (%(choices)s) or path to a custom probe SDF. '
             'Default = %(default)s',
        default='benzene',
    )
    arg_parser.add_argument(
        '--probe_ff_path', type=str, default=None,
        help='optional SMIRNOFF OFFXML for a custom probe. '
             'If omitted, OpenFF 2.2.0 is used.',
    )
    arg_parser.add_argument(
        '--concentration', type=float,
        help=PROBE_CONCENTRATION_HELP,
        default=1.0,
    )
    arg_parser.add_argument(
        '--n_replicas', type=int,
        help='number of HREMD replicas for SWISH / SWISH-X (must be even). '
             'Default = %(default)s',
        default=6,
    )
    arg_parser.add_argument(
        '--simtime_ps', type=float,
        help='per-replica production length, in picoseconds. Default = %(default)s',
        default=100.0,
    )
    arg_parser.add_argument(
        '-ts', '--time_step_fs', type=int,
        help='MD integration timestep, in femtoseconds. Default = %(default)s fs',
        default=4,
    )
    arg_parser.add_argument(
        '--forcefield',
        help='force field / water model combination. Default = %(default)s',
        choices=ALL_FORCEFIELDS,
        default='desamber',
    )
    arg_parser.add_argument(
        '-o', '--output', type=str,
        help='output CSV path. Default = %(default)s',
        default='benchmark_perf.csv',
    )
    arg_parser.add_argument(
        '--verbose',
        action='store_true',
        help='enable verbose logging output.',
        default=False,
    )
    arg_parser.add_argument(
        '-bi', '--box_info', type=str, default=None,
        help='optional JSON file with box parameters (padding, box_shape, box_size, '
             'box_edge, or box_vectors). Same format as the SandCat MD -bi option. '
             'If omitted, a 1.0 nm cubic padding is used.',
    )
    add_config_arg(arg_parser)
    return arg_parser


if __name__ == '__main__':
    parser = parse_args()
    args = parse_args_with_config(parser)
    main(
        protein=args.protein,
        probe_type=args.probe,
        conc_mol_per_litre=args.concentration,
        n_replicas=args.n_replicas,
        simtime_ps=args.simtime_ps,
        time_step_fs=args.time_step_fs,
        forcefield=args.forcefield,
        output=args.output,
        verbose=args.verbose,
        box_info_json_path=args.box_info,
        probe_ff_path=args.probe_ff_path,
    )
