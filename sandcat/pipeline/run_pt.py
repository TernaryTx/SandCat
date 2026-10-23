from __future__ import annotations

from argparse import ArgumentParser
import logging

from copy import deepcopy

from openmm import app, unit
from openmmtools import states, mcmc, multistate

from sandcat.md_variables import get_n_steps, REPORT_INTERVAL
from sandcat.structure import PdbFile
from sandcat.system_setup import create_system, minimise_equilibrate
from sandcat.protonation import ProteinProtonater
from sandcat.config import add_config_arg, parse_args_with_config
from sandcat.utils import validate_n_replicas
from sandcat.variables import PT_ITER_LENGTH, PT_OUTPUT_PREFIX


# Parallel tempering using openmmtools.

# Default MD time between replica-exchange attempts. Overridable per run via
# the ``exchange_interval_ps`` argument.
def main(
        input_protein: PdbFile,
        total_simtime_ns: float = 100.0,
        time_step_fs: int = 4,
        t_min: float = 300.0,
        t_max: float = 600.0,
        n_replicas: int = 4,
        add_hydrogens: bool = True,
        exchange_interval_ps: float | None = None,
        verbose: bool = False,
) -> None:
    """Run a parallel tempering (temperature replica exchange) simulation on a protein structure.

    The protein is solvated in TIP3P water, energy-minimised, and equilibrated before the
    replica exchange run. Replicas are distributed across an evenly spaced temperature ladder
    between t_min and t_max. Each replica exchange iteration covers a fixed 2 ps of MD time,
    with the number of iterations derived from total_simtime_ns.

    Args:
        input_protein (PdbFile): Input protein structure file.
        total_simtime_ns (float): Total simulation length per replica, in nanoseconds.
        time_step_fs (int): MD integration timestep, in femtoseconds.
        t_min (float): Temperature of the lowest replica, in Kelvin.
        t_max (float): Temperature of the highest replica, in Kelvin.
        n_replicas (int): Number of temperature replicas (must be between 2 and 8 inclusive).
        add_hydrogens (bool): Whether to protonate the input structure before simulation.
            Set to False if the structure already has hydrogens or contains phosphoresidues.
        exchange_interval_ps (float | None): MD time (ps) between replica-exchange attempts.
            If None, the default of 2 ps is used.
        verbose (bool): Whether to enable INFO-level logging to the terminal.
    """
    validate_n_replicas(n_replicas)
    iter_length = (
        PT_ITER_LENGTH
        if exchange_interval_ps is None
        else exchange_interval_ps * unit.picosecond
    )

    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    if add_hydrogens:
        # Will raise error if running with phosphoresidues;
        # prepare separately and run with add_hydrogens = False
        input_protein = ProteinProtonater().process_one(input_protein)[0]

    protein = app.PDBFile(input_protein.path)
    omm_model = app.Modeller(protein.topology, protein.positions)

    system = create_system(omm_model, add_solvent=True)
    equil_positions, _ = minimise_equilibrate(system, omm_model)

    t_min_qty = t_min * unit.kelvin
    t_max_qty = t_max * unit.kelvin
    temperatures = [
        t_min_qty + (t_max_qty - t_min_qty) * (idx / (n_replicas - 1)) for idx in range(n_replicas)
    ]

    thermo_states = [
        states.ThermodynamicState(system=system, temperature=T) for T in temperatures
    ]

    sampler_state = states.SamplerState(
        positions=equil_positions,
        box_vectors=omm_model.topology.getPeriodicBoxVectors(),
    )
    sampler_states = [deepcopy(sampler_state) for _ in range(n_replicas)]

    # steps_per_iteration is fixed so each iteration covers iter_length
    timestep_qty = time_step_fs * unit.femtosecond
    steps_per_iteration = get_n_steps(iter_length, timestep_qty)

    move = mcmc.LangevinDynamicsMove(
        timestep=timestep_qty,
        collision_rate=1.0 / unit.picosecond,
        n_steps=steps_per_iteration,
    )

    # n_iterations and checkpoint_interval are derived from simulation time, not set directly
    total_steps = get_n_steps(total_simtime_ns * unit.nanosecond, timestep_qty)
    n_iterations = total_steps // steps_per_iteration
    checkpoint_interval = get_n_steps(REPORT_INTERVAL, iter_length)

    reporter = multistate.MultiStateReporter(
        f'{PT_OUTPUT_PREFIX}.nc', checkpoint_interval=checkpoint_interval
    )

    simulation = multistate.ReplicaExchangeSampler(
        mcmc_moves=move, number_of_iterations=n_iterations
    )

    # The 'create' call handles the distribution of states across MPI ranks automatically
    simulation.create(
        thermodynamic_states=thermo_states,
        sampler_states=sampler_states,
        storage=reporter,
    )

    simulation.run()


def parse_args() -> ArgumentParser:
    """Parse command-line arguments for the parallel tempering script."""
    arg_parser = ArgumentParser(
        description='Run parallel tempering (replica exchange MD) on a PDB file using openmmtools.',
    )
    arg_parser.add_argument(
        '-p', '--protein', type=str,
        help='input protein PDB filename',
    )
    arg_parser.add_argument(
        '-t', '--total_simtime_ns', type=float,
        help='total simulation length per replica, in nanoseconds. Default = %(default)s ns',
        default=100.0,
    )
    arg_parser.add_argument(
        '-ts', '--time_step_fs', type=int,
        help='MD timestep, in femtoseconds. Default = %(default)s fs',
        default=4,
    )
    arg_parser.add_argument(
        '--t_min', type=float,
        help='minimum replica temperature in Kelvin. Default = %(default)s K',
        default=300.0,
    )
    arg_parser.add_argument(
        '--t_max', type=float,
        help='maximum replica temperature in Kelvin. Default = %(default)s K',
        default=600.0,
    )
    arg_parser.add_argument(
        '--n_replicas', type=int,
        help='number of temperature replicas. Default = %(default)s',
        default=4,
    )
    arg_parser.add_argument(
        '-ah', '--add_hydrogens',
        help="whether to add hydrogen atoms to the protein structure. "
             "Default = True. To turn off, use '-ah False'",
        choices=['True', 'False'],
        default='True',
    )
    arg_parser.add_argument(
        '--exchange_interval_ps', type=float, default=None,
        help='MD time (ps) between replica-exchange attempts. '
             'If omitted, the default of 2 ps is used.',
    )
    arg_parser.add_argument(
        '--verbose',
        action='store_true',
        help='enable verbose logging output.',
        default=False,
    )
    add_config_arg(arg_parser)
    return arg_parser


if __name__ == '__main__':
    parser = parse_args()
    args = parse_args_with_config(parser)
    main(
        input_protein=PdbFile(path=args.protein),
        total_simtime_ns=args.total_simtime_ns,
        time_step_fs=args.time_step_fs,
        t_min=args.t_min,
        t_max=args.t_max,
        n_replicas=args.n_replicas,
        add_hydrogens=args.add_hydrogens != 'False',
        exchange_interval_ps=args.exchange_interval_ps,
        verbose=args.verbose,
    )
