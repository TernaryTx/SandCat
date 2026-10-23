from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from openmm import app, unit

from sandcat.protonation import ProteinProtonater
from sandcat.structure import PdbFile
from sandcat.md_utils import (
    add_contact_map_upper_wall, add_probe_repulsion,
    can_restart_swish_production, create_swish_system, create_static_swish_systems,
    minimise_equilibrate_npt_sandcat, restart_swish_production, run_swish_production,
    validate_output_provenance, write_output_provenance,
)
from sandcat.config import add_config_arg, parse_args_with_config
from sandcat.provenance import build_output_provenance
from sandcat.utils import (
    build_mixed_solvent_system, resolve_probe_sdf_path,
    uses_historical_probe_14_correction,
)
from sandcat.variables import (
    ALL_FORCEFIELDS,
    DEFAULT_TEMPERATURE_K,
    DESAMBER_FORCEFIELDS,
    PROBE_CONCENTRATION_HELP,
    SWISH_OUTPUT_PREFIX,
)


def main(
        input_protein: PdbFile,
        probe_type: str = 'benzene',
        probe_ff_path: str | None = None,
        conc_mol_per_litre: float = 1.0,
        add_hydrogens: bool = False,
        pocket_centre_nm: tuple[float, float, float] | None = None,
        cmap_at: float = 30.0,
        n_replicas: int = 6,
        scale_max: float = 1.35,
        total_simtime_ns: float = 250.0,
        time_step_fs: int = 4,
        exchange_interval_ps: float | None = None,
        verbose: bool = False,
        output_prefix: str = SWISH_OUTPUT_PREFIX,
        forcefield: str = 'desamber',
        box_info_json_path: str | None = None,
        ensemble: str = 'NPT',
) -> None:
    """Set up and run a SWISH mixed solvent simulation from a protein PDB file.

    Implements the full SWISH protocol: mixed solvent solvation, inter-probe repulsion,
    contact map upper wall, N Hamiltonians with linearly spaced OW-apolar epsilon scaling
    from 1.0 to scale_max, per-replica NPT equilibration (restrained then unrestrained,
    50/50, so the protein relaxes before production), and Hamiltonian replica exchange
    production.

    If ``{output_prefix}.nc`` and its checkpoint already exist (e.g. pulled from S3),
    the run is resumed from where it left off and system setup / equilibration are
    skipped entirely — only the remaining iterations are performed.

    Args:
        input_protein: Input protein PDB file.
        probe_type: Built-in probe name or path to an SDF containing one molecule.
        probe_ff_path: Optional SMIRNOFF OFFXML used to parameterize the probe.
        conc_mol_per_litre: Target molar concentration of probe.
        add_hydrogens: Whether to protonate the input structure before setup.
        pocket_centre_nm: (x, y, z) in nm of the binding pocket centre used to exclude
            nearby residues from the contact map. If None, no pocket filtering is applied.
        cmap_at: Upper wall threshold for the contact map CV (default 30, matching the
            1JWP example in the SWISH bootcamp).
        n_replicas: Number of SWISH replicas. Must be even; replica 0 is unscaled (lambda=1.0).
        scale_max: Maximum lambda scaling factor (default 1.35, as used in the bootcamp).
        total_simtime_ns: Total production simulation time per replica, in nanoseconds.
        time_step_fs: MD integration timestep, in femtoseconds.
        exchange_interval_ps: MD time (ps) between replica-exchange attempts. If None,
            the SWISH default of 10 ps is used.
        verbose: Whether to enable INFO-level logging to the terminal.
        output_prefix: Prefix for the NC output files (``{prefix}.nc`` and
            ``{prefix}_checkpoint.nc``). Used for both fresh and restart runs.
        forcefield: Force field / water model combination. 'desamber' = DES-Amber-SF1.0
            + TIP4P-D; 'ff14' = AMBER ff14SB + TIP3P; 'ff19' = AMBER ff19SB + OPC.
        box_info_json_path: optional path to a JSON file specifying explicit
            box parameters.
        ensemble: production ensemble — 'NPT' adds a MonteCarloBarostat via ThermodynamicState;
            'NVT' runs without pressure coupling.
    """
    # Restart short-circuit: if both NC files are present, skip system build and
    # equilibration entirely, and resume dynamics from the stored state.
    if can_restart_swish_production(output_prefix):
        validate_output_provenance(output_prefix)
        restart_swish_production(
            total_simtime_ns=total_simtime_ns,
            time_step_fs=time_step_fs,
            exchange_interval_ps=exchange_interval_ps,
            output_prefix=output_prefix,
            verbose=verbose,
        )
        return

    original_protein_path = input_protein.path
    if add_hydrogens:
        input_protein = ProteinProtonater().process_one(input_protein)[0]

    system, omm_model, system_generator = build_mixed_solvent_system(
        forcefield, input_protein.path, probe_type, conc_mol_per_litre,
        box_info_json_path=box_info_json_path,
        probe_ff_path=probe_ff_path,
    )

    pdb_stem = Path(input_protein.path).stem
    mixed_solvent_fname = f'{pdb_stem}_mixed_solvent.pdb'
    with open(mixed_solvent_fname, 'w') as pdb_out:
        app.PDBFile.writeFile(
            omm_model.topology, omm_model.positions, pdb_out, keepIds=True,
        )

    probe_provenance = add_probe_repulsion(
        system,
        omm_model,
        correct_probe_14_exceptions=uses_historical_probe_14_correction(
            probe_type, probe_ff_path,
        ),
    )
    add_contact_map_upper_wall(
        system, omm_model, pocket_centre_nm=pocket_centre_nm, at=cmap_at,
    )

    # Build N per-lambda systems for equilibration only — each replica 
    # equilibrates under its own scaled Hamiltonian, matching the SWISH
    # bootcamp protocol.
    equil_systems = create_static_swish_systems(
        system, omm_model, n_replicas=n_replicas, scale_max=scale_max,
        forcefield_name=forcefield,
        forcefield=system_generator.forcefield,
    )

    # Equilibrate each replica independently under its own Hamiltonian.
    # Work is distributed across MPI ranks so each rank only equilibrates its share.
    if MPI is not None:
        comm = MPI.COMM_WORLD
        mpi_rank = comm.Get_rank()
        n_ranks = comm.Get_size()
    else:
        comm = None
        mpi_rank = 0
        n_ranks = 1

    local_results = {}
    for replica_idx in range(mpi_rank, n_replicas, n_ranks):
        positions, box = minimise_equilibrate_npt_sandcat(
            equil_systems[replica_idx], omm_model.topology, omm_model.positions,
        )
        local_results[replica_idx] = (positions, box)

    if comm is not None:
        all_results = comm.allgather(local_results)
        gathered = {
            replica_idx: result for rank_results in all_results 
            for replica_idx, result in rank_results.items()
        }
    else:
        gathered = local_results

    equil_positions = [gathered[replica_idx][0] for replica_idx in range(n_replicas)]
    equil_boxes = [gathered[replica_idx][1] for replica_idx in range(n_replicas)]

    # Build a single shared system with lam as a free global parameter for prod
    swish_system, scale_factors = create_swish_system(
        system, omm_model, n_replicas=n_replicas, scale_max=scale_max,
        forcefield_name=forcefield,
        forcefield=system_generator.forcefield,
    )

    if mpi_rank == 0:
        parameters = {
            'input_protein': original_protein_path,
            'probe_type': probe_type,
            'probe_ff_path': probe_ff_path,
            'concentration_mol_per_litre': conc_mol_per_litre,
            'add_hydrogens': add_hydrogens,
            'pocket_centre_nm': pocket_centre_nm,
            'contact_map_upper_wall': cmap_at,
            'n_replicas': n_replicas,
            'scale_max': scale_max,
            'scale_factors': scale_factors,
            'total_simtime_ns': total_simtime_ns,
            'time_step_fs': time_step_fs,
            'exchange_interval_ps': 10.0 if exchange_interval_ps is None else exchange_interval_ps,
            'equilibration_time_ns': 10.0,
            'temperature_kelvin': DEFAULT_TEMPERATURE_K,
            'pressure_bar': 1.0 if ensemble == 'NPT' else None,
            'collision_frequency_per_ps': 1.0,
            'output_prefix': output_prefix,
            'forcefield': forcefield,
            'box_info_json_path': box_info_json_path,
            'ensemble': ensemble,
            'use_barostat': ensemble == 'NPT',
            'verbose': verbose,
            'random_seed': None,
            'random_seed_source': 'OpenMM OS-random seed',
        }
        input_paths = {
            'input_protein': original_protein_path,
            'prepared_protein': (
                input_protein.path if input_protein.path != original_protein_path else None
            ),
            'box_info': box_info_json_path,
            'probe_definition': resolve_probe_sdf_path(probe_type),
            'probe_forcefield': probe_ff_path,
            'forcefield_definition': (
                DESAMBER_FORCEFIELDS if forcefield == 'desamber' else None
            ),
        }
        extra = dict(probe_provenance)
        extra.update({
            'workflow': 'swish',
            'probe_type': probe_type,
            'n_replicas': n_replicas,
            'scale_factors': scale_factors,
        })
        provenance = build_output_provenance(
            workflow='swish',
            parameters=parameters,
            input_paths=input_paths,
            extra=extra,
        )
        write_output_provenance(output_prefix, provenance)
    if comm is not None:
        comm.Barrier()

    run_swish_production(
        swish_system=swish_system,
        scale_factors=scale_factors,
        equil_positions=equil_positions,
        equil_boxes=equil_boxes,
        total_simtime_ns=total_simtime_ns,
        time_step_fs=time_step_fs,
        exchange_interval_ps=exchange_interval_ps,
        output_prefix=output_prefix,
        verbose=verbose,
        use_barostat=(ensemble == 'NPT'),
    )


def parse_args() -> ArgumentParser:
    """Parse command-line arguments for the SWISH setup script."""
    arg_parser = ArgumentParser(
        description='Set up a SWISH mixed solvent simulation from a PDB file.',
    )
    arg_parser.add_argument(
        '-p', '--protein', type=str,
        help='input protein PDB filename',
        required=True,
    )
    arg_parser.add_argument(
        '--probe', type=str,
        help='built-in probe name or path to a single-molecule SDF. Default = %(default)s',
        default='benzene',
    )
    arg_parser.add_argument(
        '--probe_ff_path', type=str, default=None,
        help='optional SMIRNOFF OFFXML used to parameterize the probe.',
    )
    arg_parser.add_argument(
        '--concentration', type=float,
        help=PROBE_CONCENTRATION_HELP,
        default=1.0,
    )
    arg_parser.add_argument(
        '-ah', '--add_hydrogens',
        help="whether to add hydrogen atoms to the protein structure. "
             "Default = False",
        choices=['True', 'False'],
        default='False',
    )
    arg_parser.add_argument(
        '--pocket_centre', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
        help='x y z coordinates (nm) of the binding pocket centre, used to '
             'exclude nearby residues from the contact map. If omitted, no '
             'pocket filtering is applied.',
        default=None,
    )
    arg_parser.add_argument(
        '--cmap_at', type=float,
        help='upper wall threshold for the contact map CV. Default = %(default)s',
        default=30.0,
    )
    arg_parser.add_argument(
        '--n_replicas', type=int,
        help='number of SWISH replicas (must be even). Default = %(default)s',
        default=6,
    )
    arg_parser.add_argument(
        '--scale_max', type=float,
        help='maximum OW-apolar epsilon scaling factor. Default = %(default)s',
        default=1.35,
    )
    arg_parser.add_argument(
        '-t', '--total_simtime_ns', type=float,
        help='total production simulation time per replica, in nanoseconds. ' \
             'Default = %(default)s ns',
        default=250.0,
    )
    arg_parser.add_argument(
        '-ts', '--time_step_fs', type=int,
        help='MD integration timestep, in femtoseconds. Default = %(default)s fs',
        default=4,
    )
    arg_parser.add_argument(
        '--exchange_interval_ps', type=float, default=None,
        help='MD time (ps) between replica-exchange attempts. '
             'If omitted, the SWISH default of 10 ps is used.',
    )
    arg_parser.add_argument(
        '--forcefield',
        help='force field / water model combination. Default = %(default)s',
        choices=ALL_FORCEFIELDS,
        default='desamber',
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
    arg_parser.add_argument(
        '-e', '--ensemble', type=str, default='NPT', choices=['NPT', 'NVT'],
        help='production ensemble: NPT (MonteCarloBarostat) or NVT (fixed box). Default = %(default)s',
    )
    add_config_arg(arg_parser)
    return arg_parser


if __name__ == '__main__':
    parser = parse_args()
    args = parse_args_with_config(parser)
    pocket_centre_nm = tuple(args.pocket_centre) if args.pocket_centre else None
    main(
        input_protein=PdbFile(path=args.protein),
        probe_type=args.probe,
        probe_ff_path=args.probe_ff_path,
        conc_mol_per_litre=args.concentration,
        add_hydrogens=args.add_hydrogens != 'False',
        pocket_centre_nm=pocket_centre_nm,
        cmap_at=args.cmap_at,
        n_replicas=args.n_replicas,
        scale_max=args.scale_max,
        total_simtime_ns=args.total_simtime_ns,
        time_step_fs=args.time_step_fs,
        exchange_interval_ps=args.exchange_interval_ps,
        verbose=args.verbose,
        forcefield=args.forcefield,
        box_info_json_path=args.box_info,
        ensemble=args.ensemble,
    )
