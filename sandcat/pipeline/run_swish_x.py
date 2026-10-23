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
    can_restart_swish_production, create_static_swish_systems,
    create_swish_system, minimise_equilibrate_npt_sandcat,
    restart_swish_production, run_swish_x_production,
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
    SWISH_X_OUTPUT_PREFIX,
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
        opes_t_max: float = 350.0,
        opes_n_temps: int = 30,
        opes_pace: int = 2500,
        opes_observation_steps: int = 100,
        total_simtime_ns: float = 250.0,
        time_step_fs: int = 4,
        exchange_interval_ps: float | None = None,
        equil_time_ns: float = 10.0,
        verbose: bool = False,
        output_prefix: str = SWISH_X_OUTPUT_PREFIX,
        forcefield: str = 'desamber',
        box_info_json_path: str | None = None,
        ensemble: str = 'NPT',
        collision_freq: float = 1.0,
        colvar_interval_ps: float | None = 2.0,
) -> None:
    """SWISH-X = SWISH × OPES Multithermal (Borsatto et al, JCTC 20, 3335, 2024).

    Each replica runs vanilla SWISH (water-O / apolar-C,S LJ epsilon scaled
    by per-replica λ) plus an independent OPES_EXPANDED bias on the system
    potential energy targeting [TEMPERATURE, opes_t_max]. Replicas exchange
    configurations via Hamiltonian replica exchange on λ.

    Setup is identical to plain SWISH (mixed-solvent solvation, probe-probe
    repulsion, contact-map upper wall, per-replica NPT equilibration);
    production swaps openmmtools' LangevinDynamicsMove for one
    ``OpesExpandedDynamicsMove`` per replica.

    Defaults assume dt=4 fs / pace=2500 → 1 PACE = 10 ps, matching the
    Borsatto/PLUMED SWISH-X bootcamp (PACE=5000 × 2 fs = 10 ps). The
    unbiased warmup is `opes_observation_steps × 10 ps` (default 1 ns,
    matching PLUMED's OBSERVATION_STEPS=100 default). Shorter PACE updates
    are more autocorrelated per sample, so the `1/counter` ΔF weight
    under-represents each effective sample and the per-replica ΔF curves
    drift apart, eroding HRE swap acceptance at the top of the λ ladder.

    Example:
        mpirun -np 6 python sandcat/pipeline/run_swish_x.py -p 1jwp.pdb \\
            --probe benzene --n_replicas 6 --scale_max 1.35 \\
            --opes_t_max 350 --opes_n_temps 30 -t 250 --verbose

    If ``{output_prefix}.nc`` and its checkpoint already exist (e.g. pulled
    from S3), the run resumes from the stored iteration and system setup /
    equilibration are skipped — only the remaining iterations are performed.

    Args:
        input_protein: Input protein PDB file.
        probe_type: Built-in probe name or path to an SDF containing one molecule.
        probe_ff_path: Optional SMIRNOFF OFFXML used to parameterize the probe.
        conc_mol_per_litre: Target probe molarity.
        add_hydrogens: protonate the PDB.
        pocket_centre_nm: (x,y,z) in nm; nearby residues are excluded from
            the contact-map restraint.
        cmap_at: upper-wall threshold for the CMDIST CV.
        n_replicas: HRE replicas (must be even).
        scale_max: largest SWISH λ scaling factor.
        opes_t_max: OPES multithermal upper T (Borsatto 2024: TEMPERATURE+50).
        opes_n_temps: T grid size (~50% adjacent-state ESS for ~50k-atom
            solvated systems at 30 points).
        opes_pace: ΔF update stride in MD steps.
        opes_observation_steps: unbiased warmup window in PACE units.
        total_simtime_ns, time_step_fs: production length and dt.
        equil_time_ns: per-replica total NPT equilibration before production, split
            50/50 restrained then unrestrained
        forcefield: 'desamber' = DES-Amber-SF1.0+TIP4P-D, 'ff14' = ff14SB+TIP3P,
            'ff19' = ff19SB+OPC.
        box_info_json_path: optional path to a JSON file specifying explicit
            box parameters.
        ensemble: production ensemble — 'NPT' adds a MonteCarloBarostat via ThermodynamicState;
            'NVT' runs without pressure coupling.
        collision_freq: Langevin collision frequency in inverse picoseconds.
        colvar_interval_ps: MD time between exact-context COLVAR samples.
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
            colvar_interval_ps=colvar_interval_ps,
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

    # Per-replica equilibration under its own scaled Hamiltonian (SWISH
    # bootcamp protocol; identical to run_swish).
    equil_systems = create_static_swish_systems(
        system, omm_model, n_replicas=n_replicas, scale_max=scale_max,
        forcefield_name=forcefield,
        forcefield=system_generator.forcefield,
    )

    # Distribute per-replica equilibration across MPI ranks if available.
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
            equil_time=equil_time_ns * unit.nanosecond,
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

    # Single shared SWISH system with ``lam`` as a global parameter; the
    # per-replica OPES bias is added inside run_swish_x_production.
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
            'opes_t_max_kelvin': opes_t_max,
            'opes_n_temps': opes_n_temps,
            'opes_pace_steps': opes_pace,
            'opes_observation_steps': opes_observation_steps,
            'total_simtime_ns': total_simtime_ns,
            'time_step_fs': time_step_fs,
            'exchange_interval_ps': 10.0 if exchange_interval_ps is None else exchange_interval_ps,
            'equilibration_time_ns': equil_time_ns,
            'temperature_kelvin': DEFAULT_TEMPERATURE_K,
            'pressure_bar': 1.0 if ensemble == 'NPT' else None,
            'output_prefix': output_prefix,
            'forcefield': forcefield,
            'box_info_json_path': box_info_json_path,
            'ensemble': ensemble,
            'use_barostat': ensemble == 'NPT',
            'collision_frequency_per_ps': collision_freq,
            'colvar_interval_ps': colvar_interval_ps,
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
            'workflow': 'swish-x',
            'probe_type': probe_type,
            'n_replicas': n_replicas,
            'scale_factors': scale_factors,
            'opes_enabled': True,
            'swish_x_lifecycle': 'gmx_staggered_boundary_v1',
        })
        provenance = build_output_provenance(
            workflow='swish-x',
            parameters=parameters,
            input_paths=input_paths,
            extra=extra,
        )
        write_output_provenance(output_prefix, provenance)
    if comm is not None:
        comm.Barrier()

    run_swish_x_production(
        swish_system=swish_system,
        scale_factors=scale_factors,
        equil_positions=equil_positions,
        equil_boxes=equil_boxes,
        opes_t_max=opes_t_max * unit.kelvin,
        opes_n_temps=opes_n_temps,
        opes_pace=opes_pace,
        opes_observation_steps=opes_observation_steps,
        total_simtime_ns=total_simtime_ns,
        time_step_fs=time_step_fs,
        exchange_interval_ps=exchange_interval_ps,
        output_prefix=output_prefix,
        verbose=verbose,
        use_barostat=(ensemble == 'NPT'),
        collision_freq=collision_freq,
        colvar_interval_ps=colvar_interval_ps,
    )


def parse_args() -> ArgumentParser:
    arg_parser = ArgumentParser(
        description='Set up SWISH-X (SWISH × OPES Multithermal) on a PDB.',
    )
    # ---- shared with run_swish.py ----
    arg_parser.add_argument(
        '-p', '--protein', type=str, required=True,
        help='input protein PDB filename',
    )
    arg_parser.add_argument(
        '--probe', type=str,
        default='benzene',
        help='built-in probe name or path to a single-molecule SDF. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--probe_ff_path', type=str, default=None,
        help='optional SMIRNOFF OFFXML used to parameterize the probe.',
    )
    arg_parser.add_argument(
        '--concentration', type=float, default=1.0,
        help=PROBE_CONCENTRATION_HELP,
    )
    arg_parser.add_argument(
        '-ah', '--add_hydrogens', choices=['True', 'False'], default='False',
        help="whether to add hydrogen atoms to the protein structure. Default = False",
    )
    arg_parser.add_argument(
        '--pocket_centre', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
        default=None,
        help='x y z coordinates (nm) of the pocket centre, used to exclude nearby '
             'residues from the contact map. If omitted, no pocket filtering is applied.',
    )
    arg_parser.add_argument(
        '--cmap_at', type=float, default=30.0,
        help='upper wall threshold for the contact map CV. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--n_replicas', type=int, default=6,
        help='number of SWISH replicas (must be even). Default = %(default)s',
    )
    arg_parser.add_argument(
        '--scale_max', type=float, default=1.35,
        help='maximum OW-apolar epsilon scaling factor. Default = %(default)s',
    )
    arg_parser.add_argument(
        '-t', '--total_simtime_ns', type=float, default=250.0,
        help='total per-replica simulation time, in ns. Default = %(default)s',
    )
    arg_parser.add_argument(
        '-ts', '--time_step_fs', type=int, default=4,
        help='MD timestep in fs. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--exchange_interval_ps', type=float, default=None,
        help='MD time (ps) between replica-exchange attempts. '
             'If omitted, the SWISH default of 10 ps is used.',
    )
    arg_parser.add_argument(
        '--colvar_interval_ps', type=float, default=2.0,
        help='MD time (ps) between exact-context COLVAR samples. Default = %(default)s.',
    )
    arg_parser.add_argument(
        '--forcefield', choices=ALL_FORCEFIELDS, default='desamber',
        help='force field / water model combination. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--verbose', action='store_true', default=False,
    )
    arg_parser.add_argument(
        '--opes_t_max', type=float, default=350.0,
        help='OPES_EXPANDED multithermal upper temperature in K. '
             'Default = %(default)s (Borsatto 2024: TEMPERATURE + 50 K)',
    )
    arg_parser.add_argument(
        '--opes_n_temps', type=int, default=30,
        help='OPES_EXPANDED T grid size. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--opes_pace', type=int, default=2500,
        help='OPES_EXPANDED ΔF deposition stride in MD steps. Default = %(default)s '
             '(= 10 ps at dt=4 fs, matching PLUMED SWISH-X bootcamp PACE=5000 × 2 fs).',
    )
    arg_parser.add_argument(
        '--opes_observation_steps', type=int, default=100,
        help='OPES_EXPANDED unbiased warmup in PACE units. '
             'Default = %(default)s (= 1 ns at PACE=2500, dt=4 fs; matches PLUMED default).',
    )
    arg_parser.add_argument(
        '--equil_time_ns', type=float, default=10.0,
        help='per-replica total NPT equilibration in ns before production, split '
             '50/50 restrained then unrestrained. '
             'Default = %(default)s; smoke tests use 0.01 (10 ps).',
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
    arg_parser.add_argument(
        '--collision_freq', type=float, default=1.0,
        help='Langevin thermostat friction gamma in ps^-1. Default = %(default)s',
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
        opes_t_max=args.opes_t_max,
        opes_n_temps=args.opes_n_temps,
        opes_pace=args.opes_pace,
        opes_observation_steps=args.opes_observation_steps,
        total_simtime_ns=args.total_simtime_ns,
        time_step_fs=args.time_step_fs,
        exchange_interval_ps=args.exchange_interval_ps,
        equil_time_ns=args.equil_time_ns,
        verbose=args.verbose,
        forcefield=args.forcefield,
        box_info_json_path=args.box_info,
        ensemble=args.ensemble,
        collision_freq=args.collision_freq,
        colvar_interval_ps=args.colvar_interval_ps,
    )
