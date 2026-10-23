from __future__ import annotations

import argparse
import logging
import time

import mdtraj as mdt
import numpy as np
import openmm as mm
from openmm import app, unit
from openmm.app import CheckpointReporter, DCDReporter, StateDataReporter

from openmmforcefields.generators import SystemGenerator

from sandcat.ligand import LigFile
from sandcat.md_variables import FRICTION, REPORT_INTERVAL, get_n_steps
from sandcat.structure import PdbFile
from sandcat.system_setup import (
    add_custom_metal_bonds, add_ligfile_to_omm_model, wrap_align_trajectory,
)
from sandcat.protonation import ProteinProtonater
from sandcat.md_utils import minimise_equilibrate_npt_sandcat
from sandcat.opes_expanded import (
    OPES_BIAS_FORCE_GROUP, OpesExpanded, add_opes_expanded_bias,
    build_opes_expanded_integrator,
)
from sandcat.config import add_config_arg, parse_args_with_config
from sandcat.utils import solvent_box_kwargs
from sandcat.variables import FORCEFIELD_OPTIONS, OPES_EXPANDED_OUTPUT_PREFIX


def main(
        input_protein: PdbFile,
        t_0: float = 300.0,
        t_min: float | None = None,
        t_max: float = 350.0,
        n_temps: int = 30,
        pace: int = 500,
        observation_steps: int = 250,
        equilibration_ps: int = 1000,
        total_simtime_ns: float = 5.0,
        time_step_fs: int = 4,
        add_hydrogens: bool = False,
        log_stride_ps: float = 10.0,
        verbose: bool = False,
        output_prefix: str = OPES_EXPANDED_OUTPUT_PREFIX,
        remove_solvent: bool = True,
        forcefield: str = 'desamber',
        box_info_json_path: str | None = None,
        input_ligand: LigFile | None = None,
) -> None:
    """Single-replica OPES_EXPANDED + ECV_MULTITHERMAL on a solvated protein.

    Solvates, NPT-equilibrates, then NVT-samples at t_0 with a multithermal
    bias targeting [t_min, t_max]. t_min defaults to t_0 when not specified.
    Outputs ``{prefix}.npz/_state.npz/.log`` plus DCD/checkpoint/CSV in
    MDRunner style.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    if t_min is None:
        t_min = t_0
    if t_max <= t_min:
        raise ValueError('t_max must be > t_min for a multithermal expansion')
    if n_temps < 2:
        raise ValueError('n_temps must be >= 2')

    if add_hydrogens:
        input_protein = ProteinProtonater().process_one(input_protein)[0]

    pdb = app.PDBFile(input_protein.path)
    omm_model = app.Modeller(pdb.topology, pdb.positions)

    logging.info('Building solvated system...')
    forcefields, water_model = FORCEFIELD_OPTIONS[forcefield]
    off_molecules = []
    if input_ligand is not None:
        off_molecule = add_ligfile_to_omm_model(omm_model, input_ligand)
        off_molecules.append(off_molecule)
    system_generator = SystemGenerator(
        forcefields=forcefields,
        small_molecule_forcefield='openff-2.2.0.offxml',
        molecules=off_molecules,
    )
    omm_model.addSolvent(
        system_generator.forcefield,
        model=water_model,
        ionicStrength=0.15 * unit.molar,
        **solvent_box_kwargs(box_info_json_path),
    )
    system = system_generator.forcefield.createSystem(
        omm_model.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometer,
        constraints=app.HBonds,
        hydrogenMass=4 * unit.amu,
    )
    add_custom_metal_bonds(system, omm_model)

    timestep = time_step_fs * unit.femtosecond

    # Restraints + barostat are removed in-place from ``system`` before
    # returning, so the same object is ready for NVT OPES below.
    logging.info(f'NPT equilibrating ({equilibration_ps} ps with restraints + barostat)...')
    equil_positions, equil_box = minimise_equilibrate_npt_sandcat(
        system,
        omm_model.topology,
        omm_model.positions,
        timestep=timestep,
        equil_time=equilibration_ps * unit.picosecond,
    )

    cv_force = add_opes_expanded_bias(system, n_temps=n_temps)
    opes = OpesExpanded(
        temperature=t_0 * unit.kelvin,
        t_min=t_min,
        t_max=t_max,
        n_temps=n_temps,
        pace=pace,
        observation_steps=observation_steps,
    )

    integrator = build_opes_expanded_integrator(
        timestep=timestep,
        collision_rate=FRICTION,
        temperature=t_0 * unit.kelvin,
        n_temps=n_temps,
    )
    sim = app.Simulation(
        omm_model.topology,
        system,
        integrator,
        platform=mm.Platform.getPlatformByName('CUDA'),
    )
    sim.context.setPeriodicBoxVectors(*equil_box)
    sim.context.setPositions(equil_positions)
    sim.context.setVelocitiesToTemperature(t_0 * unit.kelvin)
    opes.apply_to_context(sim.context)

    # MDRunner-style reporters (DCD + periodic State checkpoint + CSV log),
    # attached after equilibration so production output excludes the warmup.
    timestep_qty = time_step_fs * unit.femtosecond
    n_steps_total = int(total_simtime_ns * 1000.0 / (time_step_fs / 1000.0))
    if total_simtime_ns * unit.nanosecond < REPORT_INTERVAL:
        report_interval_n_steps = n_steps_total
    else:
        report_interval_n_steps = get_n_steps(REPORT_INTERVAL, timestep_qty)
    sim.reporters.append(
        DCDReporter(f'{output_prefix}.dcd', report_interval_n_steps),
    )
    sim.reporters.append(
        CheckpointReporter(
            f'{output_prefix}_state.xml', report_interval_n_steps, writeState=True,
        ),
    )
    sim.reporters.append(
        StateDataReporter(
            f'{output_prefix}_sim.log', report_interval_n_steps,
            step=True, temperature=True, progress=True,
            remainingTime=True, speed=True, potentialEnergy=True,
            density=True, volume=True,
            totalSteps=n_steps_total, separator=',',
        ),
    )

    logging.info(
        f'OPES_EXPANDED multithermal: t_0={t_0} K, range '
        f'[{opes.temps[0]:.1f}, {opes.temps[-1]:.1f}] K, N_T={len(opes.temps)} '
        f'(grid: {[round(t,1) for t in opes.temps]}); '
        f'PACE={pace}, obs_steps={observation_steps} '
        f'({pace*observation_steps*time_step_fs/1000.0:.0f} ps unbiased warmup), '
        f'total_simtime_ns={total_simtime_ns}'
    )

    log_stride_steps = max(int(log_stride_ps * 1000.0 / time_step_fs), pace)
    record_stride = pace  # one record per ΔF update

    n_records = n_steps_total // record_stride
    times_ps = np.zeros(n_records)
    u_traj = np.zeros(n_records)
    bias_traj = np.zeros(n_records)

    log_path = f'{output_prefix}.log'
    temp_col_names = ' '.join(f'DeltaF_{t:.6g}' for t in opes.temps)
    with open(log_path, 'w') as logf:
        logf.write(f'#! FIELDS time rct bias_V_kj u_kj {temp_col_names}\n')
        logf.write(f'#! SET print_stride  {record_stride}\n')
        wall_start = time.time()
        log_every_chunks = max(1, log_stride_steps // record_stride)
        chunk = record_stride

        ff_groups_mask = 0xFFFFFFFF & ~(1 << OPES_BIAS_FORCE_GROUP)
        for rec in range(n_records):
            opes.step(sim, chunk)
            ff_state = sim.context.getState(getEnergy=True, groups=ff_groups_mask)
            pot_ene = ff_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            bias_ene = opes.compute_bias_v(pot_ene)
            t_ps = sim.currentStep * time_step_fs / 1000.0
            times_ps[rec] = t_ps
            u_traj[rec] = pot_ene
            bias_traj[rec] = bias_ene
            if rec % log_every_chunks == 0:
                d = opes.get_diagnostics()
                df_str = ' '.join(f'{x:+.2f}' for x in d['deltaF_kj'])
                wall = time.time() - wall_start
                ns_per_day = (t_ps / 1000.0) / max(wall, 1) * 86400
                logf.write(
                    f'{t_ps:.2f} {d["rct_kj"]:+.3f} {bias_ene} {pot_ene} {df_str}\n'
                )
                logf.flush()
                logging.info(
                    f't={t_ps:.1f} ps  U={pot_ene} V={bias_ene}  '
                    f'rct={d["rct_kj"]:+.2f}  speed={ns_per_day:.0f} ns/day  '
                )

        wall_total = time.time() - wall_start
        logging.info(
            f'Production complete: {total_simtime_ns} ns / {wall_total:.0f} s '
            f'= {total_simtime_ns * 86400 / wall_total:.1f} ns/day'
        )

    np.savez(
        f'{output_prefix}.npz',
        times_ps=times_ps,
        u_kj=u_traj,
        bias_kj=bias_traj,
        temps_k=opes.temps,
        dbeta=opes._dbeta,
    )
    state = opes.get_state()
    np.savez(
        f'{output_prefix}_state.npz',
        deltaf_kj=np.array(state['deltaF_kj']),
        temps_k=np.array(state['temps_k']),
        dbeta=np.array(state['dbeta']),
        scalars=np.array([
            state['rct_kj'], state['counter'], state['t0_k'],
            state['t_min_k'], state['t_max_k'], state['pace'],
            int(state['initialized']),
        ]),
    )
    logging.info(f'Wrote {output_prefix}.npz, {output_prefix}_state.npz, {output_prefix}.log')

    # MDRunner-style post-process: image into the box, align to chain-0 CA,
    # strip solvent by default. Aligned DCD written alongside the raw DCD.
    dcd_path = f'{output_prefix}.dcd'
    aligned_path = f'aligned_{output_prefix}.dcd'
    last_frame_path = f'{output_prefix}_last_frame.pdb'
    logging.info(f'Wrapping/aligning trajectory (remove_solvent={remove_solvent})...')
    md_topology = mdt.Topology.from_openmm(sim.topology)
    md_traj = wrap_align_trajectory(
        dcd_path, md_topology, input_protein.path, remove_solvent,
    )
    md_traj.save(aligned_path)
    md_traj[-1].save(last_frame_path)
    logging.info(f'Wrote {aligned_path} and {last_frame_path}')


def parse_args() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description='Run a single-replica OPES_EXPANDED multithermal MD on a PDB.',
    )
    arg_parser.add_argument(
        '-p', '--protein', type=str, required=True,
        help='input protein PDB filename',
    )
    arg_parser.add_argument(
        '-T', '--temperature', dest='t_0', type=float, default=300.0,
        help='thermostat / "physical" temperature in K. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--t_max', type=float, default=350.0,
        help='upper bound of the multithermal range, K. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--t_min', type=float, default=None,
        help='lower bound of the multithermal range, K. Defaults to '
             '--temperature if not set.',
    )
    arg_parser.add_argument(
        '--n_temps', type=int, default=30,
        help='temperature grid size. Default = %(default)s ',
    )
    arg_parser.add_argument(
        '--pace', type=int, default=500,
        help='ΔF update interval in MD steps. Default = %(default)s',
    )
    arg_parser.add_argument(
        '--observation_steps', type=int, default=250,
        help='unbiased ΔF init phase in PACE units (on top of NPT equil). '
             'Default = %(default)s.',
    )
    arg_parser.add_argument(
        '--equilibration_ps', type=int, default=1000,
        help='NPT equil with restraints + barostat before OPES Expanded (ps). '
             'Default = %(default)s; solvated proteins typically need >= 1 ns.',
    )
    arg_parser.add_argument(
        '-t', '--total_simtime_ns', type=float, default=100.0,
        help='total simulation length in ns. Default = %(default)s',
    )
    arg_parser.add_argument(
        '-ts', '--time_step_fs', type=int, default=4,
        help='MD timestep in fs. Default = %(default)s',
    )
    arg_parser.add_argument(
        '-ah', '--add_hydrogens', choices=['True', 'False'], default='False',
        help="whether to protonate the input. Default = False",
    )
    arg_parser.add_argument(
        '--log_stride_ps', type=float, default=100.0,
        help='wall-clock interval for diagnostic log lines (ps). Default = %(default)s',
    )
    arg_parser.add_argument(
        '--forcefield', choices=list(FORCEFIELD_OPTIONS.keys()), default='desamber',
        help='force field / water model combination. Default = %(default)s',
    )
    arg_parser.add_argument(
        '-bi', '--box_info', type=str, default=None,
        help='optional JSON file with box parameters (padding, box_shape, box_size, '
             'box_edge, or box_vectors). If omitted, a 1.0 nm cubic padding is used.',
    )
    arg_parser.add_argument(
        '-l', '--ligand', type=str, default=None,
        help='path to ligand SDF or JSON file. If omitted, protein-only simulation is run.',
    )
    arg_parser.add_argument(
        '--output_prefix', type=str, default=OPES_EXPANDED_OUTPUT_PREFIX,
    )
    arg_parser.add_argument(
        '--verbose', action='store_true', default=False,
    )
    arg_parser.add_argument(
        '--keep_solvent', action='store_true', default=False,
        help='keep water/ions in the aligned trajectory output. Default: stripped',
    )
    add_config_arg(arg_parser)
    return arg_parser


if __name__ == '__main__':
    parser = parse_args()
    args = parse_args_with_config(parser)
    main(
        input_protein=PdbFile(path=args.protein),
        t_0=args.t_0,
        t_min=args.t_min,
        t_max=args.t_max,
        n_temps=args.n_temps,
        pace=args.pace,
        observation_steps=args.observation_steps,
        equilibration_ps=args.equilibration_ps,
        total_simtime_ns=args.total_simtime_ns,
        time_step_fs=args.time_step_fs,
        add_hydrogens=args.add_hydrogens != 'False',
        log_stride_ps=args.log_stride_ps,
        verbose=args.verbose,
        output_prefix=args.output_prefix,
        remove_solvent=not args.keep_solvent,
        forcefield=args.forcefield,
        box_info_json_path=args.box_info,
        input_ligand=LigFile(path=args.ligand) if args.ligand else None,
    )
