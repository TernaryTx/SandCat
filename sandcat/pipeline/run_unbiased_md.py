from __future__ import annotations

import logging
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path

import mdtraj as mdt
from openmm import (
    app,
    unit,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
)
from openmmforcefields.generators import SystemGenerator

from sandcat.md_variables import FRICTION, PRESSURE, TEMPERATURE, get_n_steps
from sandcat.protonation import ProteinProtonater
from sandcat.structure import PdbFile
from sandcat.system_setup import add_custom_metal_bonds, wrap_align_trajectory
from sandcat.config import add_config_arg, parse_args_with_config
from sandcat.md_utils import (
    add_probe_repulsion,
    minimise_equilibrate_npt_sandcat,
    validate_output_provenance,
    write_output_provenance,
)
from sandcat.msmd_utils import reconcile_msmd_restart
from sandcat.msmd_options import normalise_msmd_probe_options
from sandcat.provenance import build_output_provenance
from sandcat.utils import (
    build_multi_ligand_solvent_system,
    build_mixed_solvent_system,
    create_multi_probe_system_generator,
    create_probe_system_generator,
    multi_ligand_probe_resnames,
    resolve_probe_sdf_path,
    uses_historical_probe_14_correction,
)
from sandcat.variables import (
    ALL_FORCEFIELDS,
    DEFAULT_TEMPERATURE_K,
    DESAMBER_FORCEFIELDS,
    MD_OUTPUT_PREFIX,
    PROBE_CONCENTRATION_HELP,
    PROBE_RESNAME,
)

MSMD_RESTART_PARAMETER_KEYS = (
    'forcefield',
    'probe_type',
    'probe_ff_path',
    'ensemble',
    'time_step_fs',
)


def can_restart_md(output_prefix: str = MD_OUTPUT_PREFIX) -> bool:
    """Return whether an XML state, DCD, and reference PDB are available."""
    return (
        Path(f'{output_prefix}.xml').exists()
        and Path(f'{output_prefix}.dcd').exists()
        and Path(f'{output_prefix}_ref.pdb').exists()
    )


def validate_msmd_restart_parameters(
        output_prefix: str,
        expected_parameters: Mapping[str, object],
) -> dict[str, object]:
    """Validate parameters that determine the rebuilt MSMD Hamiltonian.

    Args:
        output_prefix: Prefix of the MSMD restart and provenance files.
        expected_parameters: Current resolved CLI parameters.

    Returns:
        Validated provenance record.

    Raises:
        RuntimeError: If required parameters are missing or differ from the
            original simulation.
    """
    provenance = validate_output_provenance(output_prefix)
    if provenance.get('workflow') != 'msmd':
        raise RuntimeError(
            f"MSMD restart provenance has unexpected workflow: "
            f"{provenance.get('workflow')!r}."
        )
    recorded_parameters = provenance.get('parameters')
    if not isinstance(recorded_parameters, dict):
        raise RuntimeError('MSMD restart provenance has no parameter record.')

    missing_value = object()
    parameter_keys = list(MSMD_RESTART_PARAMETER_KEYS)
    if isinstance(expected_parameters.get('probe_type'), list):
        parameter_keys.append('probe_resname_mapping')

    mismatches = {}
    for key in parameter_keys:
        recorded_value = recorded_parameters.get(key, missing_value)
        expected_value = expected_parameters.get(key, missing_value)
        if recorded_value is missing_value or recorded_value != expected_value:
            mismatches[key] = (
                '<missing>' if recorded_value is missing_value else recorded_value,
                '<missing>' if expected_value is missing_value else expected_value,
            )
    if mismatches:
        mismatch_text = ', '.join(
            f'{key}: recorded={recorded!r}, requested={requested!r}'
            for key, (recorded, requested) in mismatches.items()
        )
        raise RuntimeError(
            f'MSMD restart parameter mismatch: {mismatch_text}'
        )
    return provenance


def _require_multi_probe_ff_paths(
        probe_ff_paths: Sequence[str | None],
) -> list[str]:
    """Return multi-ligand OFFXML paths without silently dropping missing values.

    Args:
        probe_ff_paths: Force-field paths normalised alongside the probe list.

    Returns:
        The same paths narrowed to strings.

    Raises:
        ValueError: If any ligand has no paired OFFXML path.
    """
    validated_paths = []
    for probe_ff_path in probe_ff_paths:
        if probe_ff_path is None:
            raise ValueError(
                'Every ligand in multi-ligand MSMD requires an OFFXML path.',
            )
        validated_paths.append(probe_ff_path)
    return validated_paths


def build_fresh_simulation(
        input_protein: PdbFile,
        forcefield: str,
        probe_types: Sequence[str],
        conc_mol_per_litre: float | None,
        add_hydrogens: bool,
        timestep,
        seed: int | None,
        ref_pdb_path: str,
        total_steps: int,
        box_info_json_path: str | None = None,
        ensemble: str = 'NPT',
        probe_ff_paths: Sequence[str | None] = (None,),
) -> tuple[app.Simulation, int, dict[str, object]]:
    """Build an OpenMM simulation from scratch: solvate, add probes, equilibrate.

    Writes the solvated reference PDB to ref_pdb_path for checkpoint restart.

    Returns:
        Simulation, remaining production steps, and probe-Hamiltonian provenance.
    """
    if add_hydrogens:
        input_protein = ProteinProtonater().process_one(input_protein)[0]

    is_multi_ligand = len(probe_types) > 1
    multi_probe_ff_paths = (
        _require_multi_probe_ff_paths(probe_ff_paths)
        if is_multi_ligand else []
    )
    probe_resnames: str | list[str] = (
        multi_ligand_probe_resnames(len(probe_types))
        if is_multi_ligand else PROBE_RESNAME
    )
    if is_multi_ligand:
        system, omm_model, _ = build_multi_ligand_solvent_system(
            forcefield=forcefield,
            protein_pdb_path=input_protein.path,
            probe_types=probe_types,
            probe_ff_paths=multi_probe_ff_paths,
            box_info_json_path=box_info_json_path,
            seed=seed,
        )
    else:
        system, omm_model, _ = build_mixed_solvent_system(
            forcefield,
            input_protein.path,
            probe_types[0],
            conc_mol_per_litre,
            box_info_json_path=box_info_json_path,
            seed=seed,
            probe_ff_path=probe_ff_paths[0],
        )
    probe_provenance = add_probe_repulsion(
        system,
        omm_model,
        probe_resname=probe_resnames,
        correct_probe_14_exceptions=(
            not is_multi_ligand
            and uses_historical_probe_14_correction(
                probe_types[0],
                probe_ff_paths[0],
            )
        ),
    )

    # write ref PDB with probes for topology reference
    with open(ref_pdb_path, 'w') as fh:
        app.PDBFile.writeFile(omm_model.topology, omm_model.positions, fh)
    logging.info(f'Reference PDB written to {ref_pdb_path}')

    logging.info('Running NPT equilibration...')
    positions, box_vectors = minimise_equilibrate_npt_sandcat(
        system,
        omm_model.topology,
        omm_model.positions,
        timestep,
        additional_solvent_resnames=(
            probe_resnames if is_multi_ligand else ()
        ),
    )

    if ensemble == 'NPT':
        system.addForce(MonteCarloBarostat(PRESSURE, TEMPERATURE))

    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, timestep)
    simulation = app.Simulation(
        omm_model.topology, system, integrator,
        Platform.getPlatformByName('CUDA'), {'Precision': 'mixed'},
    )
    simulation.context.setPositions(positions)
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.computeVirtualSites()
    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    return simulation, total_steps, probe_provenance


def restart_simulation(
        state_xml_path: str,
        ref_pdb_path: str,
        timestep,
        total_steps: int,
        ensemble: str = 'NPT',
        system_generator: SystemGenerator | None = None,
        correct_probe_14_exceptions: bool = True,
        probe_resnames: str | Sequence[str] = PROBE_RESNAME,
) -> tuple[app.Simulation, int, dict[str, object]]:
    """Rebuild an OpenMM simulation from a saved XML state.

    Returns:
        Simulation, remaining production steps, and probe-Hamiltonian provenance.
    """
    logging.info(f'Restarting from XML state: {state_xml_path}')
    ref_pdb = app.PDBFile(ref_pdb_path)
    omm_model = app.Modeller(ref_pdb.topology, ref_pdb.positions)
    omm_model.addExtraParticles(system_generator.forcefield)
    system = system_generator.forcefield.createSystem(
        omm_model.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometer,
        constraints=app.HBonds,
        hydrogenMass=4 * unit.amu,
    )
    add_custom_metal_bonds(system, omm_model)
    probe_provenance = add_probe_repulsion(
        system,
        omm_model,
        probe_resname=probe_resnames,
        correct_probe_14_exceptions=correct_probe_14_exceptions,
    )
    if ensemble == 'NPT':
        system.addForce(MonteCarloBarostat(PRESSURE, TEMPERATURE))
    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, timestep)
    simulation = app.Simulation(
        omm_model.topology, system, integrator,
        Platform.getPlatformByName('CUDA'), {'Precision': 'mixed'},
    )
    simulation.loadState(state_xml_path)

    steps_remaining = max(0, total_steps - simulation.currentStep)
    logging.info(
        f'Resuming from step {simulation.currentStep}: '
        f'{steps_remaining} steps remaining',
    )
    return simulation, steps_remaining, probe_provenance


def add_reporters(
        simulation: app.Simulation,
        dcd_path: str,
        state_xml_path: str,
        timestep,
        verbose: bool,
) -> None:
    """Attach DCD, XML-state, and optional console reporters."""
    report_interval_steps = get_n_steps(0.1 * unit.nanosecond, timestep)

    # append mode so restarts don't overwrite existing frames
    dcd_append = Path(dcd_path).exists()
    simulation.reporters.append(
        app.DCDReporter(dcd_path, report_interval_steps, append=dcd_append),
    )
    simulation.reporters.append(
        app.CheckpointReporter(
            state_xml_path,
            report_interval_steps,
            writeState=True,
        ),
    )
    if verbose:
        simulation.reporters.append(
            app.StateDataReporter(
                'stdout', report_interval_steps,
                step=True, time=True, potentialEnergy=True, temperature=True, speed=True,
            ),
        )


def main(
        input_protein: PdbFile,
        probe_type: str | Sequence[str] = 'benzene',
        conc_mol_per_litre: float | None = None,
        add_hydrogens: bool = False,
        total_simtime_ns: float = 150.0,
        time_step_fs: int = 4,
        verbose: bool = False,
        output_prefix: str = MD_OUTPUT_PREFIX,
        forcefield: str = 'desamber',
        seed: int | None = None,
        box_info_json_path: str | None = None,
        remove_solvent: bool = True,
        ensemble: str = 'NPT',
        probe_ff_path: str | Sequence[str] | None = None,
) -> None:
    """Set up and run a standard (unbiased) mixed-solvent MD simulation.

    Uses the same system setup as run_swish.py (probe type, concentration, force field,
    equilibration protocol) but runs single-replica plain NVT production. Intended to
    generate the unbiased probe density reference for probe_density_comparison.py.

    If an XML state file and DCD from a previous run are present the simulation is
    resumed automatically, so spot-instance interruptions are recoverable.

    Args:
        input_protein: Input protein PDB file.
        probe_type: Built-in probe name, one ligand SDF, or multiple ligand SDFs.
        conc_mol_per_litre: Target concentration for a single probe. It is
            incompatible with multi-ligand mode, which adds one copy per ligand.
        add_hydrogens: Whether to protonate the input structure before setup.
        total_simtime_ns: Total production simulation time in nanoseconds.
        time_step_fs: MD integration timestep in femtoseconds.
        verbose: Whether to enable INFO-level logging to the terminal.
        output_prefix: Prefix for output files (DCD, XML state, ref PDB).
        forcefield: Force field / water model combination ('desamber', 'ff14', or 'ff19').
        seed: Random seed for probe placement. If None, a random seed is used.
        box_info_json_path: optional path to a JSON file specifying explicit
            box parameters (one of ``padding`` / ``box_size`` / ``box_edge`` /
            ``box_vectors``, plus optional ``box_shape``). Format matches the
            ``-bi`` flag in SandCat MD jobs. If omitted, defaults to a
            1.0 nm cubic padding.
        remove_solvent: strip water and ions from the aligned output trajectory.
            Transition-metal residues (ZN, FE, etc.) are always retained.
        ensemble: production ensemble — 'NPT' adds a MonteCarloBarostat; 'NVT' runs
            without pressure coupling (box volume fixed at the equilibrated size).
        probe_ff_path: Optional SMIRNOFF OFFXML, or one OFFXML per ligand.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    probe_types, probe_ff_paths, resolved_concentration = normalise_msmd_probe_options(
        probe_types=probe_type,
        probe_ff_paths=probe_ff_path,
        conc_mol_per_litre=conc_mol_per_litre,
    )
    is_multi_ligand = len(probe_types) > 1
    multi_probe_ff_paths = (
        _require_multi_probe_ff_paths(probe_ff_paths)
        if is_multi_ligand else []
    )
    probe_resnames: str | list[str] = (
        multi_ligand_probe_resnames(len(probe_types))
        if is_multi_ligand else PROBE_RESNAME
    )
    probe_parameter: str | list[str] = (
        probe_types if is_multi_ligand else probe_types[0]
    )
    probe_ff_parameter: str | list[str] | None = (
        multi_probe_ff_paths
        if is_multi_ligand else probe_ff_paths[0]
    )
    probe_definition_paths = [
        resolve_probe_sdf_path(current_probe_type)
        for current_probe_type in probe_types
    ]
    probe_definition_parameter: str | list[str] = (
        probe_definition_paths if is_multi_ligand else probe_definition_paths[0]
    )
    probe_resname_mapping = (
        [
            {'probe_type': current_probe_type, 'resname': current_resname}
            for current_probe_type, current_resname in zip(
                probe_types,
                probe_resnames,
                strict=True,
            )
        ]
        if is_multi_ligand else [
            {'probe_type': probe_types[0], 'resname': PROBE_RESNAME},
        ]
    )

    timestep = time_step_fs * unit.femtosecond
    total_steps = get_n_steps(total_simtime_ns * unit.nanosecond, timestep)

    dcd_path = f'{output_prefix}.dcd'
    state_xml_path = f'{output_prefix}.xml'
    ref_pdb_path = f'{output_prefix}_ref.pdb'
    provenance_path = f'{output_prefix}_provenance.json'

    is_restart = can_restart_md(output_prefix)
    has_restart_provenance = is_restart and Path(provenance_path).is_file()
    if has_restart_provenance:
        validate_msmd_restart_parameters(
            output_prefix,
            {
                'forcefield': forcefield,
                'probe_type': probe_parameter,
                'probe_ff_path': probe_ff_parameter,
                'probe_resname_mapping': probe_resname_mapping,
                'ensemble': ensemble,
                'time_step_fs': time_step_fs,
            },
        )
    if is_restart:
        reconcile_msmd_restart(output_prefix)
        if is_multi_ligand:
            _, system_generator = create_multi_probe_system_generator(
                forcefield=forcefield,
                probe_types=probe_types,
                probe_ff_paths=multi_probe_ff_paths,
            )
        else:
            _, system_generator = create_probe_system_generator(
                forcefield=forcefield,
                probe_type=probe_types[0],
                probe_ff_path=probe_ff_paths[0],
            )
        simulation, steps_remaining, probe_provenance = restart_simulation(
            state_xml_path=state_xml_path,
            ref_pdb_path=ref_pdb_path,
            timestep=timestep,
            total_steps=total_steps,
            ensemble=ensemble,
            system_generator=system_generator,
            probe_resnames=probe_resnames,
            correct_probe_14_exceptions=uses_historical_probe_14_correction(
                probe_types[0],
                probe_ff_paths[0],
            ) if not is_multi_ligand else False,
        )
    else:
        simulation, steps_remaining, probe_provenance = build_fresh_simulation(
            input_protein=input_protein,
            forcefield=forcefield,
            probe_types=probe_types,
            conc_mol_per_litre=resolved_concentration,
            add_hydrogens=add_hydrogens,
            timestep=timestep,
            seed=seed,
            ref_pdb_path=ref_pdb_path,
            total_steps=total_steps,
            box_info_json_path=box_info_json_path,
            ensemble=ensemble,
            probe_ff_paths=probe_ff_paths,
        )

    if not has_restart_provenance:
        parameters = {
            'input_protein': input_protein.path,
            'probe_type': probe_parameter,
            'probe_ff_path': probe_ff_parameter,
            'concentration_mol_per_litre': resolved_concentration,
            'copies_per_ligand': 1 if is_multi_ligand else None,
            'probe_resnames': probe_resnames,
            'probe_resname_mapping': probe_resname_mapping,
            'probe_template_matching': (
                'residue-name-directed' if is_multi_ligand else 'automatic'
            ),
            'add_hydrogens': add_hydrogens,
            'total_simtime_ns': total_simtime_ns,
            'time_step_fs': time_step_fs,
            'temperature_kelvin': DEFAULT_TEMPERATURE_K,
            'pressure_bar': 1.0 if ensemble == 'NPT' else None,
            'collision_frequency_per_ps': FRICTION.value_in_unit(
                unit.picosecond ** -1,
            ),
            'output_prefix': output_prefix,
            'forcefield': forcefield,
            'box_info_json_path': box_info_json_path,
            'remove_solvent': remove_solvent,
            'ensemble': ensemble,
            'use_barostat': ensemble == 'NPT',
            'system_build_seed': seed,
            'system_build_seed_source': (
                'user' if seed is not None else 'OS randomness'
            ),
            'production_random_seed': None,
            'production_random_seed_source': 'OpenMM OS-random seed',
        }
        input_paths = {
            'input_protein': input_protein.path,
            'box_info': box_info_json_path,
            'probe_definition': probe_definition_parameter,
            'probe_forcefield': probe_ff_parameter,
            'forcefield_definition': (
                DESAMBER_FORCEFIELDS if forcefield == 'desamber' else None
            ),
        }
        extra = dict(probe_provenance)
        extra.update({
            'workflow': 'msmd',
            'probe_type': probe_parameter,
        })
        if is_restart:
            extra['provenance_reconstructed_on_restart'] = True
        provenance = build_output_provenance(
            workflow='msmd',
            parameters=parameters,
            input_paths=input_paths,
            extra=extra,
        )
        write_output_provenance(output_prefix, provenance)

    add_reporters(simulation, dcd_path, state_xml_path, timestep, verbose)

    logging.info(f'Running production MD: {total_simtime_ns:.0f} ns total ({steps_remaining} steps remaining)')
    simulation.step(steps_remaining)
    simulation.saveState(state_xml_path)
    logging.info(f'Production MD complete. DCD: {dcd_path}')

    aligned_path = f'aligned_{output_prefix}.dcd'
    last_frame_path = f'{output_prefix}_last_frame.pdb'
    logging.info(f'Wrapping/aligning trajectory (remove_solvent={remove_solvent})...')
    md_topology = mdt.Topology.from_openmm(simulation.topology)
    md_traj = wrap_align_trajectory(
        dcd_path, md_topology, input_protein.path, remove_solvent,
    )
    md_traj.save(aligned_path)
    md_traj[-1].save(last_frame_path)
    logging.info(f'Wrote {aligned_path} and {last_frame_path}')


def parse_args() -> ArgumentParser:
    """Parse command-line arguments for standard mixed-solvent MD."""
    arg_parser = ArgumentParser(
        description='Run a standard (unbiased) mixed-solvent MD simulation for probe density comparison.',
    )
    arg_parser.add_argument('-p', '--protein', type=str, required=True, help='input protein PDB filename')
    arg_parser.add_argument(
        '--probe', type=str, nargs='+', default=['benzene'],
        help='built-in probe name, one ligand SDF, or a space-separated list of ligand SDFs. '
             'Multiple ligands are each inserted once.',
    )
    arg_parser.add_argument(
        '--probe_ff_path', type=str, nargs='+', default=None,
        help='optional SMIRNOFF OFFXML, or one ordered OFFXML per ligand.',
    )
    arg_parser.add_argument(
        '--concentration', type=float, default=None,
        help=PROBE_CONCENTRATION_HELP + ' Defaults to 1.0 M for one probe and is '
             'incompatible with multiple probes.',
    )
    arg_parser.add_argument('-ah', '--add_hydrogens', choices=['True', 'False'], default='False',
                            help='whether to add hydrogen atoms to the protein structure. Default = %(default)s')
    arg_parser.add_argument('-t', '--total_simtime_ns', type=float, default=150.0,
                            help='total simulation length in nanoseconds. Default = %(default)s ns')
    arg_parser.add_argument('-ts', '--time_step_fs', type=int, default=4,
                            help='MD timestep in femtoseconds. Default = %(default)s fs')
    arg_parser.add_argument('--forcefield', type=str, default='desamber', choices=ALL_FORCEFIELDS,
                            help='force field / water model combination. Default = %(default)s')
    arg_parser.add_argument('--verbose', action='store_true', default=False,
                            help='enable verbose logging output')
    arg_parser.add_argument('--seed', type=int, default=None,
                            help='random seed for probe placement. Default = random')
    arg_parser.add_argument(
        '-bi', '--box_info', type=str, default=None,
        help='optional JSON file with box parameters (padding, box_shape, box_size, '
             'box_edge, or box_vectors). Same format as the SandCat MD -bi option. '
             'If omitted, a 1.0 nm cubic padding is used.',
    )
    arg_parser.add_argument(
        '--keep_solvent', action='store_true', default=False,
        help='keep water and ions in the aligned output trajectory',
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
    main(
        input_protein=PdbFile(args.protein),
        probe_type=args.probe,
        conc_mol_per_litre=args.concentration,
        add_hydrogens=args.add_hydrogens != 'False',
        total_simtime_ns=args.total_simtime_ns,
        time_step_fs=args.time_step_fs,
        verbose=args.verbose,
        forcefield=args.forcefield,
        seed=args.seed,
        box_info_json_path=args.box_info,
        remove_solvent=not args.keep_solvent,
        ensemble=args.ensemble,
        probe_ff_path=args.probe_ff_path,
    )
