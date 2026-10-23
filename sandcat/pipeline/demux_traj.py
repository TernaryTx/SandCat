from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import mdtraj as mdt
import numpy as np
from mdtraj.utils.unitcell import box_vectors_to_lengths_and_angles
from openmm import unit
from openmmtools.multistate import MultiStateReporter

from sandcat.variables import DEMUX_OUTPUT_PREFIX

from sandcat.config import add_config_arg, parse_args_with_config


def _process_trajectory(
        traj: mdt.Trajectory,
        strip_water: bool,
        align_selection: str,
) -> mdt.Trajectory:
    """Centre on the protein, align to frame 0, and optionally strip solvent.

    Steps applied in order:
      1. make_molecules_whole  — unwrap molecules split across periodic boundaries
      2. image_molecules       — re-image all molecules into the primary cell,
                                 keeping the protein as the anchor
      3. superpose             — RMSD-fit all frames to frame 0 using align_selection
      4. atom_slice (optional) — remove water and monatomic ions

    Args:
        traj: MDTraj Trajectory with unitcell information set.
        strip_water: remove water and monatomic ions if True.
        align_selection: MDTraj DSL selection string used for superposition.

    Returns:
        Processed trajectory.
    """
    try:
        traj.make_molecules_whole(inplace=True)
        protein_atoms = traj.topology.select('protein')
        traj.image_molecules(inplace=True, anchor_molecules=[set(protein_atoms)])
    except Exception as exc:
        # make_molecules_whole/image_molecules require bond information in the topology.
        # Warn and continue rather than crash if a stripped PDB lacks CONECT records.
        logging.warning('PBC re-imaging skipped (%s). Proceeding without wrapping.', exc)

    align_atoms = traj.topology.select(align_selection)
    if len(align_atoms) == 0:
        logging.warning('align_selection "%s" matched no atoms — skipping superposition.', align_selection)
    else:
        traj.superpose(traj, 0, atom_indices=align_atoms)

    if strip_water:
        keep_atoms = traj.topology.select(
            'not (water or (resname NA CL K MG CA ZN))'
        )
        traj = traj.atom_slice(keep_atoms)

    return traj


def _derive_checkpoint_path(nc_file: str) -> str | None:
    """Return the checkpoint file path for a .nc.bak reporter, or None for plain .nc files.

    Args:
        nc_file: Path to the main reporter NC file.

    Returns:
        Checkpoint path for .nc.bak files; None to let openmmtools auto-derive.
    """
    if nc_file.endswith('.nc.bak'):
        base = nc_file[:-len('.nc.bak')]
        return base + '_checkpoint.nc.bak'
    return None


def main(
        nc_file: str,
        topology_file: str,
        output_dir: str = '.',
        output_prefix: str = DEMUX_OUTPUT_PREFIX,
        strip_water: bool = False,
        align_selection: str = 'backbone',
) -> None:
    """Demultiplex a replica exchange NetCDF trajectory into per-thermodynamic-state XTC files.

    This sorts by thermodynamic STATE, not by walker. The distinction matters because
    "demultiplex" names the opposite operation in GROMACS: ``gmx trjcat -demux`` follows
    one coordinate set as it random-walks the ladder, so each of its output frames sits at
    a different lambda. Here each output file collects every frame that occupied one state,
    contributed by whichever replica held that slot, so ``state_state0.dcd`` is a
    fixed-lambda ensemble. Replica exchange preserves the correct distribution within each
    state, so a configuration's history at other lambda values does not bias it.

    The consequence is that an output file is not one continuous trajectory: it is a relay
    of continuous segments spliced at the points where the slot changed hands. Ensemble
    averages (densities, distance distributions, rotamer populations) are unaffected.
    Anything that reads the frame ORDER, such as counting transitions or using frame 0 as a
    reference, is not well defined on these files without also reading the replica
    assignment back from the reporter.

    Reads the replica-to-state assignment at each checkpoint iteration, routes each
    replica's coordinates to the trajectory of the thermodynamic state it occupied at that
    frame, then centres on the protein, aligns to frame 0, and optionally strips solvent.

    Outputs one DCD file and one reference PDB (frame 0) per thermodynamic state,
    named <output_prefix>_state{N}.dcd and <output_prefix>_state{N}_ref.pdb.

    Supports both plain .nc files and intermediate .nc.bak backup files produced by SWISH-X.
    For .nc.bak files the companion checkpoint is auto-derived as <stem>_checkpoint.nc.bak.

    Memory: all replica positions are pre-allocated simultaneously.
    For 8 replicas × 2500 checkpoints × 50k atoms this is ~12 GB (float32).

    Args:
        nc_file: Path to the main reporter NetCDF file (.nc or .nc.bak).
        topology_file: Path to a topology file (PDB or GRO) whose atoms match the NC file.
        output_dir: Directory for output files (created if absent).
        output_prefix: Stem prefix for output filenames.
        strip_water: Remove water and monatomic ions from output trajectories.
        align_selection: MDTraj DSL selection for RMSD superposition (default: backbone).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reporter = MultiStateReporter(nc_file, checkpoint_storage=_derive_checkpoint_path(nc_file), open_mode='r')
    n_states = reporter.n_states

    # Only checkpoint iterations have sampler states (positions) stored
    checkpoint_iterations = reporter.read_checkpoint_iterations()
    n_checkpoints = len(checkpoint_iterations)

    # Shape: (total_iterations, n_replicas) — entry [iter, replica] = state index
    all_replica_states = reporter.read_replica_thermodynamic_states()
    replica_states_at_checkpoints = all_replica_states[checkpoint_iterations]

    topology = mdt.load(topology_file).topology
    n_atoms = topology.n_atoms
    logging.info('%d states, %d checkpoints, %d atoms', n_states, n_checkpoints, n_atoms)

    # float32 halves memory vs float64 and matches MDTraj's internal representation
    state_positions = {sid: np.zeros((n_checkpoints, n_atoms, 3), dtype=np.float32) for sid in range(n_states)}
    state_box_lengths = {sid: np.zeros((n_checkpoints, 3), dtype=np.float32) for sid in range(n_states)}
    state_box_angles = {sid: np.zeros((n_checkpoints, 3), dtype=np.float32) for sid in range(n_states)}

    for checkpoint_num, iteration in enumerate(checkpoint_iterations):
        if checkpoint_num % 100 == 0:
            logging.info('Reading checkpoint %d/%d...', checkpoint_num, n_checkpoints)
        sampler_states = reporter.read_sampler_states(iteration)

        for replica_idx, state_idx in enumerate(replica_states_at_checkpoints[checkpoint_num]):
            sampler_state = sampler_states[replica_idx]
            vecs = sampler_state.box_vectors.value_in_unit(unit.nanometer)
            len_a, len_b, len_c, alpha, beta, gamma = box_vectors_to_lengths_and_angles(
                np.array(vecs[0]), np.array(vecs[1]), np.array(vecs[2]),
            )
            state_positions[state_idx][checkpoint_num] = sampler_state.positions.value_in_unit(unit.nanometer)
            state_box_lengths[state_idx][checkpoint_num] = [len_a, len_b, len_c]
            state_box_angles[state_idx][checkpoint_num] = [alpha, beta, gamma]

    reporter.close()

    for state_idx in range(n_states):
        logging.info('Processing state %d...', state_idx)
        traj = mdt.Trajectory(
            state_positions[state_idx],
            topology,
            unitcell_lengths=state_box_lengths[state_idx],
            unitcell_angles=state_box_angles[state_idx],
        )

        traj = _process_trajectory(traj, strip_water, align_selection)

        dcd_path = output_path / f'{output_prefix}_state{state_idx}.dcd'
        ref_path = output_path / f'{output_prefix}_state{state_idx}_ref.pdb'
        traj.save_dcd(str(dcd_path))
        traj[0].save_pdb(str(ref_path))
        logging.info('Saved %s (%.2f GB), %s', dcd_path, dcd_path.stat().st_size / 1e9, ref_path.name)


def parse_args() -> ArgumentParser:
    """Parse command-line arguments for the trajectory demultiplexing script."""
    arg_parser = ArgumentParser(
        description='Demultiplex a SWISH replica exchange trajectory into per-state XTC files.',
    )
    arg_parser.add_argument(
        '-nc', '--nc_file', type=str, required=True,
        help='path to the main reporter NetCDF file (.nc or .nc.bak); for .nc.bak files the '
             'companion checkpoint is auto-derived as <stem>_checkpoint.nc.bak',
    )
    arg_parser.add_argument(
        '-top', '--topology_file', type=str, required=True,
        help='path to a topology file (PDB or GRO) matching the atoms in the NC file',
    )
    arg_parser.add_argument(
        '-o', '--output_dir', type=str, default='.',
        help='directory for output XTC files (default: current directory)',
    )
    arg_parser.add_argument(
        '--output_prefix', type=str, default=DEMUX_OUTPUT_PREFIX,
        help='stem prefix for output files (default: %(default)s)',
    )
    arg_parser.add_argument(
        '--strip_water', action='store_true',
        help='remove water and monatomic ions from output trajectories',
    )
    arg_parser.add_argument(
        '--align_selection', type=str, default='backbone',
        help='MDTraj DSL selection for RMSD superposition (default: %(default)s)',
    )
    add_config_arg(arg_parser)
    return arg_parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    parser = parse_args()
    args = parse_args_with_config(parser)
    main(
        nc_file=args.nc_file,
        topology_file=args.topology_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        strip_water=args.strip_water,
        align_selection=args.align_selection,
    )
