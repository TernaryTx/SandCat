"""Build the probe density maps and candidate tables from the raw trajectories.

The first stage of the prospective analysis, run by
run_prospective_analysis_pipeline. For each target it aligns the unbiased
mixed-solvent replicates and the demultiplexed SWISH-X trajectories, computes
the probe density maps, and writes the candidate table the later stages read.

Every parameter is pinned here explicitly rather than left to a default, and
the per-target input paths are recorded alongside them, so that a run is fully
described by this file. Point it at a fresh output directory and the result can
be compared against an existing one; anything that differs is either a parameter
that is not what the defaults say or a genuine non-determinism.

TEM1 is the one target whose SWISH directory carries a suffix, and this is
deliberate. The _gammaN suffix is a
label on the Langevin collision frequency, not on the structure: all five TEM1
runs share one input checksum and differ only in --collision_freq, which is 0.5,
1.0, 2.5, 5.0 and 10.0 ps-1 for _gamma05, _gamma1, _gamma25, the unsuffixed run
and _gamma10. Every other target was run at 1.0, so _gamma1 is the only TEM1
trajectory that matches them; the unsuffixed one would have put TEM1 alone at 5.0.
"""
from __future__ import annotations

import logging
import sys
import traceback
from argparse import ArgumentParser
from pathlib import Path


from sandcat.analysis.compute_probe_density import STRIDE, main as run_density
# the raw simulation tree, defined once in experiments rather than resolved
# again here, so this stage and the ones after it cannot read different trees
from sandcat.analysis.experiments import target_dir, SIM

# (unbiased directory, SWISH demux directory, apo reference, replicas).
# TEM1 points at _gamma1, its collision frequency 1.0 run, matching every other
# target. MDM2 has five replicas rather than six and a scale_max of 1.28 rather
# than 1.35.
SIMULATION_INPUTS = {
    'BCLXL':    ('BCLXL/msmd_out_benz_1R2D_input',
                 'BCLXL/sandcat_swish_x_out_benz_1R2D_input/demux',
                 'BCLXL/msmd_out_benz_1R2D_input/0/1R2D_input.pdb', 6),
    'CRBN':     ('CRBN/msmd_out_benz_9FJX_input',
                 'CRBN/sandcat_swish_x_out_benz_9FJX_input/demux',
                 'CRBN/msmd_out_benz_9FJX_input/0/9FJX_input.pdb', 6),
    'HPV11-E2': ('HPV11-E2/msmd_out_benz_1R6K_input',
                 'HPV11-E2/sandcat_swish_x_out_benz_1R6K_input/demux',
                 'HPV11-E2/msmd_out_benz_1R6K_input/0/1R6K_input.pdb', 6),
    'IL2':      ('IL2/msmd_out_benz_1M47_input',
                 'IL2/sandcat_swish_x_out_benz_1M47_input/demux',
                 'IL2/msmd_out_benz_1M47_input/0/1M47_input.pdb', 6),
    'MDM2':     ('MDM2/msmd_out_benz_1Z1M_input',
                 'MDM2/sandcat_swish_x_out_benz_1Z1M_input/demux',
                 'MDM2/msmd_out_benz_1Z1M_input/0/1Z1M_input.pdb', 5),
    'PLK1':     ('PLK1/msmd_out_benz_3FVH_input',
                 'PLK1/sandcat_swish_x_out_benz_3FVH_input/demux',
                 'PLK1/msmd_out_benz_3FVH_input/0/3FVH_input.pdb', 6),
    'TEM1':     ('TEM1/msmd_out_benz_1JWP_input',
                 'TEM1/sandcat_swish_x_out_benz_1JWP_input_gamma1/demux',
                 'TEM1/msmd_out_benz_1JWP_input/0/1JWP_input.pdb', 6),
    'UB':       ('UB/msmd_out_benz_2LJ5_input',
                 'UB/sandcat_swish_x_out_benz_2LJ5_input/demux',
                 'UB/msmd_out_benz_2LJ5_input/0/2LJ5_input.pdb', 6),
}

# Every parameter stated, none left to a default that could change underneath
# the analysis. The dry unbiased pair is used: the wet files carry water and
# ions and do not match the demuxed SWISH-X trajectories they are compared with.
UNBIASED_DCD = 'aligned_md_output.dcd'
UNBIASED_TOP = 'md_output_last_frame.pdb'
TRAJ_PREFIX = 'state_state'
SWISH_TOP_PREFIX = 'state_state'
FIT_SELECTION = 'name CA'
# Every tenth frame: the raw demuxed trajectories hold 2501 frames and the
# analysed ones 251. It must be set here rather than left to the default of 1
# in compute_probe_density.main and the CLI, which would analyse ten times the
# data and give different maps.
# The Hamiltonian ladder, so the logs name the real lambda rather than the
# state index. State 0 is lambda = 1.0, the unscaled Hamiltonian the whole
# analysis rests on; the ladder has no rung at 0.
LAMBDA_LADDER = {5: [1.0, 1.07, 1.14, 1.21, 1.28],
                 6: [1.0, 1.07, 1.14, 1.21, 1.28, 1.35]}


def unbiased_pairs(folder: Path) -> list[tuple[str, str]]:
    """One (topology, trajectory) pair per numbered replicate subdirectory.

    The dry pair is used. The wet files sitting beside them carry water and ions
    and do not match the demuxed SWISH-X trajectories they are subtracted from,
    and pairing a dry trajectory with a wet topology fails on an atom count
    rather than silently, but it is easy to do by accident.

    Args:
        folder: Unbiased MD output directory, holding one numbered
            subdirectory per replicate.

    Returns:
        (topology path, trajectory path) per replicate that has both, ordered by
        subdirectory name. A replicate missing either file is skipped.
    """
    pairs = []
    for sub in sorted(folder.iterdir(), key=lambda p: p.name):
        if not (sub.is_dir() and sub.name.isdigit()):
            continue
        dcd, top = sub / UNBIASED_DCD, sub / UNBIASED_TOP
        if dcd.exists() and top.exists():
            pairs.append((str(top), str(dcd)))
    return pairs


def parse_args() -> ArgumentParser:
    """Create the density-stage argument parser.

    Returns:
        Parser accepting an output directory and optional benchmark targets.
    """
    parser = ArgumentParser(description='Build density maps for the published benchmark.')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('targets', nargs='*', help='Target names; defaults to all targets.')
    return parser


def main() -> None:
    """Build density maps for the targets named on the command line, or all of them.

    Each target writes into its own branch directory with its own run.log. A
    target that raises is recorded and stepped over, since the targets are
    independent and finishing the rest is useful, but any failure makes the
    stage exit non-zero at the end so that no later stage runs against a
    partially updated tree.

    Raises:
        SystemExit: If any target failed, after every target has been attempted.
    """
    parser = parse_args()
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    wanted = args.targets or list(SIMULATION_INPUTS)
    unknown = set(wanted) - SIMULATION_INPUTS.keys()
    if unknown:
        parser.error(f'Unknown targets: {", ".join(sorted(unknown))}')
    out.mkdir(parents=True, exist_ok=True)
    failed = []

    for target in wanted:
        md_dir, swish_dir, apo, replicas = SIMULATION_INPUTS[target]
        branch = target_dir(target, out)
        branch.mkdir(exist_ok=True)
        logging.basicConfig(level=logging.INFO, force=True,
                            format='%(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(branch / 'run.log',
                                                          mode='w'),
                                      logging.StreamHandler(sys.stdout)])
        pairs = unbiased_pairs(SIM / md_dir)
        print(f'=== {target}: {len(pairs)} unbiased replicates, '
              f'{replicas} SWISH replicas', flush=True)
        try:
            run_density(
                md_pairs=pairs,
                swish_dir=str(SIM / swish_dir),
                swish_top_prefix=SWISH_TOP_PREFIX,
                ref_pdb=str(SIM / apo),
                output_dir=str(branch / 'density'),
                n_replicas=replicas,
                lambda_values=LAMBDA_LADDER[replicas],
                traj_prefix=TRAJ_PREFIX,
                fit_selection=FIT_SELECTION,
                stride=STRIDE,
            )
            print(f'=== {target}: done', flush=True)
        except Exception:
            failed.append(target)
            print(f'=== {target}: FAILED\n{traceback.format_exc()}', flush=True)

    if failed:
        raise SystemExit(
            f'\n{len(failed)} of {len(wanted)} targets failed: '
            f'{", ".join(failed)}. Exiting non-zero so the stages that read '
            f'these maps do not run against a partially updated tree, which on '
            f'a reused output directory would mean consuming an earlier run.')


if __name__ == '__main__':
    main()
