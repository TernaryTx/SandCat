"""Run the whole prospective analysis, from the raw trajectories to the figures.

The stage order is not recoverable from the imports: run_dynamics_all and
residue_ranking are each imported as a library by scripts downstream of them AND
required as a stage that writes files those scripts read, so importing one is not
the same as having run it. A run that skips residue_ranking fails on a missing
ranking_residues.csv two stages later, which is the failure this script exists to
make impossible.

Stages, in dependency order:

  1. density            probe density maps and the candidate tables, from the
                        raw unbiased and demuxed SWISH-X trajectories
  2. dynamics           per-residue span of closest approach at each candidate
                        -> dynamics/lambda1/<T>/pocket_dynamics.csv
  3. ranking            the blind residue ranking
                        -> ranking_residues.csv, ranking_regions.csv
  4-8. figures          the panels and the open/closed snapshots

Every parameter is pinned in the stage modules rather than here, so this script
decides only what runs and in what order.

Usage:
    python run_prospective_analysis_pipeline.py <output_dir> [--skip-density]

The output directory is exported as SANDCAT_ANALYSIS_DIR for the stages that
read it through experiments.ANALYSIS, so a run can be pointed at a clean tree and
compared against an existing one. --skip-density reuses the maps already in the
output directory, which is what you want when iterating on a figure.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ('build_density_maps', 'density maps and candidate tables'),
    ('run_dynamics_all', 'pocket_dynamics.csv and one open frame per candidate'),
    ('residue_ranking', 'ranking_residues.csv, ranking_regions.csv'),
    ('plot_ranking_figure', 'site_rank.png'),
    ('plot_site_residue_figure', 'residue_from_site_per_target/, site_residue_ranks.csv'),
    ('snapshot_frames', 'snapshots/ and snapshot_residue_changes.csv'),
    ('plot_candidate_residues', 'prospective_candidates/ and its csv'),
    ('make_overlay_pmls', 'pymol/<target>_density_overlay.pml and its maps'),
]


def run_stage(module: str, out: Path, env: dict) -> None:
    """Run one stage as a subprocess, stopping the run if it fails.

    A stage is run rather than imported so that each gets a clean interpreter
    and the failure of one cannot leave a half-initialised module behind for the
    next.

    Args:
        module: Module name in this directory, without the .py suffix.
        out: Output directory. Passed as an argument to the density stage, which
            takes it positionally; every other stage reads it from the
            environment instead.
        env: Environment for the subprocess, carrying SANDCAT_ANALYSIS_DIR.

    Raises:
        SystemExit: If the stage exits non-zero. The stages after it read what
            it writes, so the run stops rather than building figures from a
            half-made tree.
    """
    command = [sys.executable, '-m', f'sandcat.analysis.{module}']
    if module == 'build_density_maps':
        command.append(str(out))
    started = time.time()
    result = subprocess.run(command, env=env)
    if result.returncode != 0:
        raise SystemExit(
            f'\n{module} failed with exit code {result.returncode}. '
            f'The stages after it read what it writes, so the run stops here '
            f'rather than producing figures from a half-built tree.')
    print(f'--- {module}: {time.time() - started:.0f}s', flush=True)


def parse_args() -> ArgumentParser:
    """Create the benchmark pipeline argument parser.

    Returns:
        Parser for output, input-root and density-reuse options.
    """
    parser = ArgumentParser(description='Run the prospective benchmark analysis stages.')
    parser.add_argument('output_dir', type=Path, help='Analysis output directory.')
    parser.add_argument(
        '--paper-dir', type=Path,
        default=Path(os.environ.get('SANDCAT_PAPER_DIR', Path.cwd())),
        help='Root containing september2026/ and september2026_ref_pockets/. '
             'Defaults to SANDCAT_PAPER_DIR or the current directory.',
    )
    parser.add_argument('--skip-density', action='store_true',
                        help='Reuse density outputs already in output_dir.')
    return parser


def main() -> None:
    """Run the published benchmark stages into a caller-selected output tree.

    Inputs are local trajectories and held-out benchmark references beneath
    the paper-data root. Each subprocess receives absolute input and output
    roots. Stop at the first failed stage rather than consuming partial results.
    """
    parser = parse_args()
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    paper_dir = args.paper_dir.expanduser().resolve()
    for required in ('september2026', 'september2026_ref_pockets'):
        if not (paper_dir / required).is_dir():
            parser.error(f'Missing benchmark input directory: {paper_dir / required}')
    out.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, SANDCAT_ANALYSIS_DIR=str(out),
               SANDCAT_PAPER_DIR=str(paper_dir))
    stages = [s for s in STAGES
              if not (args.skip_density and s[0] == 'build_density_maps')]

    print(f'output directory: {out}')
    print(f'{len(stages)} stages\n')
    started = time.time()
    for index, (module, produces) in enumerate(stages, start=1):
        print(f'===== {index}/{len(stages)}  {module}  ->  {produces}',
              flush=True)
        run_stage(module, out, env)
    print(f'\nALL DONE in {(time.time() - started) / 60:.1f} min -> {out}')


if __name__ == '__main__':
    main()
