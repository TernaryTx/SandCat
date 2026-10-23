from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import netCDF4

NC_FILENAME = 'swish_output.nc'
NC_FILENAME_X = 'swish_x_output.nc'


def resolve_nc_path(run_dir: str) -> str:
    """Return the path to the NC file in run_dir, auto-detecting SWISH vs SWISH-X.

    Checks for swish_output.nc first, then swish_x_output.nc.

    Args:
        run_dir: Directory containing the NC file.

    Returns:
        Absolute path to the NC file.

    Raises:
        FileNotFoundError: If neither NC file is found in run_dir.
    """
    for filename in (NC_FILENAME, NC_FILENAME_X):
        candidate = Path(run_dir) / filename
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f'No NC file found in {run_dir!r}. '
        f'Expected {NC_FILENAME!r} or {NC_FILENAME_X!r}.'
    )


def compute_acceptance_rates(nc_path: str) -> None:
    """Read a SWISH NetCDF output file and log per-neighbour and overall swap acceptance rates.

    Args:
        nc_path: Path to the swish_output.nc file.
    """
    nc = netCDF4.Dataset(nc_path)
    accepted = nc.variables['accepted'][:]
    proposed = nc.variables['proposed'][:]
    nc.close()

    n_iterations = accepted.shape[0]
    n_states = accepted.shape[1]
    total_accepted = accepted.sum(axis=0)
    total_proposed = proposed.sum(axis=0)

    logging.info(f'Iterations: {n_iterations} | States: {n_states}')
    logging.info('Neighbour swap acceptance rates:')
    for idx in range(n_states - 1):
        prop = total_proposed[idx, idx + 1]
        acc = total_accepted[idx, idx + 1]
        rate = acc / prop if prop > 0 else float('nan')
        logging.info(f'  {idx} <-> {idx + 1}: {rate:.3f}  ({int(acc)}/{int(prop)})')

    overall_rate = total_accepted.sum() / total_proposed.sum()
    logging.info('Overall: %.3f', overall_rate)


def main(run_dir: str) -> None:
    """Compute and log SWISH replica swap acceptance rates from a run directory.

    Auto-detects whether the directory contains a SWISH or SWISH-X NC file and
    prints per-neighbour and overall swap acceptance rates to stdout. Healthy
    rates are 0.2-0.4; rates below 0.2 suggest the lambda spacing is too coarse.

    Args:
        run_dir: Directory containing the NC file.
    """
    nc_path = resolve_nc_path(run_dir)
    logging.info(f'Reading {nc_path}')
    compute_acceptance_rates(nc_path)


def parser():
    """Parse command-line arguments for the SWISH acceptance rate script."""
    arg_parser = ArgumentParser(description='Compute SWISH replica swap acceptance rates.')
    arg_parser.add_argument(
        '-d', '--run_dir',
        type=str,
        default='.',
        help='directory containing the NC file. Default: current directory.',
    )
    return arg_parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    args = parser()
    main(args.run_dir)
