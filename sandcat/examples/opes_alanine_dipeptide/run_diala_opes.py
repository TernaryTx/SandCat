from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from sandcat.opes import METAD, OPES, OpesVariable

# ACE.C, ALA.N, ALA.CA, ALA.C — atom indices in diala.prmtop (0-based).
PHI_ATOMS = (4, 6, 8, 14)
# ALA.N, ALA.CA, ALA.C, NME.N
PSI_ATOMS = (6, 8, 14, 16)


def _torsion_force(atoms: tuple[int, int, int, int]) -> mm.CustomTorsionForce:
    """Return a CustomTorsionForce whose energy is the dihedral angle in radians.

    The Force's potential energy is the CV value, matching the
    openmm.app.metadynamics.BiasVariable / sandcat OpesVariable convention.
    """
    force = mm.CustomTorsionForce('theta')
    force.addTorsion(*atoms, [])
    return force


def _build_variables(cv_names: list[str]) -> list[OpesVariable]:
    atoms_map = {'phi': PHI_ATOMS, 'psi': PSI_ATOMS}
    return [
        OpesVariable(
            force=_torsion_force(atoms_map[name]),
            min_value=-np.pi, max_value=np.pi, periodic=True, grid_width=200,
        )
        for name in cv_names
    ]


def _run_simulation(
        prmtop_path: str,
        inpcrd_path: str,
        cv_names: list[str],
        temperature_k: float,
        total_ns: float,
        timestep_fs: float,
        pace: int,
        barrier_kj: float,
        print_stride: int,
        platform_name: str,
        seed: int,
) -> OPES:
    """Run OPES_METAD on diala in vacuum; return the OPES driver after the run."""
    prmtop = app.AmberPrmtopFile(prmtop_path)
    inpcrd = app.AmberInpcrdFile(inpcrd_path)
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

    variables = _build_variables(cv_names)
    opes = OPES(
        system=system,
        variables=variables,
        temperature=temperature_k * unit.kelvin,
        barrier=barrier_kj * unit.kilojoules_per_mole,
        pace=pace,
        mode=METAD,
    )

    integrator = mm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(seed)

    platform = mm.Platform.getPlatformByName(platform_name)
    simulation = app.Simulation(prmtop.topology, system, integrator, platform)
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)
    simulation.minimizeEnergy()

    n_steps = int(total_ns * 1e6 / timestep_fs)
    n_chunks = max(1, n_steps // print_stride)
    for _ in range(n_chunks):
        opes.step(simulation, print_stride)

    return opes


def _grid_axis(variable) -> np.ndarray:
    """Bin centres along the OpesVariable's tabulation grid."""
    edges = np.linspace(variable.min_value, variable.max_value, variable.grid_width + 1)
    return 0.5 * (edges[1:] + edges[:-1])


def _plot_1d(centres: np.ndarray, fes: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.degrees(centres), fes, lw=2)
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel('FES (kJ/mol)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, np.nanmin([np.nanmax(fes), 60.0]))
    ax.set_title('Alanine dipeptide vacuum: OPES-reweighted FES along $\\phi$')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_2d(cx: np.ndarray, cy: np.ndarray, fes: np.ndarray, out_path: str) -> None:
    fes_clipped = np.where(np.isfinite(fes), fes, np.nan)
    fes_max = min(60.0, np.nanmax(fes_clipped))
    fig, ax = plt.subplots(figsize=(6, 5))
    levels = np.linspace(0, fes_max, 21)
    cf = ax.contourf(
        np.degrees(cx), np.degrees(cy), fes_clipped.T,
        levels=levels, cmap='viridis', extend='max',
    )
    fig.colorbar(cf, ax=ax, label='FES (kJ/mol)')
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel(r'$\psi$ (deg)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_title('Alanine dipeptide vacuum: OPES-reweighted FES on Ramachandran')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parser() -> argparse.Namespace:
    """Parse command-line arguments for the diala OPES example."""
    here = Path(__file__).resolve().parent
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--prmtop', default=str(here / 'diala.prmtop'))
    arg_parser.add_argument('--inpcrd', default=str(here / 'diala.inpcrd'))
    arg_parser.add_argument('--cvs', default='phi,psi', choices=['phi', 'phi,psi'])
    arg_parser.add_argument('--temperature', type=float, default=300.0, help='K')
    arg_parser.add_argument('--barrier', type=float, default=40.0, help='kJ/mol')
    arg_parser.add_argument('--pace', type=int, default=500)
    arg_parser.add_argument('--total_ns', type=float, default=5.0)
    arg_parser.add_argument('--timestep_fs', type=float, default=2.0)
    arg_parser.add_argument('--print_stride', type=int, default=500)
    arg_parser.add_argument('--platform', default='CPU',
                            choices=['CUDA', 'CPU', 'Reference', 'OpenCL'])
    arg_parser.add_argument('--out_prefix', default='diala_opes')
    arg_parser.add_argument('--seed', type=int, default=42)
    return arg_parser.parse_args()


def main(
        prmtop: str,
        inpcrd: str,
        cvs: str,
        temperature: float,
        barrier: float,
        pace: int,
        total_ns: float,
        timestep_fs: float,
        print_stride: int,
        platform: str,
        out_prefix: str,
        seed: int,
) -> None:
    """Run OPES_METAD on alanine dipeptide in vacuum and plot the resulting FES.

    Single CV (phi) by default → 1D FES; pass ``--cvs phi,psi`` for the 2D FES.
    The script runs a Langevin MD simulation in vacuum with the OPES_METAD bias
    applied to the chosen torsion(s), reweights the trajectory by exp(beta*V),
    histograms it, and writes the FES as a PNG and an npz.

    Usage:
        python run_diala_opes.py --cvs phi --total_ns 5
        python run_diala_opes.py --cvs phi,psi --total_ns 10
    """
    cv_names = cvs.split(',')
    print(f'Running OPES_METAD on diala vacuum: cvs={cv_names}, '
          f'total={total_ns} ns, T={temperature} K, '
          f'barrier={barrier} kJ/mol, pace={pace}')

    opes = _run_simulation(
        prmtop_path=prmtop,
        inpcrd_path=inpcrd,
        cv_names=cv_names,
        temperature_k=temperature,
        total_ns=total_ns,
        timestep_fs=timestep_fs,
        pace=pace,
        barrier_kj=barrier,
        print_stride=print_stride,
        platform_name=platform,
        seed=seed,
    )

    # FES from the deposited kernel sum: dense and smooth on the OPES tabulation grid.
    fes = opes.get_free_energy()
    if len(cv_names) == 1:
        centres = _grid_axis(opes.variables[0])
        np.savez(f'{out_prefix}_fes_1d.npz', cv=centres, fes_kj=fes)
        _plot_1d(centres, fes, f'{out_prefix}_fes_1d.png')
        print(f'Wrote {out_prefix}_fes_1d.png and .npz')
    else:
        cx = _grid_axis(opes.variables[0])
        cy = _grid_axis(opes.variables[1])
        np.savez(f'{out_prefix}_fes_2d.npz', cv_x=cx, cv_y=cy, fes_kj=fes)
        _plot_2d(cx, cy, fes, f'{out_prefix}_fes_2d.png')
        print(f'Wrote {out_prefix}_fes_2d.png and .npz')


if __name__ == '__main__':
    args = parser()
    main(
        prmtop=args.prmtop,
        inpcrd=args.inpcrd,
        cvs=args.cvs,
        temperature=args.temperature,
        barrier=args.barrier,
        pace=args.pace,
        total_ns=args.total_ns,
        timestep_fs=args.timestep_fs,
        print_stride=args.print_stride,
        platform=args.platform,
        out_prefix=args.out_prefix,
        seed=args.seed,
    )
