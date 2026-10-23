from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from sandcat.opes_expanded import (
    KB_KJ_PER_MOL_K,
    OPES_BIAS_FORCE_GROUP,
    OpesExpanded,
    add_opes_expanded_bias,
    build_opes_expanded_integrator,
)


# ACE.C, ALA.N, ALA.CA, ALA.C  /  ALA.N, ALA.CA, ALA.C, NME.N — diala.prmtop indices.
PHI_ATOMS = (4, 6, 8, 14)
PSI_ATOMS = (6, 8, 14, 16)


def _dihedral(positions_nm: np.ndarray, atoms: tuple[int, int, int, int]) -> float:
    """Signed dihedral in radians for the four-atom set (no PBC, vacuum)."""
    p0, p1, p2, p3 = positions_nm[list(atoms)]
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    # IUPAC convention: positive = clockwise looking along b1 (opposite to the
    # right-hand rule), so negate the atan2 result.
    return -float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _run_simulation(
        prmtop_path: str,
        inpcrd_path: str,
        temperature_k: float,
        t_max_k: float,
        n_temps: int,
        total_ns: float,
        timestep_fs: float,
        pace: int,
        observation_steps: int,
        record_stride: int,
        platform_name: str,
        seed: int,
) -> tuple[OpesExpanded, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run OPES_EXPANDED on diala in vacuum.

    Returns (opes, times_ps, U, bias_V, phi, psi).
    """
    prmtop = app.AmberPrmtopFile(prmtop_path)
    inpcrd = app.AmberInpcrdFile(inpcrd_path)
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

    add_opes_expanded_bias(system, n_temps=n_temps)

    opes = OpesExpanded(
        temperature=temperature_k * unit.kelvin,
        t_min=temperature_k * unit.kelvin,
        t_max=t_max_k * unit.kelvin,
        n_temps=n_temps,
        pace=pace,
        observation_steps=observation_steps,
    )

    integrator = build_opes_expanded_integrator(
        timestep=timestep_fs * unit.femtosecond,
        collision_rate=1.0 / unit.picosecond,
        temperature=temperature_k * unit.kelvin,
        n_temps=n_temps,
    )
    integrator.setRandomNumberSeed(seed)

    platform = mm.Platform.getPlatformByName(platform_name)
    simulation = app.Simulation(prmtop.topology, system, integrator, platform)
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)
    simulation.minimizeEnergy()
    opes.apply_to_context(simulation.context)

    n_steps = int(total_ns * 1e6 / timestep_fs)
    n_records = n_steps // record_stride
    times_ps = np.zeros(n_records)
    u_traj = np.zeros(n_records)
    bias_traj = np.zeros(n_records)
    phi_traj = np.zeros(n_records)
    psi_traj = np.zeros(n_records)

    ff_groups_mask = 0xFFFFFFFF & ~(1 << OPES_BIAS_FORCE_GROUP)
    for rec in range(n_records):
        opes.step(simulation, record_stride)
        times_ps[rec] = simulation.currentStep * timestep_fs / 1000.0
        ff_state = simulation.context.getState(getEnergy=True, groups=ff_groups_mask)
        u = ff_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        u_traj[rec] = u
        bias_traj[rec] = opes.compute_bias_v(u)
        positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        positions_nm = positions.value_in_unit(unit.nanometer)
        phi_traj[rec] = _dihedral(positions_nm, PHI_ATOMS)
        psi_traj[rec] = _dihedral(positions_nm, PSI_ATOMS)

    return opes, times_ps, u_traj, bias_traj, phi_traj, psi_traj


def _reweighted_deltaf(
        u_kj: np.ndarray, bias_kj: np.ndarray,
        temps_k: np.ndarray, t0_k: float,
) -> np.ndarray:
    """deltaF(T) = -kT_0 log <exp(-(beta_T - beta_0) U + beta_0 V)>.

    Per-frame log-weights computed in a numerically stable form (max-shift)
    so the average doesn't overflow for high-T points.
    """
    beta0 = 1.0 / (KB_KJ_PER_MOL_K * t0_k)
    deltaf = np.zeros_like(temps_k)
    for i, t_k in enumerate(temps_k):
        beta_t = 1.0 / (KB_KJ_PER_MOL_K * t_k)
        log_w = -(beta_t - beta0) * u_kj + beta0 * bias_kj
        m = log_w.max()
        # -kT_0 * log <w> = -kT_0 * (m + log mean(exp(log_w - m)))
        deltaf[i] = -1.0 / beta0 * (m + np.log(np.mean(np.exp(log_w - m))))
    deltaf -= deltaf[0]
    return deltaf


def _ramachandran_fes(
        phi: np.ndarray, psi: np.ndarray, log_w: np.ndarray, t0_k: float,
        n_bins: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D weighted FES on the Ramachandran plane at T_0.

    log_w[t] = beta_0 * V_t — reweights the multithermal sample back to
    the canonical T_0 ensemble. Returns (phi_centres, psi_centres, F_kj)
    with F shifted to min=0; empty bins are NaN.
    """
    weights = np.exp(log_w - log_w.max())
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _, _ = np.histogram2d(phi, psi, bins=[edges, edges], weights=weights)
    hist /= hist.sum() if hist.sum() > 0 else 1.0
    kT = KB_KJ_PER_MOL_K * t0_k
    with np.errstate(divide='ignore'):
        fes = -kT * np.log(hist)
    fes = np.where(np.isfinite(fes), fes, np.nan)
    if np.isfinite(fes).any():
        fes = fes - np.nanmin(fes)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, centres, fes


def _basin_deltaf(
        phi: np.ndarray, log_w: np.ndarray, t0_k: float,
) -> tuple[float, float, int, int]:
    """Free energy difference between negative-phi (C7eq + C5) and
    positive-phi (C7ax) basins via reweighted populations.

    Returns (deltaF_kj, sem_kj, n_neg, n_pos). deltaF > 0 means positive-phi
    is less populated (i.e. C7eq is the global minimum, as expected in
    vacuum diala).
    """
    weights = np.exp(log_w - log_w.max())
    neg_mask = phi < 0
    pos_mask = ~neg_mask
    w_neg = weights[neg_mask].sum()
    w_pos = weights[pos_mask].sum()
    if w_neg == 0 or w_pos == 0:
        return float('nan'), float('nan'), int(neg_mask.sum()), int(pos_mask.sum())
    kT = KB_KJ_PER_MOL_K * t0_k
    deltaf = -kT * np.log(w_pos / w_neg)
    # Crude SEM via block bootstrap on log-weights (~10 blocks).
    n = len(phi)
    n_blocks = max(2, min(10, n // 50))
    block_size = n // n_blocks
    boot = []
    rng = np.random.default_rng(0)
    for _ in range(200):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        sample_idx = np.concatenate([
            np.arange(b * block_size, (b + 1) * block_size) for b in idx
        ])
        wb = weights[sample_idx]
        pb = phi[sample_idx]
        wn = wb[pb < 0].sum()
        wp = wb[pb >= 0].sum()
        if wn > 0 and wp > 0:
            boot.append(-kT * np.log(wp / wn))
    sem = float(np.std(boot)) if boot else float('nan')
    return float(deltaf), sem, int(neg_mask.sum()), int(pos_mask.sum())


def _plot(
        times_ps: np.ndarray, u_traj: np.ndarray, bias_traj: np.ndarray,
        phi_traj: np.ndarray, psi_traj: np.ndarray,
        temps_k: np.ndarray, deltaf_opes: np.ndarray, deltaf_rew: np.ndarray,
        rama_phi: np.ndarray, rama_psi: np.ndarray, rama_fes: np.ndarray,
        deltaf_basin: float, sem_basin: float,
        out_path: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 0: timeseries
    axes[0, 0].plot(times_ps / 1000.0, u_traj, lw=0.5, color='C0')
    axes[0, 0].set_xlabel('time (ns)')
    axes[0, 0].set_ylabel('U (kJ/mol)')
    axes[0, 0].set_title('Potential energy timeseries')

    axes[0, 1].plot(times_ps / 1000.0, bias_traj, lw=0.5, color='C1')
    axes[0, 1].set_xlabel('time (ns)')
    axes[0, 1].set_ylabel('bias V (kJ/mol)')
    axes[0, 1].set_title('OPES_EXPANDED bias')

    axes[0, 2].plot(temps_k, deltaf_opes, 'o-', label='OPES deltaF', color='C0')
    axes[0, 2].plot(temps_k, deltaf_rew, 's--', label='reweighted', color='C2')
    axes[0, 2].set_xlabel('T (K)')
    axes[0, 2].set_ylabel('deltaF (kJ/mol)')
    axes[0, 2].legend()
    axes[0, 2].set_title('Multithermal deltaF(T) vs reweighted')

    # Row 1: dihedral timeseries + Ramachandran FES
    axes[1, 0].plot(times_ps / 1000.0, np.degrees(phi_traj), lw=0.5, color='C3')
    axes[1, 0].set_xlabel('time (ns)')
    axes[1, 0].set_ylabel(r'$\phi$ (deg)')
    axes[1, 0].set_ylim(-180, 180)
    axes[1, 0].axhline(0, color='k', lw=0.5, alpha=0.3)
    axes[1, 0].set_title(r'$\phi$ timeseries')

    axes[1, 1].plot(times_ps / 1000.0, np.degrees(psi_traj), lw=0.5, color='C4')
    axes[1, 1].set_xlabel('time (ns)')
    axes[1, 1].set_ylabel(r'$\psi$ (deg)')
    axes[1, 1].set_ylim(-180, 180)
    axes[1, 1].set_title(r'$\psi$ timeseries')

    fes_clip = np.where(np.isfinite(rama_fes), rama_fes, np.nan)
    fes_max = min(40.0, np.nanmax(fes_clip)) if np.isfinite(fes_clip).any() else 40.0
    levels = np.linspace(0, fes_max, 21)
    cf = axes[1, 2].contourf(
        np.degrees(rama_phi), np.degrees(rama_psi), fes_clip.T,
        levels=levels, cmap='viridis', extend='max',
    )
    fig.colorbar(cf, ax=axes[1, 2], label='F (kJ/mol)')
    axes[1, 2].axvline(0, color='w', lw=0.5, alpha=0.5)
    axes[1, 2].set_xlabel(r'$\phi$ (deg)')
    axes[1, 2].set_ylabel(r'$\psi$ (deg)')
    axes[1, 2].set_xlim(-180, 180)
    axes[1, 2].set_ylim(-180, 180)
    title = 'Reweighted FES at $T_0$'
    if np.isfinite(deltaf_basin):
        title += f'\n$\\Delta F$($\\phi$>0 - $\\phi$<0) = {deltaf_basin:+.2f} ± {sem_basin:.2f} kJ/mol'
    axes[1, 2].set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(
        prmtop: str,
        inpcrd: str,
        temperature: float,
        t_max: float,
        n_temps: int,
        pace: int,
        observation_steps: int,
        total_ns: float,
        timestep_fs: float,
        record_stride: int,
        platform: str,
        out_prefix: str,
        seed: int,
) -> None:
    """OPES_EXPANDED multithermal example on alanine dipeptide in vacuum.

    Sample a single trajectory at the thermostat temperature T_0, biased to
    also visit configurations characteristic of T_max via the OPES_EXPANDED
    log-sum-exp bias on the system potential energy. This is the multithermal
    counterpart of ``run_diala_opes.py`` (which biases on torsions) and uses
    the full ECV_MULTITHERMAL machinery — same single-replica recipe each
    SWISH-X replica runs internally.

    After the run, the script reweights the trajectory back to each
    temperature on the grid and reports both the OPES_EXPANDED ``deltaF`` and
    the reweighted ``deltaF_rew = -kT log <exp(-(beta_T - beta_0) U)>``
    from the recorded U trajectory. For T = T_0 both should be ~zero; the
    two estimators should agree to within sampling noise across the range.

    Usage:
        python run_diala_opes_expanded.py
        python run_diala_opes_expanded.py --total_ns 10 --t_max 1000 --n_temps 10
    """
    print(f'Running OPES_EXPANDED on diala vacuum: T_0={temperature} K, '
          f'T_max={t_max} K, N_T={n_temps}, total={total_ns} ns, pace={pace}')

    opes, times_ps, u_traj, bias_traj, phi_traj, psi_traj = _run_simulation(
        prmtop_path=prmtop,
        inpcrd_path=inpcrd,
        temperature_k=temperature,
        t_max_k=t_max,
        n_temps=n_temps,
        total_ns=total_ns,
        timestep_fs=timestep_fs,
        pace=pace,
        observation_steps=observation_steps,
        record_stride=record_stride,
        platform_name=platform,
        seed=seed,
    )

    diag = opes.get_diagnostics()
    temps_k = diag['temps_k']
    deltaf_opes = diag['deltaF_kj'].copy()
    deltaf_opes -= deltaf_opes[0]

    # Drop the unbiased observation phase from the reweighting window —
    # bias V is zero there and U is canonical at T_0, which biases the
    # reweighted average toward T_0 statistics.
    obs_end_step = pace * observation_steps
    obs_end_idx = np.searchsorted(times_ps, obs_end_step * timestep_fs / 1000.0)
    deltaf_rew = _reweighted_deltaf(
        u_traj[obs_end_idx:], bias_traj[obs_end_idx:], temps_k, temperature,
    )

    print(f'Temperature grid: {[round(t, 1) for t in temps_k]} K')
    print('  T (K)    deltaF_OPES (kJ/mol)    deltaF_rew (kJ/mol)    diff')
    for t_k, df_o, df_r in zip(temps_k, deltaf_opes, deltaf_rew):
        print(f'  {t_k:7.2f}  {df_o:+18.3f}     {df_r:+18.3f}     {df_o - df_r:+8.3f}')

    # Reweight the trajectory back to the canonical T_0 ensemble:
    # log_w = beta_0 * V_t.  Used both for the 2D Ramachandran FES and
    # the per-basin ΔF estimate.
    beta0 = 1.0 / (KB_KJ_PER_MOL_K * temperature)
    log_w = beta0 * bias_traj[obs_end_idx:]
    phi_post = phi_traj[obs_end_idx:]
    psi_post = psi_traj[obs_end_idx:]

    rama_phi, rama_psi, rama_fes = _ramachandran_fes(
        phi_post, psi_post, log_w, temperature,
    )
    deltaf_basin, sem_basin, n_neg, n_pos = _basin_deltaf(
        phi_post, log_w, temperature,
    )
    print(f'\nRamachandran basin populations (post-warmup): '
          f'phi<0 (C7eq+C5): {n_neg} frames, phi>0 (C7ax): {n_pos} frames')
    if np.isfinite(deltaf_basin):
        print(f'Reweighted ΔF(phi>0 - phi<0) at T_0={temperature} K: '
              f'{deltaf_basin:+.2f} ± {sem_basin:.2f} kJ/mol')
    else:
        print('Reweighted basin ΔF: N/A (one basin unsampled — try a longer run '
              'or a higher t_max)')

    np.savez(
        f'{out_prefix}.npz',
        times_ps=times_ps,
        U_kj=u_traj,
        bias_kj=bias_traj,
        phi_rad=phi_traj,
        psi_rad=psi_traj,
        temps_k=temps_k,
        deltaF_opes_kj=deltaf_opes,
        deltaF_rew_kj=deltaf_rew,
        rama_phi_rad=rama_phi,
        rama_psi_rad=rama_psi,
        rama_fes_kj=rama_fes,
        deltaF_basin_kj=deltaf_basin,
        deltaF_basin_sem_kj=sem_basin,
    )
    _plot(times_ps, u_traj, bias_traj, phi_traj, psi_traj,
          temps_k, deltaf_opes, deltaf_rew,
          rama_phi, rama_psi, rama_fes, deltaf_basin, sem_basin,
          f'{out_prefix}.png')
    print(f'Wrote {out_prefix}.npz and {out_prefix}.png')


def parser() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        description='OPES_EXPANDED multithermal example on alanine dipeptide.',
    )
    here = Path(__file__).resolve().parent
    arg_parser.add_argument('--prmtop', default=str(here / 'diala.prmtop'))
    arg_parser.add_argument('--inpcrd', default=str(here / 'diala.inpcrd'))
    arg_parser.add_argument('--temperature', type=float, default=300.0, help='K (T_0)')
    arg_parser.add_argument('--t_max', type=float, default=700.0, help='K (T_max)')
    arg_parser.add_argument('--n_temps', type=int, default=10)
    arg_parser.add_argument('--pace', type=int, default=500)
    arg_parser.add_argument('--observation_steps', type=int, default=20,
                            help='unbiased warmup window in PACE units')
    arg_parser.add_argument('--total_ns', type=float, default=5.0)
    arg_parser.add_argument('--timestep_fs', type=float, default=4.0)
    arg_parser.add_argument('--record_stride', type=int, default=500,
                            help='steps between (U, V) samples saved')
    arg_parser.add_argument('--platform', default='CPU',
                            choices=['CUDA', 'CPU', 'Reference', 'OpenCL'])
    arg_parser.add_argument('--out_prefix', default='diala_opes_expanded')
    arg_parser.add_argument('--seed', type=int, default=1337)
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parser()
    main(
        prmtop=args.prmtop,
        inpcrd=args.inpcrd,
        temperature=args.temperature,
        t_max=args.t_max,
        n_temps=args.n_temps,
        pace=args.pace,
        observation_steps=args.observation_steps,
        total_ns=args.total_ns,
        timestep_fs=args.timestep_fs,
        record_stride=args.record_stride,
        platform=args.platform,
        out_prefix=args.out_prefix,
        seed=args.seed,
    )
