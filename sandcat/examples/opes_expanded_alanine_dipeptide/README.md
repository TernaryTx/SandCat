# OPES_EXPANDED (multithermal) on alanine dipeptide (vacuum)

Minimal example driving `sandcat.opes_expanded.OpesExpanded` on alanine
dipeptide in vacuum. Unlike the OPES_METAD example (which biases on the
backbone dihedrals), this one biases on the **system potential energy U**
to sample an *expanded multithermal ensemble* in a single trajectory at
the thermostat temperature `T_0 = 300 K`, covering everything that would
normally need to be sampled across `[T_0, T_max]`.

This is exactly the per-replica recipe each SWISH-X replica runs
internally (Borsatto et al., J. Chem. Theory Comput. 20, 3335, 2024),
following Invernizzi, Piaggi & Parrinello (Phys. Rev. X 10, 041034, 2020).

## What "multithermal sampling" means here

The bias is a closed-form log-sum-exp over a discrete temperature grid
`{T_j}` between `T_min` and `T_max`:

```
V(U) = -kT_0 log[(1/N_T) Σ_j exp(-(β_j - β_0) U + β_0 ΔF(T_j))]
```

The running `ΔF(T_j)` values are updated each `PACE` MD steps by
self-consistent reweighting (Eq. 8 of Invernizzi 2020). Effects:

- **The marginal P(U) broadens** — at `T_0 = 300 K` you now sample U
  configurations that would normally only appear at higher `T` up to
  `T_max`. Easy to see in the `U(t)` panel of the output plot.
- **Any observable can be reweighted to any T in `[T_min, T_max]`**
  using log-weight `-(β_T - β_0) U + β_0 V` per frame. The script does
  exactly this to compute `ΔF_rew(T)` as an internal cross-check.
- **For diala in vacuum** (a tiny 22-atom system) the multithermal
  trick is overkill on `U` itself — the script is a *validation
  example*: the OPES-tracked `ΔF(T)` and the post-hoc reweighted
  `ΔF_rew(T)` should agree across the T range, confirming the bias is
  correctly inverting back to canonical statistics. On real systems
  (cryptic-pocket sampling in TEM-1, SWISH-X) the same machinery
  accelerates rare conformational events by ~5x.
- **Does it accelerate phi/psi sampling?** Indirectly, yes — but mildly,
  because the phi-flip barrier is only weakly coupled to total `U`.
  Empirically (5 ns, `--t_max 500`, 'mild' settings): plain Langevin
  at 300 K never leaves the positive-phi C7ax basin; OPES at
  `T_max = 500 K` likewise tends to stay put; **`T_max = 700 K` is
  enough to see a single recrossing into the negative-phi C7eq basin
  in 5 ns** (the global minimum in vacuum). One crossing is enough to
  produce a finite reweighted basin `ΔF` with the correct sign
  (negative — C7eq more populated than C7ax by ~5–6 kJ/mol, consistent
  with literature for ff14SB vacuum diala) but with a large SEM. To
  *converge* `ΔF`, push `--t_max 800` or longer trajectories so multiple
  independent crossings accumulate.

## Files

- `diala.prmtop`, `diala.inpcrd`, `diala.pdb` — symlinks to the AMBER
  ff14SB inputs in `../opes_alanine_dipeptide/`
- `run_diala_opes_expanded.py` — runs OPES_EXPANDED and writes a PNG
  with `U(t)`, `V(t)`, and `ΔF_OPES` vs. `ΔF_rew` over the T grid

## Running

```
python run_diala_opes_expanded.py
```

Defaults: `T_0 = 300 K`, `T_max = 700 K`, `N_T = 10`, 4 ns total at 4 fs,
`PACE = 500`, 20 PACE-blocks (10 ps) of unbiased warmup before the bias
activates. Runs on CPU — about 5 minutes on an M3 MacBook.

Useful knobs:

- `--total_ns 5` — longer runs tighten the agreement between `ΔF_OPES`
  and `ΔF_rew`. At 50 ps both estimates agree to within ~0.5 kJ/mol on
  a 9 kJ/mol scale; 2 ns brings it under 0.1 kJ/mol.
- `--t_max 1000 --n_temps 20` — wider range needs more grid points so
  adjacent `Δβ` gaps stay close enough that the LSE doesn't collapse
  onto one slot.
- `--observation_steps 5` — shortens the unbiased warmup. For a stiff
  solvated system this would be too short; for diala vacuum it's fine.

## Outputs

- `diala_opes_expanded.npz` — `times_ps`, `U_kj`, `bias_kj`, `phi_rad`,
  `psi_rad`, `temps_k`, `deltaF_opes_kj`, `deltaF_rew_kj`,
  `rama_phi_rad`, `rama_psi_rad`, `rama_fes_kj`, `deltaF_basin_kj`,
  `deltaF_basin_sem_kj`
- `diala_opes_expanded.png` — 2x3 grid:
  - **Top row** — `U(t)`, `V(t)`, and `ΔF(T)` (OPES vs reweighted overlay)
  - **Bottom row** — `phi(t)`, `psi(t)`, and the reweighted Ramachandran
    FES at `T_0` with the basin `ΔF(phi>0 - phi<0)` printed in the title
- The phi timeseries is the most informative panel for whether the run
  is long enough / hot enough — if phi never crosses zero, the basin
  `ΔF` is undefined regardless of how clean `ΔF(T)` looks. With
  `--t_max 700 --total_ns 5` you should see at least one excursion into
  negative-phi territory; the Ramachandran FES will then show a partly
  sampled C7eq basin to the left of `phi = 0`.

## How this differs from the OPES_METAD example

| | `opes_alanine_dipeptide/` | `opes_expanded_alanine_dipeptide/` |
|---|---|---|
| CV | dihedrals (phi, psi) | system potential energy U |
| Bias form | sum of Gaussian kernels | closed-form LSE over T grid |
| Module | `sandcat.opes` | `sandcat.opes_expanded` |
| Output | FES F(phi[, psi]) | ΔF(T) + reweightable U trajectory |
| Use case | rare-event sampling on chosen CVs | multithermal / replica-exchange-free temperature sampling |
