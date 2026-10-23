# OPES_METAD on alanine dipeptide (vacuum)

Minimal example showing how to drive `sandcat.opes.OPES` on alanine
dipeptide in vacuum, then reweight the trajectory to recover the 1D and
2D free-energy surfaces along the backbone dihedrals.

## Files

- `build_diala.tleap` — tleap input that builds the AMBER topology and coords
- `diala.prmtop`, `diala.inpcrd`, `diala.pdb` — pre-built AMBER ff14SB inputs
  (regenerate with `tleap -f build_diala.tleap`)
- `run_diala_opes.py` — runs OPES_METAD and writes the kernel-based FES
  (via `opes.get_free_energy()`) as a PNG and an npz

## Running

1D FES along phi:

```
python run_diala_opes.py --cvs phi --total_ns 5
```

2D FES on the Ramachandran plane:

```
python run_diala_opes.py --cvs phi,psi --total_ns 10
```

Both invocations use OPES_METAD with a 40 kJ/mol barrier, pace = 500 steps,
2 fs timestep, T = 300 K. Defaults run on CPU (the simulation is small enough to be tolerable on a CPU). 1 ns took about 90s on an M3 Macbook.

Outputs:

- `diala_opes_fes_1d.png` / `.npz` for the 1D run
- `diala_opes_fes_2d.png` / `.npz` for the 2D run

## What the script does

1. Builds a `CustomTorsionForce` per CV whose potential energy *is* the
   dihedral angle in radians, wraps it in an `OpesVariable`, and hands the
   list to `sandcat.opes.OPES`.
2. Runs Langevin MD, calling `opes.step(simulation, n)` in chunks. OPES
   deposits a kernel every `pace` steps and tabulates the bias on a grid.
3. After the run, reads the FES directly off the deposited kernels via
   `opes.get_free_energy()`. This is dense on OPES's tabulation grid
   (200 bins per CV), so the 2D Ramachandran surface looks smooth even
   for short runs — unlike a histogram-reweighted FES, which would be
   sparse with only ~10³ frames over a 100×100 grid.
