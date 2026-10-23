# DES-Amber-SF1.0 → native OpenMM ForceField XML

The force field option **`desamber`** runs through the same
`SystemGenerator`/`Modeller.addSolvent` path as `ff14`/`ff19` — no GROMACS at runtime, no separate
prep job, no topology staging. (An earlier GROMACS-built variant that read `topol.top`/`conf.gro`
at runtime has been removed; this native XML is now the only DES-Amber path.)

| file | contents |
|---|---|
| `sandcat/forcefields/des-amber-SF1.0.xml` | 83 atom types, 134 residue templates (20 AAs + protonation variants + N/C termini + zwitterions + caps + ions), 432 bonds, 3056 angles, 3621 proper quartets, 841 improper types |
| `sandcat/forcefields/desamber_tip4pd.xml` | TIP4P-D 4-site water (HOH) with the M-site `average3` virtual site |
| `sandcat/forcefields/desamber_benzene.xml` | GAFF benzene probe, resname `PRB` |

## How the conversion works

* **comb-rule 2** is exactly OpenMM's Lorentz–Berthelot mixing → plain `NonbondedForce`, no NBFIX.
* **Per-residue charges** are carried with `<UseAttributeFromResidue name="charge"/>` + per-type
  σ/ε (the mechanism `amber/protein.ff19SB.xml` and `amber14/opc.xml` use). This keeps the type
  count at 83 instead of one type per residue-atom.
* `fudgeQQ 0.8333 / fudgeLJ 0.5` → `coulomb14scale=0.8333 / lj14scale=0.5`.
* **Proper dihedrals** (gmx funct 9): all terms for a quartet consolidated into one `<Proper>`.
* **Improper dihedrals** (gmx funct 4): the rtp lists the central atom *third*, but OpenMM's
  `_matchImproper` matches the central atom against `class1`, so the converter reorders
  `(c1,c2,c3,c4) → (c3,c1,c2,c4)`. **This reordering is the single thing that makes impropers
  correct** — without it, 428 impropers are dropped and 279 mis-typed (≈30 000 kJ/mol error).
* **TIP4P-D water**: `virtual_sites3 a=b=0.131938` → `ThreeParticleAverageSite` weights
  `(0.7361, 0.1319, 0.1319)`; solvated via `addSolvent(model='tip4pew')` (shared 4-site geometry),
  M-site then placed by the template virtual site (same trick `ff19` uses for OPC).
* **Benzene probe**: bundled as the `PRB` template; for `desamber` the probe is **not**
  registered as an OpenFF molecule, so OpenMM matches each inserted copy to the GAFF template by
  graph — exact parity with the GROMACS reference (no GAFF-version drift).

## Validation (GROMACS is the oracle)

**Per-term single-point energy, OpenMM-from-XML vs `gmx -rerun`, minimized TEM-1 frame (1jwp,
63 761 particles, ~17 400 torsions):**

| term (kJ/mol) | gmx -rerun | OpenMM ffxml | Δ |
|---|---:|---:|---:|
| Bond | 798.078 | 798.078 | 0.000 |
| Angle | 2734.426 | 2734.428 | 0.002 |
| Proper+Improper | −70011.094 | −70011.101 | −0.007 |
| LJ-14 | 4209.039 | 4209.036 | −0.003 |
| Coulomb-14 | 7846.796 | 7846.799 | 0.003 |
| Nonbonded (LJ+Coul) | −965871.3 | −965908.3 | −37.0 |
| **Potential** | **−1 020 294** | **−1 020 331** | **−37 (0.0036 %)** |

Bonded + 1-4 match exactly; the −37 kJ/mol nonbonded gap is the PME spline-order convention
residual (identical to the Route-1 GromacsTopFile PoC). Built on the *same* topology, the ffxml
System and the GromacsTopFile System agree to **2×10⁻⁴ kJ/mol** (a pure parameter check).

**Other checks**
* **Capped tripeptides** `ACE-X-NME`, ffxml vs GROMACS, vacuum: **24/25 residues exact** to
  0.0000 kJ/mol, including all protonation variants (HID, HIE, HIP, ASH, GLH, CYM). LYN cannot be
  built by `pdb2gmx` due to an hdb/rtp naming inconsistency in the source GROMACS port
  (`HZ1/HZ2` vs `HZ2/HZ3`) — a defect in the upstream files, not the conversion; the LYN template
  itself is complete.
* **All 134 templates** build a complete System (116 unique matches; CYX/CYM disambiguate by the
  disulfide `SG` external bond; NH2/NHE are graph-identical amide caps resolvable via
  `residueTemplates=` — not used by standard proteins).
* **Native solvation (no gmx)**: prep → `addHydrogens` → `addSolvent('tip4pew')` → `createSystem`
  yields one M-site per water; `build_mixed_solvent_system('desamber', …)` produces
  protein + TIP4P-D + benzene (PRB) with net charge 0.
* **MD stability**: 100 ps NPT on the solvated TEM-1 system (L40S, HMR 4 fs, rigid TIP4P-D) is
  stable — steady potential energy, 300 K, density ≈ 1.02 g/cc, no virtual-site blow-ups,
  ~540 ns/day.

## Caveats

* The single `MG–OW` NBFIX is not expressible in a standard `NonbondedForce` ffxml; it is omitted
  (irrelevant without Mg²⁺). Add a post-load `CustomNonbondedForce` patch if an Mg system is run.
* The XML was generated from the DES-Amber-SF1.0 GROMACS `.ff` by a one-off converter and is now
  vendored directly; the source `.ff` and converter are no longer kept in the repo.
