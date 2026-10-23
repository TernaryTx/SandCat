# SandCat - Cryptic Pocket Discovery using Molecular Dynamics

`SandCat` is a validated OpenMM implementation of SWISH-X for discovering
previously unseen ('cryptic') pockets in proteins. SWISH-X combines SWISH
(Sampling Water Interfaces through Scaled Hamiltonians), introduced by
[Oleinikovas et al., 2016](https://pubs.acs.org/doi/10.1021/jacs.6b05425),
with OPES Multithermal enhanced sampling, following
[Borsatto et al., 2024](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c01318).
The implementation has been validated and is the primary workflow provided by
this package.

The original OpenMM SWISH port is retained as a reference implementation and
for comparison with SWISH-X. SandCat also includes a basic parallel-tempering
workflow, although parallel tempering alone has not proved effective for
cryptic-pocket discovery.

# Reproducing the paper results

To reproduce the results reported in the SandCat paper, use the provided
configuration (`.cfg`) files in the shared `data` directory. These files record
the simulation parameters used for the reported runs.

# Installation

`SandCat` runs in a conda environment with the OpenMM / OpenFF molecular
dynamics stack (`openmm`, `rdkit`, `openff-toolkit`, `openmmforcefields`,
AmberTools/`tleap`, `propka`, `mdtraj`) plus `openmmtools`. Create the
environment, install `openmmtools`, then install this package:

```commandline
conda install -c conda-forge openmmtools
pip install -e .
```

# Usage

## Command line 

### SWISH

Run a SWISH simulation on a local machine:

```commandline
mpirun -np 8 python sandcat/pipeline/run_swish.py \
    -p 1jwp.pdb \
    --probe benzene \
    --concentration 1.0 \
    --n_replicas 8 \
    --scale_max 1.35 \
    -t 250 \
    --verbose
```

`--probe` accepts the built-in `benzene` and `pyrimidine` probes or a path to
an SDF containing one molecule with 3D coordinates. Pass a corresponding
SMIRNOFF file with `--probe_ff_path ligand.offxml` when using bespoke ligand
parameters. The same single-probe options are available for SWISH-X and
unbiased mixed-solvent MD.

Use 1 M for small cosolvents such as benzene. For larger drug-like probes,
approximately 0.025 M is a useful starting point to reduce crowding and
nonspecific protein-surface saturation. The appropriate concentration remains
system-dependent, so inspect binding and unbinding behaviour and run independent
replicates.

`--n_replicas` must be even and defaults to 6. Larger replica counts reduce the
lambda spacing and can improve neighbouring exchange acceptance at additional
compute cost. The MD interval between exchange attempts defaults to 10 ps and
can be changed with `--exchange_interval_ps`.

SWISH, SWISH-X, and unbiased mixed-solvent MD support `--ensemble NPT` (the
default) or `--ensemble NVT`. Available protein/water combinations are
`desamber` (DES-Amber-SF1.0 + TIP4P-D), `ff14` (AMBER ff14SB + TIP3P), and
`ff19` (AMBER ff19SB + OPC).

If you know the binding pocket centre, pass it to exclude nearby residues from the contact
map upper wall (NOTE - units in nm):

```commandline
mpirun -np 8 python sandcat/pipeline/run_swish.py \
    -p 1jwp.pdb \
    --pocket_centre 2.3 1.8 2.9 \
    --verbose
```

### SWISH-X

SWISH-X combines SWISH with OPES Multithermal enhanced sampling. SWISH changes
the strength of water-apolar protein interactions across a replica ladder,
making it easier for water to penetrate hydrophobic regions and stabilise
transiently open pockets. OPES Multithermal adds a history-dependent bias on
the system's potential energy, encouraging broader thermal and conformational
fluctuations within every replica.

The two methods target complementary barriers. SWISH primarily promotes
wetting and hydration changes, while OPES helps the protein and solvent move
between conformational states that may otherwise remain separated on ordinary
MD timescales. Compared with standard SWISH, SWISH-X can therefore produce
more frequent pocket-opening and pocket-closing transitions and explore a
broader conformational ensemble. It does this without requiring a
pocket-specific collective variable, which makes it suitable for prospective
cryptic-pocket discovery where the location or opening motion is not already
known.

Run SWISH-X in the same way as SWISH, with the additional OPES temperature
range and grid controls:

```commandline
mpirun -np 6 python sandcat/pipeline/run_swish_x.py \
    -p 1jwp.pdb \
    --probe benzene \
    --concentration 1.0 \
    --n_replicas 6 \
    --scale_max 1.35 \
    --opes_t_max 350 \
    --opes_n_temps 30 \
    -t 250 \
    --verbose
```

Fresh runs write a versioned `<output_prefix>_provenance.json` beside the
NetCDF reporters. It records the command, resolved simulation parameters,
SandCat commit, software versions, and checksums for local inputs. Keep it
with the NetCDF files because restarts validate the recorded Hamiltonian and
lifecycle revision.

`--colvar_interval_ps` controls buffered PLUMED-style per-state `COLVAR` output
(default 2 ps). DeltaF and COLVAR rows use human-readable PLUMED formatting.
With `--verbose`, component-level exchange decisions are also written to
`<output_prefix>_exchange_diagnostics.jsonl`.

### Unbiased mixed-solvent MD

Use unbiased mixed-solvent MD to generate a probe-density reference or run a
conventional simulation with the same system construction used by SWISH:

```commandline
python sandcat/pipeline/run_unbiased_md.py \
    -p protein.pdb --probe benzene --concentration 1.0 -t 150
```

To simulate several different drug-like ligands together, provide ordered SDF
and OFFXML lists. SandCat inserts exactly one copy of each ligand, so do not pass
`--concentration` in this mode:

```commandline
python sandcat/pipeline/run_unbiased_md.py \
    -p protein.pdb \
    --probe ligand_a.sdf ligand_b.sdf \
    --probe_ff_path ligand_a.offxml ligand_b.offxml \
    -t 150
```

The SDF and OFFXML positions are paired. Multi-ligand residues are named `P00`,
`P01`, and so on in input order, and the mapping is recorded in provenance.
SandCat applies probe-probe repulsion between distinct probe residues, including
different ligand types, to discourage aggregation and simultaneous occupation
of the same confined site.

### Configuration files

All runnable `pipeline/run_*.py` commands accept `-c/--config <file>`. The
format is one `key = value` per line; `#` starts a comment, and explicit command
line arguments override values read from the file. For example:

```ini
# swish.cfg
protein = protein.pdb
probe = benzene
n_replicas = 6
scale_max = 1.35
total_simtime_ns = 250
exchange_interval_ps = 10
forcefield = desamber
ensemble = NPT
```

```commandline
mpirun -np 6 python sandcat/pipeline/run_swish.py --config swish.cfg
```

### Parallel Tempering

Run a basic on-the-machine PT simulation like this:

```commandline
python sandcat/pipeline/run_pt.py -p protein.pdb -t 1 --n_replicas 2 -ah False --verbose
```

It should take ~5 minutes on an L4. If you demux the `pt_output.nc`, you'll see that `state1`, 
the high temperature replica, shows the protein to be fairly distorted, indicating something like
melting at high temperature.

The strength of replica exchange methods is the ability to run those replicas in parallel. This is 
made easy by CUDA MPS and MPI processes:

```commandline
mpirun -np 2 python sandcat/pipeline/run_pt.py -p protein.pdb -t 1 --n_replicas 2 -ah False --verbose
```

## Processing & Analysis

`Openmmtools` writes a single, multiplexed trajectory file. To begin the analysis, 
you need demultiplex ('demux') the trajectories, writing one for each thermodynamic state.

Use `demux_traj.py` for that, with something like:

```commandline
python sandcat/pipeline/demux_traj.py -nc pt_test/pt_output.nc -top pt_test/equil.pdb
```

It will write out a separate NC file for each 'state', with *all* atoms, which you can analyse or 
visualise further.

### Known pocket analysis

The pocket exposure calculations (`analyse_known_pocket.py`) are done with `mdpocket`, installed as part of `fpocket` 4.0.

`analyse_known_pocket.py` aligns each replica trajectory to a common reference frame, runs `mdpocket --selected_pocket` to track the cryptic site across all frames, and computes two exposure metrics as defined in the SWISH and SWISH-X papers:

- **PLA-based**: `P.E. / 100% = |PLAf_pocket ∩ PLAligand| / |PLAligand|`, where PLAligand is the set of protein heavy atoms within 0.45 nm of the cryptic-site ligand in the holo structure (active-site ligand excluded).
- **Volume-based**: `P.E. = V_pocket(t) / V_holo × 100%`, where V_holo is the cryptic-site pocket volume in the holo crystal measured by `mdpocket`.

Required inputs are the per-replica trajectory DCDs and their reference PDBs, an alignment reference PDB (e.g. apo crystal structure), a pocket reference PDB with dummy atoms defining the cryptic site, and a holo structure PDB containing the cryptic-site ligand — all pre-aligned to the same coordinate frame.

The `pocket` argument requires a PDB file with the pocket dummy atoms, generated with `mdpocket` like so `mdpocket --pdb_list pdb_list_file`, with `pdb_list_file` containing the reference holo structure (e.g. 1PZO).

`--ligand_resnum 301` is specific for TEM-1. It specifies the cryptic-site ligand in 1PZO. 1PZO contains two CBT molecules: resnum 300 is bound to the active site cavity already present in the apo structure (1JWP) and is excluded per the SWISH paper, whereas resnum 301 is bound exclusively to the cryptic site and is used to define PLAligand. 

Example reference files for TEM-1 (1JWP apo / 1PZO holo) are in `sandcat/examples/reference_pdbs/`.

```commandline
python sandcat/analyse_known_pocket.py \
    -d /path/to/swish_output \
    -ref sandcat/examples/reference_pdbs/1jwp_paper_ref.pdb \
    -pocket sandcat/examples/reference_pdbs/pocket_ref_from_swish_paper.pdb \
    -o /path/to/output_dir \
    --traj_prefix state_state \
    --n_replicas 8 \
    --stride 10 \
    --fit_selection 'name CA' \
    --lambda_values 1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.35 \
    --holo_pdb sandcat/examples/reference_pdbs/1pzo_paper_ref.pdb \
    --ligand_resnum 301
```

Outputs are violin plots (`exposure_pla.png`, `exposure_vol.png`) and raw arrays (`exposure_pla.npy`, `exposure_vol.npy`) saved to `output_dir`.

### Prospective pocket analysis

The prospective detector now uses connected components of the **demuxed replica 0
SWISH-X minus unbiased MD density**, normalized to each system's bulk probe
density using its actual unit-cell volume. Candidates must exceed **6× bulk**
and contain at least **3 voxels**; they are ranked by summed density.
This replaces the clustering/enrichment-scoring workflow.

Both trajectories must retain benzene probes (residue name `PRB`). Align to the
apo input, not a known open-pocket structure. For a single target:

```bash
python -m sandcat.analysis.compute_probe_density \
    --unbiased_top /path/to/md_output_last_frame.pdb \
    --unbiased_dcd /path/to/aligned_md_output.dcd \
    --swish_dir /path/to/demux \
    --traj_prefix state_state \
    --ref_pdb /path/to/apo_input.pdb \
    --output_dir /path/to/analysis/density \
    --n_replicas 6 --stride 10
```

For independent unbiased replicates, call the Python API with one
`(topology, trajectory)` pair per replicate. They are normalized separately
before averaging, so differing probe counts are supported:

```python
import os
from pathlib import Path

from sandcat.analysis.compute_probe_density import main as compute_density
from sandcat.analysis.analyse_pocket_dynamics import run as analyse_dynamics

os.environ["SANDCAT_ANALYSIS_DIR"] = str(Path("analysis").resolve())

compute_density(
    md_pairs=[("md0_top.pdb", "md0.dcd"), ("md1_top.pdb", "md1.dcd")],
    swish_dir="demux",
    swish_top_prefix="state_state",
    traj_prefix="state_state",
    ref_pdb="apo_input.pdb",
    output_dir="analysis/density",
    n_replicas=6,
    stride=10,
)
analyse_dynamics(
    density_dir="analysis/density",
    output_dir="analysis/dynamics",
    hotspots_file="lambda1_hotspots.csv",
    apo_pdb="apo_input.pdb",
    n_candidates=10,
    n_replicas=6,
)
```

Key outputs:

- `lambda1_hotspots.csv`: ranked prospective candidates, including centroids.
- `probe_density_diff_lambda1.dx`: thermodynamic state 0 (lambda = 1) minus unbiased density, in bulk units.
- `msmd_hotspots.csv` and `probe_density_md.dx`: unbiased-map candidates and density.
- Per-state density maps (`probe_density_swish_stateN.dx` and
  `probe_density_diff_stateN.dx` for N > 0) and aligned trajectories.
- `pocket_dynamics.csv`, `candidateN_open.pdb` and
  `candidateN_open_closed.pml`: candidate residue dynamics and open/apo comparisons.

An empty candidate table is a valid detector result. There is no requirement for
a known pocket or literature residue list in the single-target detection and
dynamics calls above.

The dynamics session writer uses `SANDCAT_ANALYSIS_DIR` as its output-tree
root and copies external apo inputs into `inputs/` beneath it. Keep density
and dynamics output directories under that root so session paths are portable.
State 0 has the unscaled Hamiltonian but still carries the OPES multithermal
bias; the difference map is a biased-versus-unbiased contrast, not an
equilibrium occupancy or binding free-energy estimate.

#### Reproducing the benchmark figures

The full driver is **benchmark-specific**, not a generic new-target driver.
Supply a local data root containing `september2026/` (simulation outputs) and
`september2026_ref_pockets/` (held-out reference structures).
The target-specific relative paths are recorded in
`sandcat/analysis/build_density_maps.py` and `sandcat/analysis/experiments.py`.
These datasets are not bundled with the package.

```bash
python -m sandcat.analysis.run_prospective_analysis_pipeline \
    /path/to/analysis --paper-dir /path/to/benchmark-data
```

This runs density, dynamics, ranking, figures, snapshots and density-overlay
sessions in dependency order. A failed stage stops the driver.
`--skip-density` reuses maps already in the output tree; use a fresh output
directory for a clean reproduction.

Target outputs now live under `<analysis>/<TARGET>/`, without the older
`_branch` suffix. Rebuild an older analysis tree before using `--skip-density`.

`SANDCAT_PAPER_DIR` supplies the input root when `--paper-dir` is omitted;
otherwise the current directory is used. The driver passes absolute input and
output roots to every stage. When running benchmark stages individually, set
both `SANDCAT_PAPER_DIR` and `SANDCAT_ANALYSIS_DIR` explicitly. Benchmark ranking and validation
figures use held-out references; they are distinct from blind candidate detection.

To view a density map in PyMOL:

```text
load /path/to/apo_input.pdb, protein
load /path/to/analysis/density/probe_density_diff_lambda1.dx, dens_diff
isomesh mesh_diff, dens_diff, 6
```

### Replica swap acceptance rates

After a SWISH or SWISH-X run, check that neighbouring replicas are exchanging efficiently. Healthy acceptance rates are 0.2–0.4.

```commandline
python sandcat/swish_acceptance_rates.py -d /path/to/swish_run_dir
```

The script auto-detects whether the directory contains a SWISH (`swish_output.nc`) or SWISH-X (`swish_x_output.nc`) output file and prints per-neighbour and overall acceptance rates to stdout. `-d` defaults to the current directory if omitted.

# Tests

The unit and integration test suite under `sandcat/tests/` runs with `pytest`:

```commandline
pytest sandcat/tests/
```

Sandcat-only tests (OPES helpers, OPES_EXPANDED math, md_utils, analysis_utils) finish in a few seconds on CPU. Six integration tests in `test_opes_vs_plumed.py` and the PLUMED observer rows in `test_opes_performance.py` cross-check sandcat against PLUMED on alanine dipeptide vacuum; they skip cleanly when the PLUMED kernel library isn't present.

## Building the PLUMED kernel for the integration tests

The integration tests load `libplumedKernel.so` from `sandcat/tests/plumed2/src/lib/`. To build it:

```commandline
cd sandcat/tests
git clone https://github.com/plumed/plumed2.git
cd plumed2
./configure --disable-mpi --disable-external-blas --disable-external-lapack \
            --enable-modules=bias+colvar+core+function+generic+setup+cltools+opes
make -j 1
```

The `opes` module must be enabled — `OPES_METAD` and `OPES_EXPANDED` aren't in the default module set. Build artefacts land at `sandcat/tests/plumed2/src/lib/libplumedKernel.so`; the test conftest sets `PLUMED_KERNEL` automatically when that path exists.

The build needs `gcc`/`g++`/`make` (e.g. `apt-get install build-essential` on Ubuntu) and the `plumed` Python wrapper (`pip install plumed`). Total build time is ~10 minutes on 1 core.
