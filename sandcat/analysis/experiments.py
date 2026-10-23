"""Paths, constants and reference geometry shared by the analysis scripts.

Every value more than one script needs is defined here exactly once: the
simulation and analysis roots, each target's apo input, the distance criteria
and the residues the source papers name. TARGET_RESIDUES is the benchmark's
answer key, kept in one place so no second copy can disagree with it.

BASE and ANALYSIS come from SANDCAT_PAPER_DIR and SANDCAT_ANALYSIS_DIR, since
neither is a property of the package. A redirectable analysis root lets the
pipeline run into a clean directory and the result be compared against an
existing tree.

HIT_A (2.0 A) is how close a candidate centre must come to a reference pocket
to count as having found it. Every matched candidate is within 1.5 A of its
reference, so the cut only bites on MDM2, whose reference is 7.8 A across: at
5 A its rim blob (candidate 2, 3.6 A) would outrank the candidate in the cleft
(candidate 3, 0.45 A).

RESIDUE_A (10.0 A) is how close a candidate must come to a residue's nearest
heavy atom to reach it, applied to the apo input by the benchmark and to the
median over the lambda = 1 trajectory by the figure scripts. GATING_RADIUS_A
(8.0 A) is the tighter apo shell of residues eligible to drive a candidate's
opening coordinate, and PREFILTER_A (20.0 A) only bounds which residues are
worth computing a median for.

N_CANDIDATES bounds how far down the ranked list the dynamics stage
characterises candidates; it must not fall below the deepest matched site, 4 on
Bcl-xL and 6 on PLK1. best_candidate searches the whole list, so a site is
never reported as missing because it ranks below that bound.

true_pocket returns the held-out reference in the simulation's frame,
superposing the paper structure onto the apo input and expanding it to every
protomer of a homo-oligomer. It and best_candidate are the only things here
that touch the answer key, and they serve the validation figures rather than
the prospective path, which never reads them.
"""
from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

def required_dir(variable: str, purpose: str) -> Path:
    """A directory named by an environment variable, with no default.

    There is no default because a wrong one is silent: a run would read
    whichever tree happened to sit at that path and report results from it
    without saying so. An unset variable stops the run instead.

    Args:
        variable: Environment variable name.
        purpose: What the directory holds, used in the error message.

    Returns:
        The directory. It need not exist yet, since a run creates its own
        analysis directory.

    Raises:
        SystemExit: If the variable is unset or empty.
    """
    value = os.environ.get(variable, '').strip()
    if not value:
        raise SystemExit(
            f'{variable} is not set. It must name {purpose}.\n'
            f'    export {variable}=/path/to/directory')
    return Path(value).expanduser().resolve()


def paper_dir() -> Path:
    """The simulation tree and pocket references, from SANDCAT_PAPER_DIR."""
    return required_dir(
        'SANDCAT_PAPER_DIR',
        'the directory holding the simulation tree and the pocket references')


def analysis_dir() -> Path:
    """The tree this analysis reads and writes, from SANDCAT_ANALYSIS_DIR."""
    return required_dir(
        'SANDCAT_ANALYSIS_DIR',
        'the directory this analysis reads and writes. The pipeline runner '
        'sets it from its command-line argument; set it yourself to run one '
        'stage alone')


def truth_dir() -> Path:
    """Where the held-out pocket references live."""
    return paper_dir() / 'september2026_ref_pockets'


def sim_dir() -> Path:
    """Where the raw simulation tree lives."""
    return paper_dir() / 'september2026'


def sim_apo(target: str) -> Path:
    """The apo input one target's simulations started from."""
    return sim_dir() / SIM_APO_RELATIVE[target]


# Resolved on first use rather than at import. These four name directories
# the benchmark needs, and nothing else here does: a caller importing a
# constant or a helper must not be made to set two environment variables for
# a calculation that reads no file.
_BENCHMARK_PATHS = {
    'BASE': paper_dir, 'ANALYSIS': analysis_dir,
    'TRUTH': truth_dir, 'SIM': sim_dir,
    'SIM_APO': lambda: {t: sim_dir() / r for t, r in SIM_APO_RELATIVE.items()},
}


def __getattr__(name: str):
    """Resolve the benchmark directories on attribute access (PEP 562)."""
    if name in _BENCHMARK_PATHS:
        return _BENCHMARK_PATHS[name]()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


HIT_A = 2.0

RESIDUE_A = 10.0
# The shell of residues eligible to drive a candidate's opening coordinate,
# measured on the apo input. Deliberately tighter than RESIDUE_A: these are
# ranked by how far they move, and a wider shell admits residues that move a
# long way without ever approaching the site.
GATING_RADIUS_A = 8.0
# A cheap apo shell that only decides which residues are worth computing a
# trajectory median for. Generous on purpose: a residue whose closest apo
# heavy atom is beyond this cannot come inside RESIDUE_A on the median.
PREFILTER_A = 20.0
# Frames a rotamer bin must be held before a switch is counted. Shared by
# both counters, so a transition means the same thing in each: without it
# the counts are dominated by flicker across a bin boundary.
DWELL = 3

# The reference set: residues a source-paper figure names as opening the site.
# TEM1 takes Oleinikovas Fig S1A alone, which is the figure captioned "the key
# sidechains lining the pockets". Two other TEM1 residue groups were considered
# and are deliberately out. L286 comes from Fig S11, which marks where probe
# density accumulates, not what moves, so it answers a different question.
# R222, S223 and A224 complete the same 3-10 turn and move with the labelled
# residues, but no publication names them: they are identifiable only from a
# mobility ranking of the kind this pipeline produces, so counting them would
# score the method against its own output. Including either group changes the
# totals by at most three residues and the method ranking not at all.
#
# Fig S1 covers only TEM1, IL2 and PLK1; the other four targets keep their own
# sources, listed per target below.
TARGET_RESIDUES = {
    # Fig S1A alone, the figure captioned "the key sidechains lining the
    # pockets". L286 is named by Fig S11 and does line the cryptic pocket, 2.2 A
    # from the pocket reference, but S11 maps where benzene density accumulates
    # anywhere on TEM-1 rather than what opens the site, and the rest of that
    # figure sits 5.2 to 19.4 A away. Counting one member of it would also add
    # the single closest mobile residue to the site in the whole protein, which
    # raises recovery without the method doing anything new.
    #
    # R222, S223 and A224 are as close as the gating six, 0.5 to 1.2 A, but no
    # publication names them; they are identifiable only from a mobility
    # ranking, so counting them would score the method against its own output.
    'TEM1': [220, 221, 225, 244, 276, 279],
    'BCLXL': [45, 49, 90, 185, 189, 230],
    # Dippon 2026 Fig 2, in simulation numbering (paper numbering minus 100).
    # W264 must move to open the pocket; D265 and R283 anchor the allosteric
    # ligand (D265A/E and R283A/K each ablate allostery); C287 is the
    # neighbouring contact (Fig 2e); L273 and P274 are the D270-P277 loop
    # residues within the density (5.8 and 6.1 A). The rest of the loop
    # (10.0 to 10.8 A) and C318, L321, P424 (15 to 17 A) are excluded.
    'CRBN': [164, 165, 173, 174, 183, 187],
    'MDM2': [86, 93, 96, 100],
    'IL2': [33, 34, 37, 67],
    'HPV11-E2': [19, 32, 94],
    'PLK1': [417, 421, 481, 482],
}

SIM_APO_RELATIVE = {
    'BCLXL': 'BCLXL/msmd_out_benz_1R2D_input/0/1R2D_input.pdb',
    'CRBN': 'CRBN/msmd_out_benz_9FJX_input/0/9FJX_input.pdb',
    'HPV11-E2': 'HPV11-E2/msmd_out_benz_1R6K_input/0/1R6K_input.pdb',
    'IL2': 'IL2/msmd_out_benz_1M47_input/0/1M47_input.pdb',
    'MDM2': 'MDM2/msmd_out_benz_1Z1M_input/0/1Z1M_input.pdb',
    'PLK1': 'PLK1/msmd_out_benz_3FVH_input/0/3FVH_input.pdb',
    'TEM1': 'TEM1/msmd_out_benz_1JWP_input/0/1JWP_input.pdb',
    'UB': 'UB/msmd_out_benz_2LJ5_input/0/2LJ5_input.pdb',
}

# target -> (ground-truth dir, pocket reference file). UB has none by design.
# Named for its contents: build_density_maps holds a different per-target
# table, and one name for both invites reading the wrong one.
POCKET_REFERENCES = {
    'BCLXL': ('bcl2-xl', 'bclxl_pocket_ref.pdb'),
    'CRBN': ('crbn', 'crbn_pocket_ref.pdb'),
    'HPV11-E2': ('hpv2', 'hpv2_pocket_ref.pdb'),
    'IL2': ('il2', 'il2_pocket_ref.pdb'),
    'MDM2': ('mdm2', 'mdm2_pocket_ref.pdb'),
    'PLK1': ('plk1', 'plk1_pocket_ref.pdb'),
    'TEM1': ('tem-1', 'pocket_ref_from_swish_paper.pdb'),
    'UB': (None, None),
}

# Where a bundled copy of each apo input lives, relative to the analysis root.
APO_DIR = 'inputs'

# What the density stage writes and every later stage reads. Named here because
# a contract repeated as a literal in five modules is a contract that drifts:
# the state index in these names is the thermodynamic state, and state 0 is the
# unscaled Hamiltonian, lambda = 1.0.
PRODUCTION_STATE = 0
HOTSPOTS_FILE = 'lambda1_hotspots.csv'
STATE_TRAJECTORY = f'swish_{PRODUCTION_STATE}_aligned.dcd'
STATE_TOPOLOGY = f'density_swish_{PRODUCTION_STATE}_top.pdb'
DIFFERENCE_MAP = 'probe_density_diff_lambda1.dx'


def bundled_apo(target: str) -> Path:
    """The apo input, copied into the analysis directory, as an absolute path.

    The simulation inputs sit outside the analysis directory, and every session
    loads an apo as its closed reference. Keeping a copy inside makes the
    directory self-contained, at a few hundred kilobytes per target. The copy
    is refreshed on every call, so a stage rerun against a changed input cannot
    leave the old structure beside new frames.

    Args:
        target: Target name.

    Returns:
        The copy's path, inside the analysis directory.
    """
    return bundle_into_analysis(sim_apo(target))


def bundle_into_analysis(path: str | Path) -> Path:
    """Copy a file into the analysis directory unless it is already inside.

    A session loads its structures by a path relative to the analysis root, so
    anything it loads has to live under that root. Callers of the general API
    may hold an apo anywhere, and asking them to copy it first would make the
    benchmark wrapper the only usable entry point.

    Args:
        path: File to make available inside the analysis directory.

    Returns:
        The path to use, inside the analysis directory. A file already under
        the root is returned unchanged rather than copied onto itself.
    """
    source = Path(path).resolve()
    root = analysis_dir().resolve()
    if source.is_relative_to(root):
        return source
    destination = root / APO_DIR / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def analysis_relative(path: str | Path) -> str:
    """A path inside the analysis directory, written relative to its root.

    Args:
        path: A path under ANALYSIS.

    Returns:
        The path relative to the analysis root, with forward slashes.

    Raises:
        ValueError: If the path is not inside the analysis directory, which
            would otherwise be written into a session as a path that resolves
            nowhere once the directory is moved.
    """
    return Path(path).resolve().relative_to(
        analysis_dir().resolve()).as_posix()


def pml_preamble(pml_path: str | Path) -> list[str]:
    """Session lines that move PyMOL to the analysis root, wherever it sits.

    PyMOL resolves a relative load against its working directory and not
    against the script, so these lines set the working directory from the
    running script's own path. Every load after them is then written relative
    to the analysis root and resolves wherever the directory has been put.

    Args:
        pml_path: Where the session will be written, inside ANALYSIS.

    Returns:
        Lines to place at the top of the session, before any load.
    """
    depth = len(Path(pml_path).resolve().parent
                .relative_to(analysis_dir().resolve()).parts)
    up = '/'.join(['..'] * depth) if depth else '.'
    return [
        '# --- Paths -----------------------------------------------------',
        '# Every path below is relative to the root of this analysis',
        '# directory. PyMOL resolves a relative load against its working',
        '# directory rather than against the script, so the session moves',
        '# there itself and the directory can be opened from anywhere.',
        'python',
        'import os, pymol',
        'from pymol import cmd',
        'cmd.cd(os.path.normpath(os.path.join(',
        f"    os.path.dirname(os.path.abspath(pymol.__script__)), '{up}')))",
        'python end',
        '',
    ]


# Residues that are not part of the protein and must never be reported as one:
# solvent, ions, the benzene probe, and the ACE/NME/NH2 terminal caps, which
# carry a resSeq and would otherwise be ranked and plotted as residues. Defined
# once here because every stage that measures a residue has to exclude the same
# set; two lists meant four capping groups were reported as residues.
NON_PROTEIN = ('PRB', 'BNZ', 'HOH', 'WAT', 'TIP3', 'SOL', 'NA', 'CL', 'K',
               'MG', 'CA', 'ZN', 'ACE', 'NME', 'NH2', 'DMS', 'MSO')
# Elements that are not heavy atoms: hydrogen, and the virtual sites and
# blank-element records some force fields write, which carry no mass and must
# not contribute a distance.
NON_HEAVY_ELEMENTS = ('H', 'VS', '')


def heavy_atoms_by_resseq(topology, wanted=None) -> dict[int, list[int]]:
    """Heavy-atom indices per residue, excluding solvent, ions and caps.

    Args:
        topology: An mdtraj topology.
        wanted: resSeq values to keep, or None for every protein residue.

    Returns:
        resSeq mapped to its heavy-atom indices. Residues with no heavy atom
        are omitted, as are any whose name is in NON_PROTEIN.
    """
    out: dict[int, list[int]] = {}
    for residue in topology.residues:
        if residue.name in NON_PROTEIN:
            continue
        if wanted is not None and residue.resSeq not in wanted:
            continue
        heavy = [a.index for a in residue.atoms
                 if a.element.symbol not in NON_HEAVY_ELEMENTS]
        if heavy:
            out.setdefault(residue.resSeq, []).extend(heavy)
    return out


def target_dir(target: str, root: Path | None = None) -> Path:
    """One target's directory inside an analysis tree.

    Written once so the layout of a produced tree is decided in a single
    place: ten stages built this path from a literal and all ten had to agree.

    Args:
        target: Target name, as used throughout the analysis.
        root: Tree to resolve against. Defaults to the analysis directory,
            and is given explicitly by the stage that builds the tree.

    Returns:
        The directory holding that target's density maps and candidate tables.
    """
    return (analysis_dir() if root is None else root) / target


def distance_trace(xyz: np.ndarray, heavy: list[int],
                   centre: np.ndarray) -> np.ndarray:
    """Per-frame closest approach of one residue's heavy atoms to a point.

    The one place this measurement is written. It was spelled out in three
    modules, which is how one of them came to be given a different cutoff from
    the rest without anything failing.

    Args:
        xyz: (frames, atoms, 3) coordinates in Angstroms.
        heavy: Heavy-atom indices of the residue.
        centre: (3,) point to measure from, in Angstroms.

    Returns:
        (frames,) closest heavy-atom distance, one value per frame.
    """
    return np.linalg.norm(xyz[:, heavy, :] - centre, axis=2).min(axis=1)


def closest_approach(coords: np.ndarray, centre: np.ndarray) -> float:
    """Closest approach of a set of coordinates to a point, in one structure.

    The single-structure counterpart of distance_trace, used wherever the apo
    input decides something rather than the trajectory.

    Args:
        coords: (atoms, 3) coordinates in Angstroms.
        centre: (3,) point to measure from, in Angstroms.

    Returns:
        The smallest distance from any of those coordinates to the point.
    """
    return float(np.linalg.norm(coords - centre, axis=1).min())


def pocket_separations(centres: np.ndarray,
                       pocket: np.ndarray) -> np.ndarray:
    """Closest approach of each candidate centre to a reference pocket.

    The site-matching measurement, written once. It decides both the rank
    reported in ranking_regions.csv and the one the per-target residue figure
    draws, which were separate implementations of the same comparison.

    Args:
        centres: (candidates, 3) candidate centres in Angstroms.
        pocket: (points, 3) reference pocket coordinates in Angstroms.

    Returns:
        (candidates,) distance from each centre to the nearest pocket point.
    """
    return np.linalg.norm(
        centres[:, None, :] - pocket[None, :, :], axis=2).min(axis=1)


def residue_distance_traces(trajectory, centre: np.ndarray,
                            wanted=None) -> dict[int, np.ndarray]:
    """Per-frame closest heavy-atom distance from each residue to a point.

    This is the measurement the whole prospective analysis rests on: a
    candidate's environment, the rank at which a residue is first reached, the
    bars, the boxes and the opening coordinate are all summaries of it. It lives
    here, in one place, because a definition repeated per stage drifts, and a
    cutoff applied to a repeated definition drifts with it.

    Args:
        trajectory: Trajectory to measure over, any number of frames.
        centre: The point to measure to, in Angstroms.
        wanted: resSeq values to keep, or None for every protein residue.

    Returns:
        resSeq mapped to an array of one distance per frame, in Angstroms.
    """
    xyz = trajectory.xyz * 10.0
    return {resseq: distance_trace(xyz, heavy, centre)
            for resseq, heavy in heavy_atoms_by_resseq(
                trajectory.topology, wanted).items()}


def apo_residues(apo_pdb: str) -> dict[int, np.ndarray]:
    """Heavy-atom coordinates per residue, from the apo input structure.

    The apo input and never a trajectory frame: it defines the coordinate frame
    every density map is written in, it does not move if the equilibration is
    repeated, and it is what a prospective user holds before running anything.

    Args:
        apo_pdb: Path to the apo input structure.

    Returns:
        Simulation resSeq mapped to an (atoms, 3) array of heavy-atom
        coordinates in Angstroms. Hydrogens are dropped.
    """
    out: dict[int, list] = {}
    for line in Path(apo_pdb).read_text().splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        # the same exclusion the trajectory path applies: a capping group
        # carries a resSeq and would otherwise enter the driver pool as a
        # residue, on seven candidates across three targets
        if line[17:20].strip() in NON_PROTEIN:
            continue
        if (line[76:78].strip() or line[12:16].strip()[0]) == 'H':
            continue
        out.setdefault(int(line[22:26]), []).append(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return {resid: np.asarray(xyz) for resid, xyz in out.items()}


def kabsch(mobile: np.ndarray, target: np.ndarray):
    """Rotation and translation taking mobile onto target.
    Args:
        mobile: (N, 3) coordinates to be moved.
        target: (N, 3) coordinates to move onto, in the same atom order.

    Returns:
        (rotation, translation) such that mobile @ rotation + translation is
        the least-squares superposition onto target.
    """
    mu_m, mu_t = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile - mu_m).T @ (target - mu_t))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, mu_t - rotation @ mu_m


def true_pocket(target: str, apo_sim: str) -> np.ndarray | None:
    """Held-out pocket reference, moved into the simulation's coordinate frame.

    The ground-truth files are not all in the frame the simulation ran in. Most
    sit on it exactly, but TEM1's reference is 83 A away and PLK1's 17 A, so
    scoring against them unmoved reports that every strategy missed the site
    when in fact nothing was ever in the same space. The superposition is derived from
    the protein alone and applied to the pocket, so it moves the answer key into
    frame without telling the detector anything.

    Args:
        target: Target name.
        apo_sim: Path to the apo input the simulation started from, which
            defines the coordinate frame every density map is in.

    Returns:
        (atoms, 3) pocket coordinates in the simulation frame, with one copy per
        protein chain for an oligomer, or None if the target has no reference.

    Raises:
        ValueError: If no ground-truth apo shares residues with the
            simulation, so the superposition has nothing to fit on.
    """
    folder, pocket_file = POCKET_REFERENCES[target]
    if folder is None:
        return None
    pocket = mda.Universe(
        str(truth_dir() / folder / pocket_file)).atoms.positions
    apo_files = [f for f in (truth_dir() / folder).glob('*.pdb')
                 if 'pocket' not in f.name]
    sim = mda.Universe(apo_sim).select_atoms('name CA')
    sim_by_resid = dict(zip(sim.resids, sim.positions))

    best = None
    for apo in sorted(apo_files):
        gt = mda.Universe(str(apo)).select_atoms('name CA')
        # match on shared residue numbers: references are not always the same
        # construct as the simulation (PLK1's spans 2-603 against the
        # simulation's 371-603), so equal atom counts cannot be assumed
        shared = [r for r in gt.resids if r in sim_by_resid]
        if len(shared) < 50:
            continue
        gt_by_resid = dict(zip(gt.resids, gt.positions))
        gt_ca = np.array([gt_by_resid[r] for r in shared])
        sim_ca = np.array([sim_by_resid[r] for r in shared])
        if best is None or len(shared) > best[0]:
            best = (len(shared), gt_ca, sim_ca, apo.name)
    if best is None:
        raise ValueError(f'{target}: no ground-truth apo shares residues with the simulation')

    n_shared, gt_ca, sim_ca, name = best
    if np.sqrt(((gt_ca - sim_ca) ** 2).sum(1).mean()) < 0.5:
        moved = pocket                          # already in frame
    else:
        rotation, translation = kabsch(gt_ca, sim_ca)
        moved = pocket @ rotation.T + translation
        print(f'    {target}: pocket superposed from {name} on {n_shared} shared CA')
    return expand_symmetry_copies(moved, apo_sim, target)


N_CANDIDATES = 10


def best_candidate(target: str) -> tuple[str, int, float] | None:
    """Highest-ranked lambda = 1 difference candidate within 2 A of the reference.
    Highest-ranked rather than nearest, because a user works down the ranked
    list and reaches that one first. On three targets a lower-ranked candidate
    is marginally closer, but selecting it would be choosing on the answer key.

    Args:
        target: Target name.

    Returns:
        (detector, rank, distance in Angstroms rounded to 1 dp) for the first
        candidate within HIT_A of the reference, or None if the target has no
        reference or nothing falls inside the cutoff.
    """
    pocket = true_pocket(target, str(sim_apo(target)))
    if pocket is None:
        return None
    best = None
    # the lambda = 1 difference map only. density_hotspots.csv is the pooled
    # difference map and msmd_hotspots.csv the unbiased one; neither is the
    # production detector, and letting best_candidate choose between them put
    # PLK1 and MDM2 on the MSMD map.
    for detector, filename in (('lambda1', 'lambda1_hotspots.csv'),):
        path = target_dir(target) / 'density' / filename
        if not path.exists():
            continue
        # The whole ranked list, not the first N_CANDIDATES: that bound says
        # how deep the dynamics stage characterises candidates, and reusing it
        # here would report a site found at a lower rank as not found at all.
        hotspots = pd.read_csv(path)
        separations = pocket_separations(
            hotspots[['x', 'y', 'z']].to_numpy(float), pocket)
        for (_, row), distance in zip(hotspots.iterrows(), separations):
            distance = float(distance)
            if distance <= HIT_A:
                candidate = (detector, int(row['rank']), round(distance, 1))
                if best is None or candidate[1] < best[1]:
                    best = candidate
                break
    return best

def expand_symmetry_copies(pocket, apo_sim, target=''):
    """Add the equivalent pocket on every other protein chain.

    A homo-oligomer carries its cryptic site once per protomer, but the shipped
    reference defines only one copy. Scoring against that single copy marks
    every detection of another protomer's site as a miss: on Bcl-xL nothing
    falls within HIT_A of it, the nearest candidate being rank 7 at 8.9 A,
    while against both protomers the site is matched at rank 4 and 1.5 A.

    Chains are matched position by position, so this assumes the protomers are
    built in the same residue order, which is true of a homodimer written as one
    sequential file. A poor superposition is rejected rather than used.

    Args:
        pocket: (atoms, 3) pocket coordinates on one chain.
        apo_sim: Path to the apo input, read for its chains.
        target: Target name, used only in the log line.

    Returns:
        The pocket stacked once per protein chain, every copy the same length,
        so the result reshapes to (chains, atoms, 3). Monomers come back
        unchanged.
    """
    structure = mda.Universe(apo_sim)
    chains = {}
    for residue in structure.select_atoms('protein').residues:
        chains.setdefault(residue.segid, []).append(residue)
    chains = {k: v for k, v in chains.items() if len(v) >= 20}
    if len(chains) < 2:
        return pocket

    copies = [pocket]
    keys = sorted(chains)
    reference = keys[0]
    for other in keys[1:]:
        a, b = chains[reference], chains[other]
        # Protomers are not always written with the same leading residues:
        # Bcl-xL's chain B carries an extra GLY, so matching position by
        # position slips by one and agrees on only 8 residue names of 139.
        # Pick the offset that maximises residue-name agreement instead.
        best_offset, best_matches = 0, -1
        for offset in range(-5, 6):
            matches = 0
            for i, ra in enumerate(a):
                j = i + offset
                if 0 <= j < len(b) and ra.resname == b[j].resname:
                    matches += 1
            if matches > best_matches:
                best_offset, best_matches = offset, matches
        ca_a, ca_b = [], []
        for i, ra in enumerate(a):
            j = i + best_offset
            if not (0 <= j < len(b)) or ra.resname != b[j].resname:
                continue
            sa, sb = ra.atoms.select_atoms('name CA'), b[j].atoms.select_atoms('name CA')
            if len(sa) and len(sb):
                ca_a.append(sa.positions[0]); ca_b.append(sb.positions[0])
        if len(ca_a) < 20:
            print(f'    {target}: chains {reference}/{other} share only '
                  f'{len(ca_a)} alignable residues, not treated as copies')
            continue
        ca_a, ca_b = np.array(ca_a), np.array(ca_b)
        rotation, translation = kabsch(ca_a, ca_b)
        rmsd = np.sqrt((((ca_a @ rotation.T + translation) - ca_b) ** 2).sum(1).mean())
        if rmsd > 3.0:
            print(f'    {target}: chains {reference}/{other} superpose at '
                  f'{rmsd:.1f} A, too poor to treat as symmetry copies')
            continue
        copies.append(pocket @ rotation.T + translation)
        print(f'    {target}: added pocket copy for chain {other} '
              f'({len(ca_a)} CA, {rmsd:.2f} A)')
    return np.vstack(copies)
