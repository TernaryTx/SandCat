"""Side-by-side density sessions for a paper figure: lambda = 1 difference against MSMD.

One session per target, two copies of the same structure in the same orientation,
the left carrying the SWISH-X minus MSMD probe density from state 0 and the
right carrying the unbiased MSMD density. Colours are those of the single-map
sessions, teal for protein and purple for density, and they
are deliberately identical on both sides: the comparison is where the density
sits, so letting colour vary too would invite the reader to attribute a
difference to the wrong variable.

Both maps are cut at the same multiple of bulk, the detector's threshold. They
are directly comparable at that cut because both are normalised to bulk probe
density, so a voxel at the cut means the same occupancy in each.

The separation is applied to the coordinates and to the density origin in the
files this script writes, rather than through PyMOL's object matrix. Translating
a map object is the part that tends to break between PyMOL versions, and a
figure whose two panels have silently drifted apart by a different amount than
the protein is worse than one that fails loudly. Shifting the data means the
geometry is fixed before PyMOL ever sees it, and can be checked numerically.

Orientation is set rather than inherited. Shifting along the model's x axis and
letting PyMOL pick a camera puts the two copies at whatever angle the deposited
frame happens to carry, which is rarely the one a figure wants. Instead the
protein's principal axes are computed and the camera is set so the longest axis
is vertical and the second longest horizontal, which is the view that shows the
most of a protein. The copies are then separated along that same second axis, so
they sit level, side by side, and the pair is as compact on the page as the
molecule allows.

The density is not rotated, only translated: rotating a grid would mean
resampling it and blurring the thing being compared. The camera does the
rotating, so both maps stay exactly as computed.

The target is marked on the cartoon in yellow, on both panels, with no
sticks: the density meshes are already fine-grained and sticks underneath them
compete for the same space. CRBN is the one exception, listed in STICKS, because
its target is a single tryptophan out of 327 residues and a one-residue cartoon
segment is a few pixels wide. The cartoon is 45% transparent so the mesh reads
through it, except on the marked residues, which stay opaque so the yellow is not
washed out by the same setting that is there to reveal the density.

What gets marked is the residues the source papers name, which is the sharper
claim available and the one a reader can check against the literature, and it is
what the residue-recovery column scores. Marking every residue lining the
held-out pocket reference instead would be correct but diffuse, 59 residues on
Bcl-xL and 44 on TEM1, roughly a fifth of the protein, which reads as a region
rather than a target.

Having the marker on both sides is the point of the figure: the reader can see
whether each map puts density on the yellow or somewhere else, which is the
question the benchmark answers numerically.

Ubiquitin is included and has no yellow, because it has no cryptic pocket and no
reference. Its MSMD panel is the one that shows why the difference map is needed,
since the unbiased density is substantial on a protein with nothing to find.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from gridData import Grid
from scipy.ndimage import label as nd_label


from sandcat.analysis.compute_probe_density import write_dx_file

from sandcat.analysis.compute_probe_density import (  # noqa: E402
    THRESHOLD_FACTOR as THRESHOLD)

from sandcat.analysis.experiments import (target_dir,
                         ANALYSIS, SIM_APO, TARGET_RESIDUES, POCKET_REFERENCES,
                         true_pocket)

warnings.filterwarnings('ignore')

OUT = ANALYSIS / 'pymol'

GAP_A = 15.0          # clear space between the two copies

LIGHT_TEAL = '[0.53, 0.87, 0.87]'   # light teal: the cartoon is background
                              # here, so a pale ground lets the purple density
                              # and the yellow target carry the figure
PURPLE = '[0.52, 0.55, 0.88]'
YELLOW = '[1.00, 0.78, 0.40]'
BLUE = '[0.25, 0.56, 0.77]'    # nitrogen on the stick residues
RED = '[0.96, 0.38, 0.44]'     # oxygen, matching the snapshot sessions
FOV_DEG = 20.0        # PyMOL's default field of view, which set_view assumes
# targets drawn as sticks as well as coloured cartoon; sticks overlap the mesh being compared.
STICKS: set[str] = set()

# The residues the source papers name as gating the site, in simulation
# numbering. See tracked_residues.csv for the literature numbering, which differs
# for IL-2, CRBN and Bcl-xL.
#
# TEM1 takes Oleinikovas Fig S1A alone, the figure captioned "the key sidechains
# lining the pockets", and all six sit within 1.7 A of the pocket reference. Its
# Fig S11 hotspot residues, 70, 151, 203, 232, 266 and 286, are deliberately out:
# that figure maps where benzene density accumulates anywhere on TEM-1, not what
# opens the cryptic site, and its members spread over 32 A. L286 is the one that
# does line the pocket, 2.2 A from the reference, but it is a pocket residue
# rather than a gate and it is also the closest mobile residue to the site in
# the protein, so counting it would raise recovery for nothing. They are listed
# deliberately not marked.

# MSMD on the left, the lambda = 1 difference on the right: the figure reads as
# "here is everything the unbiased simulation proposes, here is what survives
# subtraction", which is the direction of the argument.
PANELS = [
    ('msmd', 'probe_density_md.dx', 'unbiased MSMD'),
    ('diffl1', 'probe_density_diff_lambda1.dx',
     'SWISH-X minus MSMD, state 0 (lambda = 1)'),
]


def find_map(folder: Path, filename: str) -> Path:
    """Path to one density map inside a target's density directory.

    Args:
        folder: The target's density directory, as the density stage wrote it.
        filename: Map file name, one of the entries in PANELS.

    Returns:
        The path, whether or not it exists; the caller checks and skips a target
        whose maps are missing.
    """
    return folder / filename


def regions(density, low):
    """Labelled candidate regions above the cut, ordered by total density.

    Args:
        density: the grid in bulk units, as read from the .dx map.
        low: the cut in bulk units; voxels at or below it are background.

    Returns:
        A pair (labelled, order) where labelled is the integer label grid from
        a 6-connected component analysis and order lists the labels from the
        densest region to the sparsest, which is the detector's own ranking.
    """
    labelled, n = nd_label(density > low)
    totals = [(float(density[labelled == m].sum()), m)
              for m in range(1, n + 1) if (labelled == m).any()]
    return labelled, [m for _, m in sorted(totals, key=lambda r: -r[0])]


def marked_coordinates(pdb: Path, resids: set[int]) -> np.ndarray:
    """Heavy-atom coordinates of the marked residues, for aiming the camera.

    Args:
        pdb: the structure to read.
        resids: the residue numbers to keep, in the file's own numbering.

    Returns:
        An (n_atoms, 3) array, empty with shape (0, 3) if no atom matched, so
        that the caller can fall back to a principal-axis view.
    """
    out = []
    for line in Path(pdb).read_text().splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        if int(line[22:26]) not in resids:
            continue
        if (line[76:78].strip() or line[12:16].strip()[0]) == 'H':
            continue
        out.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(out) if out else np.empty((0, 3))


def coordinates(pdb: Path) -> np.ndarray:
    """Every atom coordinate in a PDB, hydrogens included.

    Args:
        pdb: path to the structure to read.

    Returns:
        An (n_atoms, 3) array in the file's own frame, used for the centre of
        mass and the principal axes that set the camera.
    """
    return np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])]
                     for line in Path(pdb).read_text().splitlines()
                     if line.startswith(('ATOM', 'HETATM'))])


def shift_pdb(source: Path, destination: Path, shift: np.ndarray) -> None:
    """Copy a PDB with every atom translated by a vector.

    Args:
        source: the structure to read.
        destination: where to write the translated copy.
        shift: a length-3 vector added to every atom position, in angstrom.
    """
    lines = []
    for line in Path(source).read_text().splitlines():
        if line.startswith(('ATOM', 'HETATM')):
            xyz = np.array([float(line[30:38]), float(line[38:46]),
                            float(line[46:54])]) + shift
            line = (f'{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}'
                    f'{line[54:]}')
        lines.append(line)
    destination.write_text('\n'.join(lines) + '\n')


def view_frame(xyz: np.ndarray, site: np.ndarray):
    """Camera rotation facing the marked site, with the copies separated across it.

    Principal axes alone would present the protein
    well but takes no account of where the site is, so the yellow patch often
    ended up on the far side and invisible. Here the camera's z axis, the one
    pointing at the viewer, is the direction from the protein's centre to the
    site, so the marked residues always face front. The horizontal axis is then
    whatever is left of the longest principal axis once that direction is
    removed, which keeps the molecule's long dimension roughly in plane, and the
    two copies are separated along it so they stay level and the site stays
    forward on both.

    Args:
        xyz: every atom coordinate of the protein, which sets both the centre
            the camera looks at and the principal axes.
        site: the point the camera should face, the centre of the region being
            marked.

    Returns:
        A pair (rotation, horizontal). The rows of rotation are the camera's x,
        y and z axes in world coordinates, and horizontal is the axis along
        which the caller separates the two copies.
    """
    centre = xyz.mean(axis=0)
    forward = site - centre
    norm = np.linalg.norm(forward)
    centred = xyz - centre
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    if norm < 1e-6:
        forward = axes[2]
    else:
        forward = forward / norm
    # the longest principal axis with the viewing direction projected out, so
    # the horizontal is perpendicular to the camera axis by construction
    horizontal = axes[0] - np.dot(axes[0], forward) * forward
    if np.linalg.norm(horizontal) < 1e-6:
        horizontal = axes[1] - np.dot(axes[1], forward) * forward
    horizontal = horizontal / np.linalg.norm(horizontal)
    vertical = np.cross(forward, horizontal)
    rotation = np.array([horizontal, vertical, forward])
    if np.linalg.det(rotation) < 0:
        rotation[1] = -rotation[1]
    return rotation, horizontal


def set_view_block(rotation, centre, radius) -> list[str]:
    """A PyMOL set_view call for this camera, orthoscopic.

    Args:
        rotation: the 3x3 camera frame from view_frame.
        centre: the point the camera looks at, in world coordinates.
        radius: the half-extent of what must fit in view, in angstrom; it sets
            both the camera distance and the near and far clipping planes.

    Returns:
        The lines of the set_view call, to be joined into the .pml.
    """
    distance = 2.4 * radius
    r = rotation
    return view_matrix(r, centre, distance, radius, FOV_DEG)


def view_matrix(r, centre, distance: float, half_depth: float,
                fov_deg: float) -> list[str]:
    """The set_view call itself, shared by both session writers.

    Only the camera model differs between them: one frames by a bounding
    radius, the other by a field of view. The matrix they write is identical,
    and was duplicated line for line until the two drifted on the field of
    view, which one of them had hardcoded.

    Args:
        r: 3x3 rotation whose rows are the camera axes.
        centre: Point the camera looks at, in Angstroms.
        distance: Camera distance from that point.
        half_depth: Half the near-to-far clipping range.
        fov_deg: Vertical field of view in degrees.

    Returns:
        The lines of the set_view call.
    """
    return ['set_view (\\',
            f'  {r[0, 0]:10.6f},{r[0, 1]:11.6f},{r[0, 2]:11.6f},\\',
            f'  {r[1, 0]:10.6f},{r[1, 1]:11.6f},{r[1, 2]:11.6f},\\',
            f'  {r[2, 0]:10.6f},{r[2, 1]:11.6f},{r[2, 2]:11.6f},\\',
            f'  {0.0:10.6f},{0.0:11.6f},{-distance:11.4f},\\',
            f'  {centre[0]:10.4f},{centre[1]:11.4f},{centre[2]:11.4f},\\',
            f'  {distance - half_depth:10.4f},{distance + half_depth:11.4f},'
            f'{-fov_deg:11.4f} )']



def marked_residues(target: str, protein) -> tuple[list[int], np.ndarray]:
    """The residues a session highlights, and their coordinates.

    The residues a source paper names. Both session writers need the same
    answer, and the choice was duplicated between them.

    Args:
        target: Target name.
        protein: Path to the structure the session loads.

    Returns:
        (residue numbers, their heavy-atom coordinates).
    """
    site_ids = sorted(TARGET_RESIDUES.get(target, []))
    return site_ids, marked_coordinates(protein, set(site_ids))


def main() -> None:
    """Write one side-by-side session per target, skipping what is incomplete.

    A target is skipped, with a line saying why, if its structure or either of
    its two maps is absent, so that a partial analysis tree yields sessions for
    the targets it does cover rather than an error.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    for target in POCKET_REFERENCES:
        folder = target_dir(target) / 'density'
        # the apo input: the structure the maps are in frame with, and the
        # one a reader can download from the PDB to check the session against
        protein = Path(SIM_APO[target])
        if not protein.exists():
            print(f'  {target:9s} no structure, skipped')
            continue
        missing = [f for _, f, _ in PANELS if not find_map(folder, f).exists()]
        if missing:
            print(f'  {target:9s} missing {missing}, skipped')
            continue

        xyz = coordinates(protein)
        # the camera faces the marked residues, so compute them before the view
        site_ids, site_xyz = marked_residues(target, protein)
        focus = site_xyz.mean(axis=0) if len(site_xyz) else xyz.mean(axis=0)
        # Separate along world x and view down world -z, then rotate about world
        # x only. Screen x stays world x under that rotation, so the two copies
        # are horizontal by construction.
        offset = float(xyz[:, 0].max() - xyz[:, 0].min()) + GAP_A
        shift = np.array([offset, 0.0, 0.0])
        towards = focus - xyz.mean(axis=0)
        tilt = float(np.degrees(np.arctan2(towards[1], towards[2])))

        # right-hand copy of the structure, shifted in the file
        right_pdb = OUT / f'{target}_cmp_right.pdb'
        shift_pdb(protein, right_pdb, shift)



        lines = ['# ' + '=' * 70,
                 f'#  {target}: probe density, two maps side by side',
                 '#',
                 f'#  LEFT   {PANELS[0][2]}',
                 f'#  RIGHT  {PANELS[1][2]}',
                 '#',
                 '#  LIGHT TEAL  protein, both panels',
                 '#  YELLOW  the target, both panels, '
                 + ('cartoon and sticks' if target in STICKS
                    else 'cartoon only, no sticks'),
                 '#  PURPLE  density, both panels',
                 '#',
                 f'#  Both maps are cut at {THRESHOLD:g}x bulk and are comparable at',
                 '#  that cut because both are normalised to bulk probe density.',
                 f'#  The right panel is shifted {offset:.1f} A in the file, along',
                 '#  the axis the camera shows horizontally, not by a PyMOL object',
                 '#  matrix, so the separation is exact.',
                 '#',
                 '#  The view is set explicitly: the protein\'s longest principal',
                 '#  axis is vertical and its second longest horizontal. Reset the',
                 '#  camera with "orient" only if you want PyMOL to choose instead.',
                 '# ' + '=' * 70,
                 '',
                 '# --- Colours ---------------------------------------------------',
                 f'set_color ttx_light_teal, {LIGHT_TEAL}',
                 f'set_color ttx_purple, {PURPLE}',
                 f'set_color ttx_yellow, {YELLOW}',
                 f'set_color ttx_blue, {BLUE}',
                 f'set_color ttx_red, {RED}',
                 '',
                 '# --- Structures ------------------------------------------------',
                 f'load {protein}, {target}_left',
                 f'load {right_pdb}, {target}_right',
                 'hide everything',
                 f'show cartoon, {target}_left or {target}_right',
                 f'color ttx_light_teal, {target}_left or {target}_right',
                 'hide everything, elem H',
                 '']

        # the target, marked identically on both panels
        site = site_ids
        provenance = f'{len(site)} residues named by the source papers'
        if site:
            resi = '+'.join(str(r) for r in site)
            lines += ['# --- The target -------------------------------------------------',
                      f'# {provenance}',
                      f'select true_site, ({target}_left or {target}_right) '
                      f'and resi {resi}',
                      'color ttx_yellow, true_site']
            if target in STICKS:
                # CRBN alone keeps sticks: it marks TRP164 out of 327 residues,
                # and one coloured cartoon residue is a few pixels at this scale.
                lines += ['show sticks, true_site and not elem H',
                          'set stick_radius, 0.35, true_site',
                          # carbons keep the marking colour so the sticks still
                          # read as the target; N and O go to the conventional
                          # blue and red, as in the snapshot sessions, so the
                          # side chain reads as chemistry rather than as a blob
                          'color ttx_yellow, true_site and elem C',
                          'color ttx_blue, true_site and elem N',
                          'color ttx_red, true_site and elem O',
                          'color ttx_yellow, true_site and elem S']
            lines += ['set cartoon_transparency, 0.45',
                      # the marked cartoon stays opaque, otherwise the only thing
                      # carrying the target on these panels is 45% washed out
                      'set cartoon_transparency, 0.0, true_site',
                      'deselect', '']
        else:
            lines += ['# --- Nothing marked ---------------------------------------------',
                      '# Ubiquitin is the negative control: no cryptic pocket, no',
                      '# named residues, so there is nothing to mark.',
                      '']

        counts = []
        for (tag, filename, description), panel_shift in zip(
                PANELS, (np.zeros(3), shift)):
            grid = Grid(str(find_map(folder, filename)))
            density = grid.grid
            origin = np.array(grid.origin) + panel_shift
            labelled, order = regions(density, THRESHOLD)
            keep = np.zeros_like(density)
            for marker in order:
                selected = labelled == marker
                keep[selected] = density[selected]
            path = OUT / f'{target}_cmp_{tag}.dx'
            write_dx_file(keep, origin, float(grid.delta[0]), str(path))
            counts.append((tag, len(order), int((keep > 0).sum())))
            lines += [f'# --- {tag}: {description} ---',
                      f'#     {len(order)} regions, {int((keep > 0).sum())} voxels',
                      f'load {path}, dens_{tag}',
                      f'isomesh mesh_{tag}, dens_{tag}, {THRESHOLD:g}',
                      f'color ttx_purple, mesh_{tag}',
                      f'set mesh_width, 1.2, mesh_{tag}',
                      '']

        lines += ['# --- Final state -----------------------------------------------',
                  'bg_color white',
                  'set mesh_quality, 2',
                  'set ray_shadows, 0',
                  'set cartoon_fancy_helices, 1',
                  '',
                  '# --- View -------------------------------------------------------',
                  '# reset gives the identity camera, looking down world -z, so',
                  '# screen x is world x and the copies are level. The turn is',
                  '# about that same axis, which keeps them level while bringing',
                  '# the marked residues to the front.',
                  'reset',
                  f'turn x, {-tilt:.1f}',
                  f'zoom {target}_left or {target}_right, 6',
                  'set orthoscopic, 1',
                  'deselect',
                  '',
                  '# Both panels are the same structure in the same orientation, so',
                  '# any difference on the page is a difference between the maps.']
        (OUT / f'{target}_density_comparison.pml').write_text('\n'.join(lines) + '\n')

        summary = '  '.join(f'{t}: {r} regions, {v} vox' for t, r, v in counts)
        print(f'  {target:9s} offset {offset:5.1f} A  site {len(site):3d} res  '
              f'{summary}')


if __name__ == '__main__':
    main()
