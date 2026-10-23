"""Overlay density sessions: one structure, both maps on it, toggled by hand.

Same objects and same meshes as the side-by-side sessions, but with nothing
shifted. One protein, the unbiased MSMD density and the lambda = 1 difference
density both loaded at their true position, so the two meshes sit on the same
coordinates and only one is enabled at a time:

  enable mesh_msmd     disable mesh_diffl1      the MSMD panel
  disable mesh_msmd    enable mesh_diffl1       the difference panel

A figure made by toggling is better than a side-by-side render for the thing
these panels are for. Both images come from one camera on one copy of the
structure, so nothing can drift between them: no second copy to be translated,
no separation to keep level, no view that has to accommodate a pair. The only
difference between the two images is the map, which is exactly the claim the
figure makes.

Both meshes are purple and both are cut at the detector's threshold, in
multiples of bulk, deliberately. The maps are normalised to bulk probe density,
so a voxel at that cut means the same occupancy in each, and keeping the colour
identical means the two images differ in one thing only. The session opens with
MSMD showing.

The camera looks straight down the line from the centre of the protein to the
marked residues, so the yellow faces the viewer. That is simple here in a way it
is not for a pair: with one object there is no separation axis to keep
horizontal, so the view is free to face the site.

The target is marked on the cartoon in yellow with no sticks, except
CRBN, whose target is a single tryptophan out of 327 residues. The cartoon is
45% transparent so the mesh reads through it, and the marked residues are set
back to opaque so the yellow is not washed out by the same setting.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from gridData import Grid


from sandcat.analysis.compute_probe_density import write_dx_file

from sandcat.analysis.experiments import (target_dir,
                         ANALYSIS, SIM_APO, POCKET_REFERENCES, analysis_relative,
                         bundled_apo, pml_preamble)
from sandcat.analysis.make_comparison_pmls import (BLUE, FOV_DEG, PANELS,
                                  marked_residues,
                                  view_matrix,
                                  LIGHT_TEAL, PURPLE, RED, STICKS, THRESHOLD, YELLOW,
                                  coordinates, find_map,
                                  regions, view_frame)

warnings.filterwarnings('ignore')

OUT = ANALYSIS / 'pymol'

PAD_A = 4.0           # margin round the protein, roughly what the mesh adds


def set_view_block_fov(rotation, centre, half_height, depth) -> list[str]:
    """A PyMOL set_view call for this camera, orthoscopic and framed to fit.

    PyMOL's nine rotation floats are column-major, so reading them row-major
    into a 3x3 gives a matrix whose COLUMNS are the camera axes in world
    coordinates, not its rows. `rotation` is built rows-first, camera x, y then
    z, so it is emitted transposed.

    The column reading is what PyMOL itself does, which the documentation does
    not make plain: after `turn y, 40` a marker at world +z renders to the
    right of one at -z, and only the column reading predicts that. Writing the
    rows directly applies the inverse rotation, which looks plausible and is
    wrong.

    The distance is derived from the protein's own extent rather than from
    `zoom`, which fits a bounding sphere and leaves a wide margin. At the
    default field of view the visible half-height is distance * tan(fov / 2).

    Args:
        rotation: the 3x3 camera frame, rows being the camera x, y and z axes
            in world coordinates.
        centre: the point the camera looks at, in world coordinates.
        half_height: half the extent that must fit vertically, in angstrom,
            which sets the camera distance.
        depth: half the extent along the viewing direction, in angstrom, which
            sets the near and far clipping planes either side of the centre.

    Returns:
        The lines of the set_view call, to be joined into the .pml.
    """
    distance = half_height / np.tan(np.radians(FOV_DEG / 2.0)) * 1.04
    r = rotation.T
    return view_matrix(r, centre, distance, depth, FOV_DEG)


def main() -> None:
    """Write one overlay session per target, skipping what is incomplete.

    A target is skipped, with a line saying why, if its apo structure or either
    of its two maps is absent, so that a partial analysis tree yields sessions
    for the targets it does cover rather than an error.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    for target in POCKET_REFERENCES:
        folder = target_dir(target) / 'density'
        # the apo input: the structure the maps are in frame with, and the
        # one a reader can download from the PDB to check the session against.
        # Copied into the analysis directory so the session stays loadable once
        # the directory is moved or published.
        if not Path(SIM_APO[target]).exists():
            print(f'  {target:9s} no structure, skipped')
            continue
        protein = bundled_apo(target)
        missing = [f for _, f, _ in PANELS if not find_map(folder, f).exists()]
        if missing:
            print(f'  {target:9s} missing {missing}, skipped')
            continue

        xyz = coordinates(protein)
        site_ids, site_xyz = marked_residues(target, protein)
        focus = site_xyz.mean(axis=0) if len(site_xyz) else xyz.mean(axis=0)

        rotation, _ = view_frame(xyz, focus)
        centre = xyz.mean(axis=0)
        width, height, depth = (float(np.ptp((xyz - centre) @ axis)) + 2 * PAD_A
                                for axis in rotation)
        # fit the wider of the two in-plane extents, so the session is framed
        # correctly in a square window and has margin to spare in a wide one
        half_height = max(height, width) / 2.0

        lines = ['# ' + '=' * 70,
                 f'#  {target}: probe density, two maps overlaid on one structure',
                 '#',
                 '#  Nothing is shifted. Both meshes sit on the same coordinates,',
                 '#  so toggle one off to make each figure:',
                 '#',
                 '#      enable mesh_msmd    disable mesh_diffl1',
                 '#      disable mesh_msmd   enable mesh_diffl1',
                 '#',
                 f'#  mesh_msmd    {PANELS[0][2]}',
                 f'#  mesh_diffl1  {PANELS[1][2]}',
                 '#',
                 '#  LIGHT TEAL  protein',
                 '#  YELLOW  the target, '
                 + ('cartoon and sticks' if target in STICKS
                    else 'cartoon only, no sticks'),
                 '#  PURPLE  density, both maps, the same colour on purpose so',
                 '#          the two figures differ in the map and nothing else',
                 '#',
                 f'#  Both maps are cut at {THRESHOLD:g}x bulk and are comparable at',
                 '#  that cut because both are normalised to bulk probe density.',
                 '#',
                 '#  The view faces the marked residues, so the yellow is at the',
                 '#  front. "orient" or "reset" will destroy it.',
                 '# ' + '=' * 70,
                 '',
                 *pml_preamble(OUT / f'{target}_density_overlay.pml'),
                 '# --- Colours ---------------------------------------------------',
                 f'set_color ttx_light_teal, {LIGHT_TEAL}',
                 f'set_color ttx_purple, {PURPLE}',
                 f'set_color ttx_yellow, {YELLOW}',
                 f'set_color ttx_blue, {BLUE}',
                 f'set_color ttx_red, {RED}',
                 '',
                 '# --- Structure -------------------------------------------------',
                 f'load {analysis_relative(protein)}, {target}',
                 'hide everything',
                 f'show cartoon, {target}',
                 f'color ttx_light_teal, {target}',
                 'hide everything, elem H',
                 '']

        provenance = f'{len(site_ids)} residues named by the source papers'
        if site_ids:
            resi = '+'.join(str(r) for r in site_ids)
            lines += ['# --- The target -------------------------------------------------',
                      f'# {provenance}',
                      f'select true_site, {target} and resi {resi}',
                      'color ttx_yellow, true_site']
            if target in STICKS:
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
                      'set cartoon_transparency, 0.0, true_site',
                      'deselect', '']
        else:
            lines += ['# --- Nothing marked ---------------------------------------------',
                      '# Ubiquitin is the negative control: no cryptic pocket, no',
                      '# named residues, so there is nothing to mark.',
                      '']

        counts = []
        for tag, filename, description in PANELS:
            grid = Grid(str(find_map(folder, filename)))
            density = grid.grid
            labelled, order = regions(density, THRESHOLD)
            keep = np.zeros_like(density)
            for marker in order:
                selected = labelled == marker
                keep[selected] = density[selected]
            path = OUT / f'{target}_ovl_{tag}.dx'
            write_dx_file(keep, np.array(grid.origin), float(grid.delta[0]),
                          str(path))
            counts.append((tag, len(order), int((keep > 0).sum())))
            lines += [f'# --- {tag}: {description} ---',
                      f'#     {len(order)} regions, {int((keep > 0).sum())} voxels',
                      f'load {analysis_relative(path)}, dens_{tag}',
                      f'isomesh mesh_{tag}, dens_{tag}, {THRESHOLD:g}',
                      f'color ttx_purple, mesh_{tag}',
                      f'set mesh_width, 1.2, mesh_{tag}',
                      '']

        lines += ['# --- Which map is showing ---------------------------------------',
                  '# The session opens on MSMD. Swap these two lines for the',
                  '# lambda = 1 difference, or toggle them in the object panel.',
                  'enable mesh_msmd',
                  'disable mesh_diffl1',
                  '',
                  '# --- Final state -----------------------------------------------',
                  'bg_color white',
                  'set mesh_quality, 2',
                  'set ray_shadows, 0',
                  'set cartoon_fancy_helices, 1',
                  '',
                  '# --- View -------------------------------------------------------',
                  '# The camera z axis points from the centre of the protein to the',
                  '# marked residues, so the yellow is at the front.']
        lines += set_view_block_fov(rotation, centre, half_height, depth)
        lines += ['set orthoscopic, 1',
                  'deselect',
                  '',
                  '# One structure, one camera, two meshes on the same coordinates,',
                  '# so a pair of figures made by toggling differs in the map alone.']
        (OUT / f'{target}_density_overlay.pml').write_text('\n'.join(lines) + '\n')

        summary = '  '.join(f'{t}: {r} regions, {v} vox' for t, r, v in counts)
        print(f'  {target:9s} site {len(site_ids):3d} res  {summary}')


if __name__ == '__main__':
    main()
