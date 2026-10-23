"""One panel per candidate: the residues at it and how each of them moves.

This is the prospective output of the pipeline. For every candidate the lambda =
1 difference detector proposes, on every target, it draws the residues forming
that candidate's environment and the distribution of their closest approach to
it over the state 0 trajectory. Nothing here is scored against a reference and
no residue is marked as correct, because prospectively there is no answer key:
the panels say what the method proposes, not whether it was right.

A residue is included when the median over frames of its closest heavy-atom
distance to the candidate centre is within RESIDUE_A. That is deliberately the
same quantity the box itself shows and the same cut residue_from_site uses to
decide whether a residue is reached, so a residue drawn here is exactly a
residue that figure would count, and no bar can exceed the stated radius.

The earlier rule tested a static distance in the apo input and then plotted an
ensemble median, which are different measurements: it admitted residues that
start beside the site and spend the run 10 A away, TEM-1 SER223 among them at
7.3 A in the apo structure and 10.3 A across the trajectory, while excluding
residues that only approach once the pocket opens. Changing it moved the drawn
set on 31 of the 105 panels and brought in four named residues.

A wider apo shell is still used as a pre-filter, since medians only have to be
computed for residues that could plausibly qualify. Panels are ordered with the
nearest residue at the top.

Hatching carries how the residue behaves, measured on state 0 and judged
against the residue's own protein rather than an absolute scale. Diagonal
hatching is aromatic rotamer switching, an aromatic side chain in the upper half
of its protein by dwell-filtered switching. Dotted is side-chain displacement,
anything else in the upper half by either switching or displacement, which
includes non-aromatic rotamer switches. A plain box is a residue in the lower
half on both measures, present because the density reached it rather than
because it moves. The labels name what is measured rather than a mechanism: a
residue can switch rotamer throughout the run without its rotamer state
differing between open and closed frames.

Files are written as prospective_candidates/<TARGET>/rank_NN.png, one per
candidate, in the detector's own ranking order, so the series can be read the
way a user would work down the list.

Within a panel every box is the same colour. No colour distinguishes a known
residue from an unknown one, because prospectively there is no answer key to
draw one from. The only key on these panels is the hatching, which is measured
from the trajectory and is therefore available on any target. The CRBN top-five
set written for the SI varies colour between panels, not within one, so that
each panel matches its own density mesh in the companion structure figure.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mdtraj
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


from sandcat.analysis.experiments import (target_dir,
                         HOTSPOTS_FILE, NON_PROTEIN, PREFILTER_A,
                         closest_approach,
                         STATE_TOPOLOGY, STATE_TRAJECTORY,
                         ANALYSIS, RESIDUE_A, SIM_APO, POCKET_REFERENCES, apo_residues,
                         residue_distance_traces)
from sandcat.analysis.literature_numbering import labels_from_topology
from sandcat.analysis.figure_style import DARK, DISPLAY, GREY, LABEL_PT, LEGEND_PT, TICK_PT
from sandcat.analysis.plot_site_residue_figure import MOTION_HATCH, MOTION_LABEL
from sandcat.analysis.residue_motion import motion_table
from sandcat.analysis.residue_ranking import motion_class

warnings.filterwarnings('ignore')

OUT = ANALYSIS / 'figures' / 'prospective_candidates'
# The SI's CRBN figure: the five densest candidates, each drawn in the colour
# its density mesh carries in the accompanying structure panel, so a reader can
# match a box panel to a site by colour. These are the colours set as
# ttx_purple, ttx_blue, ttx_red, ttx_teal and ttx_yellow in the companion
# PyMOL script, in candidate-rank order.
TOP5_OUT = ANALYSIS / 'figures' / 'crbn_top5_candidates'
TOP5_COLOURS = ('#858CE0', '#408FC4', '#F56170', '#0DBFBF', '#FFC766')

# A residue is drawn when its trajectory median distance to the candidate is
# within RESIDUE_A, imported above rather than restated: it is the same quantity
# the box shows and the same cut residue_from_site thresholds.
# PREFILTER_A, the apo shell that bounds that computation, is imported with
# the other radii so every cutoff on this path is declared in one place.
# The plain-bar label states the inclusion radius; the shared MOTION_LABEL is
# left unchanged for the census and residue_from_site.
PANEL_MOTION_LABEL = dict(MOTION_LABEL)
PANEL_MOTION_LABEL['neither'] = (
    f'reached by proximity only (within {RESIDUE_A:g} Å)')
# residues drawn per panel, nearest first
N_SHOW = 12
TEAL = '#0DBFBF'


def panel(traces: dict[int, np.ndarray], names: dict[int, str],
          motions: dict[int, str], title: str, path: Path,
          labels_by_resseq: dict[int, str] | None = None,
          box_colour: str = TEAL) -> None:
    """Horizontal boxplots, nearest residue at the top, hatched by mechanism.
    Args:
        traces: Per-frame distances per residue, keyed by resSeq.
        names: resSeq mapped to three-letter residue name, used when a residue
            has no literature label.
        motions: resSeq mapped to its mechanism, which sets the hatching.
        title: Panel title, the candidate rank.
        path: Where to write the PNG.
        labels_by_resseq: Literature labels, so ticks match the source papers.
        box_colour: Box fill, one colour for the whole panel. Defaults to teal;
            the CRBN top-five set passes the colour of that candidate's mesh in
            the companion structure figure.
    """
    order = sorted(traces, key=lambda r: float(np.median(traces[r])))[:N_SHOW]
    # matplotlib numbers boxplot positions upwards, so reverse to put the
    # nearest residue at the top of the panel
    order = order[::-1]
    data = [traces[r] for r in order]
    labels_by_resseq = labels_by_resseq or {}
    labels = [labels_by_resseq.get(r, f'{names[r].upper()}{r}') for r in order]

    height = max(3.0, 0.42 * len(order) + 1.6)
    fig, ax = plt.subplots(figsize=(7.4, height), constrained_layout=True)
    box = ax.boxplot(data, vert=False, widths=0.62, patch_artist=True,
                     medianprops=dict(color='white', lw=1.6),
                     flierprops=dict(marker='.', markersize=3, alpha=0.35,
                                     markeredgecolor='none',
                                     markerfacecolor=GREY))
    for patch, resseq in zip(box['boxes'], order):
        patch.set_facecolor(box_colour)
        patch.set_edgecolor(DARK)
        patch.set_linewidth(0.6)
        patch.set_hatch(MOTION_HATCH[motions.get(resseq, 'neither')])

    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(labels, fontsize=TICK_PT)
    ax.tick_params(axis='x', labelsize=TICK_PT)
    ax.set_xlabel('Distance to the candidate cryptic site centre (Å)',
                  fontsize=LABEL_PT)
    ax.set_xlim(left=0)
    ax.grid(axis='x', lw=0.4, color=GREY, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=LABEL_PT, fontweight='bold')

    # Always draw all three key rows so axes height, and hence box size, is
    # the same across panels. Fill is uniform teal, so this is the only key.
    handles = [Patch(facecolor='white', edgecolor=DARK, lw=.6,
                     hatch=MOTION_HATCH[m], label=PANEL_MOTION_LABEL[m])
               for m in MOTION_HATCH]
    ax.legend(handles=handles, frameon=False, fontsize=LEGEND_PT,
              loc='lower left', bbox_to_anchor=(0.0, 1.03))
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    """Write one panel per candidate per target, and the csv behind them.

    Panels go to prospective_candidates/<TARGET>/rank_NN.png in the
    detector's own ranking order, and prospective_candidate_residues.csv
    carries one row per drawn residue so any panel can be rebuilt from it.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in POCKET_REFERENCES:
        folder = target_dir(target) / 'density'
        hotspots = folder / HOTSPOTS_FILE
        if not hotspots.exists():
            continue
        candidates = pd.read_csv(hotspots).sort_values('rank')
        if candidates.empty:
            print(f'  {DISPLAY.get(target, target):10s} no candidates, '
                  f'nothing to draw')
            continue

        # loaded once per target, not once per candidate
        trajectory = mdtraj.load(str(folder / STATE_TRAJECTORY),
                                 top=str(folder / STATE_TOPOLOGY))
        apo = apo_residues(str(SIM_APO[target]))
        names = {residue.resSeq: residue.name
                 for residue in trajectory.topology.residues
                 if residue.name not in NON_PROTEIN}
        # labels in the numbering the source papers use, not the construct's
        labels_by_resseq = labels_from_topology(target, trajectory.topology)

        motion = motion_table(folder)
        motion = motion.assign(n_residues=len(motion))
        motion['motion'] = motion.apply(motion_class, axis=1)
        motions = dict(zip(motion.resSeq, motion.motion))

        plain = OUT / target
        plain.mkdir(exist_ok=True)
        for row in candidates.itertuples():
            centre = np.array([row.x, row.y, row.z])
            nearby = {resseq for resseq, xyz in apo.items()
                      if closest_approach(xyz, centre) <= PREFILTER_A}
            traces = {resseq: trace
                      for resseq, trace in residue_distance_traces(trajectory, centre, nearby).items()
                      if float(np.median(trace)) <= RESIDUE_A}
            if not traces:
                continue
            # the target is the folder name and the candidate count is in
            # prospective_candidate_residues.csv, so the panel carries only the
            # one thing that distinguishes it from its neighbours
            title = f'candidate {int(row.rank)}'
            name = f'rank_{int(row.rank):02d}.png'
            panel(traces, names, motions, title, plain / name,
                  labels_by_resseq=labels_by_resseq)
            # The SI figure for CRBN repeats the five densest candidates, each
            # in the colour of its own mesh in the structure panel. Same data as
            # the panels above, drawn again so the two halves of the figure can
            # be read together.
            if target == 'CRBN' and int(row.rank) <= len(TOP5_COLOURS):
                TOP5_OUT.mkdir(parents=True, exist_ok=True)
                panel(traces, names, motions, title, TOP5_OUT / name,
                      labels_by_resseq=labels_by_resseq,
                      box_colour=TOP5_COLOURS[int(row.rank) - 1])
            shown = sorted(traces, key=lambda r: float(np.median(traces[r])))[:N_SHOW]
            # every drawn residue with its median and mechanism, so each panel
            # can be rebuilt or replotted from the csv alone
            for resseq in shown:
                rows.append({
                    'target': target, 'rank': int(row.rank),
                    'n_voxels': int(row.n_voxels),
                    'resSeq': int(resseq),
                    'resname': names[resseq].upper(),
                    'label': labels_by_resseq.get(resseq,
                                                  f'{names[resseq].upper()}{resseq}'),
                    'median_a': float(np.median(traces[resseq])),
                    'q1_a': float(np.percentile(traces[resseq], 25)),
                    'q3_a': float(np.percentile(traces[resseq], 75)),
                    'min_a': float(traces[resseq].min()),
                    'max_a': float(traces[resseq].max()),
                    'motion': motions.get(resseq, 'neither'),
                    'residues_within_reach': len(traces)})
        drawn = len({r['rank'] for r in rows if r['target'] == target})
        print(f'  {DISPLAY.get(target, target):10s} {drawn:3d} candidates drawn, '
              f'{len(candidates)} proposed')

    table = pd.DataFrame(rows)
    table.to_csv(ANALYSIS / 'prospective_candidate_residues.csv', index=False)
    panels = table.groupby(['target', 'rank']).size()
    print(f'\n{len(panels)} panels in {OUT.name}/, '
          f'{len(table)} drawn residues in prospective_candidate_residues.csv')
    print(f'median residues within {RESIDUE_A:g} A of a candidate: '
          f'{table.groupby(["target", "rank"]).residues_within_reach.first().median():.0f}')


if __name__ == '__main__':
    main()
