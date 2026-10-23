"""Named residues scored from the cryptic site itself, not from the first candidate.

The companion figure, residue_rank.png, reports the position of the first
candidate that comes within RESIDUE_A of each named residue. That number is a property
of the candidate list rather than of the residue, so four residues lining one
pocket all read rank 1 because a single blob covers them, and a residue can read
better than its own site when an unrelated earlier candidate happens to land
nearby: PLK1 421 reads 5 against a site rank of 6, and three Bcl-xL residues read
1 against a site rank of 4. Neither is evidence about the site.

This figure asks the question the other way round. Given the candidate that
actually matches the held-out pocket, how near the top of the residues it
implicates does each literature residue sit? Every residue in the protein is
ordered by its distance to that candidate and the bar is the named residue's
position in that ordering, so a bar of 3 means two residues of the protein lie
closer to the site than this one does.

The distance is the one the boxplot figures measure: the closest heavy-atom
distance to the candidate centre in each frame of the lambda = 1 state 0
trajectory, summarised by its median, which is the statistic those panels are
already ordered by. A bar here can therefore be traced back to a box in
<target>_residues.png. It is not the static apo-to-voxel distance, which
measures a structure rather than a gate.

Every bar is measured from the cryptic site, never from whichever candidate
happens to be nearest, so a short bar always means the cryptic site itself
reaches the residue. Measuring each residue from its own detecting candidate
would instead rank a residue well for having been found at the wrong pocket.
Colour says whether the cryptic site reaches the residue at all,
within RESIDUE_A. Which candidate reaches the ones it misses is in the csv rather
than on the bars, since a candidate index is meaningless without that target's
whole list.

Bcl-xL is folded to one protomer. It is a homodimer, so its six named residues
are three residues written twice, 45/185, 49/189 and 90/230 at a resSeq offset
of 140, and the pocket reference exists as two symmetry copies. Only one copy
was detected: chain B's carries candidate 4 at 1.48 A, while the nearest
candidate to chain A's is 9.05 A, outside the 5 A match. Scoring both protomers
therefore double counted three residues and reported the other protomer's copies
at 45 A, which measures the distance between the two protomers rather than
anything about the pocket. Each symmetry pair is now scored once, on the
protomer where the site was found.

Colour carries which site found the residue and hatching carries how it moves,
so the two questions stay separable: a residue can be found at the right pocket
without being mobile, and the figure should not merge those into one bar.
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
import MDAnalysis as mda
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


from sandcat.analysis.experiments import (target_dir,
                         HOTSPOTS_FILE, STATE_TOPOLOGY,
                         STATE_TRAJECTORY,
                         ANALYSIS, HIT_A, RESIDUE_A, SIM_APO, POCKET_REFERENCES,
                         apo_residues, pocket_separations,
                         TARGET_RESIDUES, residue_distance_traces,
                         true_pocket)

from sandcat.analysis.figure_style import (BAND_PT, DARK, DISPLAY, GREY, LABEL_PT, LEGEND_PT,
                          MIN_SLOTS, TARGET_ORDER, TICK_PT, tidy)
from sandcat.analysis.residue_ranking import first_candidate_near_coordinates
from sandcat.analysis.literature_numbering import labels_from_apo

warnings.filterwarnings('ignore')

TEAL, PURPLE = '#0DBFBF', '#858ce0'

SITE_COLOUR = {'correct': TEAL, 'other': PURPLE, 'none': GREY}
# The bars carry two independent encodings, fill colour and hatching. The
# swatches keep them apart: the colours are solid fills and the patterns have no
# fill at all, so a pattern swatch cannot be read as a third colour.
SITE_LABEL = {
    'correct': f'reached by the cryptic site (within {RESIDUE_A:g} Å)',
    'other': f'reached by another candidate site '
             f'(beyond {RESIDUE_A:g} Å of the cryptic site)',
    'none': f'reached by no candidate '
            f'(beyond {RESIDUE_A:g} Å of the cryptic site)',
}
# hatching, not a second colour: a residue can be found at the right pocket
# without moving, and merging the two questions into one bar hides that
MOTION_HATCH = {
    'aromatic rotamer switching': '///',
    'side-chain displacement': '...',
    'neither': '',
}
# the furthest residue is TEM-1 ARG244 at 11.88 A, so a 15 A ceiling holds
# every panel on one scale
YMAX = 15.0
# below this figure width a two-column legend is wider than the axes
TWO_COLUMN_WIDTH = 12.0
# The classes are named for what is measured rather than for a mechanism. A
# residue may switch rotamer throughout the trajectory without its rotamer state
# differing between the open and closed ends of the opening coordinate, so a
# switch count establishes mobility and not a contribution to pocket opening.
MOTION_LABEL = {
    'aromatic rotamer switching': 'aromatic rotamer switching',
    'side-chain displacement':
        'side-chain displacement or non-aromatic rotamer switching',
    # not "proximity to the cryptic site": the fill colour already says which
    # site reached the residue, and one plain bar, MDM2 PHE86, is 11.15 A from
    # the cryptic site and was reached by candidate 3 instead.
    'neither': 'reached by proximity only',
}


def median_distances(target: str) -> tuple[dict[int, dict[int, float]], int]:
    """Median over frames of each residue's closest distance to each candidate.

    The same measurement the boxplots show, reduced to the median. Returns the
    per-candidate tables keyed by candidate rank, and the rank of the candidate
    matching the held-out pocket.

    Args:
        target: Target name.

    Returns:
        (per-candidate tables, matched rank). Each table maps simulation resSeq
        to the median over frames of that residue's closest heavy-atom distance
        to that candidate's centre, on state 0.
    """
    folder = target_dir(target) / 'density'
    candidates = pd.read_csv(folder / HOTSPOTS_FILE).sort_values('rank')
    centres = candidates[['x', 'y', 'z']].to_numpy(float)
    site = first_candidate_near_coordinates(
        centres, true_pocket(target, str(SIM_APO[target])), HIT_A)

    trajectory = mdtraj.load(str(folder / STATE_TRAJECTORY),
                             top=str(folder / STATE_TOPOLOGY))
    tables = {}
    for rank, centre in enumerate(centres, start=1):
        tables[rank] = {
            resseq: float(np.median(trace)) for resseq, trace
            in residue_distance_traces(trajectory, centre).items()}
    return tables, site


def protomer_offset(apo_pdb: str) -> int | None:
    """resSeq offset between the protomers of a homo-oligomer, or None.

    Chains are matched by residue name at the offset that maximises agreement,
    the same rule experiments.pocket_with_symmetry_copies uses, because Bcl-xL's
    chain B carries an extra glycine and a position-by-position match slips by
    one. The offset returned is in residue numbering, so 45 and 185 are
    recognised as the same residue of the dimer.

    Args:
        apo_pdb: Path to the apo input structure.

    Returns:
        The residue-number offset between the first two protein chains, or None
        for a monomer or where the chains are not a clean repeat.
    """
    structure = mda.Universe(apo_pdb)
    chains: dict[str, list] = {}
    for residue in structure.select_atoms('protein').residues:
        chains.setdefault(residue.segid, []).append(residue)
    chains = {k: v for k, v in chains.items() if len(v) >= 20}
    if len(chains) < 2:
        return None
    keys = sorted(chains)
    first, second = chains[keys[0]], chains[keys[1]]
    best, best_matches = 0, -1
    for offset in range(-5, 6):
        matches = sum(1 for i, residue in enumerate(first)
                      if 0 <= i + offset < len(second)
                      and residue.resname == second[i + offset].resname)
        if matches > best_matches:
            best, best_matches = offset, matches
    deltas = [second[i + best].resid - residue.resid
              for i, residue in enumerate(first)
              if 0 <= i + best < len(second)
              and residue.resname == second[i + best].resname]
    return int(np.median(deltas)) if deltas else None


def fold_to_one_protomer(target: str, named: list[int],
                         distances: dict[int, float]) -> list[int]:
    """Keep one residue of each symmetry pair, on the protomer carrying the site.

    `distances` is the median distance to the cryptic site's candidate, so the
    nearer member of a pair is the copy on the protomer where the site was
    actually detected. Targets with no second chain are returned unchanged.

    Args:
        target: Target name.
        named: Residues the source paper names, in simulation numbering, which
            for an oligomer lists the same site once per protomer.
        distances: Per-candidate distance tables from median_distances.

    Returns:
        The named residues folded onto the protomer the matched candidate sits
        on, so a site found once is not scored as half a site.
    """
    offset = protomer_offset(str(SIM_APO[target]))
    if not offset:
        return named
    keep, seen = [], set()
    for resseq in named:
        if resseq in seen:
            continue
        partner = next((q for q in named
                        if q != resseq and abs(q - resseq) == abs(offset)), None)
        if partner is None:
            keep.append(resseq)
            continue
        seen.update({resseq, partner})
        best = min((resseq, partner), key=lambda q: distances.get(q, np.inf))
        keep.append(best)
        print(f'    {target}: {resseq} and {partner} are symmetry copies, '
              f'keeping {best} ({distances.get(best, float("nan")):.1f} '
              f'vs {distances.get(resseq + partner - best, float("nan")):.1f} '
              f'Å from the site)')
    return sorted(keep)


def pocket_copies(target: str):
    """The held-out pocket reference split back into its per-chain copies.

    experiments.true_pocket stacks one copy per protein chain, every copy the
    same length, so the stack reshapes back into (chains, atoms, 3). Returns
    None for a monomer.

    Args:
        target: Target name.

    Returns:
        (chain names, (chains, atoms, 3) pocket coordinates) with one copy per
        protein chain, or None for a monomer.
    """
    pocket = true_pocket(target, str(SIM_APO[target]))
    if pocket is None:
        return None
    structure = mda.Universe(str(SIM_APO[target]))
    chains: dict[str, list] = {}
    for residue in structure.select_atoms('protein').residues:
        chains.setdefault(residue.segid, []).append(residue)
    chains = {k: v for k, v in chains.items() if len(v) >= 20}
    if len(chains) < 2:
        return None
    return sorted(chains), pocket.reshape(len(chains), -1, 3)


def protomer_table(target: str) -> tuple[pd.DataFrame, dict] | None:
    """Each protomer's named residues against that protomer's own candidate.

    The folded figure scores a homodimer once, on the protomer where the site
    was found, which is right for counting but hides the more interesting fact:
    the same pocket exists twice and the detector found it once. This splits it
    back out. Each protomer's residues are measured to the candidate nearest
    that protomer's copy of the pocket reference, so no bar ever carries the
    distance between the protomers.

    Colour follows the same rule as every other panel, so the legends match: a
    residue is teal only when the cryptic site reaches it. Protomer A's copy of
    the pocket was never matched by a candidate, so none of its residues can be
    reached by the cryptic site and all three are purple, with their heights
    measured to the nearest candidate to that copy rather than to the other
    protomer's density 45 A away.

    Args:
        target: Target name.

    Returns:
        (table, labels) for an oligomer, with each named residue assigned to its
        own protomer's copy of the pocket, or None for a monomer.
    """
    split = pocket_copies(target)
    if split is None:
        return None
    names, pockets = split
    tables, _ = median_distances(target)
    folder = target_dir(target) / 'density'
    centres = (pd.read_csv(folder / HOTSPOTS_FILE)
               .sort_values('rank')[['x', 'y', 'z']].to_numpy(float))
    motions = pd.read_csv(ANALYSIS / 'ranking_residues.csv').set_index(
        ['target', 'resSeq'])
    apo = apo_residues(str(SIM_APO[target]))

    rows, labels = [], {}
    for index, name in enumerate(names):
        separations = pocket_separations(centres, pockets[index])
        matched = np.where(separations <= HIT_A)[0]
        candidate = int(matched[0]) + 1 if len(matched) else int(
            np.argmin(separations)) + 1
        group = f'protomer {name}'
        # kept short: these sit below the axis in data coordinates, and a long
        # label overflows the frame, which bbox_inches='tight' then compensates
        # for by shrinking the whole plot. The matching distances are in the
        # module docstring and belong in the caption.
        labels[group] = f'{group}\ncandidate {candidate}'
        for resseq in TARGET_RESIDUES.get(target, []):
            if resseq not in apo or resseq not in tables[candidate]:
                continue
            # a residue belongs to the protomer whose pocket copy it sits on
            nearest = int(np.argmin([
                np.linalg.norm(apo[resseq][:, None, :] - p[None, :, :],
                               axis=2).min() for p in pockets]))
            if nearest != index:
                continue
            key = (target, resseq)
            meta = motions.loc[key] if key in motions.index else None
            rows.append({
                'target': target, 'resSeq': resseq, 'group': group,
                'resname': meta.resname if meta is not None else '?',
                'motion': meta.motion if meta is not None else 'neither',
                'median_a': round(tables[candidate][resseq], 2),
                'candidate': candidate,
                # the same rule as every other panel: teal only if the cryptic
                # site itself reaches the residue. A protomer whose pocket copy
                # was never matched has no cryptic site to reach it, so all its
                # residues are "reached by another candidate", which is true.
                'source': 'correct' if (len(matched) and
                                        tables[candidate][resseq] <= RESIDUE_A)
                          else 'other'})
    return (pd.DataFrame(rows), labels) if rows else None



def table() -> pd.DataFrame:
    """One row per named residue: its rank at the cryptic site."""
    motions = pd.read_csv(ANALYSIS / 'ranking_residues.csv')
    motions = motions.set_index(['target', 'resSeq'])

    rows = []
    for target in POCKET_REFERENCES:
        folder = target_dir(target) / 'density'
        if target not in TARGET_RESIDUES or not (folder / HOTSPOTS_FILE).exists():
            continue
        tables, site = median_distances(target)
        if site is None:
            continue
        at_site = tables[site]
        # every bar is ranked against this one ordering, so the heights are
        # comparable across targets and across colours
        order = sorted(at_site, key=lambda q: at_site[q])
        named = fold_to_one_protomer(
            target, [r for r in TARGET_RESIDUES[target] if r in at_site], at_site)

        for resseq in named:
            reached = at_site[resseq] <= RESIDUE_A
            # which candidate does reach it, for the residues the site misses
            elsewhere = None if reached else next(
                (rank for rank in sorted(tables)
                 if tables[rank][resseq] <= RESIDUE_A), None)
            key = (target, resseq)
            meta = motions.loc[key] if key in motions.index else None
            rows.append({
                'target': target, 'resSeq': resseq,
                'resname': meta.resname if meta is not None else '?',
                'motion': meta.motion if meta is not None else 'neither',
                'site_rank': site, 'rank': order.index(resseq) + 1,
                'median_a': round(at_site[resseq], 2),
                'reached_by': elsewhere,
                'source': 'correct' if reached else
                          ('other' if elsewhere else 'none')})
    return pd.DataFrame(rows)


def figure(residues: pd.DataFrame, path: Path, bands: bool,
           group_col: str = 'target', labels: dict | None = None,
           site_labels: dict | None = None, band_text: bool = True) -> None:
    """One bar per residue: its rank at the candidate that found it.
    Args:
        residues: One row per residue, with its distance, mechanism and whether
            the matched candidate reached it.
        path: Where to write the PNG.
        bands: Whether to draw grouping bands under the axis.
        group_col: Column to group the bands by, normally the target.
        labels: Tick label per row. Literature numbering, not simulation.
        site_labels: Legend text per source category.
        band_text: Text under each band, or None to label by group value.
    """
    heights = residues.median_a.to_numpy(dtype=float)
    xs = np.arange(len(residues))

    width = max(7.8, 0.52 * len(residues) + 2.8)
    # the extra height is only needed to hang band names below the axis
    height = 7.2 if (bands and band_text) else 5.4
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    ax.bar(xs, heights, width=.70, zorder=3,
           color=[SITE_COLOUR[s] for s in residues.source],
           hatch=[MOTION_HATCH[m] for m in residues.motion],
           edgecolor=DARK, linewidth=.6)

    # no line at the detection cut. The colours already carry it, and a drawn
    # threshold asserts a hard boundary the method does not have.
    # which candidate reached the purple bars is in site_residue_ranks.csv,
    # not written above them: the colour already says it was not the cryptic
    # site, and a candidate index means nothing without that target's list
    ax.set_xticks(xs)
    rotation = 0 if len(residues) <= 8 else 90
    # labels in the numbering the source papers use, not the construct's
    marks = {t: labels_from_apo(t, str(SIM_APO[t])) for t in set(residues.target)}
    ax.set_xticklabels([marks[t].get(s, f'{n.upper()}{s}') for t, n, s
                        in zip(residues.target, residues.resname,
                               residues.resSeq)],
                       rotation=rotation, fontsize=TICK_PT)
    ax.tick_params(axis='y', labelsize=TICK_PT)
    # two lines: on the per-target figures, which are four bars wide, the single
    # line is taller than the axes and is clipped at the top. The wording
    # matches the boxplot panels, which measure the same thing.
    # "Distance", not "closest distance": what it is the closest of, heavy atoms
    # per frame, goes in the caption rather than onto an already long axis label
    ax.set_ylabel('Distance to the\ncryptic site centre (Å)',
                  fontsize=LABEL_PT)
    # one scale for every panel, so a bar means the same length on each and the
    # per-target figures can be read against one another
    ax.set_ylim(0, YMAX)
    ax.yaxis.set_major_locator(MultipleLocator(2))

    slots = max(len(residues), MIN_SLOTS)
    centre = (len(residues) - 1) / 2.0
    ax.set_xlim(centre - slots / 2.0, centre + slots / 2.0)
    tidy(ax)

    if bands:
        names = (TARGET_ORDER if group_col == 'target'
                 else list(dict.fromkeys(residues[group_col])))
        shown = labels or DISPLAY
        start = 0
        for name in names:
            n = int((residues[group_col] == name).sum())
            if not n:
                continue
            # axes-fraction y, so the band name clears the rotated residue
            # labels whatever the data range is
            if band_text:
                ax.text(start + (n - 1) / 2, -0.30, shown.get(name, name),
                        transform=ax.get_xaxis_transform(), ha='center',
                        va='top', fontsize=BAND_PT, fontweight='bold',
                        color=DARK, clip_on=False)
            if start:
                ax.axvline(start - .5, lw=.6, color=GREY, alpha=.6, zorder=1)
            start += n

    # One legend carrying both facts the bars encode: colour says which pocket
    # found the residue, hatching says how it moves. matplotlib fills a legend
    # column by column, so two columns put the colours in the first and the
    # hatches in the second. Two separate legends collided on the per-target
    # figures, which have four bars and no horizontal room.
    site_labels = site_labels or SITE_LABEL
    sites = [s for s in SITE_COLOUR if (residues.source == s).any()]
    # The site colours are solid fills and the motion classes carry no fill at
    # all, so a pattern swatch cannot be read as a third colour. 'neither' is
    # included, as a plain outline, because it is a class the text reports on in
    # its own right rather than merely the absence of the other two.
    motions = [m for m in MOTION_HATCH if (residues.motion == m).any()]
    handles = [Patch(facecolor=SITE_COLOUR[s], edgecolor=DARK, lw=.6,
                     label=site_labels[s]) for s in sites]
    handles += [Patch(facecolor='none', edgecolor=DARK, lw=.6,
                      hatch=MOTION_HATCH[m], label=MOTION_LABEL[m])
                for m in motions]
    ax.legend(handles=handles, frameon=False, fontsize=LEGEND_PT,
              loc='lower left', bbox_to_anchor=(0.0, 1.01),
              # two columns only once the axes are wide enough to carry them.
              # Tying this to `bands` put a two-column legend on the six-bar
              # homodimer panel, and since bbox_inches='tight' grows the canvas
              # to fit the legend, the axes shrank to a sliver.
              ncol=2 if width >= TWO_COLUMN_WIDTH else 1, borderaxespad=0.0,
              columnspacing=3.0)

    # bbox_inches='tight' so the target names below the axis are kept. They are
    # drawn in data coordinates with annotation_clip off, which constrained
    # layout does not reserve room for, and without this they are cut off.
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    """Write the per-target panels and site_residue_ranks.csv.

    One panel per target in residue_from_site_per_target/, each showing how
    far the residues the source papers name sit from the candidate that
    matched that target's pocket.
    """
    residues = table()
    residues['order'] = residues.target.map(
        {t: i for i, t in enumerate(TARGET_ORDER)})
    residues = (residues.sort_values(['order', 'resSeq'])
                .reset_index(drop=True))
    residues.drop(columns='order').to_csv(
        ANALYSIS / 'site_residue_ranks.csv', index=False)

    figures = ANALYSIS / 'figures'
    figures.mkdir(exist_ok=True)

    out = figures / 'residue_from_site_per_target'
    out.mkdir(exist_ok=True)
    for target in TARGET_ORDER:
        sub = residues[residues.target == target].sort_values('resSeq')
        if sub.empty:
            continue
        # a homodimer gets the split panel instead of the folded one: same
        # residues, but each protomer scored against its own copy of the site
        split = protomer_table(target)
        if split is not None:
            rows, labels = split
            figure(rows.sort_values(['group', 'resSeq']).reset_index(drop=True),
                   out / f'{target}.png', bands=True, group_col='group',
                   labels=labels,
                   # the divider alone separates the protomers; which is which
                   # belongs in the caption, not under the axis
                   band_text=False)
            print(f'  {target}: split into {rows.group.nunique()} protomers, '
                  f'{len(rows)} residues')
            continue
        figure(sub.reset_index(drop=True), out / f'{target}.png', bands=False)

    counts = residues.source.value_counts()
    print(f'site_residue_ranks.csv: {len(residues)} residues, '
          f'{int(counts.get("correct", 0))} found at the true site, '
          f'{int(counts.get("other", 0))} at another candidate, '
          f'{int(counts.get("none", 0))} not reached')
    found = residues[residues.source == 'correct']['rank']
    print(f'rank at the true site : median {found.median():.0f}, '
          f'range {found.min():.0f} to {found.max():.0f}')
    print(f'per target            : {len(TARGET_ORDER)} figures in {out.name}/')
    print('written               : site_residue_ranks.csv')


if __name__ == '__main__':
    main()
