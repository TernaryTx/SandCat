"""Ranking figures: where the detector put each cryptic site and each named residue.

Rank is plotted rather than a score, because rank is what a user experiences:
they work down a ranked list, so the number that matters is the position at
which each site and each named residue first appears. Residues are not
restricted to aromatics, since a site can be gated by backbone movement, as
TEM-1's is, whose rank-1 hotspot has twenty residues within 7 A of which none
is aromatic; an aromatics-only figure cannot show such a target at all.

One figure is written, site_rank.png, giving the rank of the true site on each
of the six benchmark targets. CRBN is reported on its own and the ubiquitin
negative control has no site to rank, so both are excluded; the candidate count
each bar was drawn from is printed beneath it. The panel carries no title, so it
can be placed and captioned as a figure rather than by a heading on the panel.

A site's rank is the position of the first candidate within HIT_A of the
held-out pocket reference. The companion per-residue ranks, the position of the
first candidate within RESIDUE_A of any of a residue's heavy atoms, are written
to ranking_residues.csv by residue_ranking and drawn by
plot_site_residue_figure.

Styling comes from figure_style, which every figure script in the set shares.

The detector is the production one, a flat 6x bulk cut with a 3-voxel floor on
the lambda = 1 difference map, the only configuration that finds all six
benchmark sites with no ubiquitin candidates.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sandcat.analysis.experiments import ANALYSIS

from sandcat.analysis.figure_style import (DISPLAY, LABEL_PT, SITE_EXCLUDE, TARGET_ORDER, TEAL,
                          TICK_PT, tidy)


def site_figure(sites: pd.DataFrame, path: Path) -> None:
    """One bar per target: the rank at which the true site was proposed.

    Candidates are the production detector's, a flat 6x bulk cut with the
    3-voxel floor, so the counts under each name are the list a user would
    actually work down.

    One colour throughout and no reference line: the rank is already on the y
    axis, and marking a top-5 threshold would assert a cutoff the method does
    not have.

    Args:
        sites: One row per target, carrying the rank of the candidate that
            matched its held-out pocket.
        path: Where to write the PNG.
    """
    ranks = sites.region_rank.to_numpy(dtype=float)
    x = np.arange(len(sites))
    fig, ax = plt.subplots(figsize=(9.6, 5.0), constrained_layout=True)
    ax.bar(x, ranks, width=.62, zorder=3, color=TEAL)
    ax.set_xticks(x)
    # the candidate count rides with the target name rather than floating under
    # the axis, where it overprinted the tick label
    # 'n = 20' rather than '20 candidates': at this font size the long form
    # runs into its neighbours
    ax.set_xticklabels([f'{DISPLAY[t]}\nn = {int(n)}'
                        for t, n in zip(sites.index, sites.n_candidates)],
                       fontsize=TICK_PT)
    ax.set_ylabel('Rank of the true cryptic site', fontsize=LABEL_PT)
    ax.set_ylim(0, ranks.max() + 0.6)
    ax.set_yticks(np.arange(0, ranks.max() + 1, 1))
    ax.tick_params(axis='y', labelsize=TICK_PT)
    tidy(ax)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Write site_rank.png from the region table residue_ranking produced."""
    regions = pd.read_csv(ANALYSIS / 'ranking_regions.csv')
    sites = (regions[(regions.source == 'diff_l1') & (regions.target != 'UB')]
             .set_index('target').reindex(TARGET_ORDER))

    figures = ANALYSIS / 'figures'
    figures.mkdir(exist_ok=True)
    site_figure(sites[~sites.index.isin(SITE_EXCLUDE)],
                figures / 'site_rank.png')

    shown = sites[~sites.index.isin(SITE_EXCLUDE)]
    ranks = shown.region_rank.to_numpy(dtype=float)
    print(f'site_rank.png     : {len(shown)} targets '
          f'({", ".join(sorted(SITE_EXCLUDE))} excluded), '
          f'ranks {ranks.min():.0f} to {ranks.max():.0f}')


if __name__ == '__main__':
    main()
