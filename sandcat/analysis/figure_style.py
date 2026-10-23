"""Shared styling for the prospective analysis figures.

The figure scripts are written to be composed onto one page, so the target
order, display names, palette, type scale and axis treatment are defined once
here and imported by each of them rather than restated. A panel drawn by one
script therefore sits beside a panel drawn by another without the two
disagreeing on a colour or a font size.

This module draws nothing. It exists so that no figure script has to import its
styling from another figure script, which made the dependency between them look
like a dependency on someone else's figure.
"""
from __future__ import annotations

# figure order, not alphabetical. CRBN trails because it is presented
# on its own and is excluded from the site panel entirely.
TARGET_ORDER = ['TEM1', 'IL2', 'PLK1', 'MDM2', 'BCLXL', 'HPV11-E2', 'CRBN']

# the display names used in the figures
DISPLAY = {'BCLXL': 'Bcl-xL', 'CRBN': 'CRBN', 'HPV11-E2': 'HPV-11 E2',
           'IL2': 'IL-2', 'MDM2': 'MDM2', 'PLK1': 'PLK1', 'TEM1': 'TEM-1'}

# CRBN is reported on its own, so it is left out of the
# cross-target site panel rather than shown twice
SITE_EXCLUDE = {'CRBN'}

TEAL, GREY, DARK = '#0DBFBF', '#9aa0a6', '#2b2b2b'

# narrowest axis a per-target panel may use, in bar slots. Below this a target
# with one or three named residues renders as a wall of colour. Read by
# plot_site_residue_figure, which draws those panels.
MIN_SLOTS = 6

# one type scale for every figure in the set, so panels sit together on a page
LABEL_PT, TICK_PT, LEGEND_PT, BAND_PT = 15, 14, 12, 13


def tidy(ax) -> None:
    """Apply the shared axis styling: no top or right spine, faint y grid.

    Args:
        ax: Matplotlib axes to restyle in place.
    """
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', lw=.4, color=GREY, alpha=.35, zorder=0)
