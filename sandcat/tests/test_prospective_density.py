"""Tests for the prospective (apo-only) parts of the probe-density analysis.

Two things are covered here, both of which exist to remove prior knowledge of
the pocket from the pipeline.

``to_bulk_units`` expresses a density grid as a multiple of the density the
probes would have if they were spread uniformly through the box. Raw grids scale
with how many probes a system happens to contain, and two builds of the same
protein can differ by a probe or two, so raw grids from different systems are not
comparable and cannot be subtracted. In bulk units they are, and the threshold
becomes a plain factor rather than a number meaning something different in every
box.

``density_hotspots`` derives a pocket centre from the difference map. Borsatto et
al. 2024 anchor their contact analysis on residues within 7 A of the centre of a
known cryptic pocket, which is unavailable when only an apo structure exists.
Ranking the connected components of the difference density supplies that centre
from the data instead, so the tests check that the ranking, the centroids and
the empty case all behave.
"""
from __future__ import annotations

from pathlib import Path


import numpy as np
import pytest
from MDAnalysis.lib.mdamath import box_volume

from sandcat.analysis.compute_probe_density import (density_hotspots,
                                                    to_bulk_units)


def uniform_grid(value: float, shape: tuple[int, int, int] = (4, 4, 4)) -> np.ndarray:
    """A grid holding one value everywhere."""
    return np.full(shape, float(value))

def test_uniform_density_becomes_one_times_bulk() -> None:
    """A grid at exactly the bulk density normalises to 1.0 everywhere."""
    n_probes, spacing, volume = 600, 1.0, 300000.0
    bulk = n_probes * spacing ** 3 / volume
    result = to_bulk_units(uniform_grid(bulk), n_probes, spacing, volume)
    assert np.allclose(result, 1.0)

def test_scales_linearly_with_the_raw_density() -> None:
    """Doubling the raw density doubles the value in bulk units."""
    n_probes, spacing, volume = 600, 1.0, 300000.0
    single = to_bulk_units(uniform_grid(0.01), n_probes, spacing, volume)
    double = to_bulk_units(uniform_grid(0.02), n_probes, spacing, volume)
    assert np.allclose(double, 2.0 * single)

def test_probe_count_difference_cancels() -> None:
    """Two systems differing by one probe agree once normalised.

    This is the case the function exists for: PLK1's unbiased replicate 1
    carries 276 benzenes where replicates 0 and 2 carry 277, so the raw
    grids are not on the same scale and averaging them compares slightly
    different systems.
    """
    spacing, volume = 1.0, 300000.0
    raw_276 = uniform_grid(276 * spacing ** 3 / volume)
    raw_277 = uniform_grid(277 * spacing ** 3 / volume)
    assert not np.allclose(raw_276, raw_277)
    assert np.allclose(to_bulk_units(raw_276, 276, spacing, volume),
                       to_bulk_units(raw_277, 277, spacing, volume))

def test_voxel_volume_not_edge_length_sets_the_scale() -> None:
    """Bulk density goes with spacing cubed, so halving the edge is eight-fold."""
    n_probes, volume = 600, 300000.0
    fine = to_bulk_units(uniform_grid(0.01), n_probes, 0.5, volume)
    coarse = to_bulk_units(uniform_grid(0.01), n_probes, 1.0, volume)
    assert np.allclose(fine, 8.0 * coarse)

def test_larger_box_raises_the_value_in_bulk_units() -> None:
    """The same raw density is more enriched in a box with fewer probes per volume."""
    n_probes, spacing = 600, 1.0
    big = to_bulk_units(uniform_grid(0.01), n_probes, spacing, 600000.0)
    small = to_bulk_units(uniform_grid(0.01), n_probes, spacing, 300000.0)
    assert np.all(big > small)

@pytest.mark.parametrize('n_probes, volume', [(0, 300000.0), (600, 0.0),
                                              (-1, 300000.0), (600, -1.0)])
def test_unusable_system_returns_the_grid_untouched(n_probes: int, volume: float) -> None:
    """With no probe count or no box, the grid is left in raw units.

    Returning it unchanged rather than raising keeps a run going on a
    trajectory that carries no box, and the caller is warned; the grid is
    then simply not comparable with any other system.
    """
    grid = uniform_grid(0.01)
    result = to_bulk_units(grid, n_probes, 1.0, volume)
    assert np.allclose(result, grid)


def grid_with_two_blobs() -> np.ndarray:
    """A weak 2x2x2 blob and a strong 3x3x3 blob, well separated."""
    density = np.zeros((30, 30, 30))
    density[2:4, 2:4, 2:4] = 0.05        # 8 voxels, total 0.40
    density[20:23, 20:23, 20:23] = 0.10  # 27 voxels, total 2.70
    return density

def test_ranks_components_by_total_density() -> None:
    hotspots = density_hotspots(
        grid_with_two_blobs(), np.zeros(3), 1.0, threshold=0.01,
    )
    assert len(hotspots) == 2
    assert list(hotspots['rank']) == [1, 2]
    assert hotspots.n_voxels.tolist() == [27, 8]
    assert hotspots.total_density.iloc[0] == pytest.approx(2.70)
    assert hotspots.total_density.iloc[1] == pytest.approx(0.40)

def test_centroid_lands_in_the_blob() -> None:
    """The rank-1 centroid must sit at the centre of the strong blob."""
    hotspots = density_hotspots(
        grid_with_two_blobs(), np.zeros(3), 1.0, threshold=0.01,
    )
    top = hotspots.iloc[0]
    assert (top.x, top.y, top.z) == pytest.approx((21.0, 21.0, 21.0))

def test_origin_offsets_the_centroid() -> None:
    """Centroids are reported in Angstroms in the grid frame, not voxel indices."""
    origin = np.array([-10.0, 5.0, 100.0])
    hotspots = density_hotspots(
        grid_with_two_blobs(), origin, 1.0, threshold=0.01,
    )
    top = hotspots.iloc[0]
    assert (top.x, top.y, top.z) == pytest.approx((11.0, 26.0, 121.0))

def test_centroid_is_density_weighted() -> None:
    """An asymmetric blob pulls the centre toward its dense end."""
    density = np.zeros((20, 20, 20))
    density[5, 5, 5] = 0.10
    density[6, 5, 5] = 0.90          # ten times heavier
    hotspots = density_hotspots(density, np.zeros(3), 1.0, threshold=0.01,
                                min_voxels=1)
    # unweighted midpoint would be 5.5; weighting pulls it to 5.9
    assert hotspots.x.iloc[0] == pytest.approx(5.9)

def test_volume_follows_voxel_size() -> None:
    hotspots = density_hotspots(
        grid_with_two_blobs(), np.zeros(3), 2.0, threshold=0.01,
    )
    assert hotspots.volume_a3.iloc[0] == pytest.approx(27 * 8.0)

def test_threshold_excludes_the_weak_blob() -> None:
    hotspots = density_hotspots(
        grid_with_two_blobs(), np.zeros(3), 1.0, threshold=0.07,
    )
    assert len(hotspots) == 1
    assert hotspots.n_voxels.iloc[0] == 27

def test_no_hotspot_returns_empty_not_an_error() -> None:
    """A target with no cryptic site must produce no centre, not a crash.

    This is the negative-control path: ubiquitin is expected to land here,
    and the pipeline has to survive it and say so.
    """
    hotspots = density_hotspots(
        grid_with_two_blobs(), np.zeros(3), 1.0, threshold=1.0,
    )
    assert hotspots.empty
    assert 'total_density' in hotspots.columns


# Bulk normalisation must use the true cell volume, not a*b*c.
#
# Every system in this benchmark is solvated in a truncated octahedron, cell
# angles 60, 60 and 90 degrees, where the product of the edge lengths
# overestimates the volume by sqrt(2). Normalising with it would make the bulk
# density too small by that factor and inflate every map by 41%, so a cut
# nominally at 8x bulk would in truth sit at 5.7x.
#

def test_product_of_edges_overestimates_a_sheared_cell() -> None:
    """The case from the review: 10, 10, 10 at 90, 90, 60 degrees."""
    dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 60.0])
    assert float(np.prod(dimensions[:3])) == pytest.approx(1000.0)
    assert float(box_volume(dimensions)) == pytest.approx(866.025, abs=1e-3)

def test_truncated_octahedron_is_overestimated_by_root_two() -> None:
    """The geometry these simulations actually use."""
    dimensions = np.array([85.1, 85.1, 85.1, 60.0, 60.0, 90.0])
    ratio = float(np.prod(dimensions[:3])) / float(box_volume(dimensions))
    assert ratio == pytest.approx(2.0 ** 0.5, abs=1e-4)

def test_orthogonal_cell_is_unaffected() -> None:
    """Where the angles are all 90, the two agree and nothing changes."""
    dimensions = np.array([30.0, 40.0, 50.0, 90.0, 90.0, 90.0])
    assert float(box_volume(dimensions)) == pytest.approx(60000.0)

def test_overestimated_volume_inflates_the_normalised_density() -> None:
    """What the error does downstream: a 41% inflation of every voxel.

    A voxel reported at the 8x cut would really be at 5.7x bulk, so the
    threshold does not mean what it says and components cross it that
    should not.
    """
    dimensions = np.array([85.1, 85.1, 85.1, 60.0, 60.0, 90.0])
    grid = np.full((3, 3, 3), 0.01)
    wrong = to_bulk_units(grid, 600, 1.0, float(np.prod(dimensions[:3])))
    right = to_bulk_units(grid, 600, 1.0, float(box_volume(dimensions)))
    assert np.allclose(wrong / right, 2.0 ** 0.5, atol=1e-4)
