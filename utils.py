"""
Utility functions for the sports analytics dashboard.

The Voronoi computation implemented here is adapted from the coverage control
module in the `ice_hockey_simulator` project【124951572244772†L0-L40】.  It adds
virtual points around the pitch to ensure that all Voronoi cells are bounded,
computes the Voronoi diagram using SciPy, and then clips the resulting
polygons to the field boundaries using Shapely.

This module can easily be extended to include additional analytics such as
density‑weighted centroids and nominal velocities【124951572244772†L42-L106】.
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon as ShapelyPolygon, box

def compute_voronoi(positions: List[List[float]], field_bounds: Dict[str, float]) -> Dict[int, Optional[List[List[float]]]]:
    """
    Compute Voronoi partitions for a set of points within a rectangular field.

    This function adds four virtual points outside the field to bound the Voronoi
    diagram【124951572244772†L15-L22】.  It then computes the Voronoi diagram using
    SciPy and clips each cell to the field boundaries using Shapely.

    Args:
        positions: A list of ``[x, y]`` pairs representing player positions.
        field_bounds: A dictionary defining the rectangle to clip the Voronoi
            cells to.  Expected keys: ``"x_min"``, ``"x_max"``, ``"y_min"``, ``"y_max"``.

    Returns:
        A dictionary mapping each position index to a list of ``[x, y]`` points
        representing the clipped Voronoi cell.  If a cell is unbounded or
        completely outside the field, the value will be ``None``.
    """
    # Add virtual points far outside the field to bound the Voronoi regions
    virtual_points = [
        [field_bounds["x_min"] - 50, field_bounds["y_min"] - 50],
        [field_bounds["x_min"] - 50, field_bounds["y_max"] + 50],
        [field_bounds["x_max"] + 50, field_bounds["y_min"] - 50],
        [field_bounds["x_max"] + 50, field_bounds["y_max"] + 50],
    ]
    all_positions = positions + virtual_points

    # Compute the Voronoi diagram
    voronoi = Voronoi(all_positions)

    voronoi_regions: Dict[int, Optional[List[List[float]]]] = {}
    # Only extract regions corresponding to the actual input positions
    for idx, region_idx in enumerate(voronoi.point_region[: len(positions)]):
        region = voronoi.regions[region_idx]
        # Skip regions that are open (contain a vertex at infinity) or empty
        if not region or -1 in region:
            voronoi_regions[idx] = None
            continue
        # Extract the vertices of the Voronoi cell
        polygon = [voronoi.vertices[i] for i in region]
        # Clip the polygon to the field boundaries
        clipped_polygon = clip_to_bounds(polygon, field_bounds)
        voronoi_regions[idx] = clipped_polygon
    return voronoi_regions

def clip_to_bounds(polygon: List[List[float]], field_bounds: Dict[str, float]) -> Optional[List[List[float]]]:
    """
    Clip a polygon to a rectangular bounding box defined by ``field_bounds``.

    This uses Shapely to perform the clipping operation.  If the polygon lies
    completely outside the bounding box, ``None`` is returned【124951572244772†L108-L134】.

    Args:
        polygon: List of ``[x, y]`` coordinates representing a polygon.
        field_bounds: Dictionary with keys ``"x_min"``, ``"x_max"``, ``"y_min"``, ``"y_max"``.

    Returns:
        A list of clipped polygon vertices (closed), or ``None`` if empty.
    """
    field_box = box(
        field_bounds["x_min"],
        field_bounds["y_min"],
        field_bounds["x_max"],
        field_bounds["y_max"],
    )
    shapely_polygon = ShapelyPolygon(polygon)
    clipped = shapely_polygon.intersection(field_box)
    if clipped.is_empty:
        return None
    # Convert to list of [x, y] pairs.  Shapely returns a closed ring where the
    # first and last points are the same; we remove the duplicate last point.
    coords = list(clipped.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [[float(x), float(y)] for x, y in coords]