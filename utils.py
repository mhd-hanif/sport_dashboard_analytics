"""
Utilities for the Sunbears Dashboard
- Voronoi partition (bounded + clipped)
- Hockey rink shapes (NHL, simplified)
"""

from typing import Dict, List, Optional
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon as ShapelyPolygon, box


def compute_voronoi(positions: List[List[float]], field_bounds: Dict[str, float]) -> Dict[int, Optional[List[List[float]]]]:
    """Compute clipped Voronoi regions for positions within rectangular bounds."""
    # Virtual points outside the field to bound the diagram
    virtual_points = [
        [field_bounds["x_min"] - 50, field_bounds["y_min"] - 50],
        [field_bounds["x_min"] - 50, field_bounds["y_max"] + 50],
        [field_bounds["x_max"] + 50, field_bounds["y_min"] - 50],
        [field_bounds["x_max"] + 50, field_bounds["y_max"] + 50],
    ]
    all_positions = positions + virtual_points

    vor = Voronoi(all_positions)
    regions: Dict[int, Optional[List[List[float]]]] = {}

    for idx, region_idx in enumerate(vor.point_region[: len(positions)]):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            regions[idx] = None
            continue
        polygon = [vor.vertices[i] for i in region]
        regions[idx] = clip_to_bounds(polygon, field_bounds)
    return regions


def clip_to_bounds(polygon: List[List[float]], field_bounds: Dict[str, float]) -> Optional[List[List[float]]]:
    """Clip polygon to rectangular box defined by field_bounds."""
    field_box = box(
        field_bounds["x_min"],
        field_bounds["y_min"],
        field_bounds["x_max"],
        field_bounds["y_max"],
    )
    shp = ShapelyPolygon(polygon)
    clipped = shp.intersection(field_box)
    if clipped.is_empty:
        return None
    coords = list(clipped.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [[float(x), float(y)] for x, y in coords]


def create_rink_shapes(bounds: Dict[str, float]) -> list:
    """Return Plotly shapes for a simplified NHL rink in meters.

    Includes: outer rectangle (no rounded corners), center line, blue lines,
    goal lines, and faceoff spots (as small circles).
    """
    x0, x1 = bounds["x_min"], bounds["x_max"]
    y0, y1 = bounds["y_min"], bounds["y_max"]
    y_mid = (y0 + y1) / 2.0
    shapes = []

    # Outer rectangle
    shapes.append(
        dict(
            type="rect",
            x0=x0, x1=x1, y0=y0, y1=y1,
            line=dict(color="black", width=2),
            fillcolor="rgba(200, 200, 200, 0.05)",
        )
    )

    # Center red line
    shapes.append(dict(type="line", x0=(x0+x1)/2, x1=(x0+x1)/2, y0=y0, y1=y1, line=dict(color="red", width=2)))

    # Blue lines (±25 ft from center -> 7.62 m)
    blue_offset = 7.62
    for sgn in (-1, 1):
        x = (x0 + x1)/2 + sgn * blue_offset
        shapes.append(dict(type="line", x0=x, x1=x, y0=y0, y1=y1, line=dict(color="blue", width=2)))

    # Goal lines (11 ft from ends -> 3.3528 m)
    goal_offset = 3.3528
    shapes.append(dict(type="line", x0=x0 + goal_offset, x1=x0 + goal_offset, y0=y0, y1=y1, line=dict(color="red", width=2)))
    shapes.append(dict(type="line", x0=x1 - goal_offset, x1=x1 - goal_offset, y0=y0, y1=y1, line=dict(color="red", width=2)))

    # Faceoff circles (simplified as small circles at standard y positions)
    # We'll draw as thin rings using many-point polygons (small visual hint)
    def faceoff_circle(cx, cy, r=1.8, n=36, color="blue"):
        theta = np.linspace(0, 2*np.pi, n)
        xs = cx + r*np.cos(theta)
        ys = cy + r*np.sin(theta)
        return dict(
            type="path",
            path="M " + " L ".join(f"{x},{y}" for x, y in zip(xs, ys)) + " Z",
            line=dict(color=color, width=1),
        )

    # Typical offensive/defensive zone circle y positions
    circle_y_offsets = [y_mid - 6.7, y_mid + 6.7]  # ~22 ft from center line vertically
    circle_x_left = x0 + 13.716  # ~45 ft from end boards
    circle_x_right = x1 - 13.716
    for cy in circle_y_offsets:
        shapes.append(faceoff_circle(circle_x_left, cy))
        shapes.append(faceoff_circle(circle_x_right, cy))

    return shapes
