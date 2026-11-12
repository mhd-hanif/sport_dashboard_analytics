"""Rink geometry and evaluation grid.

Everything in the analytics package is expressed in SI units (metres, seconds)
in a canonical rink frame: ``x`` runs along the length of the rink (0 to 61 m),
``y`` across it (0 to 30 m), and the defended goal is on the left-hand side.
``analytics.engine`` mirrors a clip into this frame when the defended goal is
on the right.

Shared with the ``ice_hockey_simulator`` repository (branch
``experimental/pitch-control-coverage``).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Rink:
    """Geometry of the rink and of the defended goal.

    Attributes:
        length (float): Rink length along x [m].
        width (float): Rink width along y [m].
        goal (tuple): Centre of the defended goal mouth (x, y) [m].
        goal_width (float): Distance between the goal posts [m].
        blue_line_x (float): x-coordinate of the defended blue line [m].
        house_radius (float): Radius of the "house" sector in front of the
            goal, as in Tanpipat et al. [m].
    """

    length: float = 61.0
    width: float = 30.0
    goal: tuple = (4.3, 15.0)
    goal_width: float = 1.83
    blue_line_x: float = 21.8
    house_radius: float = 15.0

    @property
    def bounds(self):
        """Field bounds in the dictionary format used by the dashboard."""
        return {"x_min": 0.0, "x_max": self.length, "y_min": 0.0, "y_max": self.width}

    @property
    def goal_xy(self):
        return np.asarray(self.goal, dtype=float)

    def in_house(self, points):
        """Boolean mask of the points inside the house.

        The house is the sector in front of the goal bounded by the two
        45-degree lines through the goal and by a circle of radius
        ``house_radius`` centred on it.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        dx = points[:, 0] - self.goal[0]
        dy = points[:, 1] - self.goal[1]
        return (dx >= 0.0) & (np.abs(dy) <= dx) & (dx**2 + dy**2 <= self.house_radius**2)

    def in_defensive_zone(self, points):
        """Boolean mask of the points between the end boards and the blue line."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return points[:, 0] <= self.blue_line_x


@dataclass(frozen=True)
class Grid:
    """Regular grid of cell centres covering a rink.

    Attributes:
        rink (Rink): The rink that is discretised.
        cell (float): Side of the square cells [m].
        x_limit (float|None): If given, only the part of the rink with
            ``x <= x_limit`` is discretised (e.g. the defended half).
    """

    rink: Rink = field(default_factory=Rink)
    cell: float = 0.5
    x_limit: float = None

    @property
    def xs(self):
        length = self.rink.length if self.x_limit is None else self.x_limit
        n = int(round(length / self.cell))
        return (np.arange(n) + 0.5) * length / n

    @property
    def ys(self):
        n = int(round(self.rink.width / self.cell))
        return (np.arange(n) + 0.5) * self.rink.width / n

    @property
    def shape(self):
        """Shape (ny, nx) of a field defined on the grid."""
        return (len(self.ys), len(self.xs))

    @property
    def points(self):
        """Cell centres as an array of shape (ny * nx, 2)."""
        gx, gy = np.meshgrid(self.xs, self.ys)
        return np.column_stack([gx.ravel(), gy.ravel()])

    @property
    def cell_area(self):
        xs, ys = self.xs, self.ys
        return (xs[1] - xs[0]) * (ys[1] - ys[0])

    @property
    def extent(self):
        """Extent (x_min, x_max, y_max, y_min) for ``imshow`` with y pointing down."""
        length = self.rink.length if self.x_limit is None else self.x_limit
        return (0.0, length, self.rink.width, 0.0)

    def as_image(self, values):
        """Reshape a flat field of length ny * nx into an (ny, nx) image."""
        return np.asarray(values).reshape(self.shape)
