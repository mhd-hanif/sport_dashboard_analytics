"""Scoring threat of the attack.

The threat of an attacking situation is measured with the off-ball scoring
opportunity (OBSO) of Spearman [1], called here off-puck scoring opportunity:

    OBSO = sum_q  S(q) * C_att(q) * T(q),

where, for a location q, ``T`` is the probability density that the next puck
event happens at q, ``C_att`` the probability that the attack controls the puck
there, and ``S`` the probability of scoring from there. It is the probability
that the next puck event ends in a goal; a lower value means a better defense.

Shared with the ``ice_hockey_simulator`` repository (branch
``experimental/pitch-control-coverage``).

Reference:
    [1] W. Spearman, "Beyond expected goals," MIT Sloan Sports Analytics
        Conference, 2018.
"""

from dataclasses import dataclass, field

import numpy as np

from analytics.pitch_control import (
    PitchControlParams,
    control_from_times,
    time_to_intercept,
)
from analytics.rink import Rink


@dataclass(frozen=True)
class ScoringModel:
    """Stylised scoring-probability (expected-goal) surface.

    The probability of scoring from q grows with the angle ``psi(q)`` under
    which the goal mouth is seen from q, the dominant geometric predictor of
    shot-location expected-goal models:

        S(q) = min(s_max, s_ref * (psi(q) / psi_ref) ** gamma),

    and is zero behind the goal line.

    Attributes:
        rink (Rink): Rink geometry.
        s_ref (float): Scoring probability at the reference angle.
        d_ref (float): Distance in front of the goal that defines the
            reference angle [m].
        gamma (float): Exponent of the angle.
        s_max (float): Upper bound of the scoring probability.
    """

    rink: Rink = field(default_factory=Rink)
    s_ref: float = 0.25
    d_ref: float = 3.0
    gamma: float = 1.2
    s_max: float = 0.35

    def opening_angle(self, points):
        """Angle under which the goal mouth is seen from each point [rad]."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        gx, gy = self.rink.goal
        half = 0.5 * self.rink.goal_width
        a = np.column_stack([gx - points[:, 0], gy + half - points[:, 1]])
        b = np.column_stack([gx - points[:, 0], gy - half - points[:, 1]])
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        dot = (a * b).sum(axis=1)
        psi = np.abs(np.arctan2(cross, dot))
        psi[points[:, 0] <= gx] = 0.0
        return psi

    def __call__(self, points):
        psi_ref = 2.0 * np.arctan(0.5 * self.rink.goal_width / self.d_ref)
        s = self.s_ref * (self.opening_angle(points) / psi_ref) ** self.gamma
        return np.minimum(s, self.s_max)


@dataclass(frozen=True)
class TransitionModel:
    """Density of the location of the next puck event.

    Following [1], the next event happens at a location drawn from an
    isotropic normal distribution centred on the puck.

    Attributes:
        sigma (float): Standard deviation of the distribution [m].
    """

    sigma: float = 8.0

    def __call__(self, puck_pos, points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        sq = ((points - np.asarray(puck_pos, dtype=float)) ** 2).sum(axis=1)
        return np.exp(-sq / (2.0 * self.sigma**2)) / (2.0 * np.pi * self.sigma**2)


@dataclass(frozen=True)
class ThreatModel:
    """Scoring and transition models, i.e. everything but the control model."""

    scoring: ScoringModel = field(default_factory=ScoringModel)
    transition: TransitionModel = field(default_factory=TransitionModel)

    def density(self, puck_pos, points):
        """Threat density ``S(q) * T(q)`` at each point."""
        return self.scoring(points) * self.transition(puck_pos, points)


def evaluate_obso(att_pos, att_vel, def_pos, def_vel, carriers, grid,
                  threat=None, params=None):
    """Off-puck scoring opportunity of the attack for one frame.

    The puck is not always tracked. ``carriers`` lists the attackers who may
    have it; the result is the average over these hypotheses, each with the
    puck on the stick of that attacker.

    Args:
        att_pos, att_vel (ndarray): Attackers' positions and velocities, (na, 2).
        def_pos, def_vel (ndarray): Defenders' positions and velocities, (nd, 2).
        carriers (sequence of int): Indices of the possible puck carriers
            among the attackers.
        grid (Grid): Evaluation grid.
        threat (ThreatModel): Scoring and transition models.
        params (PitchControlParams): Parameters of the control model.

    Returns:
        dict: ``obso`` (float), ``per_attacker`` (threat controlled by each
        attacker, shape (na,), sums to ``obso``), and the fields on the grid
        ``surface`` (threat density ``S T C_att``, per square metre) and
        ``C_att`` (attack control; both are zero where the scoring
        probability is zero).
    """
    threat = threat or ThreatModel()
    params = params or PitchControlParams()
    att_pos = np.asarray(att_pos, dtype=float).reshape(-1, 2)
    pts = grid.points
    area = grid.cell_area
    na = len(att_pos)

    out = {"obso": 0.0, "per_attacker": np.zeros(na),
           "surface": np.zeros(len(pts)), "C_att": np.zeros(len(pts))}
    carriers = list(carriers)
    if na == 0 or not carriers:
        return out

    # Only the locations with a chance of scoring contribute.
    s_all = threat.scoring(pts)
    live = s_all > 0.0
    pts, s = pts[live], s_all[live]
    surface = np.zeros(len(pts))
    c_att_sum = np.zeros(len(pts))
    tau_a = time_to_intercept(att_pos, att_vel, pts, params)
    tau_d = time_to_intercept(def_pos, def_vel, pts, params)
    for c in carriers:
        puck = att_pos[c]
        puck_time = np.linalg.norm(pts - puck, axis=1) / params.puck_speed
        ppcf_att, _ = control_from_times(tau_a, tau_d, puck_time, params)
        dens = s * threat.transition(puck, pts)
        out["per_attacker"] += (ppcf_att * dens[None, :]).sum(axis=1) * area
        c_att = ppcf_att.sum(axis=0)
        surface += dens * c_att
        c_att_sum += c_att
    out["surface"][live] = surface
    out["C_att"][live] = c_att_sum
    n = float(len(carriers))
    out["per_attacker"] /= n
    out["surface"] /= n
    out["C_att"] /= n
    out["obso"] = float(out["per_attacker"].sum())
    return out
