"""Pitch-control coverage: a reference defense that descends the scoring threat.

Classical coverage control gives every defender the part of the rink that is
closest to it and sends the defender to the centroid of a hand-designed density
over that Voronoi cell. Here both ingredients are quantities with a meaning in
sports analytics: the density is the scoring threat of the attack and the
partition is a soft pitch-control field shared by the players of both teams.

Design model
------------
The defenders descend a smooth surrogate of the off-puck scoring opportunity,

    Theta(x) = int phi(q) C_att(q; x) dq,      phi(q) = S(q) T(q),

where ``S`` is the scoring probability, ``T`` the transition density of the
puck (a mixture centred on the carrier and on the possible receivers) and
``C_att`` the probability that the attack controls a puck sent to q. Every
player k of both teams still has to skate a gap ``g_k(q)`` when the puck
reaches q: the distance to q minus what the player covers while the puck is in
flight and minus the stick reach. Control is shared by a soft minimum of the
times to intercept, ``p_k = softmax(-beta g_k / v_max)``.

Move-to-centroid law
--------------------
The descent direction of defender i is

    -dTheta/dx_i = int phi p_i C_att (beta / v_max) e_i dq,

with ``e_i`` the unit vector from the defender to q. Dividing by the weight of
the defender turns it into a displacement towards a generalised centroid, which
is iterated a few times per control step. It is the move-to-centroid law of
statistical coverage control (Arslan, 2019) with the control of the opponents
as an additional weight, and it tends to a Lloyd law on the Voronoi partition
of all the players as the kernel gets sharper.

This is the configuration evaluated in the ``ice_hockey_simulator`` repository
(branch ``experimental/pitch-control-coverage``), without the barrier-function
layer for pass lanes and formations, so that the dashboard only needs numpy.
"""

from dataclasses import dataclass, field

import numpy as np

from analytics.rink import Grid
from analytics.threat import ThreatModel


@dataclass
class PitchControlCoverage:
    """Nominal controller of the defenders.

    Attributes:
        grid (Grid): Grid on which the integrals are evaluated.
        threat (ThreatModel): Scoring and transition models.
        beta (float): Sharpness of the soft minimum of the times [1/s].
        min_gap (float): Smallest distance used to convert the gradient into
            a displacement [m].
        reach (float): Stick reach [m].
        puck_speed (float): Speed of a pass [m/s].
        t_react (float): Reaction time before a player moves to the puck [s].
        lookahead (float): Players are placed where they will be after this
            time at their current velocity [s].
        carrier_weight (float): Probability that the puck is still with the
            carrier when the defenders have repositioned; the rest is shared
            among the other attackers.
        iterations (int): Number of centroid iterations per control step.
        gain (float): Proportional gain of the move-to-centroid law [1/s].
        v_max (float): Speed limit of the command [m/s].
    """

    grid: Grid = field(default_factory=lambda: Grid(cell=0.75, x_limit=30.0))
    threat: ThreatModel = field(default_factory=ThreatModel)
    beta: float = 4.0
    min_gap: float = 0.5
    reach: float = 1.5
    puck_speed: float = 20.0
    t_react: float = 0.5
    lookahead: float = 0.5
    carrier_weight: float = 0.8
    iterations: int = 5
    gain: float = 4.0
    v_max: float = 8.0

    def __post_init__(self):
        # Every integrand carries the scoring probability: keep the cells where it is positive.
        pts = self.grid.points
        scoring = self.threat.scoring(pts)
        self._pts = pts[scoring > 0.0]
        self._scoring = scoring[scoring > 0.0]

    # ------------------------------------------------------------------ model
    def threat_density(self, xa, puck_idx):
        """Threat density ``phi = S * T`` with the mixture transition model."""
        pts = self._pts
        trans = self.threat.transition(xa[puck_idx], pts)
        if self.carrier_weight < 1.0 and len(xa) > 1:
            others = sum(self.threat.transition(xj, pts)
                         for j, xj in enumerate(xa) if j != puck_idx)
            trans = self.carrier_weight * trans \
                + (1.0 - self.carrier_weight) * others / (len(xa) - 1)
        return self._scoring * trans

    def _terms(self, xd, xa, puck_idx):
        """Numerator and denominator of the centroid step for one carrier."""
        pts = self._pts
        nd = len(xd)
        phi = self.threat_density(xa, puck_idx)

        flight = np.linalg.norm(pts - xa[puck_idx], axis=1) / self.puck_speed
        free = self.reach + self.v_max * np.maximum(flight - self.t_react, 0.0)

        pos = np.vstack([xd, xa])
        diff = pts[None, :, :] - pos[:, None, :]                     # (n, m, 2)
        dist = np.linalg.norm(diff, axis=2)
        gap = np.maximum(dist - free[None, :], 0.0)
        logit = -self.beta * gap / self.v_max
        logit -= logit.max(axis=0, keepdims=True)
        p = np.exp(logit)
        p /= p.sum(axis=0, keepdims=True)
        c_att = p[nd:].sum(axis=0)

        slope = np.where(gap[:nd] > 0.0, self.beta / self.v_max, 0.0)
        unit = diff[:nd] / np.maximum(dist[:nd], 1e-9)[:, :, None]
        w = phi[None, :] * p[:nd] * c_att[None, :] * slope
        num = np.einsum("im,imj->ij", w, unit)
        den = (w / np.maximum(gap[:nd], self.min_gap)).sum(axis=1)
        theta = float((phi * c_att).sum() * self.grid.cell_area)
        return num, den, theta

    def objective(self, def_pos, att_pos, carriers):
        """Surrogate threat averaged over the possible carriers."""
        xd = np.asarray(def_pos, dtype=float).reshape(-1, 2)
        xa = np.asarray(att_pos, dtype=float).reshape(-1, 2)
        return float(np.mean([self._terms(xd, xa, c)[2] for c in carriers]))

    def _step(self, xd, xa, carriers):
        num = np.zeros_like(xd)
        den = np.zeros(len(xd))
        for c in carriers:
            n_c, d_c, _ = self._terms(xd, xa, c)
            num += n_c
            den += d_c
        step = num / np.maximum(den, 1e-15)[:, None]
        step[den <= 1e-15] = 0.0
        return step

    # ---------------------------------------------------------------- command
    def targets(self, def_pos, att_pos, carriers, iterations=None):
        """Positions the centroid iteration moves the defenders to, (nd, 2)."""
        target = np.array(def_pos, dtype=float).reshape(-1, 2)
        xa = np.asarray(att_pos, dtype=float).reshape(-1, 2)
        carriers = list(carriers)
        if len(target) == 0 or len(xa) == 0 or not carriers:
            return target
        for _ in range(max(1, iterations or self.iterations)):
            target = target + self._step(target, xa, carriers)
        return target

    def nominal(self, def_pos, def_vel, att_pos, att_vel, carriers):
        """Move-to-centroid command of every defender.

        Args:
            def_pos, def_vel (ndarray): Defenders' positions and velocities, (nd, 2).
            att_pos, att_vel (ndarray): Attackers' positions and velocities, (na, 2).
            carriers (sequence of int): Indices of the possible puck carriers
                among the attackers; the command descends the average threat
                over these hypotheses.

        Returns:
            tuple: ``(u, target)`` with the command ``u`` (nd, 2) [m/s] and
            the points the defenders are sent to (nd, 2).
        """
        xd = np.asarray(def_pos, dtype=float) + self.lookahead * np.asarray(def_vel, dtype=float)
        xa = np.asarray(att_pos, dtype=float) + self.lookahead * np.asarray(att_vel, dtype=float)
        target = self.targets(xd, xa, carriers)
        u = self.gain * (target - xd)
        speed = np.linalg.norm(u, axis=1, keepdims=True)
        u = np.where(speed > self.v_max, u * self.v_max / np.maximum(speed, 1e-12), u)
        return u, target
