"""Physics-based space control ("pitch control").

``spearman_control`` is the potential pitch control field of Spearman [1, 2]
in the form popularised by Shaw's open implementation [3]: each player reaches
a location after a time to intercept, and the probability that a team controls
the puck there is obtained by integrating a control rate from the arrival of
the puck onwards.

Shared with the ``ice_hockey_simulator`` repository (branch
``experimental/pitch-control-coverage``), where it is the evaluation model.

References:
    [1] W. Spearman, A. Basye, G. Dick, R. Hotovy, and P. Pop, "Physics-based
        modeling of pass probabilities in soccer," MIT Sloan Sports Analytics
        Conference, 2017.
    [2] W. Spearman, "Beyond expected goals," MIT Sloan Sports Analytics
        Conference, 2018.
    [3] L. Shaw, "LaurieOnTracking," Friends of Tracking, 2020.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchControlParams:
    """Parameters of the control model (ice-hockey values).

    Attributes:
        v_max (float): Maximum skating speed [m/s].
        t_react (float): Reaction time during which a player keeps its
            current velocity [s].
        reach (float): Stick reach; a player controls the puck when it is
            within this distance [m].
        sigma (float): Standard deviation of the arrival-time uncertainty [s].
        lam (float): Control rate of a player that has arrived [1/s].
        kappa_def (float): Advantage of the defending team in the control rate.
        puck_speed (float): Average speed of a pass [m/s].
        dt (float): Integration step of the control rate [s].
        t_max (float): Integration horizon after the arrival of the puck [s].
        tol (float): The integration stops once the uncontrolled probability
            is below this value everywhere.
    """

    v_max: float = 8.0
    t_react: float = 0.5
    reach: float = 1.5
    sigma: float = 0.45
    lam: float = 4.3
    kappa_def: float = 1.0
    puck_speed: float = 20.0
    dt: float = 0.04
    t_max: float = 6.0
    tol: float = 0.01


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def time_to_intercept(pos, vel, targets, params, use_reach=True):
    """Expected time for every player to reach every target.

    A player keeps its current velocity during the reaction time and then
    skates straight to the target at maximum speed [1].

    Args:
        pos (ndarray): Player positions, shape (n, 2) [m].
        vel (ndarray): Player velocities, shape (n, 2) [m/s].
        targets (ndarray): Target locations, shape (m, 2) [m].
        params (PitchControlParams): Model parameters.
        use_reach (bool): Subtract the stick reach from the distance to skate.

    Returns:
        ndarray: Times to intercept, shape (n, m) [s].
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    vel = np.asarray(vel, dtype=float).reshape(-1, 2)
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    start = pos + vel * params.t_react
    dist = np.linalg.norm(targets[None, :, :] - start[:, None, :], axis=2)
    if use_reach:
        dist = np.maximum(dist - params.reach, 0.0)
    return params.t_react + dist / params.v_max


def control_from_times(tau_att, tau_def, puck_time, params):
    """Integrate the control rate given the times to intercept.

    Args:
        tau_att (ndarray): Attackers' times to intercept, shape (na, m) [s].
        tau_def (ndarray): Defenders' times to intercept, shape (nd, m) [s].
        puck_time (ndarray): Time for the puck to reach each target, (m,) [s].
        params (PitchControlParams): Model parameters.

    Returns:
        tuple: ``(ppcf_att, ppcf_def)``, see ``spearman_control``.
    """
    tau = np.vstack([tau_att, tau_def])
    na = tau_att.shape[0]
    lam = np.concatenate([
        np.full(na, params.lam),
        np.full(tau_def.shape[0], params.lam * params.kappa_def),
    ])[:, None]
    scale = np.pi / (np.sqrt(3.0) * params.sigma)

    ppcf = np.zeros_like(tau)
    remaining = np.ones(tau.shape[1])
    n_steps = int(np.ceil(params.t_max / params.dt))
    for step in range(n_steps):
        t = puck_time + step * params.dt
        arrived = _sigmoid(scale * (t[None, :] - tau))
        d_ppcf = remaining[None, :] * arrived * lam * params.dt
        # Never hand out more probability than what is left.
        total = d_ppcf.sum(axis=0)
        over = total > remaining
        if np.any(over):
            d_ppcf[:, over] *= (remaining[over] / total[over])[None, :]
        ppcf += d_ppcf
        remaining = 1.0 - ppcf.sum(axis=0)
        if remaining.max() < params.tol:
            break

    ppcf /= np.maximum(ppcf.sum(axis=0, keepdims=True), 1e-12)
    return ppcf[:na], ppcf[na:]


def spearman_control(att_pos, att_vel, def_pos, def_vel, puck_pos, targets, params):
    """Potential pitch control field of both teams [2].

    Args:
        att_pos, att_vel (ndarray): Attackers' positions and velocities, (na, 2).
        def_pos, def_vel (ndarray): Defenders' positions and velocities, (nd, 2).
        puck_pos (ndarray): Position of the puck, shape (2,).
        targets (ndarray): Locations where control is evaluated, (m, 2).
        params (PitchControlParams): Model parameters.

    Returns:
        tuple: ``(ppcf_att, ppcf_def)`` where ``ppcf_att`` has shape (na, m)
        and ``ppcf_def`` has shape (nd, m). Entry (k, q) is the probability
        that player k is the one who controls a puck sent to target q; the
        team control is the sum over the players of the team, and the two team
        controls add up to one at every target.
    """
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    tau_a = time_to_intercept(att_pos, att_vel, targets, params)
    tau_d = time_to_intercept(def_pos, def_vel, targets, params)
    puck_time = np.linalg.norm(targets - np.asarray(puck_pos, dtype=float), axis=1) / params.puck_speed
    return control_from_times(tau_a, tau_d, puck_time, params)
