"""Physical motion limits for output trajectories (jump suppression).

The mower drives at most ~1.1 m/s; Basalt's largest step over the 47 bags is 0.96 m/s / 0.135 m per step.
Our steps beyond that are artifacts (vision failing while the robot is lifted or turned fast), so outputs are
held to V_MAX and S_MAX (90 % of twice Basalt's maximum):
  - limit_window (offline, SLAM): each violating step is replaced by the smallest window around it whose
    straight-line motion respects the limits, positions inside it interpolated in time; samples outside the
    window (and the first and last pose, hence the closure error) are unchanged.
  - limit_causal (online, front-end VIO): every step is clamped to the limit and the remainder is caught up
    at the limit in later steps; uses no future data.
Rotations are held to W_MAX / A_MAX the same way (limit_window_rot, limit_causal_rot): at 16 Hz the only
violations are real hand rotations of the lifted mower (up to ~360 deg/s, gyro-measured).
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

V_MAX = 1.7    # m/s   (2 x 0.96 = 1.92)
S_MAX = 0.24   # m per step (2 x 0.135 = 0.27)
W_MAX = 192.0  # deg/s (2 x 106.7 = 213)
A_MAX = 24.0   # deg per step (2 x 13.3 = 26.7)


def _step_limit(dt):
    return np.minimum(V_MAX * dt, S_MAX)


def violations(t, p):
    dt = np.diff(t)
    return np.where(np.linalg.norm(np.diff(p, axis=0), axis=1) > _step_limit(dt) + 1e-9)[0]


def limit_window(t, p, max_iter=10000):
    """t [s], p (N,3) -> limited copy of p and the number of windows rewritten."""
    p = p.copy()
    n_win = 0
    for _ in range(max_iter):
        bad = violations(t, p)
        if not len(bad):
            break
        a, b = bad[0], bad[0] + 1
        while True:
            dt = np.diff(t[a:b + 1])
            if np.linalg.norm(p[b] - p[a]) <= min(V_MAX * (t[b] - t[a]), S_MAX * (t[b] - t[a]) / dt.max()) or \
                    (a == 0 and b == len(t) - 1):
                break
            a, b = max(a - 1, 0), min(b + 1, len(t) - 1)  # grow the window both ways
        w = (t[a:b + 1] - t[a]) / (t[b] - t[a])
        p[a:b + 1] = p[a] + w[:, None] * (p[b] - p[a])
        n_win += 1
    return p, n_win


def limit_causal(t, p):
    """Clamp each step to the limit; the clamped-off part is caught up later at the limit."""
    out = p.copy()
    lim = _step_limit(np.diff(t))
    n = 0
    for k in range(1, len(t)):
        d = p[k] - out[k - 1]
        L = np.linalg.norm(d)
        if L > lim[k - 1]:
            d *= lim[k - 1] / L
            n += 1
        out[k] = out[k - 1] + d
    return out, n


def _rot_limit(dt):
    return np.radians(np.minimum(W_MAX * dt, A_MAX))


def limit_window_rot(t, quats, max_iter=10000):
    """Offline: each step turning faster than the limit is replaced by the smallest window around it whose
    end-to-end rotation, slerped in time, respects it. Returns limited quaternions (x, y, z, w) and #windows."""
    q = quats.copy()
    n_win = 0
    for _ in range(max_iter):
        R = Rotation.from_quat(q)
        step = (R[:-1].inv() * R[1:]).magnitude()
        bad = np.where(step > _rot_limit(np.diff(t)) + 1e-9)[0]
        if not len(bad):
            break
        a, b = bad[0], bad[0] + 1
        while True:
            ang = (R[a].inv() * R[b]).magnitude()
            dt = np.diff(t[a:b + 1])
            if ang <= np.radians(min(W_MAX * (t[b] - t[a]), A_MAX * (t[b] - t[a]) / dt.max())) or \
                    (a == 0 and b == len(t) - 1):
                break
            a, b = max(a - 1, 0), min(b + 1, len(t) - 1)
        q[a:b + 1] = Slerp([t[a], t[b]], Rotation.from_quat([q[a], q[b]]))(t[a:b + 1]).as_quat()
        n_win += 1
    return q, n_win


def limit_causal_rot(t, quats):
    """Online: each output rotation step is clamped to the limit; the remainder is caught up later."""
    R_in = Rotation.from_quat(quats)
    out = [R_in[0]]
    lim = _rot_limit(np.diff(t))
    n = 0
    for k in range(1, len(t)):
        d = (out[-1].inv() * R_in[k]).as_rotvec()
        ang = np.linalg.norm(d)
        if ang > lim[k - 1]:
            d *= lim[k - 1] / ang
            n += 1
        out.append(out[-1] * Rotation.from_rotvec(d))
    return Rotation.concatenate(out).as_quat(), n
