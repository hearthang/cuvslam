"""Physical motion limits for output trajectories (jump suppression).

The mower drives at most ~1.1 m/s; Basalt's largest step over the 47 bags is 0.96 m/s / 0.135 m per step.
Our steps beyond that are artifacts (vision failing while the robot is lifted or turned fast), so outputs are
held to V_MAX and S_MAX (90 % of twice Basalt's maximum):
  - limit_window (offline, SLAM): each violating step is replaced by the smallest window around it whose
    straight-line motion respects the limits, positions inside it interpolated in time; samples outside the
    window (and the first and last pose, hence the closure error) are unchanged.
  - limit_causal (online, front-end VIO): every step is clamped to the limit and the remainder is caught up
    at the limit in later steps; uses no future data.
Rotations are left as they are (they already stay within twice Basalt's maximum).
"""
import numpy as np

V_MAX = 1.7    # m/s   (2 x 0.96 = 1.92)
S_MAX = 0.24   # m per step (2 x 0.135 = 0.27)


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
