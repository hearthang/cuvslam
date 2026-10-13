"""Sparse conditional position optimization with measured relative poses and acceleration regularization.

Rotation is held at the full SE(3) solution. All positions are jointly solved;
there is no endpoint equality constraint, speed clipping, or Basalt input.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


def solve_positions(t, positions, rotations, increments, step_sigma, loop_ij, loop_t,
                    loop_sigma=.05, acceleration_sigma=1., loop_scale=2., iterations=15):
    t = np.asarray(t, float)
    dt = np.diff(t)
    n = len(t)
    if n < 3 or np.any(dt <= 0) or not np.isfinite(positions).all():
        raise ValueError('Position optimization needs at least three finite, ordered poses')
    if acceleration_sigma <= 0 or loop_sigma <= 0 or np.any(np.asarray(step_sigma) <= 0):
        raise ValueError('Noise sigmas must be positive')
    loop_ij = np.asarray(loop_ij, int).reshape(-1, 2)
    m = len(loop_ij)
    rows, cols, data = [], [], []
    def add(row, index, weights):
        rows.extend([row]*len(index)); cols.extend(index); data.extend(weights)
    for i in range(n-1):
        add(i, [i, i+1], [-1, 1])
    for j, (a, b) in enumerate(loop_ij):
        add(n-1+j, [int(a), int(b)], [-1, 1])
    for i in range(1, n-1):
        w = 2 / (dt[i-1] + dt[i]) * np.array([1/dt[i-1], -1/dt[i-1]-1/dt[i], 1/dt[i]])
        add(n-1+m+i-1, [i-1, i, i+1], w)
    A = coo_matrix((data, (rows, cols)), shape=(2*n-3+m, n)).tocsr()
    target_steps = np.einsum('nij,nj->ni', rotations[:-1], increments)
    target_loops = np.einsum('nij,nj->ni', rotations[loop_ij[:, 0]], np.asarray(loop_t).reshape(-1, 3))
    target = np.vstack([target_steps, target_loops, np.zeros((n-2, 3))])
    sigma = np.r_[np.broadcast_to(step_sigma, n-1), np.full(m, loop_sigma), np.full(n-2, acceleration_sigma)]
    # Fix only the global translation gauge, never the final pose.
    fixed = np.asarray(positions[0]).copy()
    rhs = target - A[:, :1].toarray() * fixed
    Ar = A[:, 1:]
    p = np.asarray(positions).copy()
    history = []
    for _ in range(iterations):
        residual = (A @ p - target) / sigma[:, None]
        norm = np.linalg.norm(residual, axis=1)
        weight = np.ones(len(sigma))
        if m:
            weight[n-1:n-1+m] = 1 / (1 + (norm[n-1:n-1+m] / loop_scale)**2)
        # Huber acceleration: ordinary drive is smoothed, abrupt real motion can remain.
        weight[n-1+m:] = np.minimum(1, 2 / np.maximum(norm[n-1+m:], 1e-12))
        W = diags(weight / sigma**2)
        lhs = (Ar.T @ W @ Ar).tocsc()
        pnew = np.vstack([fixed, spsolve(lhs, Ar.T @ W @ rhs)])
        change = float(np.max(np.linalg.norm(pnew-p, axis=1)))
        p = pnew
        history.append(change)
        if change < 1e-5:
            break
    if not np.isfinite(p).all():
        raise ValueError('Sparse position solve produced nonfinite output')
    return p, history
