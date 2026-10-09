"""Front-end odometry output (fixed cuVSLAM front end, no loop closure) as TUM files, causally held to the motion
limits of traj_limit.py (twice Basalt's largest jump). Writes FE_DIR/<tag>_frontend.txt.

Usage: python export_frontend_traj.py [FE_DIR=results/fe5] [--odom in]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

fe = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "results/fe5")
mode = sys.argv[sys.argv.index("--odom") + 1] if "--odom" in sys.argv else "in"
sys.argv = sys.argv[:1]
import backend as B  # noqa: E402
from traj_limit import limit_causal, limit_causal_rot  # noqa: E402

for f in sorted(fe.glob("*_fe.npz")):
    tag = f.name[:-7]
    z = np.load(f)
    t = z["t"] * 1e-9
    P = B.frontend_odometry(z, mode)
    pos, n_p = limit_causal(t, P[:, :3, 3].copy())
    quat, n_r = limit_causal_rot(t, Rotation.from_matrix(P[:, :3, :3]).as_quat())
    np.savetxt(fe / f"{tag}_frontend.txt", np.c_[t, pos, quat], fmt="%.6f", header="timestamp tx ty tz qx qy qz qw")
    print(f"{tag}: closure {np.linalg.norm(pos[-1] - pos[0]):.3f} m, clamped steps: position {n_p}, rotation {n_r}")
