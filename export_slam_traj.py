"""Export the SLAM-optimised trajectories in Basalt's format, next to Basalt's vio_basalt_*.txt files.

Basalt format: no header, "timestamp tx ty tz qx qy qz qw" per line, timestamp in seconds (9 decimals, IMU
clock), the rest 6 decimals; pose of imu_link in a gravity-aligned world whose origin is the first pose.
The back end optimises keyframes. Every front-end frame (~5.3 Hz) gets the odometry pose times a correction
(optimised keyframe pose x inverse odometry keyframe pose) interpolated between its two keyframes (translation
linear, rotation slerp, by time), so keyframe poses are kept exactly and the trajectory has no steps where one
keyframe's correction hands over to the next. Steps beyond the mower's physical limits (twice Basalt's largest
step, see traj_limit.py) are smoothed over the smallest window that respects them.

Usage: python export_slam_traj.py [FE_DIR=results/fe5] [SUFFIX=_v10] [--odom fe_in] [--dry-run]
       -> /mnt/f/data_set/slam/<date>[/<bag>]/slam_cuvslam_<bag>.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ap = argparse.ArgumentParser()
ap.add_argument("fe_dir", nargs="?", default="results/fe5")
ap.add_argument("suffix", nargs="?", default="_v10")
ap.add_argument("--odom", default="fe_in", help="odometry the back end was run with (fe_in for the fixed front end)")
ap.add_argument("--data", default="/mnt/f/data_set/slam")
ap.add_argument("--dry-run", action="store_true")
cli = ap.parse_args()
sys.argv = sys.argv[:1]
import backend as B  # noqa: E402
from traj_limit import limit_window, limit_window_rot  # noqa: E402

DATA = Path(cli.data)


def basalt_file(tag):
    date, bag = tag.split("_", 1)
    name = bag.removeprefix("basalt_")  # 9-21 bag dirs carry a doubled "basalt_" prefix, its files do not
    for c in (DATA / date / f"vio_basalt_{name}.txt", DATA / date / bag / f"vio_basalt_{name}.txt"):
        if c.exists():
            return c, name
    raise FileNotFoundError(tag)


def mat(row):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
    M[:3, 3] = row[1:4]
    return M


fe = Path(cli.fe_dir)
for f in sorted(fe.glob("*_fe.npz")):
    tag = f.name[:-7]
    z = np.load(f)
    t = z["t"]
    if cli.odom == "vote":
        P = B.vote_odometry(z)
    elif cli.odom.startswith("fe_"):
        P = B.frontend_odometry(z, cli.odom[3:])
    else:
        P = B.build_odometry(z["pose_in"], z["pose_st"], cli.odom, t_ns=t)[0]
    kf = z["kf_idx"]
    opt = np.loadtxt(fe / f"{tag}_be{cli.suffix}.txt")
    assert len(opt) == len(kf) and np.allclose(opt[:, 0], t[kf] * 1e-9, atol=1e-5), tag
    C = np.array([mat(opt[k]) @ np.linalg.inv(P[kf[k]]) for k in range(len(kf))])  # per-keyframe correction
    seg = np.clip(np.searchsorted(kf, np.arange(len(t)), side="right") - 1, 0, len(kf) - 1)
    out = []
    for i in range(len(t)):
        k = seg[i]
        if k + 1 < len(kf):
            a = (t[i] - t[kf[k]]) / max(t[kf[k + 1]] - t[kf[k]], 1)
            Ci = np.eye(4)
            Ci[:3, 3] = (1 - a) * C[k][:3, 3] + a * C[k + 1][:3, 3]
            Ci[:3, :3] = Slerp([0, 1], Rotation.from_matrix([C[k][:3, :3], C[k + 1][:3, :3]]))([a]).as_matrix()[0]
        else:
            Ci = C[k]
        M = Ci @ P[i]
        out.append([t[i] * 1e-9, *M[:3, 3], *Rotation.from_matrix(M[:3, :3]).as_quat()])
    out = np.array(out)
    out[:, 1:4], n_win = limit_window(out[:, 0], out[:, 1:4])  # jump suppression, see traj_limit.py
    out[:, 4:8], n_win_rot = limit_window_rot(out[:, 0], out[:, 4:8])
    n_win += n_win_rot
    ref, name = basalt_file(tag)
    dst = ref.parent / f"slam_cuvslam_{name}.txt"
    closure = np.linalg.norm(out[-1, 1:4] - out[0, 1:4])
    kf_closure = np.linalg.norm(opt[-1, 1:4] - opt[0, 1:4])
    print(f"{tag}: {len(out)} poses, closure {closure:.3f} m (keyframes {kf_closure:.3f}), "
          f"{n_win} jump windows smoothed -> {dst}")
    if not cli.dry_run:
        with open(dst, "w") as fh:
            for r in out:
                fh.write(f"{r[0]:.9f} " + " ".join(f"{v:.6f}" for v in r[1:]) + "\n")
