"""Export the SLAM-optimised trajectories in Basalt's format, next to Basalt's vio_basalt_*.txt files.

Basalt format: no header, "timestamp tx ty tz qx qy qz qw" per line, timestamp in seconds (9 decimals, IMU
clock), the rest 6 decimals; pose of imu_link in a gravity-aligned world whose origin is the first pose.
The back end optimises keyframes; every front-end frame (~5.3 Hz) gets its keyframe's optimised pose plus the
odometry motion since that keyframe.

Usage: python export_slam_traj.py [FE_DIR=results/fe3] [SUFFIX=_v8] [--odom vote] [--dry-run]
       -> /mnt/f/data_set/slam/<date>[/<bag>]/slam_cuvslam_<bag>.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("fe_dir", nargs="?", default="results/fe3")
ap.add_argument("suffix", nargs="?", default="_v8")
ap.add_argument("--odom", default="vote", help="odometry the back end was run with")
ap.add_argument("--data", default="/mnt/f/data_set/slam")
ap.add_argument("--dry-run", action="store_true")
cli = ap.parse_args()
sys.argv = sys.argv[:1]
import backend as B  # noqa: E402

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
    P = B.vote_odometry(z) if cli.odom == "vote" else B.build_odometry(z["pose_in"], z["pose_st"], cli.odom, t_ns=t)[0]
    kf = z["kf_idx"]
    opt = np.loadtxt(fe / f"{tag}_be{cli.suffix}.txt")
    assert len(opt) == len(kf) and np.allclose(opt[:, 0], t[kf] * 1e-9, atol=1e-5), tag
    seg = np.clip(np.searchsorted(kf, np.arange(len(t)), side="right") - 1, 0, len(kf) - 1)
    out = []
    for i in range(len(t)):
        k = seg[i]
        M = mat(opt[k]) @ np.linalg.inv(P[kf[k]]) @ P[i]
        out.append([t[i] * 1e-9, *M[:3, 3], *Rotation.from_matrix(M[:3, :3]).as_quat()])
    out = np.array(out)
    ref, name = basalt_file(tag)
    dst = ref.parent / f"slam_cuvslam_{name}.txt"
    closure = np.linalg.norm(out[-1, 1:4] - out[0, 1:4])
    kf_closure = np.linalg.norm(opt[-1, 1:4] - opt[0, 1:4])
    print(f"{tag}: {len(out)} poses, closure {closure:.3f} m (keyframes {kf_closure:.3f}) -> {dst}")
    if not cli.dry_run:
        with open(dst, "w") as fh:
            for r in out:
                fh.write(f"{r[0]:.9f} " + " ".join(f"{v:.6f}" for v in r[1:]) + "\n")
