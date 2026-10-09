"""Loosely-coupled gyro heading fusion on top of cuVSLAM visual odometry.

VO keeps its translation increments (expressed in the body frame) and its roll/pitch, but the heading
is replaced by gyro integration: the bias-corrected angular rate is projected on the world vertical
(tilt taken from VO) and integrated. The residual gyro bias is estimated from the stationary period
at the start of the run (detected from VO). Everything is causal, so it can run online.

Usage: python gyro_fusion.py RESULTS_DIR [--suffix _stereo_slam] [--calib calibration.json]
Writes <tag>_gyrovo.txt next to each <tag>_vo.txt.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation


def load_gyro(bag_dir, calib):
    """imu.csv exported next to each bag: (t [s], gyro xyz [rad/s]) with calibrated bias removed."""
    a = np.loadtxt(os.path.join(bag_dir, "imu.csv"), delimiter=",", skiprows=1, usecols=(0, 3, 4, 5))
    a[:, 0] *= 1e-9
    a[:, 1:] -= np.array(calib["calib_gyro_bias"][:3])
    return a


def static_bias(vo, gyro, max_move=0.01, min_len=1.0):
    """Mean gyro over the initial window where VO has not moved yet (None if too short)."""
    moved = np.linalg.norm(vo[:, 1:4] - vo[0, 1:4], axis=1) > max_move
    t_end = vo[np.argmax(moved), 0] if moved.any() else vo[-1, 0]
    t_end -= 0.3  # margin before motion onset
    sel = (gyro[:, 0] >= vo[0, 0]) & (gyro[:, 0] < t_end)
    if t_end - vo[0, 0] < min_len or sel.sum() < 100:
        return np.zeros(3), 0.0
    return gyro[sel, 1:].mean(0), float(t_end - vo[0, 0])


def fuse(vo, gyro, bias):
    """Return a VO trajectory whose heading comes from the vertical gyro rate."""
    R_vo = Rotation.from_quat(vo[:, 4:8])
    # vertical axis in body coordinates from VO tilt (world z is up: world = first imu pose)
    g = gyro[(gyro[:, 0] >= vo[0, 0]) & (gyro[:, 0] <= vo[-1, 0])]
    idx = np.clip(np.searchsorted(vo[:, 0], g[:, 0]), 0, len(vo) - 1)
    up_body = R_vo[idx].inv().apply([0, 0, 1])
    yaw_rate = np.einsum("ij,ij->i", g[:, 1:] - bias, up_body)
    yaw_gyro = np.concatenate([[0], np.cumsum(0.5 * (yaw_rate[1:] + yaw_rate[:-1]) * np.diff(g[:, 0]))])
    yaw_gyro_at_vo = np.interp(vo[:, 0], g[:, 0], yaw_gyro)

    yaw_vo = np.unwrap(R_vo.as_euler("zyx")[:, 0])
    corr = Rotation.from_euler("z", yaw_gyro_at_vo - (yaw_vo - yaw_vo[0]))  # world-frame yaw correction
    R_new = corr * R_vo
    # re-chain VO body-frame translation increments with the corrected orientation
    d_body = R_vo[:-1].inv().apply(np.diff(vo[:, 1:4], axis=0))
    p = np.vstack([vo[:1, 1:4], vo[0, 1:4] + np.cumsum(R_new[:-1].apply(d_body), axis=0)])
    return np.column_stack([vo[:, 0], p, R_new.as_quat()])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--suffix", default="_stereo_slam")
    ap.add_argument("--calib", default="/home/cj/data/slam/calibration_front_small.json")
    args = ap.parse_args()
    calib = json.load(open(args.calib))["value0"]
    for f in sorted(glob.glob(f"{args.results}/*{args.suffix}_summary.json")):
        prefix = f[: -len("_summary.json")]
        bag = json.load(open(f))["bag"]
        vo = np.loadtxt(f"{prefix}_vo.txt")
        gyro = load_gyro(bag, calib)
        bias, static_s = static_bias(vo, gyro)
        fused = fuse(vo, gyro, bias)
        np.savetxt(f"{prefix}_gyrovo.txt", fused, fmt="%.6f", header="timestamp tx ty tz qx qy qz qw")
        print(f"{os.path.basename(prefix)}: static {static_s:.1f}s, residual bias z {1e3 * bias[2]:+.3f} mrad/s")
