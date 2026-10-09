"""Headless cuVSLAM RGB-D run on TUM fr3 + ATE evaluation against ground truth.

Usage (in the `cuvslam` conda env):
    python run_tum_eval.py [dataset_dir]
"""
import os
import sys
import time

import numpy as np
import yaml

import cuvslam

EXAMPLES_TUM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuVSLAM/examples/tum")
sys.path.insert(0, EXAMPLES_TUM)
from dataset_utils import load_frame, get_matched_rgbd_pairs  # noqa: E402

dataset = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    EXAMPLES_TUM, "dataset/rgbd_dataset_freiburg3_long_office_household")

with open(os.path.join(dataset, "freiburg3_rig.yaml")) as f:
    cfg = yaml.safe_load(f)

camera = cuvslam.Camera()
camera.size = (cfg["rgb_camera"]["image_width"], cfg["rgb_camera"]["image_height"])
camera.principal = cfg["rgb_camera"]["principal_point"]
camera.focal = cfg["rgb_camera"]["focal_length"]
camera.border_top, camera.border_bottom = 20, 20
camera.border_left, camera.border_right = 10, 50

rgbd = cuvslam.Tracker.OdometryRGBDSettings()
rgbd.depth_scale_factor = cfg["depth_camera"]["scale"]
rgbd.depth_camera_id = 0
rgbd.enable_depth_stereo_tracking = False

odom_cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=False,  # offline run: deterministic, no frame dropping
    odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
    rgbd_settings=rgbd,
)
tracker = cuvslam.Tracker(cuvslam.Rig([camera]), odom_cfg)

pairs = get_matched_rgbd_pairs(dataset, max_time_diff=0.02, max_gap=0.5)
print(f"{len(pairs)} RGB-D pairs")

frames = [(t, load_frame(r), load_frame(d)) for t, r, d in pairs]  # preload so we time tracking only
est, failed, track_time = [], 0, 0.0
for t, color, depth in frames:
    t0 = time.perf_counter()
    pose_est, _ = tracker.track(int(t * 1e9), images=[color], depths=[depth])
    track_time += time.perf_counter() - t0
    if pose_est.world_from_rig is None:
        failed += 1
        continue
    p = pose_est.world_from_rig.pose
    est.append([t, *p.translation, *p.rotation])  # rotation is (qx, qy, qz, qw)

est = np.array(est)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tum_fr3_cuvslam_traj.txt")
np.savetxt(out, est, fmt="%.6f", header="timestamp tx ty tz qx qy qz qw")
print(f"tracked {len(est)}/{len(frames)} frames, failed {failed}")
print(f"avg track() time: {1000 * track_time / len(frames):.2f} ms  ({len(frames) / track_time:.0f} FPS)")
print(f"trajectory saved to {out}")

# ---- ATE: associate by timestamp, Umeyama SE(3) alignment (RGB-D has metric scale) ----
gt = np.loadtxt(os.path.join(dataset, "groundtruth.txt"), comments="#")
idx = np.clip(np.searchsorted(gt[:, 0], est[:, 0]), 1, len(gt) - 1)
idx -= (est[:, 0] - gt[idx - 1, 0]) < (gt[idx, 0] - est[:, 0])
ok = np.abs(gt[idx, 0] - est[:, 0]) < 0.02
A, B = est[ok, 1:4], gt[idx[ok], 1:4]

mu_a, mu_b = A.mean(0), B.mean(0)
U, _, Vt = np.linalg.svd((B - mu_b).T @ (A - mu_a))
S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
R = U @ S @ Vt
err = np.linalg.norm((A - mu_a) @ R.T + mu_b - B, axis=1)
length = np.linalg.norm(np.diff(B, axis=0), axis=1).sum()
print(f"ATE over {ok.sum()} poses (GT path {length:.2f} m): "
      f"RMSE {100 * np.sqrt((err ** 2).mean()):.2f} cm, mean {100 * err.mean():.2f} cm, max {100 * err.max():.2f} cm")
