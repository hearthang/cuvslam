"""Run cuVSLAM on the front stereo camera (+ IMU) of a ROS2 bag from /mnt/e/slam.

Reads the sqlite3 bag directly with `rosbags` (no ROS install needed), uses the basalt-style
calibration in calibration_front_small.json, and writes TUM-format trajectories plus a
top-down comparison plot against wheel odometry.

Usage (in the `cuvslam` conda env):
    python run_bag.py /mnt/e/slam/8-7/2026_04_28-04_23_58 [--mode stereo|inertial] [--slam]
"""
import argparse
import heapq
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

import cuvslam
from loop_check import closure_error, measure_first_last, traj_first_last

LEFT = "/sensor/camera/front_bottom_left/compressed"
RIGHT = "/sensor/camera/front_bottom_right/compressed"
IMU = "/sensor/gyro"
ODOM = "/sensor/odom"
REORDER_WINDOW_NS = 500_000_000  # bag is ordered by arrival time; reorder by sensor stamp

ap = argparse.ArgumentParser()
ap.add_argument("bag")
ap.add_argument("--calib", default="/mnt/e/slam/calibration_front_small.json")
ap.add_argument("--mode", choices=["stereo", "inertial"], default="stereo")
ap.add_argument("--slam", action="store_true", help="enable loop closure / pose graph")
ap.add_argument("--mask-bottom", type=int, default=90, help="rows hidden by the robot body")
ap.add_argument("--max-frames", type=int, default=0)
# Mower vibration while driving is ~20x the static IMU noise; with the calibrated values the
# inertial mode diverges within ~30 s, x20 keeps it stable (tested on 8-7/2026_04_28-04_23_58).
ap.add_argument("--imu-noise-scale", type=float, default=20.0, help="inflate calibrated IMU noise")
ap.add_argument("--imu-avg", type=int, default=1,
                help="box-average N consecutive IMU samples before feeding cuVSLAM (low-pass vs mower vibration)")
ap.add_argument("--rig-frame", choices=["imu", "level-cam"], default="imu",
                help="level-cam: x right, y down, z forward, levelled with imu_link. Needed for --planar, because "
                     "cuVSLAM's planar constraint uses the rig y axis as the ground normal")
ap.add_argument("--planar", action="store_true", help="SLAM planar (ground) constraint")
ap.add_argument("--max-map-size", type=int, default=300, help="SLAM pose-graph size limit, 0 = unlimited")
ap.add_argument("--denoise", action="store_true", help="odometry image denoising")
ap.add_argument("--tag-suffix", default="", help="appended to output file names")
ap.add_argument("--out", default="results")
args = ap.parse_args()

# ---------------- rig from basalt calibration ----------------
if args.planar and args.rig_frame != "level-cam":
    ap.error("--planar needs --rig-frame level-cam")
calib = json.load(open(args.calib))["value0"]
# T_rig_imu: identity, or the levelled camera-style frame (rig x = -imu y, rig y = -imu z, rig z = imu x)
T_rig_imu = np.eye(4)
if args.rig_frame == "level-cam":
    T_rig_imu[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)


def to_pose(M):
    return cuvslam.Pose(rotation=Rotation.from_matrix(M[:3, :3]).as_quat().tolist(), translation=M[:3, 3].tolist())


def rig_to_imu_row(t, pose):
    """TUM row of the imu_link pose, whatever rig frame cuVSLAM was run in."""
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(pose.rotation).as_matrix()
    M[:3, 3] = pose.translation
    M = T_rig_imu.T @ M @ T_rig_imu  # world expressed in imu axes too, so plots/eval stay z-up
    return [t, *M[:3, 3], *Rotation.from_matrix(M[:3, :3]).as_quat()]

cameras = []
for T, intr, res in zip(calib["T_imu_cam"], calib["intrinsics"], calib["resolution"]):
    assert intr["camera_type"] == "kb4"
    k = intr["intrinsics"]
    cam = cuvslam.Camera()
    cam.size = tuple(res)
    cam.focal = (k["fx"], k["fy"])
    cam.principal = (k["cx"], k["cy"])
    cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Fisheye, [k["k1"], k["k2"], k["k3"], k["k4"]])
    T_imu_cam = np.eye(4)
    T_imu_cam[:3, :3] = Rotation.from_quat([T["qx"], T["qy"], T["qz"], T["qw"]]).as_matrix()
    T_imu_cam[:3, 3] = [T["px"], T["py"], T["pz"]]
    cam.rig_from_camera = to_pose(T_rig_imu @ T_imu_cam)
    cam.border_bottom = args.mask_bottom
    cameras.append(cam)

rig = cuvslam.Rig()
rig.cameras = cameras
accel_bias = np.array(calib["calib_accel_bias"][:3])
gyro_bias = np.array(calib["calib_gyro_bias"][:3])
if args.mode == "inertial":
    imu = cuvslam.ImuCalibration()
    imu.rig_from_imu = to_pose(T_rig_imu)
    imu.gyroscope_noise_density = calib["gyro_noise_std"][0] * args.imu_noise_scale
    imu.gyroscope_random_walk = calib["gyro_bias_std"][0] * args.imu_noise_scale
    imu.accelerometer_noise_density = calib["accel_noise_std"][0] * args.imu_noise_scale
    imu.accelerometer_random_walk = calib["accel_bias_std"][0] * args.imu_noise_scale
    imu.frequency = calib["imu_update_rate"] / args.imu_avg
    rig.imus = [imu]
cam_offset_ns = int(calib["cam_time_offset_ns"])  # t_imu = t_cam + offset

odom_cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=False,
    use_denoising=args.denoise,
    enable_observations_export=False,
    enable_landmarks_export=False,
    odometry_mode=(cuvslam.Tracker.OdometryMode.Inertial if args.mode == "inertial"
                   else cuvslam.Tracker.OdometryMode.Multicamera),
)
slam_cfg = (cuvslam.Tracker.SlamConfig(sync_mode=True, planar_constraints=args.planar, max_map_size=args.max_map_size)
            if args.slam else None)
tracker = cuvslam.Tracker(rig, odom_cfg, slam_cfg)


# ---------------- bag reading, reordered by sensor timestamp ----------------
def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def sensor_events(bag):
    """Yield (t_ns, kind, payload) in sensor-time order."""
    heap, seq, newest = [], 0, 0
    pending = {}  # stamp -> {topic: jpeg bytes}, to pair left/right
    with AnyReader([Path(bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        conns = [c for c in r.connections if c.topic in (LEFT, RIGHT, IMU, ODOM)]
        for conn, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            t = stamp_ns(m)
            if conn.topic == IMU:
                item = (t, "imu", (np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]),
                                   np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])))
            elif conn.topic == ODOM:
                p, q = m.pose.pose.position, m.pose.pose.orientation
                item = (t, "odom", (p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            else:
                if len(m.data) == 0:
                    continue
                pair = pending.setdefault(t, {})
                pair[conn.topic] = bytes(m.data)
                if len(pair) < 2:
                    continue
                del pending[t]
                item = (t + cam_offset_ns, "stereo", (pair[LEFT], pair[RIGHT]))
            heapq.heappush(heap, (item[0], seq, item[1], item[2]))
            seq += 1
            newest = max(newest, t)
            while heap and heap[0][0] < newest - REORDER_WINDOW_NS:
                yield _pop(heap)
            if len(pending) > 50:  # drop unpaired frames
                for k in sorted(pending)[:-10]:
                    del pending[k]
        while heap:
            yield _pop(heap)


def _pop(heap):
    t, _, kind, payload = heapq.heappop(heap)
    return t, kind, payload


# ---------------- main loop ----------------
os.makedirs(args.out, exist_ok=True)
tag = f"{Path(args.bag).parent.name}_{Path(args.bag).name}_{args.mode}{'_slam' if args.slam else ''}{args.tag_suffix}"
odom_traj, wheel_traj, gyro, imu_buf = [], [], [], []
n_frames = n_fail = n_imu = 0
t_track = t_decode = 0.0
last_frame_t = -1
first_imgs = last_left = None
wall0 = time.perf_counter()

for t, kind, payload in sensor_events(args.bag):
    if kind == "odom":
        wheel_traj.append([t * 1e-9, *payload])
        continue
    if kind == "imu":
        gyro.append([t * 1e-9, *(payload[1] - gyro_bias)])
        if args.mode == "inertial":
            imu_buf.append((t, payload[0] - accel_bias, payload[1] - gyro_bias))
            if len(imu_buf) < args.imu_avg:
                continue
            t_avg = int(np.mean([b[0] for b in imu_buf]))
            if t_avg > last_frame_t:
                meas = cuvslam.ImuMeasurement()
                meas.timestamp_ns = t_avg
                meas.linear_accelerations = np.mean([b[1] for b in imu_buf], axis=0)
                meas.angular_velocities = np.mean([b[2] for b in imu_buf], axis=0)
                tracker.register_imu_measurement(0, meas)
                n_imu += 1
            imu_buf.clear()
        continue

    t0 = time.perf_counter()
    imgs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) for b in payload]
    t1 = time.perf_counter()
    pose_est, _ = tracker.track(t, imgs)
    if args.slam and n_frames == 0:
        tracker.get_pose_graph()  # first call switches on pose-graph recording, read again at the end
    t_track += time.perf_counter() - t1
    t_decode += t1 - t0
    last_frame_t = t
    n_frames += 1
    if pose_est.world_from_rig is None:
        n_fail += 1
    else:
        odom_traj.append(rig_to_imu_row(t * 1e-9, pose_est.world_from_rig.pose))
        if first_imgs is None:
            first_imgs = imgs
        last_left = imgs[0]
    if n_frames % 1000 == 0:
        print(f"  {n_frames} frames, {n_fail} failed, {n_frames / (time.perf_counter() - wall0):.0f} frames/s wall")
    if args.max_frames and n_frames >= args.max_frames:
        break

wall = time.perf_counter() - wall0
header = "timestamp tx ty tz qx qy qz qw"
odom_traj, wheel_traj = np.array(odom_traj), np.array(wheel_traj)
np.savetxt(f"{args.out}/{tag}_vo.txt", odom_traj, fmt="%.6f", header=header)
np.savetxt(f"{args.out}/{tag}_wheel.txt", wheel_traj, fmt="%.6f", header=header)
np.save(f"{args.out}/{tag}_gyro.npy", np.array(gyro, np.float64))  # bias-corrected, imu_link, for offline fusion
trajs = {"cuVSLAM odometry": odom_traj}
if args.slam:
    slam = np.array([rig_to_imu_row(ps.timestamp_ns * 1e-9, ps.pose) for ps in tracker.get_all_slam_poses()])
    try:  # raw pose graph (rig frame) for offline loop-closure checking / re-optimisation
        pg = tracker.get_pose_graph()
        np.savez(f"{args.out}/{tag}_posegraph.npz",
                 node_id=np.array([n.id for n in pg.nodes]),
                 node_pose=np.array([[*n.node_pose.translation, *n.node_pose.rotation] for n in pg.nodes]),
                 edge_ft=np.array([[e.node_from, e.node_to] for e in pg.edges]),
                 edge_tf=np.array([[*e.transform.translation, *e.transform.rotation] for e in pg.edges]),
                 edge_cov=np.array([np.asarray(e.covariance, float).ravel() for e in pg.edges]),
                 T_rig_imu=T_rig_imu)
    except Exception as ex:
        print(f"pose graph export failed: {ex}")
    np.savetxt(f"{args.out}/{tag}_slam.txt", slam, fmt="%.6f", header=header)
    trajs["cuVSLAM SLAM"] = slam

print(f"\n{n_frames} stereo frames ({n_imu} IMU samples), tracking failed on {n_fail}")
print(f"track(): {1000 * t_track / n_frames:.2f} ms/frame; jpeg decode {1000 * t_decode / n_frames:.2f} ms/frame; "
      f"wall {wall:.0f}s for {(odom_traj[-1, 0] - odom_traj[0, 0]):.0f}s of data")


# ---------------- evaluation vs wheel odometry ----------------
def path_len(xyz):
    return np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()


def to_2d_start(tr):
    """Express trajectory relative to its first pose, projected on the gravity-ish XY plane of that pose."""
    R0 = Rotation.from_quat(tr[0, 4:8])
    xyz = R0.inv().apply(tr[:, 1:4] - tr[0, 1:4])
    return xyz


def align_2d(src, dst):
    """Best 2D rigid transform (yaw + xy) mapping src->dst (Umeyama without scale)."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    return (src - ms) @ R.T + md


loop = measure_first_last(calib, first_imgs, last_left, args.mask_bottom)
T_meas = loop["T_first_last"]
print(f"first/last image check: {loop['fund_inliers']} essential inliers, {loop['pnp_inliers']} PnP inliers -> "
      + (f"same place, measured first->last offset {np.linalg.norm(T_meas[:2, 3]):.2f} m" if loop["same_place"]
         else "NOT recognised as the start place"))
trajs_eval = {"vo": (odom_traj, odom_traj)}
if len(wheel_traj):  # some bags (e.g. 9-17, 9-21) have no /sensor/odom
    # wheel odometry sampled at the first/last camera frame times
    wi = np.clip(np.searchsorted(wheel_traj[:, 0], [odom_traj[0, 0], odom_traj[-1, 0]]), 0, len(wheel_traj) - 1)
    trajs_eval["wheel"] = (wheel_traj, wheel_traj[wi[0]:wi[1] + 1])
if args.slam:
    trajs_eval["slam"] = (slam, slam)
summary = {"bag": str(args.bag), "mode": args.mode, "slam": args.slam, "frames": n_frames, "failed": n_fail,
           "duration_s": float(odom_traj[-1, 0] - odom_traj[0, 0]), "track_ms": 1000 * t_track / n_frames,
           "wall_s": wall, "fund_inliers": loop["fund_inliers"], "pnp_inliers": loop["pnp_inliers"],
           "same_place": loop["same_place"],
           "measured_offset_m": float(np.linalg.norm(T_meas[:2, 3])) if loop["same_place"] else None}
for key, (full, seg) in trajs_eval.items():
    rel = to_2d_start(seg)
    d = {"path_m": float(path_len(rel)), "gap_m": float(np.linalg.norm(rel[-1, :2]))}
    if loop["same_place"]:
        d["closure_m"], d["closure_xy_m"], d["closure_yaw_deg"] = closure_error(traj_first_last(seg), T_meas)
    summary[key] = d
    print(f"{key}: " + ", ".join(f"{k} {v:.3f}" for k, v in d.items()))
with open(f"{args.out}/{tag}_summary.json", "w") as f:
    json.dump(summary, f, indent=1)

wheel_rel = to_2d_start(wheel_traj) if len(wheel_traj) else None
plots = {}
for name, tr in trajs.items():
    rel = to_2d_start(tr)
    if wheel_rel is None:
        print(f"{name}: path {path_len(rel):.1f} m, end-start gap {np.linalg.norm(rel[-1, :2]):.2f} m")
        plots[name] = rel[:, :2]
        continue
    idx = np.clip(np.searchsorted(wheel_traj[:, 0], tr[:, 0]), 0, len(wheel_traj) - 1)
    aligned = align_2d(rel[:, :2], wheel_rel[idx, :2])
    dev = np.linalg.norm(aligned - wheel_rel[idx, :2], axis=1)
    print(f"{name}: path {path_len(rel):.1f} m, end-start gap {np.linalg.norm(rel[-1, :2]):.2f} m, "
          f"z range {rel[:, 2].min():.2f}..{rel[:, 2].max():.2f} m, "
          f"RMS deviation from wheel odom (2D aligned) {np.sqrt((dev ** 2).mean()):.2f} m")
    plots[name] = aligned

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(9, 9))
if wheel_rel is not None:
    ax.plot(wheel_rel[:, 0], wheel_rel[:, 1], color="0.6", lw=1, label="wheel odometry")
for (name, xy), c in zip(plots.items(), ["C0", "C3"]):
    ax.plot(xy[:, 0], xy[:, 1], color=c, lw=1, label=name)
ax.plot(0, 0, "k*", ms=12, label="start")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.legend()
ax.set_title(tag)
fig.savefig(f"{args.out}/{tag}.png", dpi=120, bbox_inches="tight")
print(f"outputs in {args.out}/{tag}_*")
