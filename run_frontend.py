"""Front end for the cuVSLAM + own-backend SLAM: one pass over a bag, two cuVSLAM trackers side by side.

  * inertial tracker (stereo + IMU, OdometryMode.Inertial)
  * stereo tracker   (stereo only, OdometryMode.Multicamera), used as a divergence guard / fallback
Saves per-frame poses of both (imu_link, z-up world), the raw IMU stream and stereo keyframe JPEGs
(selected on stereo-odometry motion) for the offline loop-closure / pose-graph back end (backend.py).

Usage (cuvslam env): python run_frontend.py BAG [--calib ...] [--out results/fe]
"""
import argparse
import heapq
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

import cuvslam

LEFT = "/sensor/camera/front_bottom_left/compressed"
RIGHT = "/sensor/camera/front_bottom_right/compressed"
IMU = "/sensor/gyro"
REORDER_WINDOW_NS = 500_000_000

ap = argparse.ArgumentParser()
ap.add_argument("bag")
ap.add_argument("--calib", default="/mnt/f/data_set/slam/calibration_front_small.json")
ap.add_argument("--mask-bottom", type=int, default=90)
ap.add_argument("--imu-noise-scale", type=float, default=0.3)
ap.add_argument("--kf-dist", type=float, default=0.25, help="keyframe every N m of stereo-odometry travel")
ap.add_argument("--kf-angle", type=float, default=8.0, help="... or every N deg of rotation")
ap.add_argument("--kf-time", type=float, default=3.0, help="... or every N s")
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--out", default="results/fe")
args = ap.parse_args()

calib = json.load(open(args.calib))["value0"]


def to_pose(M):
    return cuvslam.Pose(rotation=Rotation.from_matrix(M[:3, :3]).as_quat().tolist(), translation=M[:3, 3].tolist())


cameras = []
for T, intr, res in zip(calib["T_imu_cam"], calib["intrinsics"], calib["resolution"]):
    k = intr["intrinsics"]
    cam = cuvslam.Camera()
    cam.size = tuple(res)
    cam.focal = (k["fx"], k["fy"])
    cam.principal = (k["cx"], k["cy"])
    cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Fisheye, [k["k1"], k["k2"], k["k3"], k["k4"]])
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat([T["qx"], T["qy"], T["qz"], T["qw"]]).as_matrix()
    M[:3, 3] = [T["px"], T["py"], T["pz"]]
    cam.rig_from_camera = to_pose(M)  # rig = imu_link
    cam.border_bottom = args.mask_bottom
    cameras.append(cam)

accel_bias = np.array(calib["calib_accel_bias"][:3])
gyro_bias = np.array(calib["calib_gyro_bias"][:3])
imu = cuvslam.ImuCalibration()
imu.rig_from_imu = to_pose(np.eye(4))
s = args.imu_noise_scale
imu.gyroscope_noise_density = calib["gyro_noise_std"][0] * s
imu.gyroscope_random_walk = calib["gyro_bias_std"][0] * s
imu.accelerometer_noise_density = calib["accel_noise_std"][0] * s
imu.accelerometer_random_walk = calib["accel_bias_std"][0] * s
imu.frequency = calib["imu_update_rate"]
cam_offset_ns = int(calib["cam_time_offset_ns"])  # t_imu = t_cam + offset


def make_tracker(inertial):
    rig = cuvslam.Rig()
    rig.cameras = cameras
    if inertial:
        rig.imus = [imu]
    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False, enable_observations_export=False, enable_landmarks_export=False,
        odometry_mode=cuvslam.Tracker.OdometryMode.Inertial if inertial else cuvslam.Tracker.OdometryMode.Multicamera)
    return cuvslam.Tracker(rig, cfg, None)


trk_in, trk_st = make_tracker(True), make_tracker(False)


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def sensor_events(bag):
    """Yield (t_ns, kind, payload) in sensor-time order (bag is in arrival order)."""
    heap, seq, newest, pending = [], 0, 0, {}
    with AnyReader([Path(bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        conns = [c for c in r.connections if c.topic in (LEFT, RIGHT, IMU)]
        for conn, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, conn.msgtype)
            t = stamp_ns(m)
            if conn.topic == IMU:
                item = (t, "imu", (np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]),
                                   np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])))
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
                t0, _, kind, payload = heapq.heappop(heap)
                yield t0, kind, payload
            if len(pending) > 50:
                for k in sorted(pending)[:-10]:
                    del pending[k]
        while heap:
            t0, _, kind, payload = heapq.heappop(heap)
            yield t0, kind, payload


def row(pose):
    if pose is None:
        return [np.nan] * 7
    return [*pose.translation, *pose.rotation]


Path(args.out).mkdir(parents=True, exist_ok=True)
tag = f"{Path(args.bag).parent.name}_{Path(args.bag).name}"
frame_t, pin, pst, imu_rows, kfs = [], [], [], [], []
last_frame_t, last_kf = -1, None
wall0 = time.perf_counter()
for t, kind, payload in sensor_events(args.bag):
    if kind == "imu":
        imu_rows.append([t, *payload[0], *payload[1]])
        if t > last_frame_t:
            meas = cuvslam.ImuMeasurement()
            meas.timestamp_ns = t
            meas.linear_accelerations = payload[0] - accel_bias
            meas.angular_velocities = payload[1] - gyro_bias
            trk_in.register_imu_measurement(0, meas)
        continue
    last_pair = payload
    imgs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) for b in payload]
    e_in, _ = trk_in.track(t, imgs)
    e_st, _ = trk_st.track(t, imgs)
    last_frame_t = t
    r_in = row(e_in.world_from_rig.pose if e_in.world_from_rig is not None else None)
    r_st = row(e_st.world_from_rig.pose if e_st.world_from_rig is not None else None)
    frame_t.append(t)
    pin.append(r_in)
    pst.append(r_st)
    # keyframes on stereo-odometry motion (robust; inertial may glitch)
    if not np.isnan(r_st[0]):
        p, q = np.array(r_st[:3]), Rotation.from_quat(r_st[3:])
        if (last_kf is None or np.linalg.norm(p - last_kf[1]) > args.kf_dist
                or (last_kf[2].inv() * q).magnitude() > np.radians(args.kf_angle)
                or (t - last_kf[0]) * 1e-9 > args.kf_time):
            kfs.append((len(frame_t) - 1, payload[0], payload[1]))
            last_kf = (t, p, q)
    if len(frame_t) % 2000 == 0:
        print(f"  {len(frame_t)} frames, {len(kfs)} keyframes, {len(frame_t) / (time.perf_counter() - wall0):.0f} fps",
              flush=True)
    if args.max_frames and len(frame_t) >= args.max_frames:
        break
if kfs[-1][0] != len(frame_t) - 1:  # always keep the last frame as a keyframe (last_pair: the last event may be IMU)
    kfs.append((len(frame_t) - 1, last_pair[0], last_pair[1]))

np.savez(f"{args.out}/{tag}_fe.npz", t=np.array(frame_t, np.int64), pose_in=np.array(pin), pose_st=np.array(pst),
         imu=np.array(imu_rows), kf_idx=np.array([k[0] for k in kfs]))
with open(f"{args.out}/{tag}_kf.pkl", "wb") as f:
    pickle.dump([(k[1], k[2]) for k in kfs], f)
n_nan_in = int(np.isnan(np.array(pin)[:, 0]).sum())
print(f"{len(frame_t)} frames, {len(kfs)} keyframes, inertial lost {n_nan_in}, wall {time.perf_counter() - wall0:.0f}s")
print("FE_DONE")
