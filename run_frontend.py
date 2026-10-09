"""Front end for the cuVSLAM + own-backend SLAM: one pass over a bag, two cuVSLAM trackers side by side.

  * inertial tracker (stereo + IMU, OdometryMode.Inertial)
  * stereo tracker   (stereo only, OdometryMode.Multicamera), used as a divergence guard / fallback
Saves per-frame poses of both (imu_link, z-up world), the raw IMU stream and stereo keyframe JPEGs
(selected on stereo-odometry motion) for the offline loop-closure / pose-graph back end (backend.py).

Usage (cuvslam env): python run_frontend.py BAG [--calib ...] [--out results/fe]
"""
import argparse
import collections
import heapq
import itertools
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
ap.add_argument("--imu-noise-scale", type=float, default=0.3, help="scale of the calibrated IMU noise (gyro, and accel too unless --acc-noise-scale)")
ap.add_argument("--acc-noise-scale", type=float, default=None,
                help="separate scale for the accelerometer noise (mower vibration is ~1 m/s^2, ~7x the x0.3 value)")
ap.add_argument("--kf-dist", type=float, default=0.25, help="keyframe every N m of stereo-odometry travel")
ap.add_argument("--kf-angle", type=float, default=8.0, help="... or every N deg of rotation")
ap.add_argument("--kf-time", type=float, default=3.0, help="... or every N s")
ap.add_argument("--max-frames", type=int, default=0)
# front-end experiment switches
ap.add_argument("--mask-lr", type=int, default=0, help="left/right image border to ignore [px]")
ap.add_argument("--mask-top", type=int, default=0)
ap.add_argument("--rectify", action="store_true", help="undistort kb4 fisheye to pinhole before cuVSLAM")
ap.add_argument("--skip", type=int, default=3,
                help="use every N-th stereo frame (16 Hz camera; 5.3 Hz roughly halves cuVSLAM's drift here, 4 Hz breaks)")
ap.add_argument("--denoise", action="store_true")
ap.add_argument("--async-sba", action="store_true")
ap.add_argument("--no-motion-model", action="store_true")
ap.add_argument("--multicam-mode", default="Precision", choices=["Precision", "Moderate", "Performance"])
ap.add_argument("--gyro-bias", default="static", choices=["calib", "static"],
                help="static: estimate gyro bias from the first 1.5 s (the mower is parked at the start)")
ap.add_argument("--gyro-scale-z", type=float, default=0.0,
                help="gyro z scale-factor error to remove, e.g. 0.0085 (+0.85%%, estimated from stereo rotation on turns)")
ap.add_argument("--time-offset-ns", type=int, default=None, help="override calib cam_time_offset_ns (t_imu = t_cam + offset)")
ap.add_argument("--cam1-fscale", type=float, default=-0.00038,
                help="relative focal-length correction of cam1 (right). Visual yaw drifts while driving straight and is "
                     "very sensitive to this (0.2%% -> ~50 deg / 500 s); -0.038%% is self-calibrated with "
                     "selfcal_yaw.py (visual vs gyro yaw on straight segments, no ground truth)")
ap.add_argument("--cam1-dcx", type=float, default=0.0, help="principal-point x shift of cam1 [px]")
ap.add_argument("--inertial-phases", type=int, default=3,
                help="number of stereo+IMU trackers, each on a different frame phase (0..skip-1); the extra ones are "
                     "redundant estimators for per-window voting (cuVSLAM inertial occasionally drifts in velocity)")
ap.add_argument("--imu-pnp-robust", type=float, default=None, help="inertial PnP robustifier scale (cuVSLAM default 0.4)")
ap.add_argument("--reinit-handling", action="store_true",
                help="re-create the stereo+IMU tracker(s) once the mower is back on the ground after being lifted or "
                     "tipped by hand (accelerometer tilt > 25 deg, released after 1 s below 12 deg): cuVSLAM's IMU "
                     "state is corrupted by the handling and does not recover; poses are chained across the restart")
ap.add_argument("--trackers", default="both", choices=["both", "inertial", "stereo"])
ap.add_argument("--no-kf", action="store_true", help="do not save keyframe images (experiments)")
ap.add_argument("--out", default="results/fe")
args = ap.parse_args()

calib = json.load(open(args.calib))["value0"]


def to_pose(M):
    return cuvslam.Pose(rotation=Rotation.from_matrix(M[:3, :3]).as_quat().tolist(), translation=M[:3, 3].tolist())


cameras, rect_maps = [], []
for ci, (T, intr, res) in enumerate(zip(calib["T_imu_cam"], calib["intrinsics"], calib["resolution"])):
    k = dict(intr["intrinsics"])
    if ci == 1:
        k["fx"] *= 1 + args.cam1_fscale
        k["fy"] *= 1 + args.cam1_fscale
        k["cx"] += args.cam1_dcx
    cam = cuvslam.Camera()
    cam.size = tuple(res)
    border_bottom = args.mask_bottom
    if args.rectify:
        K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
        D = np.array([k["k1"], k["k2"], k["k3"], k["k4"]])
        Kn = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, tuple(res), np.eye(3), balance=0.3)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kn, tuple(res), cv2.CV_16SC2)
        rect_maps.append((m1, m2))
        cam.focal = (Kn[0, 0], Kn[1, 1])
        cam.principal = (Kn[0, 2], Kn[1, 2])
        cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)
        body = np.zeros((res[1], res[0]), np.uint8)
        body[res[1] - args.mask_bottom:] = 255
        rows = np.where(cv2.remap(body, m1, m2, cv2.INTER_NEAREST).max(axis=1) > 0)[0]
        border_bottom = int(res[1] - rows.min()) if len(rows) else 0
    else:
        cam.focal = (k["fx"], k["fy"])
        cam.principal = (k["cx"], k["cy"])
        cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Fisheye, [k["k1"], k["k2"], k["k3"], k["k4"]])
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat([T["qx"], T["qy"], T["qz"], T["qw"]]).as_matrix()
    M[:3, 3] = [T["px"], T["py"], T["pz"]]
    cam.rig_from_camera = to_pose(M)  # rig = imu_link
    cam.border_bottom = border_bottom
    cam.border_left = cam.border_right = args.mask_lr
    cam.border_top = args.mask_top
    cameras.append(cam)

accel_bias = np.array(calib["calib_accel_bias"][:3])
gyro_bias = np.array(calib["calib_gyro_bias"][:3])
imu = cuvslam.ImuCalibration()
imu.rig_from_imu = to_pose(np.eye(4))
s = args.imu_noise_scale
imu.gyroscope_noise_density = calib["gyro_noise_std"][0] * s
imu.gyroscope_random_walk = calib["gyro_bias_std"][0] * s
sa = s if args.acc_noise_scale is None else args.acc_noise_scale
imu.accelerometer_noise_density = calib["accel_noise_std"][0] * sa
imu.accelerometer_random_walk = calib["accel_bias_std"][0] * sa
imu.frequency = calib["imu_update_rate"]
cam_offset_ns = int(calib["cam_time_offset_ns"]) if args.time_offset_ns is None else args.time_offset_ns


def make_tracker(inertial):
    rig = cuvslam.Rig()
    rig.cameras = cameras
    if inertial:
        rig.imus = [imu]
    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=args.async_sba, use_denoising=args.denoise, use_motion_model=not args.no_motion_model,
        multicam_mode=getattr(cuvslam.Tracker.MulticameraMode, args.multicam_mode),
        enable_observations_export=False, enable_landmarks_export=False,
        odometry_mode=cuvslam.Tracker.OdometryMode.Inertial if inertial else cuvslam.Tracker.OdometryMode.Multicamera)
    for attempt in range(5):  # many processes initialising CUDA at once can get "device busy"
        try:
            return cuvslam.Tracker(rig, cfg, None)
        except RuntimeError as ex:
            if attempt == 4:
                raise
            print(f"tracker init failed ({ex}), retrying", flush=True)
            time.sleep(5 + 5 * attempt)


internals = None
if args.imu_pnp_robust is not None:
    internals = cuvslam.Tracker.Internals()
    internals.imu_pnp_robustifier_scale = args.imu_pnp_robust
trk_in = make_tracker(True) if args.trackers != "stereo" else None
trk_st = make_tracker(False) if args.trackers != "inertial" else None
n_aux = min(args.inertial_phases, args.skip) - 1 if trk_in is not None else 0
trk_aux = [make_tracker(True) for _ in range(n_aux)]  # phase k+1 frames
aux_t, aux_pose, aux_last_t = [[] for _ in range(n_aux)], [[] for _ in range(n_aux)], [-1] * n_aux


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
events = sensor_events(args.bag)
if args.gyro_bias == "static":
    # the mower is parked when a recording starts: use the first 1.5 s of gyro as this run's bias
    head = []
    for ev in events:
        head.append(ev)
        if ev[0] - head[0][0] > 1.5e9:
            break
    g = np.array([p[1] for t, k, p in head if k == "imu"])
    a = np.array([p[0] for t, k, p in head if k == "imu"])
    if len(g) > 300 and a.std(0).max() < 0.15:
        gyro_bias = g.mean(0)
    print(f"gyro bias {gyro_bias} (calib {calib['calib_gyro_bias'][:3]})", flush=True)
    events = itertools.chain(head, events)
n_stereo = 0
acc_win = collections.deque()  # (t, accel) over the last 0.5 s, for the off-the-ground detector
handling_on, low_since, reinit_times = False, None, []
T_off = np.eye(4)                # inertial-tracker output = T_off @ tracker pose (changes at each restart)
pending_offset, last_out = False, None


def to_mat(r):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(r[3:7]).as_matrix()
    M[:3, 3] = r[:3]
    return M


for t, kind, payload in events:
    if kind == "imu":
        imu_rows.append([t, *payload[0], *payload[1]])
        acc_win.append((t, payload[0]))
        while acc_win and acc_win[0][0] < t - 5e8:
            acc_win.popleft()
        if trk_in is not None:
            meas = cuvslam.ImuMeasurement()
            meas.timestamp_ns = t
            meas.linear_accelerations = payload[0] - accel_bias
            meas.angular_velocities = (payload[1] - gyro_bias) / np.array([1, 1, 1 + args.gyro_scale_z])
            if t > last_frame_t:
                trk_in.register_imu_measurement(0, meas)
            for k, trk in enumerate(trk_aux):
                if t > aux_last_t[k]:
                    trk.register_imu_measurement(0, meas)
        continue
    n_stereo += 1
    phase = (n_stereo - 1) % args.skip
    if 1 <= phase <= n_aux:
        imgs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) for b in payload]
        if args.rectify:
            imgs = [cv2.remap(im, *mp, cv2.INTER_LINEAR) for im, mp in zip(imgs, rect_maps)]
        e = trk_aux[phase - 1].track(t, imgs)[0]
        aux_last_t[phase - 1] = t
        aux_t[phase - 1].append(t)
        aux_pose[phase - 1].append(row(e.world_from_rig.pose if e.world_from_rig is not None else None))
        continue
    if phase:
        continue
    last_pair = payload
    imgs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) for b in payload]
    if args.rectify:
        imgs = [cv2.remap(im, *mp, cv2.INTER_LINEAR) for im, mp in zip(imgs, rect_maps)]
    if args.reinit_handling and trk_in is not None and acc_win:
        g = np.mean([a for _, a in acc_win], axis=0)
        tilt = np.degrees(np.arccos(np.clip(g[2] / np.linalg.norm(g), -1, 1)))
        if tilt > 25:
            handling_on, low_since = True, None
        elif handling_on:
            if tilt < 12:
                low_since = t if low_since is None else low_since
                if t - low_since > 1e9:  # back on the ground: restart the inertial tracker(s)
                    handling_on = False
                    trk_in = make_tracker(True)
                    trk_aux = [make_tracker(True) for _ in range(n_aux)]
                    pending_offset = True
                    reinit_times.append(t)
                    print(f"  re-initialised stereo+IMU tracker at {(t - frame_t[0]) * 1e-9:.1f} s", flush=True)
            else:
                low_since = None
    e_in = trk_in.track(t, imgs, internals=internals)[0] if trk_in is not None else None
    e_st = trk_st.track(t, imgs)[0] if trk_st is not None else None
    last_frame_t = t
    r_in = row(e_in.world_from_rig.pose if e_in is not None and e_in.world_from_rig is not None else None)
    if not np.isnan(r_in[0]):
        if pending_offset and last_out is not None:  # chain the restarted tracker onto the last output pose
            T_off = last_out @ np.linalg.inv(to_mat(r_in))
            pending_offset = False
        M = T_off @ to_mat(r_in)
        last_out = M
        r_in = [*M[:3, 3], *Rotation.from_matrix(M[:3, :3]).as_quat()]
    r_st = row(e_st.world_from_rig.pose if e_st is not None and e_st.world_from_rig is not None else None)
    frame_t.append(t)
    pin.append(r_in)
    pst.append(r_st)
    # keyframes on stereo-odometry motion (robust; inertial may glitch)
    if not args.no_kf and not np.isnan(r_st[0]):
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
if not args.no_kf and kfs[-1][0] != len(frame_t) - 1:  # always keep the last frame as a keyframe (last_pair: the last event may be IMU)
    kfs.append((len(frame_t) - 1, last_pair[0], last_pair[1]))

np.savez(f"{args.out}/{tag}_fe.npz", t=np.array(frame_t, np.int64), pose_in=np.array(pin), pose_st=np.array(pst),
         imu=np.array(imu_rows), kf_idx=np.array([k[0] for k in kfs]), reinit_t=np.array(reinit_times, np.int64),
         **{f"aux_t{k}": np.array(aux_t[k], np.int64) for k in range(n_aux)},
         **{f"aux_pose{k}": np.array(aux_pose[k]).reshape(-1, 7) for k in range(n_aux)})
if not args.no_kf:
    with open(f"{args.out}/{tag}_kf.pkl", "wb") as f:
        pickle.dump([(k[1], k[2]) for k in kfs], f)
n_nan_in = int(np.isnan(np.array(pin)[:, 0]).sum())
print(f"{len(frame_t)} frames, {len(kfs)} keyframes, inertial lost {n_nan_in}, wall {time.perf_counter() - wall0:.0f}s")
print("FE_DONE")
