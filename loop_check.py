"""Measure the true first->last relative pose of a run from images alone.

The robot only returns to the start *region*, so end-start distance of a trajectory mixes real
parking offset with drift. Here the first stereo pair is triangulated and PnP'd against the last
left image, giving the metric rig pose of the last frame relative to the first. Closure error of a
trajectory is then the difference between its own first->last relative pose and this measurement.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def _pose(T):
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat([T["qx"], T["qy"], T["qz"], T["qw"]]).as_matrix()
    M[:3, 3] = [T["px"], T["py"], T["pz"]]
    return M


def _kd(intr):
    k = intr["intrinsics"]
    K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
    return K, np.array([k["k1"], k["k2"], k["k3"], k["k4"]])


def _features(orb, img, mask_bottom):
    mask = np.full(img.shape, 255, np.uint8)
    mask[img.shape[0] - mask_bottom:] = 0
    kp, des = orb.detectAndCompute(img, mask)
    return np.float32([k.pt for k in kp]), des


def _match(d1, d2, ratio=0.8):
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    return [(m.queryIdx, m.trainIdx) for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]


def _norm(pts, K, D):
    return cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2), K, D).reshape(-1, 2)


def measure_first_last(calib, first_pair, last_left, mask_bottom=90):
    """Return dict with loop-check stats and T_first_last (4x4, rig frame) or None if not the same place."""
    T_ic0, T_ic1 = _pose(calib["T_imu_cam"][0]), _pose(calib["T_imu_cam"][1])
    T_c1c0 = np.linalg.inv(T_ic1) @ T_ic0
    (K0, D0), (K1, D1) = _kd(calib["intrinsics"][0]), _kd(calib["intrinsics"][1])
    f = K0[0, 0]

    orb = cv2.ORB_create(4000)
    pL, dL = _features(orb, first_pair[0], mask_bottom)
    pR, dR = _features(orb, first_pair[1], mask_bottom)
    pE, dE = _features(orb, last_left, mask_bottom)
    out = {"fund_inliers": 0, "pnp_inliers": 0, "same_place": False, "T_first_last": None}
    if dL is None or dR is None or dE is None:
        return out

    # same-place check, like slam_dataset_review.csv: geometrically consistent first/last matches
    m_fe = _match(dL, dE)
    if len(m_fe) >= 8:
        a, b = _norm(pL[[i for i, _ in m_fe]], K0, D0), _norm(pE[[j for _, j in m_fe]], K0, D0)
        _, inl = cv2.findEssentialMat(a, b, np.eye(3), cv2.RANSAC, 0.999, 1.5 / f)
        out["fund_inliers"] = int(inl.sum()) if inl is not None else 0

    # triangulate first stereo pair
    m_lr = _match(dL, dR)
    if len(m_lr) < 20:
        return out
    iL = np.array([i for i, _ in m_lr])
    a, b = _norm(pL[iL], K0, D0), _norm(pR[[j for _, j in m_lr]], K1, D1)
    X = cv2.triangulatePoints(np.eye(3, 4), T_c1c0[:3], a.T, b.T)
    X = (X[:3] / X[3]).T
    X1 = X @ T_c1c0[:3, :3].T + T_c1c0[:3, 3]
    ok = (X[:, 2] > 0.3) & (X[:, 2] < 15) & (X1[:, 2] > 0.3)
    ok &= np.linalg.norm(X[:, :2] / X[:, 2:] - a, axis=1) < 1.5 / f
    ok &= np.linalg.norm(X1[:, :2] / X1[:, 2:] - b, axis=1) < 1.5 / f
    pts3d = {int(i): x for i, x, o in zip(iL, X, ok) if o}

    # PnP of the last left image against first-frame 3D points
    corr = [(pts3d[i], j) for i, j in m_fe if i in pts3d]
    if len(corr) < 15:
        return out
    obj = np.float64([c[0] for c in corr])
    img = _norm(pE[[c[1] for c in corr]], K0, D0).astype(np.float64)
    okp, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, np.eye(3), None, reprojectionError=2.0 / f,
                                              iterationsCount=2000, flags=cv2.SOLVEPNP_EPNP)
    if not okp or inl is None or len(inl) < 15:
        return out
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl[:, 0]], img[inl[:, 0]], np.eye(3), None, rvec, tvec)
    T_c0last_c0first = np.eye(4)
    T_c0last_c0first[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_c0last_c0first[:3, 3] = tvec.ravel()
    T_c_first_last = np.linalg.inv(T_c0last_c0first)
    out["pnp_inliers"] = int(len(inl))
    out["same_place"] = out["pnp_inliers"] >= 30
    out["T_first_last"] = T_ic0 @ T_c_first_last @ np.linalg.inv(T_ic0)  # rig (imu) frame
    return out


def traj_first_last(tr):
    """First->last relative rig pose of a TUM-format trajectory array."""
    def T(row):
        M = np.eye(4)
        M[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
        M[:3, 3] = row[1:4]
        return M
    return np.linalg.inv(T(tr[0])) @ T(tr[-1])


def closure_error(T_est, T_meas):
    """Translation (m, 3D and horizontal) and yaw (deg) of T_meas^-1 * T_est."""
    E = np.linalg.inv(T_meas) @ T_est
    return (float(np.linalg.norm(E[:3, 3])), float(np.linalg.norm(E[:2, 3])),
            float(abs(np.degrees(Rotation.from_matrix(E[:3, :3]).as_euler("zyx")[0]))))
