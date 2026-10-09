"""Jump statistics of trajectories (Basalt format files or TUM with header).

Physical jumps: a step whose speed > 1.5 m/s or turn rate > 150 deg/s (the mower never exceeds 1.1 m/s, 86 deg/s).
Step anomaly (needs a local reference trajectory at the same timestamps, e.g. the odometry): |dp - dp_ref| per step.
Usage: python traj_jumps.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.argv = sys.argv[:1]
import backend as B  # noqa: E402

D = Path("/mnt/f/data_set/slam")
FE = Path("results/fe3")


def phys(a):
    dt = np.diff(a[:, 0])
    ok = dt > 1e-4
    v = np.linalg.norm(np.diff(a[:, 1:4], axis=0), axis=1)[ok] / dt[ok]
    R = Rotation.from_quat(a[:, 4:8])
    w = np.degrees((R[:-1].inv() * R[1:]).magnitude())[ok] / dt[ok]
    return int(((v > 1.5) | (w > 150)).sum()), float(v.max()), float(w.max())


rows = []
for npz in sorted(FE.glob("*_fe.npz")):
    tag = npz.name[:-7]
    date, bag = tag.split("_", 1)
    name = bag.removeprefix("basalt_")
    bas = next(p for p in (D / date / f"vio_basalt_{name}.txt", D / date / bag / f"vio_basalt_{name}.txt") if p.exists())
    slam = np.loadtxt(bas.parent / f"slam_cuvslam_{name}.txt")
    vio = np.loadtxt(FE / f"{tag}_vio.txt")
    b = np.loadtxt(bas)
    z = np.load(npz)
    P = B.vote_odometry(z)
    # step anomaly of the SLAM export vs the odometry step expressed in the same (SLAM) frame
    Ms = np.array([B.mat(np.r_[r[1:4], r[4:8]]) for r in slam])
    an = []
    for i in range(len(Ms) - 1):
        step_ref = np.linalg.inv(P[i]) @ P[i + 1]
        step = np.linalg.inv(Ms[i]) @ Ms[i + 1]
        E = np.linalg.inv(step_ref) @ step
        an.append((np.linalg.norm(E[:3, 3]), np.degrees(Rotation.from_matrix(E[:3, :3]).magnitude())))
    an = np.array(an)
    rows.append(dict(tag=tag, n=len(slam), b=phys(b), s=phys(slam), v=phys(vio),
                     a2=int((an[:, 0] > 0.02).sum()), a5=int((an[:, 0] > 0.05).sum()), a10=int((an[:, 0] > 0.10).sum()),
                     amax=float(an[:, 0].max()), ymax=float(an[:, 1].max()), y1=int((an[:, 1] > 1).sum())))

def tot(key, idx):
    return sum(r[key][idx] for r in rows)
print(f"{'':28s} {'phys jumps':>10s} {'bags with':>9s} {'max v m/s':>10s} {'max w deg/s':>11s}")
for k, name in (("b", "Basalt VIO"), ("v", "cuSLAM front end (VIO)"), ("s", "cuSLAM SLAM export")):
    print(f"{name:28s} {tot(k, 0):10d} {sum(r[k][0] > 0 for r in rows):9d} {max(r[k][1] for r in rows):10.2f} {max(r[k][2] for r in rows):11.1f}")
n = sum(r["n"] for r in rows)
print(f"\nSLAM export step anomaly vs odometry ({n} steps): >2 cm {sum(r['a2'] for r in rows)}, >5 cm {sum(r['a5'] for r in rows)}, "
      f">10 cm {sum(r['a10'] for r in rows)}, max {max(r['amax'] for r in rows):.3f} m; yaw >1 deg {sum(r['y1'] for r in rows)}, max {max(r['ymax'] for r in rows):.2f} deg")
for r in sorted(rows, key=lambda r: -r["amax"])[:5]:
    print(f"  {r['tag']:38s} max step anomaly {r['amax']:.3f} m, {r['a5']} steps >5 cm, yaw max {r['ymax']:.2f} deg")
