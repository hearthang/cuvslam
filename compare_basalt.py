"""Closure-error comparison of cuVSLAM vs Basalt VIO on the F:/data_set/slam bags (output of run_f.sh).

Metric (same as basalt_closure_ratios_*.csv): the run starts and ends at the same spot, so
closure error = |p_last - p_first| (3D, imu_link position), ratio = error / 3D path length.
An odometry trajectory with any frame-to-frame speed > 2 m/s (mower drives ~0.3 m/s) is flagged as
diverged/glitched; SLAM trajectories are not flagged since loop-closure corrections jump by design.

Usage: python compare_basalt.py [RESULTS=results/f_cmp] [DATA=/mnt/f/data_set/slam]
"""
import csv
import sys
from pathlib import Path

import numpy as np

res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/f_cmp")
data = Path(sys.argv[2] if len(sys.argv) > 2 else "/mnt/f/data_set/slam")
METHODS = {  # key: (label, file suffix after "{date}_{bag}_")
    "basalt": ("Basalt VIO (stereo+IMU)", None),
    "A_vo": ("cuVSLAM VIO (stereo+IMU)", "inertial_slam_A_vo.txt"),
    "A_slam": ("cuVSLAM SLAM (stereo+IMU+loop closure)", "inertial_slam_A_slam.txt"),
    "B_vo": ("cuVSLAM VIO, level-cam rig", "inertial_slam_B_vo.txt"),
    "B_slam": ("cuVSLAM SLAM, level-cam rig", "inertial_slam_B_slam.txt"),
    "C_vo": ("cuVSLAM stereo VO (no IMU)", "stereo_slam_C_vo.txt"),
    "C_slam": ("cuVSLAM stereo SLAM (no IMU)", "stereo_slam_C_slam.txt"),
}
MAX_SPEED = 2.0


def closure(path):
    a = np.loadtxt(path, ndmin=2)
    p = a[:, 1:4]
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    speed = step / np.maximum(np.diff(a[:, 0]), 1e-3)
    err = np.linalg.norm(p[-1] - p[0])
    return {"err": err, "len": step.sum(), "pct": 100 * err / step.sum(), "jumps": int((speed > MAX_SPEED).sum())}


def basalt_traj(date, bag):
    name = bag.removeprefix("basalt_")  # 9-21 vio files drop the doubled "basalt_" prefix
    for c in (data / date / f"vio_basalt_{name}.txt", data / date / bag / f"vio_basalt_{name}.txt"):
        if c.exists():
            return c
    return None


rows = []
for meta in sorted(data.glob("*/*/metadata.yaml"), key=lambda m: (m.parent.parent.name.split("-")[0].zfill(2),
                                                                  m.parent.parent.name.split("-")[1].zfill(2), m.parent.name)):
    date, bag = meta.parent.parent.name, meta.parent.name
    r = {"date": date, "bag": bag}
    for k, (_, suffix) in METHODS.items():
        f = basalt_traj(date, bag) if suffix is None else res / f"{date}_{bag}_{suffix}"
        if f and f.exists():
            r[k] = closure(f)
    rows.append(r)
present = [k for k in METHODS if any(k in r for r in rows)]

with open(res / "closure_vs_basalt.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "bag"] + [f"{k}_{m}" for k in present for m in ("err_m", "len_m", "pct", "jumps")])
    for r in rows:
        w.writerow([r["date"], r["bag"]] + [("" if k not in r else f"{r[k][m]:.4f}" if m != "jumps" else r[k][m])
                                            for k in present for m in ("err", "len", "pct", "jumps")])


def cell(r, k):
    if k not in r:
        return "—"
    d = r[k]
    return f"{d['err']:.3f} ({d['pct']:.2f}%)" + (" ⚠" if d["jumps"] and not k.endswith("slam") else "")


main = [k for k in ("basalt", "A_vo", "A_slam") if k in present]
lines = [f"Per-bag closure error, metres (% of path). ⚠ = odometry has jumps > {MAX_SPEED} m/s (diverged/glitched).", "",
         "| date | bag | " + " | ".join(METHODS[k][0] for k in present) + " |", "|---|---|" + "---|" * len(present)]
for r in rows:
    lines.append(f"| {r['date']} | {r['bag']} | " + " | ".join(cell(r, k) for k in present) + " |")

full = [r for r in rows if all(k in r for k in present)]
lines += ["", f"## Summary over {len(full)} bags that have every result", "",
          "| method | mean err m | median err m | mean % | median % | max err m | bags with jumps | wins vs Basalt |",
          "|---|---|---|---|---|---|---|---|"]
for k in present:
    e = np.array([r[k]["err"] for r in full])
    p = np.array([r[k]["pct"] for r in full])
    wins = "" if k == "basalt" else f"{sum(r[k]['err'] < r['basalt']['err'] for r in full)}/{len(full)}"
    lines.append(f"| {METHODS[k][0]} | {e.mean():.3f} | {np.median(e):.3f} | {p.mean():.3f} | {np.median(p):.3f} | "
                 f"{e.max():.3f} | {'n/a' if k.endswith('slam') else sum(r[k]['jumps'] > 0 for r in full)} | {wins} |")
if full and "A_vo" in present:
    lines += ["", "Path length ratio cuVSLAM VIO / Basalt, median (scale sanity): "
              f"{np.median([r['A_vo']['len'] / r['basalt']['len'] for r in full]):.3f}"]
(res / "closure_vs_basalt.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
