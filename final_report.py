"""Closure-error comparison (first/last pose assumed coincident) of the optimisation against baseline.

Columns per bag (horizontal closure m / yaw deg):
  base run1 SLAM   results/all    (original baseline run)
  base run2 SLAM   results/base2  (same config re-run: run-to-run noise)
  base run2 +PGO   results/base2  (loop verification on the same run)
  final SLAM       results/final  (level-cam rig + planar constraint)
  final +PGO       results/final  (planar + loop verification)   <- proposed
Writes results/final_report.md
"""
import glob
import os

import numpy as np

from loop_check import closure_error, traj_first_last

COLS = [("base run1 SLAM", "results/all", "slam"),
        ("base run2 SLAM", "results/base2", "slam"),
        ("base run2 +PGO", "results/base2", "pgo"),
        ("final SLAM", "results/final", "slam"),
        ("final +PGO", "results/final", "pgo"),
        ("final VO", "results/final", "vo")]


def bag_tags(d):
    return {os.path.basename(f)[: -len("_stereo_slam_summary.json")]
            for f in glob.glob(f"{d}/*_stereo_slam_summary.json")}


tags = sorted(set.intersection(*[bag_tags(d) for _, d, _ in COLS]))
E = np.full((len(tags), len(COLS), 3), np.nan)  # xy, yaw, z-range
for i, t in enumerate(tags):
    for j, (_, d, kind) in enumerate(COLS):
        f = f"{d}/{t}_stereo_slam_{kind}.txt"
        if os.path.exists(f):
            tr = np.loadtxt(f)
            _, exy, yaw = closure_error(traj_first_last(tr), np.eye(4))
            E[i, j] = exy, yaw, np.ptp(tr[:, 3])

md = [f"闭合误差（假设首尾严格重合，水平位移 m / 航向 °），{len(tags)} 个 bag", "",
      "| bag | " + " | ".join(c[0] for c in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
for i, t in enumerate(tags):
    md.append(f"| {t.replace('_2026_04_28-', '/')} | " + " | ".join(
        "-" if np.isnan(E[i, j, 0]) else f"{E[i, j, 0]:.2f} / {E[i, j, 1]:.1f}" for j in range(len(COLS))) + " |")

md += ["", "| 统计 | " + " | ".join(c[0] for c in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
for name, fn in [("中位数 m", np.nanmedian), ("均值 m", np.nanmean), ("最大 m", np.nanmax)]:
    md.append(f"| {name} | " + " | ".join(f"{fn(E[:, j, 0]):.2f}" for j in range(len(COLS))) + " |")
md.append("| >0.5 m 的 bag 数 | " + " | ".join(f"{int(np.sum(E[:, j, 0] > 0.5))}" for j in range(len(COLS))) + " |")
md.append("| >1.0 m 的 bag 数 | " + " | ".join(f"{int(np.sum(E[:, j, 0] > 1.0))}" for j in range(len(COLS))) + " |")
md.append("| 航向中位数 ° | " + " | ".join(f"{np.nanmedian(E[:, j, 1]):.1f}" for j in range(len(COLS))) + " |")
md.append("| 高度范围中位数 m | " + " | ".join(f"{np.nanmedian(E[:, j, 2]):.2f}" for j in range(len(COLS))) + " |")


def paired(a, b, label):
    d = E[:, b, 0] - E[:, a, 0]
    better, worse = int(np.sum(d < -0.05)), int(np.sum(d > 0.05))
    return (f"| {label} | {np.nanmean(d):+.2f} | {np.nanmedian(d):+.2f} | {better} | {worse} | "
            f"{len(d) - better - worse} |")


md += ["", "配对差值（后者 − 前者，负数表示变好；±5 cm 以内记为持平）", "",
       "| 对比 | 均值差 m | 中位数差 m | 变好 | 变差 | 持平 |", "|---|---|---|---|---|---|",
       paired(0, 1, "基线 run1 → 基线 run2（运行间随机波动）"),
       paired(1, 2, "基线 run2 → 同一次运行 +PGO"),
       paired(3, 4, "final → 同一次运行 +PGO"),
       paired(1, 3, "基线 run2 → final（平面约束，不同运行）"),
       paired(0, 4, "基线 run1 → final +PGO（最终方案）"),
       paired(1, 4, "基线 run2 → final +PGO（最终方案）")]
text = "\n".join(md)
open("results/final_report.md", "w").write(text + "\n")
print(text)
