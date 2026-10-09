"""Collect *_summary.json from run_bag.py into a closure-error table (CSV + markdown)."""
import glob
import json
import sys

import numpy as np

from loop_check import closure_error, traj_first_last

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = args[0] if args else "results/all"
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/*_summary.json"))]


def summarize_assume_closed():
    """Treat every run's first and last pose as exactly coincident (no image-based offset measurement):
    closure error = first->last relative pose of each trajectory itself."""
    def path_len(tr):
        return float(np.linalg.norm(np.diff(tr[:, 1:4], axis=0), axis=1).sum())

    table = []
    for f in sorted(glob.glob(f"{out}/*_summary.json")):
        prefix = f[: -len("_summary.json")]
        vo, slam = np.loadtxt(f"{prefix}_vo.txt"), np.loadtxt(f"{prefix}_slam.txt")
        wheel = np.loadtxt(f"{prefix}_wheel.txt")
        wi = np.clip(np.searchsorted(wheel[:, 0], [vo[0, 0], vo[-1, 0]]), 0, len(wheel) - 1)
        row = {"bag": name(json.load(open(f)))}
        for key, tr in [("wheel", wheel[wi[0]:wi[1] + 1]), ("vo", vo), ("slam", slam)]:
            e3, exy, yaw = closure_error(traj_first_last(tr), np.eye(4))
            row[key] = {"closure_xy_m": exy, "closure_3d_m": e3, "yaw_deg": yaw, "path_m": path_len(tr),
                        "pct": 100 * exy / path_len(tr)}
        table.append(row)

    keys = ["wheel", "vo", "slam"]
    with open(f"{out}/closure_summary_assume_closed.csv", "w") as fcsv:
        fcsv.write("bag," + ",".join(f"{k}_{m}" for k in keys
                                     for m in ["closure_xy_m", "closure_3d_m", "yaw_deg", "path_m", "pct"]) + "\n")
        for r in table:
            fcsv.write(r["bag"] + "," + ",".join(f"{r[k][m]:.4f}" for k in keys
                                                 for m in ["closure_xy_m", "closure_3d_m", "yaw_deg", "path_m", "pct"]) + "\n")

    md = ["假设每个 bag 首尾位姿严格重合：闭合误差 = 轨迹自身首→末相对位姿（水平平移 / 航向）。", "",
          "| bag | 轮速计 m (%) | 轮速计 航向° | VO m (%) | VO 航向° | SLAM m (%) | SLAM 航向° | 路径 VO/轮速计 m |",
          "|---|---|---|---|---|---|---|---|"]
    for r in table:
        cell = lambda k: f"{r[k]['closure_xy_m']:.2f} ({r[k]['pct']:.2f}%) | {r[k]['yaw_deg']:.1f}"
        md.append(f"| {r['bag']} | {cell('wheel')} | {cell('vo')} | {cell('slam')} | "
                  f"{r['vo']['path_m']:.0f} / {r['wheel']['path_m']:.0f} |")
    md += ["", f"全部 {len(table)} 个 bag：", "",
           "| 方法 | 闭合误差 中位数 m | 均值 m | 最大 m | 中位数 % 路径 | 均值 % 路径 | 航向误差 中位数 ° | 航向误差 最大 ° | 3D 闭合 中位数 m |",
           "|---|---|---|---|---|---|---|---|---|"]
    for label, k in [("轮速计", "wheel"), ("cuVSLAM 双目 VO", "vo"), ("cuVSLAM 双目 SLAM", "slam")]:
        g = lambda m: np.array([r[k][m] for r in table])
        a, p, y = g("closure_xy_m"), g("pct"), g("yaw_deg")
        md.append(f"| {label} | {np.median(a):.2f} | {a.mean():.2f} | {a.max():.2f} | {np.median(p):.2f} | "
                  f"{p.mean():.2f} | {np.median(y):.1f} | {y.max():.1f} | {np.median(g('closure_3d_m')):.2f} |")
    better = sum(r["slam"]["closure_xy_m"] < r["vo"]["closure_xy_m"] for r in table)
    md += ["", f"SLAM 闭合误差小于 VO 的 bag：{better}/{len(table)}"]
    text = "\n".join(md)
    open(f"{out}/closure_summary_assume_closed.md", "w").write(text + "\n")
    print(text)



def name(r):
    parts = r["bag"].rstrip("/").split("/")
    return f"{parts[-2]}/{parts[-1]}"


if "--assume-closed" in sys.argv:
    summarize_assume_closed()
    sys.exit()


def fmt(v, p=2):
    return "-" if v is None else f"{v:.{p}f}"


cols = ["bag", "duration_s", "frames", "failed", "track_ms", "same_place", "pnp_inliers", "measured_offset_m",
        "wheel_path_m", "wheel_closure_m", "vo_path_m", "vo_closure_m", "vo_closure_pct", "vo_yaw_deg",
        "slam_closure_m", "slam_closure_pct", "slam_yaw_deg", "wheel_gap_m", "vo_gap_m", "slam_gap_m"]
table = []
for r in rows:
    w, v, s = r["wheel"], r["vo"], r["slam"]
    table.append({
        "bag": name(r), "duration_s": r["duration_s"], "frames": r["frames"], "failed": r["failed"],
        "track_ms": r["track_ms"], "same_place": r["same_place"], "pnp_inliers": r["pnp_inliers"],
        "measured_offset_m": r["measured_offset_m"],
        "wheel_path_m": w["path_m"], "wheel_closure_m": w.get("closure_xy_m"),
        "vo_path_m": v["path_m"], "vo_closure_m": v.get("closure_xy_m"),
        "vo_closure_pct": 100 * v["closure_xy_m"] / v["path_m"] if "closure_xy_m" in v else None,
        "vo_yaw_deg": v.get("closure_yaw_deg"),
        "slam_closure_m": s.get("closure_xy_m"),
        "slam_closure_pct": 100 * s["closure_xy_m"] / s["path_m"] if "closure_xy_m" in s else None,
        "slam_yaw_deg": s.get("closure_yaw_deg"),
        "wheel_gap_m": w["gap_m"], "vo_gap_m": v["gap_m"], "slam_gap_m": s["gap_m"],
    })

with open(f"{out}/closure_summary.csv", "w") as f:
    f.write(",".join(cols) + "\n")
    for t in table:
        f.write(",".join("" if t[c] is None else (f"{t[c]:.4f}" if isinstance(t[c], float) else str(t[c])) for c in cols) + "\n")

md = ["| bag | 时长 s | 失败帧 | PnP 内点 | 实测首末偏移 m | 轮速计 闭合 m | VO 闭合 m (%) | SLAM 闭合 m (%) | VO/SLAM 航向误差 ° | 路径 VO/轮速计 m |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for t in table:
    pct = lambda m, p: "-" if m is None else f"{m:.2f} ({p:.2f}%)"
    md.append(f"| {t['bag']} | {t['duration_s']:.0f} | {t['failed']} | {t['pnp_inliers']} | {fmt(t['measured_offset_m'])} | "
              f"{fmt(t['wheel_closure_m'])} | {pct(t['vo_closure_m'], t['vo_closure_pct'])} | "
              f"{pct(t['slam_closure_m'], t['slam_closure_pct'])} | {fmt(t['vo_yaw_deg'], 1)} / {fmt(t['slam_yaw_deg'], 1)} | "
              f"{t['vo_path_m']:.0f} / {t['wheel_path_m']:.0f} |")
ok = [t for t in table if t["same_place"]]
if ok:
    stat = lambda k: np.array([t[k] for t in ok])
    md += ["", f"首末帧确认回到起点的 bag：{len(ok)}/{len(table)}（仅这些参与统计）", "",
           "| 方法 | 闭合误差 中位数 m | 均值 m | 最大 m | 中位数 % 路径 | 航向误差中位数 ° |", "|---|---|---|---|---|---|"]
    for label, k, yk in [("轮速计", "wheel_closure_m", None), ("cuVSLAM 双目 VO", "vo_closure_m", "vo_yaw_deg"),
                         ("cuVSLAM 双目 SLAM", "slam_closure_m", "slam_yaw_deg")]:
        a = stat(k)
        pk = {"wheel_closure_m": None, "vo_closure_m": "vo_closure_pct", "slam_closure_m": "slam_closure_pct"}[k]
        pct = np.median(stat(pk)) if pk else np.median(100 * a / stat("wheel_path_m"))
        md.append(f"| {label} | {np.median(a):.2f} | {a.mean():.2f} | {a.max():.2f} | {pct:.2f} | "
                  f"{'-' if yk is None else f'{np.median(stat(yk)):.1f}'} |")
bad = [(t, r) for t, r in zip(table, rows) if not t["same_place"]]
if bad:
    md += ["", "以下 bag 首末帧本质矩阵内点充足（同一地点），但近处可三角化的点太少，PnP 测不出首末真实偏移，"
           "只能给原始首尾距离（含真实停车偏移，偏大）：", "",
           "| bag | 本质矩阵内点 | PnP 内点 | 轮速计 首尾距离 m | VO 首尾距离 m | SLAM 首尾距离 m |", "|---|---|---|---|---|---|"]
    for t, r in bad:
        md.append(f"| {t['bag']} | {r['fund_inliers']} | {r['pnp_inliers']} | {t['wheel_gap_m']:.2f} | "
                  f"{t['vo_gap_m']:.2f} | {t['slam_gap_m']:.2f} |")
text = "\n".join(md)
open(f"{out}/closure_summary.md", "w").write(text + "\n")
print(text)
