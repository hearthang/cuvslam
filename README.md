# nv_slam：离线轨迹参考生成

输入割草机双目相机、IMU 的 ROS bag，输出 IMU 坐标系下、重力对齐世界坐标中的完整轨迹。
核心用途是离线分析和算法对比。轨迹由传感器约束估计，不是测量真值。

## 使用现有 47 段数据

服务器已有运行环境，无须重新安装：

```bash
cd /home/cj/code/nv_slam
~/anaconda3/envs/cuvslam/bin/python offline_reference.py \
  --fe-dir results/fe5 \
  --calib /mnt/f/data_set/slam/calibration_front_small.json \
  --reuse-backend --backend-suffix _v11 \
  --output results/reference_v1 --jobs 6 --resume
```

`--reuse-backend` 明确导入现有关键帧图及回环测量；程序记录这些输入的 SHA-256。
去掉该选项会运行当前 `backend.py`，生成新的关键帧图。
新配置或新代码使用新的 `--output` 目录。`--resume` 只复用参数、代码、输入和产物校验均一致的已完成阶段。
`--only TAG` 可以只生成一个片段；TAG 是前端缓存文件去掉 `_fe.npz` 的名字，也支持子串。

## 从新 bag 开始

```bash
~/anaconda3/envs/cuvslam/bin/python offline_reference.py \
  --bag /path/to/date/recording \
  --frontend-python /home/cj/anaconda3/envs/cuvslam-dev/bin/python \
  --calib /path/to/calibration.json \
  --fe-dir results/new_capture --output results/new_reference --jobs 4
```

可以重复指定 `--bag`。前端需要 CUDA 和本仓库修改过的 cuVSLAM；后端、补充回环、全局优化使用 CPU。
当前前端是针对本设备的适配：相机、IMU topic、KB4 标定结构、图像遮罩和右相机焦距微调见 `run_frontend.py`。
换设备时必须检查这些配置，不能直接套用本设备的标定与运动假设。

## 输出

```text
results/reference_v1/
  trajectories/<tag>.txt   # timestamp tx ty tz qx qy qz qw
  quality/<tag>.json        # 收敛信息、参数、回环数量、搬动时段
  manifest.json            # 运行状态、命令、源码/标定/输入/输出校验值
  calibration.json
  source/                  # 本次生成使用的源码快照
  logs/                    # 每个阶段的完整日志
```

时间戳为秒、IMU 时钟；位置为米；四元数顺序为 `x y z w`。保持前端的每个时间戳，通常约 16 Hz。
质量文件的 `error_init/error_final` 对应六自由度图优化阶段，`position_iterations` 记录后续位置求解的迭代变化；最终输出的误差和平滑性以独立评测为准。
生成过程不读取 Basalt 文件，不使用首尾强制相等约束，也不按 Basalt 跳变幅度限幅。
首次运行、后端输入不匹配、缺失回环文件或无效轨迹都会报错，避免静默生成不完整结果。

## 实现

1. **前端**：修订版 cuVSLAM 双目+IMU；逐帧使用视觉 PnP，IMU 仍参与关键帧优化。
2. **基础回环图**：`backend.py` 提供 ORB/KLT 双目几何及关键帧回环。特征缓存现在按标定内容及焦距修正区分。
3. **补充回环**：`augment_loops.py` 对录制开头和结尾的图像进行更充分的离线检索，使用 RootSIFT、互相最近邻匹配、双向立体 PnP 和闭环一致性验证。约束保留测得的相对位姿；没有匹配时不会强加闭合。
4. **全帧优化**：`reference.py` 使用 GTSAM 同时优化每帧六自由度位姿，融合视觉增量、陀螺旋转、重力方向及回环。降低过强的逐帧视觉置信度，通过逐步收紧的 Cauchy 核保留合理回环。
5. **平移连续性**：`translation_refine.py` 固定已优化的旋转，联合求解全部位置与测得的相对位移、回环位移及 Huber 加速度正则。只固定首帧作为坐标原点；末帧自由，不通过首尾拉直或逐步截断改善指标。

推荐使用 `offline_reference.py`。`refine.py`、`eval_ref.py` 保留原实验代码，旧导出脚本和 v10 报告也保留，方便对照。
`reference.py` 中完整 IMU 预积分和直接六自由度加速度因子属于可选实验路径；默认生成流程使用已经全量评测的陀螺约束和稀疏位置优化。

## 评测

```bash
~/anaconda3/envs/cuvslam/bin/python evaluate_reference.py \
  --source v10=slam --source previous=_b \
  --source reference='results/reference_v1/trajectories/{tag}.txt' \
  --output results/reference_v1/evaluation.json

~/anaconda3/envs/cuvslam/bin/python -m unittest -v test_reference
```

评测程序在每段数据的共同时间范围、相同 Basalt 时间戳上比较：

- 首尾距离的均值、中位数、P90、最大值。
- 平移加速度和 jerk 的 P95，以及路径长度。
- 完整 SO(3) 旋转差与原始陀螺积分的一致性，包含转轴和方向。
- 相对 Basalt 的逐步位移/旋转异常数，作为辅助比较。

闭合误差仅适用于实际回到同一位置的数据；开放轨迹的首尾距离不是定位误差。
Basalt 是比较对象，不是真值。导出质量文件标记人工搬动时段；目前仍沿用“搬动后放回原位”的数据集假设，这些时段不宜作为独立位置真值。
参数在现有数据上开发；未完成新场地或新设备的外部验证。详细实测结果见 `results/reference_report.md`。

## 环境

已验证的 CPU 环境为 Python 3.12、GTSAM 4.3、NumPy 2.2.6、SciPy 1.15.3、OpenCV 4.11。
`requirements-reference.txt` 记录本机版本；cuVSLAM 前端仍使用独立的 `cuvslam-dev` 环境和修订分支。
