## original-使用随机点生成数据的方式
1.生成两组随机点，分别作为P1_global点和P2_global点
2.将local点进行旋转和平移，作为本地坐标系下的P2_local点
3.计算P1_global点和P2_global点的方向向量，并将方向向量通过同样的旋转进入P2坐标系中
4.将方向向量转换成角度，得到俯仰角和方位角，在这两个角度上增加高斯噪声
5.将得到的角度转换为方向向量，结果作为观测数据

基于miluv数据集合成bearing数据的流程
1. 在一个时间窗口内选取第一个时刻的姿态四元数作为B2坐标系，在此时刻之后的姿态可以利用mocap提供的姿态真值直接作为已知的B2到B4的姿态变换
2. 将目标无人机的轨迹转换到B4坐标系下面，然后在此坐标系下添加角度噪声，再转换到B2坐标系下作为直接观测
3. 虽然B2到B4是已知的，但是相比于将角度噪声直接加到B2坐标系的bearing观测来说，由B4转换到B2更符合真是情况，噪声经过非线性变换后已经不符合高斯假设


# # `miluv_to_bearing.py` 观测量生成调用流程
---

## 2. 整体流程

`miluv_to_bearing.py` 的整体职责可以分成四层：

1. 读取三机 `mocap.csv`
2. 在公共时间轴上同步三机状态
3. 对每条有向边合成 bearing 观测
4. 将结果保存为 `.npz`

对应的主流程是：

```text
main()
  -> build_argparser()
  -> synthesize_three_robot_bearing_observations()
       -> sync_three_robot_states()
            -> load_three_robot_mocap()
                 -> load_miluv_mocap_csv()
            -> build_common_timestamps()
                 -> estimate_common_dt()
            -> interpolate_robot_state()   [对每个机器人执行]
            -> load_experiment_timeshift_seconds()
       -> synthesize_directed_bearing_edge()  [对每条有向边执行]
            -> bearing_to_angles()
            -> wrap_to_pi()
            -> angles_to_bearing()
            -> compute_b3_from_b4_rotation_sequence()
            -> rotate_bearing_sequence()
  -> save_three_robot_bearing_dataset()
```

---

## 3. 按层理解调用关系

## 3.1 入口层

### `build_argparser()`

作用：
- 定义命令行参数
- 让脚本可以直接从终端运行

主要输入：
- `--experiment-dir`
- `--output`
- `--dt`
- `--max-samples`
- `--sigma-azimuth-deg`
- `--sigma-elevation-deg`
- `--reference-index`
- `--seed`

主要输出：
- `argparse.Namespace`

### `main()`

作用：
- 解析命令行参数
- 调用主生成函数
- 再把结果保存到磁盘

对应关系：
- `main()` 是命令行入口
- 真正的数据生成工作由 `synthesize_three_robot_bearing_observations()` 完成

---

## 3.2 数据读取层

### `load_miluv_mocap_csv(experiment_dir, robot_id)`

作用：
- 读取某一台机器人的 `mocap.csv`
- 检查时间戳、位置、四元数列是否完整
- 按时间排序并去重

输入：
- 实验目录
- 机器人编号，例如 `ifo001`

输出：
- 单台机器人的 `pandas.DataFrame`

### `load_three_robot_mocap(experiment_dir, robot_ids)`

作用：
- 对三台机器人分别调用 `load_miluv_mocap_csv`
- 把三机 mocap 数据整理成字典

输入：
- 实验目录
- 三机编号列表

输出：
- `dict[str, DataFrame]`

### `load_experiment_timeshift_seconds(experiment_dir)`

作用：
- 读取 `timeshift.yaml`
- 把时间偏移作为元数据保留到输出里

说明：
- 当前实现不会再次平移 `mocap.csv` 时间戳
- 这里只是记录实验原始 `timeshift`

---

## 3.3 时间同步层

### `estimate_common_dt(mocap_frames)`

作用：
- 根据三机原始时间戳估计一个公共采样间隔

输入：
- 三机 mocap 表

输出：
- `dt: float`

### `build_common_timestamps(mocap_frames, dt, max_samples)`

作用：
- 找到三机时间重叠区间
- 构造统一采样时间轴

输入：
- 三机 mocap 表
- 采样步长 `dt`
- 最大样本数 `max_samples`

输出：
- `timestamps: np.ndarray`

### `interpolate_robot_state(mocap_frame, query_timestamps)`

作用：
- 把单台机器人的离散 mocap 轨迹插值到公共时间轴上

输入：
- 单机 mocap 表
- 公共时间轴

输出：
- 插值后的位置 `(3, N)`
- 插值后四元数 `(4, N)`
- 插值后旋转矩阵 `(N, 3, 3)`

### `sync_three_robot_states(experiment_dir, robot_ids, dt, max_samples)`

作用：
- 封装完整的三机时间同步流程
- 生成后续合成观测所需的统一状态数据

输入：
- 实验目录
- 机器人编号
- `dt`
- `max_samples`

输出：
- 一个同步状态字典，包含：
  - `timestamps`
  - 每台机器人在世界系下的位置
  - 每台机器人的四元数
  - 每台机器人的 `rotation_wb`
  - `timeshift_s`

它在整个流程中的位置可以理解为：
- 上游接原始 mocap
- 下游接观测合成

---

## 3.4 观测生成层

### `bearing_to_angles(bearing_b4)`

作用：
- 把单位方向向量转为方位角/俯仰角

### `angles_to_bearing(angles)`

作用：
- 把方位角/俯仰角再转回单位方向向量

### `wrap_to_pi(angles)`

作用：
- 对方位角做包裹，保持在标准区间内

这三个函数的关系是：
- 先由无噪声 bearing 得到角度
- 在角度域加噪声
- 再由角度恢复带噪 bearing

### `compute_b3_from_b4_rotation_sequence(observer_rotation_wb, reference_index)`

作用：
- 计算从每个时刻 `B4(k)` 到冻结参考系 `B3(k0)` 的相对旋转序列

用途：
- 支持 Russell 风格的窗口冻结参考系观测建模

### `rotate_bearing_sequence(bearing_vectors, rotation_target_source)`

作用：
- 用一组旋转矩阵把一串 bearing 向量从源坐标系旋转到目标坐标系

用途：
- 将 `B4` 下的 bearing 转为 `B3` 下的最终观测量

### `synthesize_directed_bearing_edge(...)`

作用：
- 对单条有向边 `observer <- target`，在整条同步时间轴上生成 bearing 观测

内部流程：
1. 在世界系计算相对位移
2. 转到观测者 `B4`
3. 得到无噪声 bearing
4. 转为角度并加噪声
5. 再恢复为带噪 bearing
6. 计算 `B4 -> B3` 旋转
7. 把 `B4` bearing 转到冻结 `B3`

输入：
- 观测者位置
- 观测者姿态
- 目标位置
- 噪声参数
- `reference_index`

输出：
- 单条边的观测字典，例如：
  - `bearing_b4`
  - `angles_b4`
  - `bearing_b3`
  - `angles_b3`
  - `bearing`
  - `angles`
  - `rotation_b3_b4`

### `synthesize_three_robot_bearing_observations(...)`

作用：
- 这是 Python 侧最主要的观测生成函数
- 它先调用 `sync_three_robot_states(...)`
- 再对每条有向边调用 `synthesize_directed_bearing_edge(...)`
- 最终把三机状态和四条边观测统一打包

输入：
- 实验目录
- 锚点编号
- 机器人列表
- 有向边列表
- 时间同步参数
- 噪声参数

输出：
- 顶层数据集字典：
  - `timestamps`
  - `states`
  - `observations`
  - `anchor_id`
  - `reference_index`
  - 其他元数据

这是整个模块最适合在代码中直接调用的主函数。

---

## 3.5 保存层

### `save_three_robot_bearing_dataset(dataset, output_path)`

作用：
- 把前面生成的嵌套字典打平
- 保存为 `.npz`

输入：
- `dataset`
- 输出路径

输出：
- 磁盘上的 `.npz` 文件

它对应的是流程的最后一步，不参与观测计算，只负责持久化。

---

## 4. 生成观测量时最常用的两个入口

## 4.1 命令行入口

如果只想直接生成一个 `.npz` 文件，调用顺序是：

```text
main()
  -> synthesize_three_robot_bearing_observations()
  -> save_three_robot_bearing_dataset()
```

示例：

```powershell
& 'D:\programfiles\anaconda\install\envs\CL_AOA\python.exe' src\miluv_to_bearing.py `
  --experiment-dir D:\3-1cooperativeLocalization\dataset\miluv\data\3robot\default_3_random_0 `
  --output data\miluv_bearing\default_3_random_0_b3_bearing.npz `
  --dt 0.05 `
  --max-samples 50 `
  --reference-index 0 `
  --seed 7
```

## 4.2 Python 调用入口

如果要在别的脚本里继续处理数据，最常用的是：

```python
from pathlib import Path
from src.miluv_to_bearing import synthesize_three_robot_bearing_observations


dataset = synthesize_three_robot_bearing_observations(
    experiment_dir=Path(r"D:\3-1cooperativeLocalization\dataset\miluv\data\3robot\default_3_random_0"),
    anchor_id="ifo001",
    dt=0.05,
    max_samples=50,
    reference_index=0,
    random_seed=7,
)
```

这时调用顺序是：

```text
synthesize_three_robot_bearing_observations()
  -> sync_three_robot_states()
  -> synthesize_directed_bearing_edge()  [四次]
```

---

## 5. 一句话理解整个代码库中的位置

`src/miluv_to_bearing.py` 在整个项目流程中的位置是：

```text
MILUV mocap 真值
  -> 三机时间同步
  -> 三机有向 bearing 合成
  -> B3/B4 观测输出
  -> 后续接 BO-EPNP / GN-BP / 其他实验脚本
```

也就是说，它是一个**实验数据准备模块**，负责把真实轨迹真值变成可供后续协同定位算法直接使用的 bearing 观测数据。




