# GABP Solver Call Outline

  

## 1. 入口函数

- `BgpnpGabpSolver.solve(sample: MultiAgentWindowSample, primary_anchor_id: str, prior_covariance: np.ndarray, sol_iter: bool = True, enforce_manifold: bool = True) -> tuple[dict[str, SolverEstimate], GaBPRefinementResult]`

  - 输入：

    - `sample`: 三机或多机窗口样本，包含锚点轨迹、待估节点局部轨迹、节点间 bearing

    - `primary_anchor_id`: 主锚点 id，例如 `ifo001`

    - `prior_covariance`: GABP 先验协方差，形状 `(6, 6)`

    - `sol_iter`: 是否启用 BGPnP 交替优化

    - `enforce_manifold`: 是否在 BGPnP 中强制流形约束

  - 输出：

    - `initial_estimates`: 每个待估节点的前端初始化结果 `dict[str, SolverEstimate]`

    - `refinement`: GABP 联合优化结果 `GaBPRefinementResult`

  - 调用：

  

### 1.1 待估节点逐个做 BGPnP 初始化

- `bgpnp.solve(p1: np.ndarray, p2: np.ndarray, bearing: np.ndarray, sol_iter: bool = True, enforce_manifold: bool = False) -> tuple[tuple[np.ndarray, np.ndarray, float], float]`

  - 输入：

    - `p1`: 锚点轨迹，形状 `(K, 3)`

    - `p2`: 当前待估节点局部轨迹，形状 `(K, 3)`

    - `bearing`: 当前节点对锚点的 bearing，形状 `(K, 3)`

    - `sol_iter`: 是否执行交替优化

    - `enforce_manifold`: 是否执行流形约束

  - 输出：

    - `rotation`: 世界到局部系旋转 `(3, 3)`

    - `translation`: 世界到局部系平移 `(3,)`

    - `err`: 求解残差或内部误差量

    - `runtime`: 前端运行时间

  - 被 `BgpnpGabpSolver.solve(...)` 调用一次或多次：

    - 对 `sample.estimated_nodes` 中每个待估节点调用一次

  

### 1.2 构造初始状态容器

- `PoseState(rotation: np.ndarray, translation: np.ndarray)`

  - 输入：

    - `rotation`: BGPnP 输出的 `(3, 3)` 旋转

    - `translation`: BGPnP 输出的 `(3,)` 平移

  - 输出：

    - `PoseState`

  - 作用：

    - 作为 `refine_with_gabp(...)` 的初值状态字典元素

  

### 1.3 构造锚点因子列表

- `BgpnpGabpSolver._build_anchor_factors(sample: MultiAgentWindowSample, primary_anchor_id: str) -> list[AnchorBearingFactor]`

  - 输入：

    - `sample`

    - `primary_anchor_id`

  - 输出：

    - `AnchorBearingFactor` 列表，每个关键点一个因子

  - 调用：

  

#### 1.3.1 单个关键点的 bearing 转角度

- `vector_to_az_el(vector: np.ndarray) -> np.ndarray`

  - 输入：

    - `vector`: 单个 bearing 方向向量 `(3,)`

  - 输出：

    - `angles`: `[azimuth, elevation]`，形状 `(2,)`

  - 作用：

    - 把 `bearing_observation` 中的方向向量变成 GABP 因子使用的角度观测

  

#### 1.3.2 构造锚点因子对象

- `AnchorBearingFactor(observer_id: str, anchor_position_world: np.ndarray, measured_angles: np.ndarray, observer_point_local: np.ndarray | None = None, sigma_azimuth: float = DEFAULT_SIGMA_AZIMUTH, sigma_elevation: float = DEFAULT_SIGMA_ELEVATION, observer_extrinsic_rotation: np.ndarray | None = None, observer_extrinsic_translation: np.ndarray | None = None)`

  - 输入：

    - `observer_id`: 当前待估节点 id

    - `anchor_position_world`: 锚点在当前关键点时刻的世界坐标 `(3,)`

    - `measured_angles`: 由 bearing 转换得到的角度观测 `(2,)`

    - `observer_point_local`: 当前待估节点局部轨迹点 `(3,)`

  - 输出：

    - `AnchorBearingFactor`

  

### 1.4 构造待估节点间因子列表

- `BgpnpGabpSolver._build_pairwise_factors(sample: MultiAgentWindowSample) -> list[PairwiseBearingFactor]`

  - 输入：

    - `sample`

  - 输出：

    - `PairwiseBearingFactor` 列表，每个节点间关键点一个因子

  - 调用：

  

#### 1.4.1 单个关键点的 bearing 转角度

- `vector_to_az_el(vector: np.ndarray) -> np.ndarray`

  - 输入：

    - `vector`: 单个 pairwise bearing 方向向量 `(3,)`

  - 输出：

    - `angles`: `[azimuth, elevation]`，形状 `(2,)`

  

#### 1.4.2 构造双变量因子对象

- `PairwiseBearingFactor(observer_id: str, target_id: str, target_point_local: np.ndarray, measured_angles: np.ndarray, observer_point_local: np.ndarray | None = None, sigma_azimuth: float = DEFAULT_SIGMA_AZIMUTH, sigma_elevation: float = DEFAULT_SIGMA_ELEVATION, observer_extrinsic_rotation: np.ndarray | None = None, observer_extrinsic_translation: np.ndarray | None = None)`

  - 输入：

    - `observer_id`: 观测者节点 id

    - `target_id`: 被观测节点 id

    - `target_point_local`: 被观测节点局部轨迹点 `(3,)`

    - `measured_angles`: pairwise bearing 转换后的角度 `(2,)`

    - `observer_point_local`: 观测者局部轨迹点 `(3,)`

  - 输出：

    - `PairwiseBearingFactor`

  

### 1.5 进入 GABP 联合优化主循环

- `refine_with_gabp(initial_states: dict[str, PoseState], anchor_factors: Iterable[AnchorBearingFactor], pairwise_factors: Iterable[PairwiseBearingFactor], prior_states: dict[str, PoseState] | None = None, prior_covariance: np.ndarray | None = None, settings: GaBPSettings | None = None) -> GaBPRefinementResult`

  - 输入：

    - `initial_states`: 所有待估节点的初始位姿

    - `anchor_factors`: 锚点因子列表

    - `pairwise_factors`: 节点间因子列表

    - `prior_states`: 先验位姿，当前实现默认就是初始位姿

    - `prior_covariance`: `(6, 6)` 切空间协方差

    - `settings`: GN/BP 迭代设置

  - 输出：

    - `GaBPRefinementResult(states: dict[str, PoseState], history: list[GaBPHistoryEntry])`

  - 调用：

  

## 2. `refine_with_gabp(...)` 内部调用顺序

- `refine_with_gabp(...)`

  - 调用：

  

### 2.1 初始化与状态合法性检查

- `_validate_pose_state(state: PoseState) -> PoseState`

  - 输入：

    - `state`

  - 输出：

    - 形状校验后的 `PoseState`

  - 调用：

  

#### 2.1.1 旋转矩阵检查

- `_as_rotation(rotation: np.ndarray, name: str) -> np.ndarray`

  - 输入：`rotation`, `name`

  - 输出：`(3, 3)` 旋转矩阵

  

#### 2.1.2 平移向量检查

- `_as_vector3(vector: np.ndarray, name: str) -> np.ndarray`

  - 输入：`vector`, `name`

  - 输出：`(3,)` 向量

  

### 2.2 线性化单变量因子

- `_linearize_unary_factors(factors: list[AnchorBearingFactor], states: dict[str, PoseState]) -> list[UnaryInfoFactor]`

  - 输入：

    - `AnchorBearingFactor` 列表

    - 当前状态字典

  - 输出：

    - `UnaryInfoFactor` 列表

  - 调用：

  

#### 2.2.1 线性化单个锚点因子

- `_linearize_anchor_factor(factor: AnchorBearingFactor, states: dict[str, PoseState]) -> UnaryInfoFactor`

  - 输入：

    - 单个锚点因子

    - 当前状态

  - 输出：

    - 单变量信息因子 `UnaryInfoFactor`

  - 调用：

  

##### 2.2.1.1 当前姿态取数组

- `_pose_arrays(state: PoseState) -> tuple[np.ndarray, np.ndarray]`

  - 输入：`PoseState`

  - 输出：`(rotation(3,3), translation(3,))`

  - 调用：

    - `_as_rotation(...)`

    - `_as_vector3(...)`

  

##### 2.2.1.2 外参默认值处理

- `_extrinsic_rotation(rotation: np.ndarray | None) -> np.ndarray`

  - 输入：可选外参旋转

  - 输出：`(3, 3)`，若为空则单位阵

  - 调用：`_as_rotation(...)`

- `_extrinsic_translation(translation: np.ndarray | None) -> np.ndarray`

  - 输入：可选外参平移

  - 输出：`(3,)`，若为空则零向量

  - 调用：`_as_vector3(...)`

- `_optional_vector3(vector: np.ndarray | None, name: str) -> np.ndarray`

  - 输入：可选局部点

  - 输出：`(3,)`，若为空则零向量

  - 调用：`_as_vector3(...)`

  

##### 2.2.1.3 角度残差与解析雅可比

- `_angle_residual_and_jacobian(s: np.ndarray, measured_angles: np.ndarray, sigma_azimuth: float, sigma_elevation: float) -> tuple[np.ndarray, np.ndarray]`

  - 输入：

    - `s`: 预测 LOS 向量 `(3,)`

    - `measured_angles`: 观测角度 `(2,)`

    - `sigma_azimuth`, `sigma_elevation`: 角度噪声标准差

  - 输出：

    - `residual`: `(2,)`

    - `J_angle`: `(2, 3)`

  - 调用：

  

###### 2.2.1.3.1 向量/角度合法性检查

- `_as_vector3(vector: np.ndarray, name: str) -> np.ndarray`

- `_as_angles(angles: np.ndarray) -> np.ndarray`

  

###### 2.2.1.3.2 预测角度计算

- `vector_to_az_el(vector: np.ndarray) -> np.ndarray`

  - 输入：`(3,)` LOS 向量

  - 输出：`(2,)` 角度 `[azimuth, elevation]`

  - 调用：

    - `_as_vector3(...)`

  

###### 2.2.1.3.3 方位角 wrap

- `wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float`

  - 输入：角度或角度数组

  - 输出：wrap 到 `[-pi, pi)` 的角度

  

##### 2.2.1.4 李代数几何雅可比中的叉乘矩阵

- `skew(vector: np.ndarray) -> np.ndarray`

  - 输入：`(3,)` 向量

  - 输出：`(3, 3)` 反对称矩阵

  - 调用：

    - `_as_vector3(...)`

  

### 2.3 线性化先验因子

- `_build_prior_info_factors(prior_states: dict[str, PoseState], current_states: dict[str, PoseState], covariance: np.ndarray) -> list[UnaryInfoFactor]`

  - 输入：

    - `prior_states`

    - `current_states`

    - `covariance(6,6)`

  - 输出：

    - 先验单变量信息因子列表

  - 调用：

  

#### 2.3.1 协方差检查

- `_as_covariance(covariance: np.ndarray) -> np.ndarray`

  - 输入：`(6, 6)` 协方差

  - 输出：对称化协方差

  - 调用：

    - `_symmetrize(matrix: np.ndarray) -> np.ndarray`

  

#### 2.3.2 姿态误差映射到切空间

- `so3_log(rotation: np.ndarray) -> np.ndarray`

  - 输入：`(3, 3)` 旋转矩阵

  - 输出：`(3,)` 李代数误差

  - 调用：

    - `_as_rotation(...)`

- `_as_vector3(vector: np.ndarray, name: str) -> np.ndarray`

  

### 2.4 线性化双变量因子

- `_linearize_pairwise_factors(factors: list[PairwiseBearingFactor], states: dict[str, PoseState]) -> list[PairwiseInfoFactor]`

  - 输入：

    - 双变量 bearing 因子列表

    - 当前状态

  - 输出：

    - `PairwiseInfoFactor` 列表

  - 调用：

  

#### 2.4.1 线性化单个双变量因子

- `_linearize_pairwise_factor(factor: PairwiseBearingFactor, states: dict[str, PoseState]) -> PairwiseInfoFactor`

  - 输入：

    - 单个双变量因子

    - 当前状态

  - 输出：

    - `PairwiseInfoFactor`

  - 调用：

  

##### 2.4.1.1 当前 observer/target 姿态取数组

- `_pose_arrays(state: PoseState) -> tuple[np.ndarray, np.ndarray]`

  

##### 2.4.1.2 外参与局部点检查

- `_extrinsic_rotation(...) -> np.ndarray`

- `_extrinsic_translation(...) -> np.ndarray`

- `_optional_vector3(...) -> np.ndarray`

- `_as_vector3(...) -> np.ndarray`

  

##### 2.4.1.3 角度残差与解析雅可比

- `_angle_residual_and_jacobian(...) -> tuple[np.ndarray, np.ndarray]`

  - 调用链同 2.2.1.3

  

##### 2.4.1.4 旋转增量雅可比用到的叉乘矩阵

- `skew(vector: np.ndarray) -> np.ndarray`

  

### 2.5 解当前 GN 线性子问题

- `solve_linearized_gabp(variable_ids: Iterable[str], unary_factors: Iterable[UnaryInfoFactor], pairwise_factors: Iterable[PairwiseInfoFactor], settings: GaBPSettings | None = None) -> tuple[dict[str, np.ndarray], int]`

  - 输入：

    - `variable_ids`: 待估节点 id 列表

    - `unary_factors`: 单变量信息因子

    - `pairwise_factors`: 双变量信息因子

    - `settings`: BP 参数

  - 输出：

    - `deltas`: 每个节点的切空间增量 `(6,)`

    - `bp_iteration_count`: 本轮 BP 迭代次数

  - 调用：

  

#### 2.5.1 单变量因子汇总

- `_sum_unary_information(variable_ids: list[str], unary_factors: Iterable[UnaryInfoFactor]) -> dict[str, tuple[np.ndarray, np.ndarray]]`

  - 输入：

    - 变量 id 列表

    - 单变量信息因子

  - 输出：

    - 每个节点的 `(information(6,6), eta(6,))`

  - 调用：

    - `_zero_message() -> tuple[np.ndarray, np.ndarray]`

    - `_as_information(information: np.ndarray, name: str) -> np.ndarray`

      - 调用：`_symmetrize(...)`

    - `_as_state_vector(vector: np.ndarray, name: str) -> np.ndarray`

  

#### 2.5.2 初始化因子到变量消息

- `_zero_message() -> tuple[np.ndarray, np.ndarray]`

  - 输入：无

  - 输出：

    - 零信息矩阵 `(6,6)`

    - 零信息向量 `(6,)`

  

#### 2.5.3 构造变量到因子消息

- `_copy_message(message: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]`

  - 输入：旧消息

  - 输出：消息副本

- `_factor_contains(factor: PairwiseInfoFactor, variable_id: str) -> bool`

  - 输入：双变量因子、变量 id

  - 输出：是否连接

  

#### 2.5.4 构造因子到变量消息

- `_factor_to_first_message(factor: PairwiseInfoFactor, incoming_second: tuple[np.ndarray, np.ndarray], regularization: float) -> tuple[np.ndarray, np.ndarray]`

  - 输入：

    - 当前因子

    - 来自 second 变量的聚合消息

    - 对角正则

  - 输出：发给 first 变量的消息 `(Lambda, eta)`

  - 调用：

    - `_symmetrize(...)`

- `_factor_to_second_message(factor: PairwiseInfoFactor, incoming_first: tuple[np.ndarray, np.ndarray], regularization: float) -> tuple[np.ndarray, np.ndarray]`

  - 输入：

    - 当前因子

    - 来自 first 变量的聚合消息

    - 对角正则

  - 输出：发给 second 变量的消息 `(Lambda, eta)`

  - 调用：

    - `_symmetrize(...)`

  

#### 2.5.5 消息阻尼

- `_damp_message(old: tuple[np.ndarray, np.ndarray], new: tuple[np.ndarray, np.ndarray], damping: float) -> tuple[np.ndarray, np.ndarray]`

  - 输入：

    - `old`: 上一轮消息

    - `new`: 当前新消息

    - `damping`: 阻尼系数

  - 输出：

    - 阻尼后的消息 `(Lambda, eta)`

  - 调用：

    - `_symmetrize(...)`

  

#### 2.5.6 求每个变量的切空间增量

- `_factor_contains(factor: PairwiseInfoFactor, variable_id: str) -> bool`

  - 输入：双变量因子、变量 id

  - 输出：是否连接

  

### 2.6 将切空间增量更新到状态流形上

- `_apply_tangent_update(state: PoseState, delta: np.ndarray) -> None`

  - 输入：

    - `state`: 当前节点位姿

    - `delta`: 切空间增量 `(6,)`

  - 输出：

    - 无，原地更新 `state`

  - 调用：

  

#### 2.6.1 增量合法性检查

- `_as_state_vector(vector: np.ndarray, name: str) -> np.ndarray`

  - 输入：`(6,)` 向量

  - 输出：合法的状态增量向量

  

#### 2.6.2 SO(3) 指数映射更新旋转

- `so3_exp(omega: np.ndarray) -> np.ndarray`

  - 输入：旋转增量 `(3,)`

  - 输出：`(3, 3)` 旋转增量矩阵

  - 调用：

    - `_as_vector3(...)`

    - `skew(...)`

  

## 3. 输出结构

- `initial_estimates: dict[str, SolverEstimate]`

  - 每个待估节点一个前端初始化结果

  - `SolverEstimate` 包含：

    - `rotation: np.ndarray`

    - `translation: np.ndarray`

    - `runtime: float`

- `GaBPRefinementResult`

  - `states: dict[str, PoseState]`

  - `history: list[GaBPHistoryEntry]`

  

## 4. 当前边界

- 当前 `BgpnpGabpSolver.solve(...)` 使用单主锚点 `primary_anchor_id`。

- 当前默认角度噪声来自 `multi_agent_bearing_gabp.py` 中的：

  - `DEFAULT_SIGMA_AZIMUTH = 0.5 deg`

  - `DEFAULT_SIGMA_ELEVATION = 2.0 deg`

- 当前 `bgPnp_GABP.py` 还没有把 observation batch 中的 `sigma_azimuth_deg / sigma_elevation_deg` 显式传入每个 GABP 因子。