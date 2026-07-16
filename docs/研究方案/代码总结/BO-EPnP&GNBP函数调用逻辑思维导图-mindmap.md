---

mindmap-plugin: markdown

---


# GABP Solver Call Outline


## 1. 入口函数

- `BgpnpGabpSolver.solve(sample: MultiAgentWindowSample, primary_anchor_id: str, prior_covariance: np.ndarray, sol_iter: bool = True, enforce_manifold: bool = True) -> tuple[dict[str, SolverEstimate], GaBPRefinementResult]`

### 1.1 待估节点逐个做 BGPnP 初始化

- `bgpnp.solve(p1: np.ndarray, p2: np.ndarray, bearing: np.ndarray, sol_iter: bool = True, enforce_manifold: bool = False) -> tuple[tuple[np.ndarray, np.ndarray, float], float]`

### 1.2 构造初始状态容器

- `PoseState(rotation: np.ndarray, translation: np.ndarray)`

### 1.3 构造锚点因子列表

- `BgpnpGabpSolver._build_anchor_factors(sample: MultiAgentWindowSample, primary_anchor_id: str) -> list[AnchorBearingFactor]`

#### 1.3.1 单个关键点的 bearing 转角度

- `vector_to_az_el(vector: np.ndarray) -> np.ndarray`

#### 1.3.2 构造锚点因子对象

- `AnchorBearingFactor(observer_id: str, anchor_position_world: np.ndarray, measured_angles: np.ndarray, observer_point_local: np.ndarray | None = None, sigma_azimuth: float = DEFAULT_SIGMA_AZIMUTH, sigma_elevation: float = DEFAULT_SIGMA_ELEVATION, observer_extrinsic_rotation: np.ndarray | None = None, observer_extrinsic_translation: np.ndarray | None = None)`

### 1.4 构造待估节点间因子列表

- `BgpnpGabpSolver._build_pairwise_factors(sample: MultiAgentWindowSample) -> list[PairwiseBearingFactor]`

#### 1.4.1 单个关键点的 bearing 转角度

- `vector_to_az_el(vector: np.ndarray) -> np.ndarray`

#### 1.4.2 构造双变量因子对象

- `PairwiseBearingFactor(observer_id: str, target_id: str, target_point_local: np.ndarray, measured_angles: np.ndarray, observer_point_local: np.ndarray | None = None, sigma_azimuth: float = DEFAULT_SIGMA_AZIMUTH, sigma_elevation: float = DEFAULT_SIGMA_ELEVATION, observer_extrinsic_rotation: np.ndarray | None = None, observer_extrinsic_translation: np.ndarray | None = None)`

### 1.5 进入 GABP 联合优化主循环

- `refine_with_gabp(initial_states: dict[str, PoseState], anchor_factors: Iterable[AnchorBearingFactor], pairwise_factors: Iterable[PairwiseBearingFactor], prior_states: dict[str, PoseState] | None = None, prior_covariance: np.ndarray | None = None, settings: GaBPSettings | None = None) -> GaBPRefinementResult`

## 2. `refine_with_gabp(...)` 内部调用顺序

- `refine_with_gabp(...)`

### 2.1 初始化与状态合法性检查

- `_validate_pose_state(state: PoseState) -> PoseState`

#### 2.1.1 旋转矩阵检查

- `_as_rotation(rotation: np.ndarray, name: str) -> np.ndarray`

#### 2.1.2 平移向量检查

- `_as_vector3(vector: np.ndarray, name: str) -> np.ndarray`

### 2.2 线性化单变量因子

- `_linearize_unary_factors(factors: list[AnchorBearingFactor], states: dict[str, PoseState]) -> list[UnaryInfoFactor]`

#### 2.2.1 线性化单个锚点因子

- `_linearize_anchor_factor(factor: AnchorBearingFactor, states: dict[str, PoseState]) -> UnaryInfoFactor`

##### 2.2.1.1 当前姿态取数组

- `_pose_arrays(state: PoseState) -> tuple[np.ndarray, np.ndarray]`

##### 2.2.1.2 外参默认值处理

- `_extrinsic_rotation(rotation: np.ndarray | None) -> np.ndarray`
- 输入：可选外参旋转
- 输出：`(3, 3)`，若为空则单位阵

##### 2.2.1.3 角度残差与解析雅可比

- `_angle_residual_and_jacobian(s: np.ndarray, measured_angles: np.ndarray, sigma_azimuth: float, sigma_elevation: float) -> tuple[np.ndarray, np.ndarray]`

###### 2.2.1.3.1 向量/角度合法性检查

- `_as_vector3(vector: np.ndarray, name: str) -> np.ndarray`
- `_as_angles(angles: np.ndarray) -> np.ndarray`

###### 2.2.1.3.2 预测角度计算

- `vector_to_az_el(vector: np.ndarray) -> np.ndarray`

###### 2.2.1.3.3 方位角 wrap

- `wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float`

##### 2.2.1.4 李代数几何雅可比中的叉乘矩阵

- `skew(vector: np.ndarray) -> np.ndarray`

### 2.3 线性化先验因子

- `_build_prior_info_factors(prior_states: dict[str, PoseState], current_states: dict[str, PoseState], covariance: np.ndarray) -> list[UnaryInfoFactor]`

#### 2.3.1 协方差检查

- `_as_covariance(covariance: np.ndarray) -> np.ndarray`

#### 2.3.2 姿态误差映射到切空间

- `so3_log(rotation: np.ndarray) -> np.ndarray`
- 输入：`(3, 3)` 旋转矩阵

### 2.4 线性化双变量因子

- `_linearize_pairwise_factors(factors: list[PairwiseBearingFactor], states: dict[str, PoseState]) -> list[PairwiseInfoFactor]`

#### 2.4.1 线性化单个双变量因子

- `_linearize_pairwise_factor(factor: PairwiseBearingFactor, states: dict[str, PoseState]) -> PairwiseInfoFactor`

##### 2.4.1.1 当前 observer/target 姿态取数组

- `_pose_arrays(state: PoseState) -> tuple[np.ndarray, np.ndarray]`

##### 2.4.1.2 外参与局部点检查

- `_extrinsic_rotation(...) -> np.ndarray`
- `_extrinsic_translation(...) -> np.ndarray`
- `_optional_vector3(...) -> np.ndarray`
- `_as_vector3(...) -> np.ndarray`

##### 2.4.1.3 角度残差与解析雅可比

- `_angle_residual_and_jacobian(...) -> tuple[np.ndarray, np.ndarray]`

##### 2.4.1.4 旋转增量雅可比用到的叉乘矩阵

- `skew(vector: np.ndarray) -> np.ndarray`

### 2.5 解当前 GN 线性子问题

- `solve_linearized_gabp(variable_ids: Iterable[str], unary_factors: Iterable[UnaryInfoFactor], pairwise_factors: Iterable[PairwiseInfoFactor], settings: GaBPSettings | None = None) -> tuple[dict[str, np.ndarray], int]`

#### 2.5.1 单变量因子汇总

- `_sum_unary_information(variable_ids: list[str], unary_factors: Iterable[UnaryInfoFactor]) -> dict[str, tuple[np.ndarray, np.ndarray]]`

#### 2.5.2 初始化因子到变量消息

- `_zero_message() -> tuple[np.ndarray, np.ndarray]`

#### 2.5.3 构造变量到因子消息

- `_copy_message(message: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]`
- 输入：旧消息

#### 2.5.4 构造因子到变量消息

- `_factor_to_first_message(factor: PairwiseInfoFactor, incoming_second: tuple[np.ndarray, np.ndarray], regularization: float) -> tuple[np.ndarray, np.ndarray]`
- 输入：

#### 2.5.5 消息阻尼

- `_damp_message(old: tuple[np.ndarray, np.ndarray], new: tuple[np.ndarray, np.ndarray], damping: float) -> tuple[np.ndarray, np.ndarray]`
    - 阻尼可以看做是一个低通滤波器

#### 2.5.6 求每个变量的切空间增量

- `_factor_contains(factor: PairwiseInfoFactor, variable_id: str) -> bool`

### 2.6 将切空间增量更新到状态流形上

- `_apply_tangent_update(state: PoseState, delta: np.ndarray) -> None`

#### 2.6.1 增量合法性检查

- `_as_state_vector(vector: np.ndarray, name: str) -> np.ndarray`

#### 2.6.2 SO(3) 指数映射更新旋转

- `so3_exp(omega: np.ndarray) -> np.ndarray`

## 3. 输出结构

- `initial_estimates: dict[str, SolverEstimate]`
- 每个待估节点一个前端初始化结果

## 4. 当前边界

- 当前 `BgpnpGabpSolver.solve(...)` 使用单主锚点 `primary_anchor_id`。
- 当前默认角度噪声来自 `multi_agent_bearing_gabp.py` 中的：
- `DEFAULT_SIGMA_AZIMUTH = 0.5 deg`
