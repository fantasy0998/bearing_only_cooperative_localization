## 1. 目标

  

本文档设计一个用于 BO-EPnP 前端的关键点选择策略。适用场景为：
  

- 输入为一个时间窗口内的稠密轨迹点和 bearing-only 观测。

- BO-EPnP 尚未执行，因此未知本地坐标系与全局坐标系之间的真实或估计 `(R,t)`。

- 需要从窗口内选择少量关键点，降低计算量，并避免弱可观、病态或近退化配置。

  

现有 `FIM_ANALYSIS.md` 与 `src/fisher_information_matrix.py` 中的 FIM 计算需要已知或已有估计 `(R,t)`，例如：

  

```python
FisherInformationAnalyzer.analyze_observability(uvw, R, t)
```

因此，BO-EPnP 前不能直接计算严格意义上的 pose-space FIM。本文采用 **FIM-inspired pose-free keypoint selection**：用 bearing 方向多样性、局部轨迹几何扩展性和 BO-EPnP 线性系统条件数构造无初值可观性代理指标。

## 2. 输入与假设

对一个窗口内的候选时刻集合：

$$
\mathcal K=\{1,2,\dots,N\}
$$
已知：
锚机或目标机全局轨迹点：
$$
p_A^W(k)\in\mathbb R^3
$$
待估无人机在窗口冻结局部系中的轨迹点：
$$
p_B^L(k)\in\mathbb R^3
$$
同一局部观测系下的 bearing 单位向量：
$$
b_{B\leftarrow A}(k)\in\mathbb S^2
$$
注意：若数据生成流程同时保存 `bearing_b4` 与 `bearing_b3`，关键点选择和 BO-EPnP 前端应优先使用与局部轨迹点一致的 `bearing_b3 / B2` 观测。`B2` 与 `B3` 只差平移，对方向向量不产生影响。

目标是选择：
$$
\mathcal S\subset\mathcal K,\qquad |\mathcal S|=M
$$
使得所选关键点在 bearing 分布、轨迹几何和 BO-EPnP 线性系统上都尽量可观。
## 3. 指标一：Bearing 方向多样性

对每个 bearing 向量定义投影矩阵：


$$
P_k=I-b_k b_k^\top
$$
它表示该 bearing 对垂直于射线方向的约束。对候选集合 $(\mathcal S)$，构造 bearing 信息代理矩阵：
$$
S_b(\mathcal S)=\sum_{k\in\mathcal S} P_k
$$

  

可观性评分可采用：
$$
J_b(\mathcal S)=\log\det(S_b+\epsilon I)
$$
或：
$$
J_b(\mathcal S)=\lambda_{\min}(S_b)
$$
直观解释：

- 若所有 bearing 方向近似平行，$(S_b)$ 会病态。

- 若 bearing 分布覆盖多个方向，$(S_b)$ 的特征值更均衡。

- 该指标不依赖 $(R,t)$，适合 BO-EPnP 前使用。

  

## 4. 指标二：局部轨迹几何扩展性

  

BO-EPnP 使用控制点和重心坐标表示局部轨迹点。如果所选局部轨迹点集中在很小区域，或者近似共线，则控制点参数化会变差。

对局部轨迹点集合：
$$
\bar p_L=\frac{1}{|\mathcal S|}\sum_{k\in\mathcal S}p_B^L(k)
$$
构造局部轨迹协方差：
$$

C_p(\mathcal S)=

\sum_{k\in\mathcal S}

(p_B^L(k)-\bar p_L)(p_B^L(k)-\bar p_L)^\top

$$
轨迹几何评分：
$$
J_p(\mathcal S)=\log\det(C_p+\epsilon I)
$$
或：
$$

J_p(\mathcal S)=\lambda_{\min}(C_p)
$$
若轨迹主要是平面运动，可使用二维化指标：

- 轨迹包围面积

- 点间最大距离

- PCA 前两个特征值的乘积

- 曲率或方向变化量

  

## 5. 指标三：BO-EPnP 线性系统条件数

这是最贴近前端求解器的指标。
BO-EPnP 线性阶段可写为：
  
$$
A_{\mathcal S} c = y_{\mathcal S}
$$
其中 \(c\) 为控制点在目标坐标系或全局系中的堆叠坐标。对单个时刻 \(k\)，bearing 共线约束可写为：
$$

P_k\left(p_A^W(k)-B_k c\right)=0

$$
其中：
$$

P_k=I-b_kb_k^\top

$$
$$
B_k=\alpha_k\otimes I_3
$$
$(\alpha_k)$是局部轨迹点 $(p_B^L(k))$对控制点的重心坐标。于是：
$$

A_k=P_kB_k,\qquad y_k=P_kp_A^W(k)
$$
堆叠所选关键点得到：
$$
A_{\mathcal S}=

\begin{bmatrix}

A_{k_1}\\

A_{k_2}\\

\vdots\\

A_{k_M}

\end{bmatrix}
$$
线性系统评分可取：
$$
J_A(\mathcal S)=\log(\sigma_{\min}^2(A_{\mathcal S})+\epsilon)
$$
也可直接使用：

$$

J_A(\mathcal S)=-\log(\operatorname{cond}(A_{\mathcal S})+\epsilon)

$$
这个指标不需要已知最终 `(R,t)`，但能直接反映 BO-EPnP 初始化是否病态。
## 6. 综合目标函数

推荐将三类指标组合为：
$$

J(\mathcal S)

=

w_b J_b(\mathcal S)

+

w_p J_p(\mathcal S)

+

w_A J_A(\mathcal S)

-

w_r J_r(\mathcal S)

$$
  

其中 \(J_r\) 是冗余惩罚项，例如：
$$

J_r(\mathcal S)=

\sum_{i\neq j}

\exp\left(-\frac{|t_i-t_j|}{\tau_t}\right)

+

\sum_{i\neq j}

\left|b_i^\top b_j\right|

$$
 含义：

  

- 第一项惩罚时间过近的点。

- 第二项惩罚 bearing 方向过相似的点。

  

建议默认权重：

```python

w_b = 1.0

w_p = 0.5

w_A = 1.0

w_r = 0.1

```


若 BO-EPnP 初始化稳定性是核心目标，可提高 $w_A$。
## 7. 贪心选择算法
直接枚举所有组合复杂度为：
$$
\binom{N}{M}
$$
实际应采用贪心近似。
### 7.1 初始化
建议固定选择：
- 窗口起点

- 窗口终点

- 与起点 bearing 夹角最大的点

如果只需要最小点数，可从空集开始。
### 7.2 逐点添加

每轮从剩余候选点中选择使综合评分增益最大的点：

$$

k^*=

\arg\max_{k\in\mathcal K\setminus\mathcal S}

J(\mathcal S\cup\{k\})

$$

  

然后更新：
$$

\mathcal S\leftarrow \mathcal S\cup\{k^*\}

$$
直到：
$$
|\mathcal S|=M
$$
## 8. 伪代码

```text

Algorithm: Pose-Free Observability-Aware Keypoint Selection

  

Input:

  timestamps: shape (N,)

  anchor_position_w: shape (3, N)

  local_position_l: shape (3, N)

  bearing_l: shape (3, N)

  num_keypoints: M

  weights: wb, wp, wA, wr

  

Output:

  selected_indices: shape (M,)

  

1. Normalize all bearing vectors.

  

2. Build local control points from local_position_l.

  

3. Compute barycentric coefficients alpha[k] for all candidate points.

  

4. Initialize selected set S:

   if M >= 2:

      add first index

      add last index

   if M >= 3:

      add index with largest angular distance from first bearing

  

5. While |S| < M:

     best_score = -inf

     best_idx = None

  

     For each candidate k not in S:

        S_tmp = S union {k}

  

        Sb = sum(I - b[i] b[i]^T for i in S_tmp)

        Jb = logdet(Sb + eps * I)

  

        Cp = covariance(local_position_l[:, S_tmp])

        Jp = logdet(Cp + eps * I)

  

        A = build_bo_epnp_linear_matrix(

              anchor_position_w[:, S_tmp],

              local_position_l[:, S_tmp],

              bearing_l[:, S_tmp],

              alpha[S_tmp],

            )

        JA = log(sigma_min(A)^2 + eps)

  

        Jr = redundancy_penalty(timestamps[S_tmp], bearing_l[:, S_tmp])

  

        score = wb * Jb + wp * Jp + wA * JA - wr * Jr

  

        If score > best_score:

            best_score = score

            best_idx = k

  

     Add best_idx to S

  

6. Sort S by timestamp.

  

7. Return S.

```

  

## 9. 建议代码结构

  

建议新增模块：

  

```text

src/keypoint_selection.py

```

  

### 9.1 配置类

  

```python

@dataclass(frozen=True)

class KeypointSelectionConfig:

    num_keypoints: int

    bearing_weight: float = 1.0

    trajectory_weight: float = 0.5

    linear_system_weight: float = 1.0

    redundancy_weight: float = 0.1

    epsilon: float = 1e-9

    force_endpoints: bool = True

```

  

### 9.2 主要函数

  

```python

def select_observability_keypoints(

    timestamps: np.ndarray,

    anchor_position_w: np.ndarray,

    local_position_l: np.ndarray,

    bearing_l: np.ndarray,

    config: KeypointSelectionConfig,

) -> np.ndarray:

    """Return selected keypoint indices."""

```

  

### 9.3 辅助函数

  

```python

def compute_bearing_diversity_score(

    bearings: np.ndarray,

    epsilon: float,

) -> float:

    ...

```

  

```python

def compute_trajectory_geometry_score(

    local_points: np.ndarray,

    epsilon: float,

) -> float:

    ...

```

  

```python

def build_bo_epnp_proxy_matrix(

    anchor_points_w: np.ndarray,

    local_points_l: np.ndarray,

    bearings_l: np.ndarray,

) -> np.ndarray:

    ...

```

  

```python

def compute_linear_system_score(

    A: np.ndarray,

    epsilon: float,

) -> float:

    ...

```

  

```python

def compute_redundancy_penalty(

    timestamps: np.ndarray,

    bearings: np.ndarray,

    time_scale: float,

) -> float:

    ...

```

  

## 10. 与现有代码的关系

  

可复用：

  
- `src/fisher_information_matrix.py`

  - 作为 BO-EPnP 后的真实 FIM 验证工具。

  - 不直接用于 BO-EPnP 前的无初值关键点选择。
 

- `src/bo_epnp_solver.py`

- 复用控制点构造、重心坐标和线性系统构造逻辑。

  

- `src/bgPnp_GABP.py`

- 在构造 `MultiAgentWindowSample` 后，用本文策略替换或补充当前的分散选点逻辑。

  

- `src/miluv_to_bearing.py`

- 使用已经生成的 `bearing_b3` / `bearing` 字段作为选择输入。

  

## 11. 推荐实验验证

  

为了证明该策略有效，建议比较：

  

1. 均匀选点

2. 随机选点

3. 只用 bearing 多样性选点

4. 只用轨迹几何选点

5. 只用 BO-EPnP 线性系统条件数选点

6. 三指标联合选点

  

评估指标：

  

- BO-EPnP 初始化旋转误差

- BO-EPnP 初始化平移误差

- GNBP refinement 后旋转误差

- GNBP refinement 后平移误差

- BO-EPnP 线性矩阵条件数

- 后验 FIM 条件数

- 不同噪声水平下的成功率

  

关键结论应验证：

  

- 联合选点比均匀/随机选点更稳定。

- 线性系统条件数与 BO-EPnP 初始化误差有相关性。

- BO-EPnP 后计算的真实 FIM 能解释后端 GNBP 的收敛质量。

  

## 12. 已知限制

  

- 该策略不是严格 FIM 最优设计，因为 BO-EPnP 前没有 `(R,t)` 线性化点。

- 该策略更准确的表述是 FIM-inspired 或 observability-aware surrogate selection。

- 若窗口轨迹近似共线或 bearing 长期近似平行，任何关键点选择策略都无法从根本上恢复不可观方向。

- 多机情况下应对每个待估节点分别选点，或对所有边构造联合评分。