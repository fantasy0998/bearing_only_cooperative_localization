# GN-BP Multi-Agent Bearing Solver Design

## 1. 目标与建模范围

本文档重写多机协同定位第二阶段 `ML refinement` 的建模方式，严格对齐 Russell 论文 *Cooperative Localization of a GPS-Denied UAV Using Direction-of-Arrival Measurements* 的坐标系定义：

- `A1`：全局坐标系，由锚节点 `A` 持有
- `Bi2`：待估无人机 `i` 的局部 INS 坐标系
- `Bi4`：待估无人机 `i` 的机体/测量坐标系，实际 `DOA` 角度观测发生在该坐标系
- `Bi2 -> Bi4` 的旋转和平移在短时间内视为已知

本文只讨论第二阶段：
- 第一阶段已经通过 `BO-EPNP` 或其他前端方法，为各待估无人机提供初值
- 第二阶段在流形上做 `Gauss-Newton + Belief Propagation (GN-BP)` 最大似然优化

本文用三机场景说明：
- `A`：锚节点，已知全局轨迹，坐标在 `A1`
- `B`：待估无人机，状态定义为 `A1 -> B2` 的刚体变换
- `C`：待估无人机，状态定义为 `A1 -> C2` 的刚体变换

---

## 2. 坐标系、状态与观测

## 2.1 状态变量

按照 Russell 的定义，对每个 GPS-denied 无人机 `i`，优化变量不是“机体中心在全局系的直接位姿”，而是：

$$
X_i = (R_i, t_i), \qquad R_i := R_{i2\,A1}, \quad t_i := t_{A1\,i2}
$$

它表示从全局系 `A1` 到 `i` 的局部 INS 系 `i2` 的坐标变换：

$$
p^{i2} = R_i p^{A1} + t_i
$$

因此对 `i` 自身局部轨迹点 $p_i^{i2}(k)$，其全局坐标恢复式为：

$$
p_i^{A1}(k) = R_i^\top \bigl(p_i^{i2}(k) - t_i\bigr)
$$

---

## 2.2 观测发生在 `i4`，不是 `i2`

Russell 文中：
- 线性 `LS / SDP` 阶段通常在 `i2` 中写方向约束
- `MLE` 阶段改用 `i4`，因为机体系方位角和俯仰角噪声更适合独立高斯建模

对每个观测者 `i` 和时刻 `k`，定义已知外参：

$$
E_i(k) := R_{i4\,i2}(k), \qquad e_i(k) := t_{i2\,i4}(k)
$$

则任意一点在 `i4` 中的坐标满足：

$$
p^{i4} = E_i(k)\, p^{i2} + e_i(k)
$$

这正对应 Russell 论文公式 `(22)` / `(23)` 的推广形式。

---

## 2.3 三机 A-B-C 场景的变量节点与因子节点

在三机场景中，仅 `B,C` 为待估变量：

$$
X_B=(R_B,t_B), \qquad X_C=(R_C,t_C)
$$

局部切空间增量写为：

$$
\delta \xi_i =

\begin{bmatrix}
\delta \theta_i \\
\delta t_i
\end{bmatrix}

\in \mathbb{R}^6,
\qquad i\in\{B,C\}
$$

整体增量向量：

$$
\delta x =

\begin{bmatrix}
\delta \xi_B \\
\delta \xi_C
\end{bmatrix}

\in \mathbb{R}^{12}
$$

最小因子图可写为：

- 变量节点：`X_B`, `X_C`
- 因子节点：
  - $f_{B\leftarrow A}(X_B)$
  - $f_{B\leftarrow C}(X_B,X_C)$
  - $f_{C\leftarrow A}(X_C)$
  - $f_{C\leftarrow B}(X_C,X_B)$
  - $f_B^p(X_B)$
  - $f_C^p(X_C)$

其中：
- $i \leftarrow j$表示“i 在本地测到 j”
- bearing 测量始终定义在观测者自己的 `i4` 坐标系下

---

## 3. 预测观测与负对数似然

## 3.1 锚节点观测 $i \leftarrow A$

设锚节点 `A` 在全局系中的广播位置为：

$$
p_A^{A1}(k)
$$

则 `A` 在观测者 `i` 的 `i2` 系中的坐标为：

$$
p_A^{i2}(k; X_i) = R_i p_A^{A1}(k) + t_i
$$

进一步投影到实际测量坐标系 `i4`：

$$
p_A^{i4}(k; X_i) = E_i(k)\bigl(R_i p_A^{A1}(k) + t_i\bigr) + e_i(k)
$$

这就是 Russell 论文中 $p_A^{B4}(k,R,t)$ 的多机推广版。

---

## 3.2 节点间观测 $i \leftarrow j$

若 $j$ 也是 GPS-denied 节点，则先由 `X_j` 恢复 `j` 的全局位置：

$$
p_j^{A1}(k; X_j) = R_j^\top \bigl(p_j^{j2}(k) - t_j\bigr)
$$

再变换到观测者 `i` 的 `i2` 中：

$$
p_j^{i2}(k; X_i, X_j) = R_i\, p_j^{A1}(k; X_j) + t_i
$$

即：

$$
p_j^{i2}(k; X_i, X_j)
=
R_i R_j^\top \bigl(p_j^{j2}(k) - t_j\bigr) + t_i
$$

最后投影到测量坐标系 `i4`：

$$
p_j^{i4}(k; X_i, X_j)
=
E_i(k)\Bigl(R_i R_j^\top \bigl(p_j^{j2}(k) - t_j\bigr) + t_i\Bigr) + e_i(k)
$$

这就是 $B \leftarrow C$、$C \leftarrow B$ 两类因子的严格预测模型。

---

## 3.3 角度观测模型

记：

$$
s_{i\leftarrow j}(k) := p_j^{i4}(k)
=

\begin{bmatrix}
s_x \\
s_y \\
s_z
\end{bmatrix}

\in \mathbb{R}^3
$$

定义单位方向向量：

$$
n_{i\leftarrow j}(k) = \frac{s_{i\leftarrow j}(k)}{\|s_{i\leftarrow j}(k)\|}
=

\begin{bmatrix}
n_x \\
n_y \\
n_z
\end{bmatrix}
$$


按照 Russell 文中的定义：

- elevation：

$$
\theta_{i4}(k) = \arcsin(n_z)
$$

- azimuth：

$$
\phi_{i4}(k) = \operatorname{atan2}(n_y, n_x)
$$

因此对观测 `i \leftarrow j`，预测角度写为：

$$
h_{i\leftarrow j}(X_i, X_j)
=
\begin{bmatrix}
\theta_{i\leftarrow j}(X_i, X_j) \\
\phi_{i\leftarrow j}(X_i, X_j)
\end{bmatrix}
$$
---

## 3.4 A-B-C 三机场景下的负对数似然

记在时刻集合 $\mathcal{K}$ 上的可用观测为：

- $B \leftarrow A$
- $B \leftarrow C$
- $C \leftarrow A$
- $C \leftarrow B$

对应测量角度分别为：

$$
z_{B\leftarrow A}(k) =

\begin{bmatrix}
\theta_{B\leftarrow A}^m(k) \\
\phi_{B\leftarrow A}^m(k)
\end{bmatrix}
$$


$$
z_{B\leftarrow C}(k) =

\begin{bmatrix}
\theta_{B\leftarrow C}^m(k) \\
\phi_{B\leftarrow C}^m(k)
\end{bmatrix}
$$


$$
z_{C\leftarrow A}(k) =

\begin{bmatrix}
\theta_{C\leftarrow A}^m(k) \\
\phi_{C\leftarrow A}^m(k)
\end{bmatrix}
$$


$$
z_{C\leftarrow B}(k) =

\begin{bmatrix}
\theta_{C\leftarrow B}^m(k) \\
\phi_{C\leftarrow B}^m(k)
\end{bmatrix}
$$


对单个观测定义归一化残差：

$$
r_{i\leftarrow j}(k) =

\begin{bmatrix}
\dfrac{\theta_{i\leftarrow j}^m(k) - \theta_{i\leftarrow j}(k)}{\sigma_{\theta,i}} \\
\dfrac{\operatorname{wrap}\!\bigl(\phi_{i\leftarrow j}^m(k) - \phi_{i\leftarrow j}(k)\bigr)}{\sigma_{\phi,i}}
\end{bmatrix}

\in \mathbb{R}^2
$$

其中：
- $\sigma_{\theta,i}$：观测者 `i` 的 elevation 噪声标准差
- $\sigma_{\phi,i}$：观测者 `i` 的 azimuth 噪声标准差

于是 A-B-C 三机场景的负对数似然可写成：

$$
\mathcal{L}(X_B, X_C)
=
\frac{1}{2}
\sum_{k\in\mathcal{K}}
\Bigl(
\|r_{B\leftarrow A}(k)\|^2

\|r_{B\leftarrow C}(k)\|^2

\|r_{C\leftarrow A}(k)\|^2

\|r_{C\leftarrow B}(k)\|^2
\Bigr)

\frac{1}{2}\|r_B^p(X_B)\|_{W_B^p}^2

\frac{1}{2}\|r_C^p(X_C)\|_{W_C^p}^2
$$

其中 `prior factor` 可以来自：
- 上一历元 INS 传播结果
- BO-EPNP 初值
- 其他弱正则项

这个目标函数就是当前历元 $ML refinement$ 的优化对象。

---

## 4. 流形上的 Gauss-Newton 线性化

## 4.1 流形增量更新

第 `m` 次外层 GN 迭代时，当前估计为：

$$
X_B^m=(R_B^m,t_B^m), \qquad X_C^m=(R_C^m,t_C^m)
$$

采用左乘扰动：

$$
R_i = \exp([\delta \theta_i]_\times) R_i^m,
\qquad
t_i = t_i^m + \delta t_i
$$

这样可保证每步更新后 $R_i \in SO(3)$。

---

## 4.2 链式求导结构

对任意观测 $i \leftarrow j$，观测函数链路为：
$$
(R_i, t_i, R_j, t_j)
\longrightarrow
s_{i\leftarrow j}
\longrightarrow
n_{i\leftarrow j}
\longrightarrow
(\theta_{i\leftarrow j}, \phi_{i\leftarrow j})
\longrightarrow
r_{i\leftarrow j}
$$
因此雅可比统一写成：
$$
J_{i\leftarrow j}^{(\cdot)}
=
\frac{\partial r_{i\leftarrow j}}{\partial (\cdot)}
=
J_r^{ang}\,
J_{ang}^{n}\,
J_n^{s}\,
J_s^{(\cdot)}
$$
下面逐项展开。

---

## 4.3 从角度残差到单位向量的导数

记：

$$
r =

\begin{bmatrix}
(\theta^m - \theta)/\sigma_\theta \\
\operatorname{wrap}(\phi^m - \phi)/\sigma_\phi
\end{bmatrix}
$$


在线性化点附近，`wrap` 的局部导数按 `1` 处理，因此：

$$
J_r^{ang}
=
\frac{\partial r}{\partial (\theta,\phi)}
=

\begin{bmatrix}
-1/\sigma_\theta & 0 \\
0 & -1/\sigma_\phi
\end{bmatrix}
$$

对单位向量 $n=[n_x,n_y,n_z]^\top$，

$$
\theta = \arcsin(n_z),
\qquad
\phi = \operatorname{atan2}(n_y,n_x)
$$

于是：

$$
J_{ang}^{n}
=
\frac{\partial (\theta,\phi)}{\partial n}
=
\begin{bmatrix}
0 & 0 & \dfrac{1}{\sqrt{1-n_z^2}} \\
-\dfrac{n_y}{n_x^2+n_y^2} & \dfrac{n_x}{n_x^2+n_y^2} & 0
\end{bmatrix}
$$

---

## 4.4 单位化雅可比

对：

$$
n = \frac{s}{\|s\|}, \qquad \rho = \|s\|
$$

有：

$$
J_n^{s}
=
\frac{\partial n}{\partial s}
=
\frac{1}{\rho}\left(I_3 - nn^\top\right)
$$

---

## 4.5 锚节点观测 $i \leftarrow A$ 的几何雅可比

对锚节点观测：
$$
s_{i\leftarrow A}(k)
=
E_i(k)\bigl(R_i p_A^{A1}(k) + t_i\bigr) + e_i(k)
$$

记：
$$
a_{iA}(k) := R_i p_A^{A1}(k)
$$
则：
$$
s_{i\leftarrow A}(k) = E_i(k)\bigl(a_{iA}(k) + t_i\bigr) + e_i(k)
$$

对观测者状态增量 $\delta \xi_i$，有：
$$
R_i' = \exp([\delta\theta_i]_\times) R_i
\approx
(I + [\delta\theta_i]_\times)R_i
$$

因此：
$$
\delta a_{iA}
=
[\delta\theta_i]_\times a_{iA}
=
-[a_{iA}]_\times \delta\theta_i
$$

所以：
$$
\delta s_{i\leftarrow A}
=
E_i(k)\bigl(-[a_{iA}]_\times \delta\theta_i + \delta t_i\bigr)
$$

最终得到：

$$
J_{s,i\leftarrow A}^{(i)}
=
\frac{\partial s_{i\leftarrow A}}{\partial \delta\xi_i}
=
E_i(k)

\begin{bmatrix}
-[a_{iA}(k)]_\times & I_3
\end{bmatrix}
$$
---

## 4.6 节点间观测 `i \leftarrow j` 的几何雅可比

先定义：

$$
q_j(k) := p_j^{j2}(k) - t_j
$$

$$
g_j(k) := R_j^\top q_j(k) = p_j^{A1}(k)
$$
则：
$$
s_{i\leftarrow j}(k)
=
E_i(k)\Bigl(R_i g_j(k) + t_i\Bigr) + e_i(k)
$$
再定义：

$$
a_{ij}(k) := R_i g_j(k)
$$
于是：
$$
s_{i\leftarrow j}(k)
=
E_i(k)\bigl(a_{ij}(k)+t_i\bigr)+e_i(k)
$$
### 4.6.1 对观测者 `i` 的导数

与锚节点情况同理：

$$
\delta a_{ij}
=
-[a_{ij}]_\times \delta\theta_i
$$

所以：

$$
\delta s_{i\leftarrow j}^{(i)}
=
E_i(k)\bigl(-[a_{ij}]_\times \delta\theta_i + \delta t_i\bigr)
$$

即：

$$
J_{s,i\leftarrow j}^{(i)}
=
E_i(k)

\begin{bmatrix}
-[a_{ij}(k)]_\times & I_3
\end{bmatrix}
$$
### 4.6.2 对被观测节点 `j` 的导数
由：

$$
g_j = R_j^\top q_j, \qquad q_j = p_j^{j2} - t_j
$$
对 `j` 的增量做扰动：
$$
R_j' = \exp([\delta\theta_j]_\times)R_j
\approx
(I + [\delta\theta_j]_\times)R_j
$$
于是：

$$
R_j'^\top \approx R_j^\top(I - [\delta\theta_j]_\times)
$$

并且：

$$
q_j' = p_j^{j2} - (t_j + \delta t_j)
=
q_j - \delta t_j
$$

因此：

$$
g_j'
=
R_j'^\top q_j'
\approx
R_j^\top(I - [\delta\theta_j]_\times)(q_j - \delta t_j)
$$

忽略二阶小量后：

$$
\delta g_j
=
R_j^\top [q_j]_\times \delta\theta_j - R_j^\top \delta t_j
$$

再左乘观测者 `i` 的变换，可得：

$$
\delta s_{i \leftarrow j}^{(j)} = E_i(k)\,R_i\left( R_j^\top [q_j]_\times \,\delta\theta_j - R_j^\top \delta t_j \right)
$$

因此：

$$
J_{s,i\leftarrow j}^{(j)}
=
\frac{\partial s_{i\leftarrow j}}{\partial \delta\xi_j}
=
E_i(k)\,R_i

\begin{bmatrix}
R_j^\top [q_j(k)]_\times & -R_j^\top
\end{bmatrix}
$$
---

## 4.7 最终残差雅可比

综上，对锚节点观测 `i \leftarrow A`：

$$
J_{i\leftarrow A}^{(i)}
=
J_r^{ang}\,
J_{ang}^{n}\,
J_n^{s}\,
J_{s,i\leftarrow A}^{(i)}
$$

对节点间观测 `i \leftarrow j`：

$$
J_{i\leftarrow j}^{(i)}
=
J_r^{ang}\,
J_{ang}^{n}\,
J_n^{s}\,
J_{s,i\leftarrow j}^{(i)}
$$

$$
J_{i\leftarrow j}^{(j)}
=
J_r^{ang}\,
J_{ang}^{n}\,
J_n^{s}\,
J_{s,i\leftarrow j}^{(j)}
$$

这就是 GN 线性化时真正应进入正规方程的解析雅可比。

---

## 5. A-B-C 三机场景的 GN 线性系统

把所有时刻和所有因子残差堆叠，得到：

$$
r(\delta x) \approx r^m + J\,\delta x
$$

其中：

$$
\delta x =
\begin{bmatrix}
\delta \xi_B \\
\delta \xi_C
\end{bmatrix}
$$
Gauss-Newton 子问题为：

$$
\min_{\delta x}\ \|r^m + J\delta x\|^2
$$

正规方程：
$$
H \delta x = -g
$$

$$
H = J^\top W J, \qquad g = J^\top W r^m
$$

在 A-B-C 场景中，未知只有 `B,C`，因此有块结构：

$$
\begin{bmatrix}
H_{BB} & H_{BC} \\
H_{CB} & H_{CC}
\end{bmatrix}


\begin{bmatrix}
\delta \xi_B \\
\delta \xi_C
\end{bmatrix}

=
-

\begin{bmatrix}
g_B \\
g_C
\end{bmatrix}
$$

其中：
- `$H_{BB}`、`$H_{CC}` 由各自锚点因子、互观因子、自身先验因子累加
- `$H_{BC}`、`$H_{CB}` 来自 `B \leftarrow C` 与 `C \leftarrow B` 的交叉雅可比

---

## 6. GN-BP 如何求解该线性系统

GN-BP 的结构不变：

- 外层：在当前流形线性化点反复做 `Gauss-Newton`
- 内层：把当前线性高斯子问题转成信息形式，在因子图上做 `Belief Propagation`

这里需要先明确：
- `A` 是已知锚节点，不作为随机变量节点进入 BP
- 因此锚节点不会像 `B,C` 一样维护后验 belief
- 锚节点只通过“已知状态 + 已知观测”构造单变量因子

因此三机图中：
- `B,C` 是变量节点
- `f_{B\leftarrow A}, f_{C\leftarrow A}` 是单变量因子
- `f_{B\leftarrow C}, f_{C\leftarrow B}` 是双变量因子
- `f_B^p, f_C^p` 是单变量先验因子

---

## 6.1 变量节点在切空间中的信息形式

这里需要先明确：BP 中的变量节点不是原始位姿 `X_i=(R_i,t_i)`，而是当前 GN 线性化点附近的切空间增量：

$$
\delta \xi_i =

\begin{bmatrix}
\delta \theta_i \\
\delta t_i
\end{bmatrix}

\in \mathbb{R}^6
$$

因此对每个待估节点 `i`，变量节点的 belief 写成一个 6 维高斯分布，并用信息形式表示为：

$$
b_i(\delta\xi_i) \propto
\exp\!\left(
-\frac12 \, \delta\xi_i^\top \Lambda_i \, \delta\xi_i
+ \eta_i^\top \delta\xi_i
\right)
$$

其中：

$$
\Lambda_i \in \mathbb{R}^{6\times 6}, \qquad
\eta_i \in \mathbb{R}^{6}
$$

它与均值/协方差的关系为：

$$
\Sigma_i = \Lambda_i^{-1}, \qquad
\mu_i = \Lambda_i^{-1}\eta_i
$$

也就是说，变量节点在 BP 中真正维护的是：
- 当前切空间增量的高斯 belief
- 对应的信息矩阵 `\Lambda_i`
- 对应的信息向量 `\eta_i`

而不是直接维护 `SE(3)` 上的位姿高斯。

---

## 6.2 BO-EPNP 初值如何变成变量节点先验

设第 1 步 BO-EPNP 给出节点 `i` 的位姿初值：

$$
X_i^{bo} = (R_i^{bo}, t_i^{bo})
$$

在第 `m` 次 GN 外迭代中，当前线性化点为：

$$
X_i^m = (R_i^m, t_i^m)
$$

则 BO-EPNP 先验需要先写成当前切空间中的均值：

$$
\mu_i^{p,m} =
\begin{bmatrix}
\mu_{\theta,i}^{p,m} \\
\mu_{t,i}^{p,m}
\end{bmatrix}

=

\begin{bmatrix}
\Log\!\bigl(R_i^{bo}(R_i^m)^\top\bigr) \\
t_i^{bo} - t_i^m
\end{bmatrix}

$$

其中：
- 旋转部分是当前姿态到 BO-EPNP 初值的李代数误差
- 平移部分是当前平移到 BO-EPNP 初值的差值

若给定 BO-EPNP 先验协方差：

$$
\Sigma_i^p \in \mathbb{R}^{6\times 6}
$$

则其信息矩阵为：

$$
\Lambda_i^p = (\Sigma_i^p)^{-1}
$$

对应先验因子的高斯信息形式为：

$$
\phi_i^p(\delta\xi_i)
\propto
\exp\!\left(
-\frac12
(\delta\xi_i-\mu_i^{p,m})^\top
\Lambda_i^p
(\delta\xi_i-\mu_i^{p,m})
\right)
$$

展开后可写成 BP 直接可用的信息形式：

$$
\phi_i^p(\delta\xi_i)
\propto
\exp\!\left(
-\frac12 \, \delta\xi_i^\top \Lambda_i^p \, \delta\xi_i
+ (\eta_i^p)^\top \delta\xi_i
\right)
$$

其中：

$$
\eta_i^p = \Lambda_i^p \, \mu_i^{p,m}
$$

因此，BO-EPNP 初值进入 BP 的最终形式就是：

$$
\boxed{
\Lambda_i^p = (\Sigma_i^p)^{-1}, \qquad
\eta_i^p = \Lambda_i^p

\begin{bmatrix}
\Log\!\bigl(R_i^{bo}(R_i^m)^\top\bigr) \\
t_i^{bo} - t_i^m
\end{bmatrix}

}
$$

### 6.2.1 第一次 GN-BP 迭代的常用简化

如果第一次外层 GN 迭代直接把 BO-EPNP 结果当作线性化点：

$$
R_i^m = R_i^{bo}, \qquad t_i^m = t_i^{bo}
$$

则：

$$
\mu_i^{p,m} = 0
$$

于是信息向量可以直接设为：

$$
\eta_i^p = 0
$$

而信息矩阵仍由先验协方差决定：

$$
\Lambda_i^p = (\Sigma_i^p)^{-1}
$$

这等价于在当前切空间中使用一个零均值高斯先验：

$$
\delta\xi_i \sim \mathcal{N}(0, \Sigma_i^p)
$$

这是第一次 GN-BP 迭代里最常见、也最稳定的设置方式。

---

## 6.3 线性化后每类因子的局部信息形式

对任意因子 `f`，在当前 GN 线性化点处都可以写成：

$$
\|r_f^m + J_f \delta x_f\|^2
$$

其局部信息矩阵和信息向量为：

$$
\Omega_f = J_f^\top W_f J_f, \qquad \xi_f = J_f^\top W_f r_f^m
$$

### 6.3.1 单变量因子

对锚点因子或先验因子，只涉及单个变量 `i`：

$$
\|r_f^m + J_f \delta \xi_i\|^2
$$

其信息形式为：

$$
\Omega_f \in \mathbb{R}^{6\times 6}, \qquad \xi_f \in \mathbb{R}^{6}
$$

例如：
- $f_{B\leftarrow A}$
- $f_{C\leftarrow A}$
- $f_B^p$
- $f_C^p$

### 6.3.2 双变量因子

对待估节点之间的 bearing 因子，例如 $f_{B\leftarrow C}$ 或 $f_{C\leftarrow B}$，线性化后写成：

$$
\|r_f^m + J_f^{(i)}\delta \xi_i + J_f^{(j)}\delta \xi_j\|^2
$$

局部信息矩阵分块为：

$$
\Omega_f=

\begin{bmatrix}
\Omega_{ii} & \Omega_{ij} \\
\Omega_{ji} & \Omega_{jj}
\end{bmatrix}

=

\begin{bmatrix}
(J_f^{(i)})^\top W_f J_f^{(i)} & (J_f^{(i)})^\top W_f J_f^{(j)} \\
(J_f^{(j)})^\top W_f J_f^{(i)} & (J_f^{(j)})^\top W_f J_f^{(j)}
\end{bmatrix}
$$
$$

\xi_f=

\begin{bmatrix}
\xi_i^f \\
\xi_j^f
\end{bmatrix}

=

\begin{bmatrix}
(J_f^{(i)})^\top W_f r_f^m \\
(J_f^{(j)})^\top W_f r_f^m
\end{bmatrix}
$$


---

## 6.4 变量节点到因子节点的消息

消息统一在切空间增量上以信息形式传递：

$$
(\Lambda,\eta)
$$

其中：

$$
\Lambda = \Sigma^{-1}, \qquad \eta = \Lambda \mu
$$

这里的 `\mu` 不是位姿本身，而是当前线性化点附近的增量均值 $\delta \xi$。

设变量节点 `i` 的邻接因子集合为 $\mathcal N(i)$。在第 $\ell$ 次 BP 内迭代中，变量节点 `i` 发给因子 `f` 的消息为：

$$
\Lambda_{i\to f}^{(\ell)} = \Lambda_i^{prior} + \sum_{h\in \mathcal N(i)\setminus f} \Lambda_{h\to i}^{(\ell-1)}
$$

$$
\eta_{i\to f}^{(\ell)} = \eta_i^{prior} + \sum_{h\in \mathcal N(i)\setminus f} \eta_{h\to i}^{(\ell-1)}
$$

其中：
- `prior` 可以为空，也可以来自上一历元传播形成的初始信息
- 如果先验已经显式建成因子 `f_i^p`，也可以把 `\Lambda_i^{prior}, \eta_i^{prior}` 取为零，完全通过先验因子消息体现

变量节点本质上做的是：
- **把除目标因子外的所有入边信息加起来，再发给目标因子**

---

## 6.5 锚节点和待估节点之间的消息

这部分最容易混淆。严格来说，锚节点 `A` 不发送 BP 意义上的 belief 消息，因为：
- `A` 没有待估状态
- `A` 不是变量节点
- `A` 的全局状态在当前历元是已知常量

因此锚节点相关的消息传递分成两层理解。

### 6.5.1 物理通信层

锚节点 `A` 向待估节点广播：
- 当前时刻全局位置/轨迹点 $p_A^{A1}(k)$
- 时间戳
- 可能的姿态或其他辅助量（如果系统实现需要）

待估节点 `B` 和 `C` 在本地接收这些已知量后，分别构造锚点因子：
- $f_{B\leftarrow A}(X_B)$
- $f_{C\leftarrow A}(X_C)$

### 6.5.2 因子图消息层

因为锚点因子是单变量因子，所以它到变量节点的消息不依赖其他待估节点：

$$
\Lambda_{f_{B\leftarrow A}\to B} = \Omega_{B\leftarrow A}
$$

$$
\eta_{f_{B\leftarrow A}\to B} = \xi_{B\leftarrow A}
$$

同理：

$$
\Lambda_{f_{C\leftarrow A}\to C} = \Omega_{C\leftarrow A}
$$

$$
\eta_{f_{C\leftarrow A}\to C} = \xi_{C\leftarrow A}
$$

所以锚节点相关消息的特点是：
- **它们是直接消息**
- **不需要 Schur 补**
- **不需要等待其他待估节点的 belief**
- **在一轮 BP 内可以看成常量输入**

先验因子 `f_B^p, f_C^p` 也是同样的结构：

$$
\Lambda_{f_B^p\to B} = \Omega_B^p, \qquad \eta_{f_B^p\to B} = \xi_B^p
$$

$$
\Lambda_{f_C^p\to C} = \Omega_C^p, \qquad \eta_{f_C^p\to C} = \xi_C^p
$$

---

## 6.6 待估节点之间的双变量消息

这部分才是 BP 的核心。

考虑任意一个双变量因子 `f(i,j)`，例如：
- $f_{B\leftarrow C}$
- $f_{C\leftarrow B}$

其局部信息分块为：

$$
\Omega_f=

\begin{bmatrix}
\Omega_{ii} & \Omega_{ij} \\
\Omega_{ji} & \Omega_{jj}
\end{bmatrix}
,
\qquad
\xi_f=

\begin{bmatrix}
\xi_i \\
\xi_j
\end{bmatrix}

$$

### 6.6.1 变量到双变量因子的消息

变量节点 `i` 给双变量因子 `f(i,j)` 发送：

$$
\Lambda_{i\to f}, \eta_{i\to f}
$$

它代表：
- 除去 `f(i,j)` 以外
- 变量 `i` 目前从其他因子收到的全部信息总结

### 6.6.2 双变量因子到变量的消息

因子 `f(i,j)` 给变量 `i` 的消息，通过消去 `j` 得到：

$$
\Lambda_{f\to i}
=
\Omega_{ii} - \Omega_{ij}(\Lambda_{j\to f}+\Omega_{jj})^{-1}\Omega_{ji}
$$

$$
\eta_{f\to i}
=
\xi_i - \Omega_{ij}(\Lambda_{j\to f}+\Omega_{jj})^{-1}(\eta_{j\to f}+\xi_j)
$$

对变量 `j` 同理：

$$
\Lambda_{f\to j}
=
\Omega_{jj} - \Omega_{ji}(\Lambda_{i\to f}+\Omega_{ii})^{-1}\Omega_{ij}
$$

$$
\eta_{f\to j}
=
\xi_j - \Omega_{ji}(\Lambda_{i\to f}+\Omega_{ii})^{-1}(\eta_{i\to f}+\xi_i)
$$

这一步的物理意义是：
- 因子 `f(i,j)` 先读取“另一节点”的当前 belief
- 再将该节点边缘化掉
- 最后把压缩后的信息反馈给本节点

所以待估节点之间真正传递的不是原始角度，而是：
- 当前线性化点下的局部信息矩阵
- 当前线性化点下的信息向量
- 或等价的增量高斯 belief

---

## 6.7 A-B-C 三机场景下的一轮消息传递细节

在三机例子中，变量节点只有：
- `B`
- `C`

因子节点为：
- `f_{B\leftarrow A}`
- `f_{C\leftarrow A}`
- `f_{B\leftarrow C}`
- `f_{C\leftarrow B}`
- `f_B^p`
- `f_C^p`

### 6.7.1 `B` 节点的邻接因子

$$
\mathcal N(B)=\{f_{B\leftarrow A},\ f_{B\leftarrow C},\ f_{C\leftarrow B},\ f_B^p\}
$$

因此 `B` 的后验 belief 由四部分组成：
- 锚点因子 `f_{B\leftarrow A}`
- 先验因子 `f_B^p`
- `B \leftarrow C` 因子返回的消息
- `C \leftarrow B` 因子返回的消息

汇总后：

$$
\Lambda_B = \Lambda_{f_{B\leftarrow A}\to B} + \Lambda_{f_B^p\to B} + \Lambda_{f_{B\leftarrow C}\to B} + \Lambda_{f_{C\leftarrow B}\to B}
$$

$$
\eta_B = \eta_{f_{B\leftarrow A}\to B} + \eta_{f_B^p\to B} + \eta_{f_{B\leftarrow C}\to B} + \eta_{f_{C\leftarrow B}\to B}
$$

### 6.7.2 `C` 节点的邻接因子

同理：

$$
\mathcal N(C)=\{f_{C\leftarrow A},\ f_{B\leftarrow C},\ f_{C\leftarrow B},\ f_C^p\}
$$

$$
\Lambda_C = \Lambda_{f_{C\leftarrow A}\to C} + \Lambda_{f_C^p\to C} + \Lambda_{f_{B\leftarrow C}\to C} + \Lambda_{f_{C\leftarrow B}\to C}
$$

$$
\eta_C = \eta_{f_{C\leftarrow A}\to C} + \eta_{f_C^p\to C} + \eta_{f_{B\leftarrow C}\to C} + \eta_{f_{C\leftarrow B}\to C}
$$

### 6.7.3 一轮 BP 内迭代的推荐顺序

对第 `\ell` 次 BP 迭代，可写成：

1. 锚点广播当前时刻的已知状态给 `B,C`
2. `B,C` 各自在本地线性化，构造各自相关因子的 $\Omega_f, \xi_f$
3. 单变量因子直接发送固定消息：
   - $f_{B\leftarrow A}\to B$
   - $f_{C\leftarrow A}\to C$
   - $f_B^p\to B$
   - $f_C^p\to C$
4. `B` 根据除目标因子外的其他消息，形成：
   - $B\to f_{B\leftarrow C}$
   - $B\to f_{C\leftarrow B}$
5. `C` 同理形成：
   - $C\to f_{B\leftarrow C}$
   - $C\to f_{C\leftarrow B}$
6. 双变量因子做 Schur 补，返回：
   - $f_{B\leftarrow C}\to B,\ f_{B\leftarrow C}\to C$
   - $f_{C\leftarrow B}\to B,\ f_{C\leftarrow B}\to C$
7. `B,C` 汇总所有入边消息，得到当前后验 belief
8. 若内层 BP 未收敛，则继续下一轮；否则输出当前 GN 增量

---

## 6.8 BP 收敛后的后验增量与流形更新

消息收敛后，变量节点 `i` 的后验 belief 为：

$$
\Lambda_i = \sum_{f\in \mathcal N(i)} \Lambda_{f\to i}
$$

$$
\eta_i = \sum_{f\in \mathcal N(i)} \eta_{f\to i}
$$

因此当前 GN 外迭代对应的最优增量近似为：

$$
\delta \xi_i = -\Lambda_i^{-1}\eta_i
$$

再做流形更新：

$$
R_i^{m+1} = \exp([\delta \theta_i]_\times) R_i^m,
\qquad
t_i^{m+1} = t_i^m + \delta t_i
$$

在实现上，可加入：
- BP 消息阻尼
- GN 增量阻尼
- 退化边剔除
- 边信息权重重标定

以改善含环图上的稳定性。

---

## 7. 与本文档改写直接相关的结论

这次重写后的关键结论是：

1. Russell 文中的 `MLE` 观测模型不在 `i2`，而在 `i4`
2. 状态变量仍定义为 `A1 -> i2` 的变换
3. 因此预测观测必须经过：
   - 全局系 `A1`
   - 局部 INS 系 `i2`
   - 机体系/传感器系 `i4`
4. 解析雅可比中，`Bi4 <- Bi2` 的已知外参会直接左乘到几何雅可比上
5. 这比简化的

$$
u = R_i^\top (t_j - t_i)
$$

更符合 Russell 论文的严格建模

---

## 8. 与当前工程的对应关系

直接可复用：

- `src/bo_epnp_solver.py`
  - `SO(3)` 指数映射更新思想
  - Russell 风格角度残差的实现经验
- `docs/ml_refinement_algorithm.md`
  - 单机 ML 角度残差定义
- `docs/ml_refinement_russell_jacobian.md`
  - 单机链式求导框架，可作为当前多机推导的基础

建议后续代码模块：

- `src/multi_agent_bearing_factors.py`
  - 实现带 `i2/i4` 外参的残差与 Jacobian
- `src/distributed_gn_bp_solver.py`
  - 实现 GN 外迭代 + BP 内迭代
- `src/se3_ops.py`
  - 统一 `hat / Exp / Log`

---

## 9. 小结

对于三机 `A-B-C` 的第二阶段最大似然 refinement：

- `A` 提供全局锚点信息
- `B,C` 的优化变量是 `A1 -> B2`、`A1 -> C2` 的刚体变换
- 实际 DOA 角度比较发生在 `B4/C4`
- 因而预测观测必须先经过 `i2` 再到 `i4`
- GN 线性化后的雅可比，必须把 `i4 <- i2` 的已知外参显式带入

这套写法与 Russell 论文的建模方式保持一致，也更适合作为后续 `GN-BP` 多机实现的数学基础。

