# GN-BP Multi-Agent Bearing Solver Design

## 1. 目标与场景

本文档只讨论多机协同定位的第二阶段：
- 第一阶段已经通过 `BO-EPNP` 为待估无人机提供了位姿初值
- 第二阶段利用新的 bearing-only 观测做最大似然 refinement
- 求解器采用 `Gauss-Newton + Belief Propagation (GN-BP)`
- 旋转更新严格在 `SO(3)` 流形上进行

本文用三机协同场景说明：
- `A` 机：锚点，能广播全局位姿，记为 $X_A=(R_A,t_A)$，已知
- `B` 机：待估，状态为 $X_B=(R_B,t_B)$
- `C` 机：待估，状态为 $X_C=(R_C,t_C)$
- `B,C` 已经由 BO-EPnP 获得初值 $X_B^0, X_C^0$
- 当前历元获得新的 bearing 观测，进行第二阶段 ML refinement

本文中默认 bearing 观测始终定义在“观测者自己的本地机体系”中。这一点和前端 BO-EPNP 的近似公共坐标系写法不同，后端 ML refinement 必须严格遵守这一点。

## 2. 观测记号与因子图构建

## 2.1 观测记号

为避免歧义，本文统一采用：

$$
z_{i\leftarrow j}
$$


表示“节点 `i` 观测到节点 `j` 的 bearing 测量”，其中：
- `i` 是观测者
- `j` 是被观测者
- 该观测定义在 `i` 的本地机体系中

例如三机场景中，若 `B` 和 `C` 都能观测到其他节点，则典型观测集合为：

$$
\mathcal M = \{B\leftarrow A,\ B\leftarrow C,\ C\leftarrow A,\ C\leftarrow B\}
$$
它们对应的角度观测分别是：

$$
z_{B\leftarrow A}=(\alpha_{B\leftarrow A}^m,\beta_{B\leftarrow A}^m)
$$
$$
z_{B\leftarrow C}=(\alpha_{B\leftarrow C}^m,\beta_{B\leftarrow C}^m)
$$
$$
z_{C\leftarrow A}=(\alpha_{C\leftarrow A}^m,\beta_{C\leftarrow A}^m)
$$
$$
z_{C\leftarrow B}=(\alpha_{C\leftarrow B}^m,\beta_{C\leftarrow B}^m)
$$
## 2.2 状态变量

本历元中仅对待估节点建变量：

$$
X_B=(R_B,t_B), \qquad X_C=(R_C,t_C)
$$


每个状态位于 `SE(3)`，局部增量定义为：

$$
\delta\xi_i =
\begin{bmatrix}
\delta\theta_i \\
\delta t_i
\end{bmatrix}
\in \mathbb{R}^6,
\qquad i\in\{B,C\}
$$
因此整体系数向量为：

$$
\delta x=
\begin{bmatrix}
\delta\xi_B \\
\delta\xi_C
\end{bmatrix}

\in\mathbb{R}^{12}
$$
## 2.3 因子图节点

在 A-B-C 例子中，最小因子图可写为：
- 变量节点：`X_B, X_C`
- 因子节点：
  - $f_{B\leftarrow A}(X_B)$
  - $f_{B\leftarrow C}(X_B,X_C)$
  - $f_{C\leftarrow A}(X_C)$
  - $f_{C\leftarrow B}(X_C,X_B)$
  - $f_B^p(X_B)$
  - $f_C^p(X_C)$

其中：
- $f_{B\leftarrow A}$ 和 $f_{C\leftarrow A}$ 是单变量锚点 bearing 因子
- $f_{B\leftarrow C}$ 和 $f_{C\leftarrow B}$ 是双变量相互观测因子
- $f_B^p, f_C^p$ 是先验因子，用于抑制退化并稳定迭代

## 3. 预测 bearing 与负对数似然

## 3.1 预测 LOS 必须投影到观测者本地系

这是本文档需要特别修正的地方。

在后端 ML refinement 中，不能把本地 bearing 观测直接和全局系方向比较；正确做法是：
- 先在全局系中形成 LOS 向量
- 再把 LOS 投影到观测者的本地机体系
- 最后与该本地系下的 bearing 观测比较
补充：

$R_{A1}^{B2}\left ( P_{A}^{A1}-R_{B2}^{A1}P_{B}^{B2}-t_{B2}^{A1}\right )$
$\left (R_{A1}^{B2}P_{A}^{A1}+t_{A1}^{B2}\right )-P_{B}^{B2}$
两者是等价的，因为$t_{A1}^{B2}=-R_{A1}^{B2}t_{B2}^{A1}$，$t_{B2}^{A1}$ 表示在A1坐标系下B2坐标系原点的位置，因此先旋转到A坐标系下再取负号就得到在B2坐标系下A1原点的位置
  
### Step 1: 全局系 LOS

$$
d_{i\leftarrow j} = t_j - t_i
$$
### Step 2: 投影到观测者 `i` 的本地机体系
$$
u_{i\leftarrow j} = R_i^\top d_{i\leftarrow j}
$$
### Step 3: 单位化
$$
\hat u_{i\leftarrow j} = \frac{u_{i\leftarrow j}}{\|u_{i\leftarrow j}\|}
$$
记：
$$
\hat u_{i\leftarrow j} = [x_{i\leftarrow j},y_{i\leftarrow j},z_{i\leftarrow j}]^\top
$$
### Step 4: 由本地向量得到预测角度

$$
\alpha_{i\leftarrow j} = \operatorname{atan2}(y_{i\leftarrow j},x_{i\leftarrow j})
$$
$$
\beta_{i\leftarrow j} = \operatorname{atan2}\left(z_{i\leftarrow j}, \sqrt{x_{i\leftarrow j}^2+y_{i\leftarrow j}^2}\right)
$$
因此，后端严格模型里并不是“把测量角度转到全局系中比较”，而是“把预测 LOS 转到观测者本地系中比较”。

把本地 bearing 用 INS 姿态先旋到全局系，是前端近似线性化时的可选做法，不是这里的严格 ML 写法。
## 3.2 单个观测因子的角度残差

对观测 $i \leftarrow j$，观测角度记为：
$$
z_{i\leftarrow j}=(\alpha_{i\leftarrow j}^m,\beta_{i\leftarrow j}^m)
$$
则 bearing 残差定义为：

$$
r_{i\leftarrow j}(X_i,X_j)=

\begin{bmatrix}
r_{az,i\leftarrow j} \\
r_{el,i\leftarrow j}
\end{bmatrix}

=

\begin{bmatrix}
\mathrm{wrap}(\alpha_{i\leftarrow j}-\alpha_{i\leftarrow j}^m)/\sigma_{az} \\
(\beta_{i\leftarrow j}-\beta_{i\leftarrow j}^m)/\sigma_{el}
\end{bmatrix}

\in\mathbb{R}^2
$$
其中：
- $wrap(·)$ 把方位角残差限制在 `(-\pi,\pi]`
- $\sigma_{az},\sigma_{el}$ 分别是方位角和俯仰角标准差
## 3.3 A-B-C 三机场景的负对数似然

若当前历元可用观测是：
- `B` 观测 `A`
- `B` 观测 `C`
- `C` 观测 `A`
- `C` 观测 `B`

则 ML 目标函数应写成：

$$
\min_{X_B,X_C}
\|r_{B\leftarrow A}(X_B)\|^2
+
\|r_{B\leftarrow C}(X_B,X_C)\|^2
+
\|r_{C\leftarrow A}(X_C)\|^2
+
\|r_{C\leftarrow B}(X_C,X_B)\|^2
+
\|r_B^p(X_B)\|^2
+
\|r_C^p(X_C)\|^2
$$
更完整的加权形式为：
$$
\mathcal L(X_B,X_C)
=
\frac{1}{2}r_{B\leftarrow A}^\top W_{B\leftarrow A} r_{B\leftarrow A}
+
\frac{1}{2}r_{B\leftarrow C}^\top W_{B\leftarrow C} r_{B\leftarrow C}
+
\frac{1}{2}r_{C\leftarrow A}^\top W_{C\leftarrow A} r_{C\leftarrow A}
+
\frac{1}{2}r_{C\leftarrow B}^\top W_{C\leftarrow B} r_{C\leftarrow B}
+
\frac{1}{2}r_B^{p\top}W_B^p r_B^p
+
\frac{1}{2}r_C^{p\top}W_C^p r_C^p
$$


其中每个 bearing 因子通常取：

$$
W_{i\leftarrow j}=I_2
$$

因为残差里已经除以 $\sigma_{az},\sigma_{el}$ 做了归一化；若不在残差中归一化，则可把噪声写进 $W_{i\leftarrow j}=\Sigma_{i\leftarrow j}^{-1}$。

这就是 A-B-C 三机场景下，基于本地 bearing 观测的负对数似然最小化。

## 4. 在流形上的 Gauss-Newton 线性化

## 4.1 流形增量更新

第 $k$ 次外层 GN 迭代时，当前估计为：

$$
X_B^k=(R_B^k,t_B^k),
\qquad
X_C^k=(R_C^k,t_C^k)
$$
采用左乘扰动更新：

$$
R_i = \exp([\delta\theta_i]_\times) R_i^k,
\qquad
t_i = t_i^k + \delta t_i,
\qquad i\in\{B,C\}
$$
这保证每次更新后仍有 $R_i \in SO(3)$。
## 4.2 单个 bearing 因子的逐步推导

以下对一般观测 $i \leftarrow j$ 推导。最终可直接套到：
- $B \leftarrow A$
- $B \leftarrow C$
- $C \leftarrow A$
- $C \leftarrow B$

### Step 1: 全局相对位置向量

$$
d = t_j - t_i
$$
### Step 2: 观测者本地坐标系中的相对向量
$$
u = R_i^\top d
$$
### Step 3: 单位化
$$
\hat u = \frac{u}{\|u\|}
$$
### Step 4: 由单位向量得到角度
记：

$$
\hat u = [x,y,z]^\top
$$
则：
$$
\alpha = \operatorname{atan2}(y,x)
$$
$$
\beta = \operatorname{atan2}\left(z,\sqrt{x^2+y^2}\right)
$$
### Step 5: 形成残差
$$
r=

\begin{bmatrix}
\mathrm{wrap}(\alpha-\alpha^m)/\sigma_{az} \\
(\beta-\beta^m)/\sigma_{el}
\end{bmatrix}

$$


因此这是一个明确的链式结构：

$$
(\delta\xi_i,\delta\xi_j)
\rightarrow
d
\rightarrow
u
\rightarrow
\hat u
\rightarrow
(\alpha,\beta)
\rightarrow
r
$$
## 4.3 几何项对状态增量的导数

采用左乘扰动，线性化 $u = R_i^\top (t_j-t_i)$。

设当前估计处：
$$
d^k = t_j^k - t_i^k,
\qquad
u^k = (R_i^k)^\top d^k
$$
则一阶扰动下：

$$
\delta u
=
(R_i^k)^\top(\delta t_j - \delta t_i)
+
(R_i^k)^\top[d^k]_\times\,\delta\theta_i
$$
因此：

$$
\frac{\partial u}{\partial \delta\xi_i}
=
\begin{bmatrix}
(R_i^k)^\top[d^k]_\times & -(R_i^k)^\top
\end{bmatrix}

\in \mathbb{R}^{3\times 6}
$$
$$
\frac{\partial u}{\partial \delta\xi_j}
=
\begin{bmatrix}
0_{3\times 3} & (R_i^k)^\top
\end{bmatrix}

\in \mathbb{R}^{3\times 6}
$$
这里要注意：
- 该因子对观测者 `i` 的姿态和位置都敏感
- 若把被观测者 `j` 视为一个点目标中心，则该因子只对 `t_j` 敏感，对 `R_j` 不敏感

所以对于纯中心点 bearing 模型，被观测节点的姿态部分雅可比为零。
## 4.4 单位化雅可比

设：
$$
\hat u = \frac{u}{\|u\|}
$$
则单位化的雅可比为：

$$
J_{norm}
=
\frac{\partial \hat u}{\partial u}
=
\frac{1}{\|u\|}\left(I_3-\hat u\hat u^\top\right)
\in \mathbb{R}^{3\times 3}
$$
## 4.5 角度对单位向量的导数

记：
$$
\hat u=[x,y,z]^\top,
\qquad
\rho^2=x^2+y^2,
\qquad
\rho=\sqrt{\rho^2},
\qquad
q=x^2+y^2+z^2
$$
则：
$$
J_{ang}
=
\frac{\partial(\alpha,\beta)}{\partial \hat u}
=

\begin{bmatrix}
-y/\rho^2 & x/\rho^2 & 0 \\
-xz/(\rho q) & -yz/(\rho q) & \rho/q
\end{bmatrix}

\in \mathbb{R}^{2\times 3}
$$
若 $\hat u$ 已归一化，则 `q=1`，但保留一般写法更清晰。
## 4.6 残差权重雅可比

残差定义中含有归一化角度噪声，因此：

$$
J_w =
\begin{bmatrix}
1/\sigma_{az} & 0 \\
0 & 1/\sigma_{el}
\end{bmatrix}
\in\mathbb{R}^{2\times 2}
$$


对 $wrap(·)$，工程上默认在不跨越 `\pm\pi` 跳变点附近时，其局部导数取 `1`。

## 4.7 最终雅可比矩阵

综上，对观测 $i \leftarrow j$，残差对观测者和被观测者状态增量的雅可比分别为：

$$
J_{i\leftarrow j}^{(i)}
=
J_w\,J_{ang}\,J_{norm}

\begin{bmatrix}
(R_i^k)^\top[d^k]_\times & -(R_i^k)^\top
\end{bmatrix}

\in\mathbb{R}^{2\times 6}
$$


$$
J_{i\leftarrow j}^{(j)}
=
J_w\,J_{ang}\,J_{norm}

\begin{bmatrix}
0_{3\times 3} & (R_i^k)^\top
\end{bmatrix}

\in\mathbb{R}^{2\times 6}
$$


这就是 bearing-only 残差从角度一步步推到 `(R,t)` 的最终链式雅可比。

## 4.8 四个观测因子的具体形式

### $B \leftarrow A$

$$
d_{B\leftarrow A}=t_A-t_B,
\qquad
u_{B\leftarrow A}=R_B^\top d_{B\leftarrow A}
$$

雅可比只有对 $B$：
$$
J_{B\leftarrow A}^{(B)}
=
J_w J_{ang} J_{norm}
\begin{bmatrix}
R_B^\top[d_{B\leftarrow A}]_\times & -R_B^\top
\end{bmatrix}

$$


### $B \leftarrow C$

$$
d_{B\leftarrow C}=t_C-t_B,
\qquad
u_{B\leftarrow C}=R_B^\top d_{B\leftarrow C}
$$


$$
J_{B\leftarrow C}^{(B)}
=
J_w J_{ang} J_{norm}
\begin{bmatrix}
R_B^\top[d_{B\leftarrow C}]_\times & -R_B^\top
\end{bmatrix}

$$


$$
J_{B\leftarrow C}^{(C)}
=
J_w J_{ang} J_{norm}

\begin{bmatrix}
0_{3\times 3} & R_B^\top
\end{bmatrix}

$$


### $C \leftarrow A$

$$
d_{C\leftarrow A}=t_A-t_C,
\qquad
u_{C\leftarrow A}=R_C^\top d_{C\leftarrow A}
$$


$$
J_{C\leftarrow A}^{(C)}
=
J_w J_{ang} J_{norm}
\begin{bmatrix}
R_C^\top[d_{C\leftarrow A}]_\times & -R_C^\top
\end{bmatrix}
$$


### $C \leftarrow B$

$$
d_{C\leftarrow B}=t_B-t_C,
\qquad
u_{C\leftarrow B}=R_C^\top d_{C\leftarrow B}
$$


$$
J_{C\leftarrow B}^{(C)}
=
J_w J_{ang} J_{norm}

\begin{bmatrix}
R_C^\top[d_{C\leftarrow B}]_\times & -R_C^\top
\end{bmatrix}

$$


$$
J_{C\leftarrow B}^{(B)}
=
J_w J_{ang} J_{norm}
\begin{bmatrix}
0_{3\times 3} & R_C^\top
\end{bmatrix}

$$


## 5. A-B-C 场景下的 GN 线性系统

将所有因子残差堆叠成：

$$
r(\delta x)
\approx
r^k + J\delta x
$$


其中：

$$
r^k=

\begin{bmatrix}
r_{B\leftarrow A} \\
r_{B\leftarrow C} \\
r_{C\leftarrow A} \\
r_{C\leftarrow B} \\
r_B^p \\
r_C^p
\end{bmatrix}

$$


整体 GN 子问题：

$$
\min_{\delta x}\ \|r^k+J\delta x\|^2
$$


正规方程：

$$
H\delta x=-g
$$


其中：

$$
H=J^\top WJ,
\qquad
g=J^\top Wr^k
$$


在 A-B-C 场景下，其块结构为：

$$
H=

\begin{bmatrix}
H_{BB} & H_{BC} \\
H_{CB} & H_{CC}
\end{bmatrix}
,
\qquad
\delta x=

\begin{bmatrix}
\delta\xi_B \\
\delta\xi_C
\end{bmatrix}
,
\qquad
g=

\begin{bmatrix}
g_B \\
g_C
\end{bmatrix}

$$


各块由各因子累加：

$$
H_{BB} =
(J_{B\leftarrow A}^{(B)})^\top W_{B\leftarrow A} J_{B\leftarrow A}^{(B)}
+
(J_{B\leftarrow C}^{(B)})^\top W_{B\leftarrow C} J_{B\leftarrow C}^{(B)}
+
(J_{C\leftarrow B}^{(B)})^\top W_{C\leftarrow B} J_{C\leftarrow B}^{(B)}
+
(J_B^p)^\top W_B^p J_B^p
$$


$$
H_{CC} =
(J_{C\leftarrow A}^{(C)})^\top W_{C\leftarrow A} J_{C\leftarrow A}^{(C)}
+
(J_{B\leftarrow C}^{(C)})^\top W_{B\leftarrow C} J_{B\leftarrow C}^{(C)}
+
(J_{C\leftarrow B}^{(C)})^\top W_{C\leftarrow B} J_{C\leftarrow B}^{(C)}
+
(J_C^p)^\top W_C^p J_C^p
$$


$$
H_{BC} =
(J_{B\leftarrow C}^{(B)})^\top W_{B\leftarrow C} J_{B\leftarrow C}^{(C)}
+
(J_{C\leftarrow B}^{(B)})^\top W_{C\leftarrow B} J_{C\leftarrow B}^{(C)}
$$


$$
H_{CB}=H_{BC}^\top
$$


$$
g_B =
(J_{B\leftarrow A}^{(B)})^\top W_{B\leftarrow A} r_{B\leftarrow A}
+
(J_{B\leftarrow C}^{(B)})^\top W_{B\leftarrow C} r_{B\leftarrow C}
+
(J_{C\leftarrow B}^{(B)})^\top W_{C\leftarrow B} r_{C\leftarrow B}
+
(J_B^p)^\top W_B^p r_B^p
$$


$$
g_C =
(J_{C\leftarrow A}^{(C)})^\top W_{C\leftarrow A} r_{C\leftarrow A}
+
(J_{B\leftarrow C}^{(C)})^\top W_{B\leftarrow C} r_{B\leftarrow C}
+
(J_{C\leftarrow B}^{(C)})^\top W_{C\leftarrow B} r_{C\leftarrow B}
+
(J_C^p)^\top W_C^p r_C^p
$$


因此 GN 线性方程为：

$$

\begin{bmatrix}
H_{BB} & H_{BC} \\
H_{CB} & H_{CC}
\end{bmatrix}


\begin{bmatrix}
\delta\xi_B \\
\delta\xi_C
\end{bmatrix}

=
-

\begin{bmatrix}
g_B \\
g_C
\end{bmatrix}

$$
这就是 A-B-C 三机场景下当前历元 ML refinement 的核心线性系统。

## 6. GN-BP 如何求解这个线性系统

GN-BP 的思路不是集中式直接求逆，而是：
- 外层：Gauss-Newton 不断重线性化
- 内层：对当前线性高斯子问题做 Belief Propagation

## 6.1 因子图上的信息形式表示

对每个因子 `f`，线性化后都能写成：

$$
\|r_f^k + J_f\delta x_f\|^2
$$


对应一个局部信息矩阵和信息向量：

$$
\Omega_f = J_f^\top W_f J_f,
\qquad
\xi_f = J_f^\top W_f r_f^k
$$


## 6.2 变量到因子的消息

设 $\Lambda_{i\to f}, \eta_{i\to f}$ 为变量节点 $i$ 发送到因子 $f$ 的信息矩阵与信息向量，则：

$$
\Lambda_{i\to f} = \Lambda_i^{prior} + \sum_{h\in\mathcal N(i)\setminus f} \Lambda_{h\to i}
$$
$$
\eta_{i\to f} = \eta_i^{prior} + \sum_{h\in\mathcal N(i)\setminus f} \eta_{h\to i}
$$
这些消息定义在切空间增量 $\delta\xi_i$ 上，而不是直接定义在 $SE(3)$ 本体上。

## 6.3 因子到变量的消息

对双变量因子 `f(i,j)`，其线性化信息块写成：

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

则因子到变量 `i` 的消息为：

$$
\Lambda_{f\to i}
=
\Omega_{ii}
-
\Omega_{ij}(\Lambda_{j\to f}+\Omega_{jj})^{-1}\Omega_{ji}
$$


$$
\eta_{f\to i}
=
\xi_i
-
\Omega_{ij}(\Lambda_{j\to f}+\Omega_{jj})^{-1}(\eta_{j\to f}+\xi_j)
$$


对 `j` 同理。

## 6.4 后验增量估计

当消息收敛后，变量节点 `i` 的后验 belief 为：
$$
\Lambda_i = \Lambda_i^{prior} + \sum_{f\in\mathcal N(i)} \Lambda_{f\to i}
$$
$$
\eta_i = \eta_i^{prior} + \sum_{f\in\mathcal N(i)} \eta_{f\to i}
$$
于是本轮 GN 增量为：

$$
\delta\xi_i = -\Lambda_i^{-1}\eta_i
$$


## 7. 流形上的更新步骤

在第 `k` 次 GN 外迭代中，得到增量 $\delta\xi_B, \delta\xi_C$ 后，更新为：
$$
R_B^{k+1}=\exp([\delta\theta_B]_\times)R_B^k,
\qquad
t_B^{k+1}=t_B^k+\delta t_B
$$


$$
R_C^{k+1}=\exp([\delta\theta_C]_\times)R_C^k,
\qquad
t_C^{k+1}=t_C^k+\delta t_C
$$


这就是“流形上的 GN-BP”：
- 高斯消息在切空间增量上定义
- 位姿更新通过李群指数映射回到 `SO(3)` / `SE(3)`

## 8. 迭代流程总结

```text
Algorithm: GN-BP ML Refinement for A-B-C Bearing Localization
Input:
  - known anchor state X_A
  - BO-EPNP initial states X_B^0, X_C^0
  - current epoch local-frame bearing measurements:
      z_{B<-A}, z_{B<-C}, z_{C<-A}, z_{C<-B}
  - optional prior factors
Output:
  - refined states X_B, X_C

for k = 0,1,2,... until GN convergence:
  1. At current states X_B^k, X_C^k, compute residuals
     r_{B<-A}, r_{B<-C}, r_{C<-A}, r_{C<-B}, r_B^p, r_C^p

  2. Compute analytic Jacobians for each factor

  3. Build linearized factor graph in information form

  4. Run inner BP iterations on tangent-space variables
     delta_xi_B, delta_xi_C

  5. Recover GN increments:
     delta_xi_B = -Lambda_B^{-1} eta_B
     delta_xi_C = -Lambda_C^{-1} eta_C

  6. Update on SE(3):
     R_B <- Exp(hat(delta_theta_B)) R_B
     t_B <- t_B + delta_t_B
     R_C <- Exp(hat(delta_theta_C)) R_C
     t_C <- t_C + delta_t_C

  7. If increments or cost reduction are below threshold, stop
```

## 9. 与当前工程的对应关系

直接可复用：
- `src/bo_epnp_solver.py`
  - bearing 到 az/el 的转换
  - `wrap_to_pi`
  - 基于 `SO(3)` 指数映射的流形更新思想
- `docs/ml_refinement_algorithm.md`
  - 单机 ML refinement 的角度残差和 Jacobian 结构
- `docs/ml_refinement_russell_jacobian.md`
  - 链式求导框架，可扩展到双变量因子

建议代码模块拆分：
- `src/multi_agent_bearing_factors.py`
  - 实现 `f_{B<-A}, f_{B<-C}, f_{C<-A}, f_{C<-B}` 的残差与 Jacobian
- `src/distributed_gn_bp_solver.py`
  - 实现 GN 外迭代与 BP 内迭代
- `src/se3_ops.py`
  - 统一管理 `hat`, `Exp`, `Log` 等流形操作

## 10. 结论

对多机 bearing-only 协同定位，若 BO-EPNP 已经提供 `(R,t)` 初值，则第二阶段最大似然 refinement 可以严格写成：
- 状态变量：所有待估无人机的 `SE(3)` 位姿
- 因子：定义在观测者本地坐标系中的锚点 bearing 因子、相邻无人机 bearing 因子、先验因子
- 外层求解：Gauss-Newton 线性化
- 内层求解：Belief Propagation 求解当前切空间中的线性高斯子问题
- 更新方式：在 `SO(3)` 流形上用李群指数映射更新旋转

A-B-C 三机场景中，未知仅有 `B,C` 两个节点，因此当前历元 ML refinement 的核心线性系统是一个 `12×12` 的块稀疏 GN 系统，而 GN-BP 的作用就是分布式地逼近求解这个系统。