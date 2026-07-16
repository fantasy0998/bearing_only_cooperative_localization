# Multi-Agent BO-EPNP Solver

## 1. 文档目标

本文档基于以下两部分内容整理：
- 项目中现有的双机 BO-EPNP 实现思路：`src/bo_epnp_solver.py`
- 外部解析文档：`E:\research_obsidian\2026_paper_survey\bearing_localization\BO-EPNP求解器解析.md` 中的 `BO-EPNP 算法解析` 部分

目标是把双机场景下的 BO-EPNP 前端推广到多机场景，并给出一种适合工程落地的分布式前端求解方式：
- 每个节点仍然通过控制点参数化来表示局部轨迹点
- 节点之间通过 bearing-only 观测构造耦合线性方程
- 分布式求解时，两无人机之间传递的消息定义为当前轮的控制点坐标估计
- 线性阶段采用分布块最小二乘，收敛后再恢复各节点位姿 `(R_i, t_i)`

本文档只讨论前端初始化阶段，不展开后端 GN-BP 流形优化；后端见：
- `docs/gn_bp_multi_agent_bearing_solver_design.md`

## 2. 双机 BO-EPNP 的核心形式回顾

### 2.1 控制点参数化

对双机场景中的待估目标轨迹点 $p_1^\ell$，先选取 4 个局部控制点 $C_r$：

$$
p_1^\ell = \sum_{r=1}^4 \alpha_{\ell r} C_r,
\qquad
\sum_{r=1}^4 \alpha_{\ell r}=1
$$
设这 4 个控制点在全局坐标系中的坐标为 $C_r'$，堆叠为：
$$
c' =
\begin{bmatrix}
(C_1')^\top & (C_2')^\top & (C_3')^\top & (C_4')^\top
\end{bmatrix}
^\top \in \mathbb{R}^{12}
$$
则第 $\ell$ 个局部轨迹点在全局系中的位置可写为：
$$
x^\ell = R p_1^\ell + t = \sum_{r=1}^4 \alpha_{\ell r} C_r' = B^\ell c'
$$
其中
$$
B^\ell = \alpha^\ell \otimes I_3 \in \mathbb{R}^{3\times 12}
$$
### 2.2 bearing-only 共线约束

双机场景下，若锚点轨迹 $q^\ell$ 已知，且观测到 bearing $b^\ell \in \mathbb{S}^2$，则有：
$$
q^\ell - x^\ell \parallel b^\ell
$$
利用球面投影矩阵
$$
P^\ell = I_3 - b^\ell (b^\ell)^\top
$$
得到线性约束：
$$
P^\ell(q^\ell - x^\ell)=0
$$
代入 $x^\ell = B^\ell c'$，得到：
$$
P^\ell B^\ell c' = P^\ell q^\ell
$$
记
$$
A^\ell = P^\ell B^\ell,
\qquad
y^\ell = P^\ell q^\ell
$$
则所有观测叠加成双机 BO-EPnP 的线性最小二乘问题：

$$
\min_{c'} \sum_\ell \|A^\ell c' - y^\ell\|^2
$$
线性阶段求得 $c'$ 后，再用 Procrustes 从 $\{C_r\}$$ 与 $\{C_r'\}$ 恢复 $(R,t)$。

## 3. 多机场景建模

### 3.1 场景设定

考虑 $N+1$ 台无人机，编号为 $0,1,\dots,N$：
- $agent 0$ 为移动锚点，其全局位姿已知
- $agent i, i=1,\dots,N$ 的位姿 $X_i=(R_i,t_i)$ 为待估
- 所有无人机可以两两通信，并交换相邻时刻或当前时刻的 bearing 观测
- 每个节点有自身 INS，可在一个短滑窗内提供局部轨迹点 $p_i^\ell$

对每个待估节点 `i`，选取本地 4 个控制点 $C_{ir}$，使得：
$$
p_i^\ell = \sum_{r=1}^4 \alpha_{i\ell r} C_{ir}
\qquad \ell=1,\dots,K_i
$$
定义对应的全局控制点：
$$
C_{ir}' = R_i C_{ir} + t_i
$$
并堆叠为：
$$
c_i' =
\begin{bmatrix}
(C_{i1}')^\top & (C_{i2}')^\top & (C_{i3}')^\top & (C_{i4}')^\top
\end{bmatrix}
^\top \in \mathbb{R}^{12}
$$
则本地第 $\ell$ 个轨迹点在全局系中的位置为：
$$
x_i^\ell = R_i p_i^\ell + t_i = B_i^\ell c_i'
$$
其中
$$
B_i^\ell = \alpha_i^\ell \otimes I_3 \in \mathbb{R}^{3\times 12}
$$
## 4. 多机 BO-EPNP 的控制点线性方程

### 4.1 锚点边方程

若节点 `i` 在时刻 $\ell$ 观测到锚点 `0`，bearing 为$b_{i0}^\ell$，且锚点全局位置$x_0^\ell$ 已知，则：

$$
x_0^\ell - x_i^\ell \parallel b_{i0}^\ell
$$
令

$$
P_{i0}^\ell = I_3 - b_{i0}^\ell (b_{i0}^\ell)^\top
$$
则有

$$
P_{i0}^\ell (x_0^\ell - x_i^\ell)=0
$$
代入 $x_i^\ell = B_i^\ell c_i'$：

$$
P_{i0}^\ell B_i^\ell c_i' = P_{i0}^\ell x_0^\ell
$$


记

$$
A_{i0}^\ell = P_{i0}^\ell B_i^\ell,
\qquad
y_{i0}^\ell = P_{i0}^\ell x_0^\ell
$$


则锚点边提供的线性方程为：

$$
A_{i0}^\ell c_i' = y_{i0}^\ell
$$


### 4.2 邻接边方程
若节点 `i` 在时刻 $\ell$ 观测到节点 `j`，bearing 为 $b_{ij}^\ell$，则：

$$
x_j^\ell - x_i^\ell \parallel b_{ij}^\ell
$$
令
$$
P_{ij}^\ell = I_3 - b_{ij}^\ell (b_{ij}^\ell)^\top
$$
得到：
$$
P_{ij}^\ell(x_j^\ell - x_i^\ell)=0
$$
代入两端控制点表示：
$$
P_{ij}^\ell B_j^\ell c_j' - P_{ij}^\ell B_i^\ell c_i' = 0
$$
记
$$
A_{ij}^{\ell,(i)} = -P_{ij}^\ell B_i^\ell,
\qquad
A_{ij}^{\ell,(j)} = P_{ij}^\ell B_j^\ell
$$
则邻接边方程为：
$$
A_{ij}^{\ell,(i)} c_i' + A_{ij}^{\ell,(j)} c_j' = 0
$$
这一步是多机扩展的关键：
- 双机场景中，右端是已知锚点位置
- 多机场景中，邻居不是已知量，而是另一个未知控制点向量 $c_j'$
- 因此多机前端不再是单节点 LS，而是图上的耦合块最小二乘

## 5. 全图加权块最小二乘模型

设：
- $\mathcal{E}_a$ 为锚点边集合
- $\mathcal{E}_n$ 为邻接边集合

则前端的全局目标函数为：

$$
\min_{\{c_i'\}}
\sum_{(i,0,\ell)\in\mathcal{E}_a}
\|A_{i0}^\ell c_i' - y_{i0}^\ell\|_{W_{i0}^\ell}^2
+
\sum_{(i,j,\ell)\in\mathcal{E}_n}
\|A_{ij}^{\ell,(i)} c_i' + A_{ij}^{\ell,(j)} c_j'\|_{W_{ij}^\ell}^2
$$


其中 `W_{i0}^\ell, W_{ij}^\ell` 为观测权矩阵，可由 bearing 噪声协方差、Fisher 条件数或经验权重给出。

把所有控制点堆叠为：

$$
c=
$$
\begin{bmatrix}
(c_1')^\top & (c_2')^\top & \cdots & (c_N')^\top
\end{bmatrix}
$$^\top
$$


则该问题是一个稀疏块最小二乘问题，每个变量块的维数为 `12`。

## 6. 分布块最小二乘求解

### 6.1 每个节点维护的状态

对每个待估节点 `i`，第 `s` 轮迭代维护：

$$
c_i'^{(s)} \in \mathbb{R}^{12}
$$


这就是节点 `i` 当前的全局控制点坐标估计。

### 6.2 节点间传递的消息

本方法中，两无人机之间传递的核心消息定义为：

$$
m_{i\to j}^{(s)} = c_i'^{(s)}
$$


也就是：
- 节点 `i` 向邻居 `j` 广播自己当前轮的控制点坐标估计
- 邻居 `j` 用这个消息把耦合边中的 `c_i'` 当作已知量，更新自己的 `c_j'`

如需增强数值稳定性，也可以附带：
- 本地正规矩阵 `H_i`
- 本地残差范数
- 边有效性标志

但最小实现里只传 `c_i'` 就足够。

### 6.3 固定邻居消息后的局部子问题

在第 `s` 轮，节点 `i` 固定邻居的上一轮消息 `c_j'^{(s)}`，解本地子问题：

$$
\min_{c_i'}
\sum_{(i,0,\ell)\in\mathcal{E}_{i,a}}
\|A_{i0}^\ell c_i' - y_{i0}^\ell\|_{W_{i0}^\ell}^2
+
\sum_{(i,j,\ell)\in\mathcal{E}_{i,n}}
\|A_{ij}^{\ell,(i)} c_i' + A_{ij}^{\ell,(j)} c_j'^{(s)}\|_{W_{ij}^\ell}^2
$$


其正规方程为：

$$
H_i c_i'^{(s+1)} = g_i^{(s)}
$$


其中

$$
H_i =
\sum_{(i,0,\ell)\in\mathcal{E}_{i,a}} (A_{i0}^\ell)^\top W_{i0}^\ell A_{i0}^\ell
+
\sum_{(i,j,\ell)\in\mathcal{E}_{i,n}} (A_{ij}^{\ell,(i)})^\top W_{ij}^\ell A_{ij}^{\ell,(i)}
+ \lambda I
$$


$$
g_i^{(s)} =
\sum_{(i,0,\ell)\in\mathcal{E}_{i,a}} (A_{i0}^\ell)^\top W_{i0}^\ell y_{i0}^\ell
-
\sum_{(i,j,\ell)\in\mathcal{E}_{i,n}} (A_{ij}^{\ell,(i)})^\top W_{ij}^\ell A_{ij}^{\ell,(j)} c_j'^{(s)}
$$


因此更新为：

$$
c_i'^{(s+1)} = H_i^{-1} g_i^{(s)}
$$


其中 `\lambda I` 是轻度 Tikhonov 正则项，用于抑制弱可观条件下的数值病态。

### 6.4 阻尼更新

为了避免图中环路较多时的振荡，可采用阻尼：

$$
c_i'^{(s+1)} \leftarrow (1-\gamma)c_i'^{(s)} + \gamma H_i^{-1}g_i^{(s)},
\qquad \gamma\in(0,1]
$$


### 6.5 两种实现方式

#### `Block Jacobi`

所有节点并行更新：
- 每轮都使用邻居上一轮消息 `c_j'^{(s)}`
- 易并行，实现最简单
- 适合作为第一版多机 BO-EPNP 前端

#### `Block Gauss-Seidel`

节点异步或顺序更新：
- 尽量使用邻居最新消息
- 收敛可能更快
- 但实现和通信调度更复杂

对当前项目，前端优先建议：
- `Block Jacobi + damping`

## 7. 从控制点恢复位姿 `(R_i, t_i)`

当分布块最小二乘收敛后，节点 `i` 已得到：

$$
\{C_{ir}'\}_{r=1}^4
$$


再用本地控制点 `\{C_{ir}\}` 与全局控制点 `\{C_{ir}'\}` 做 Procrustes：

$$
\min_{R_i\in SO(3),\ t_i}
\sum_{r=1}^4
\|C_{ir}' - (R_i C_{ir} + t_i)\|^2
$$


即可恢复前端初值：

$$
(R_i^{(0)}, t_i^{(0)})
$$


这一步与双机 BO-EPNP 的 `KernelPnP -> myProcrustes` 思路一致，只是现在相机系控制点换成了每个无人机节点自己的全局控制点估计。

## 8. 分布式 BO-EPNP 前端伪代码

```text
Algorithm: Multi-Agent BO-EPNP via Block Least Squares
Input:
  - anchor trajectory {x_0^l}
  - local INS trajectory points {p_i^l}
  - pairwise bearing measurements {b_i0^l, b_ij^l}
  - local control points {C_ir}
Output:
  - initial poses {(R_i^(0), t_i^(0))}

1. For each agent i:
   build barycentric coefficients alpha_i^l
   form B_i^l = alpha_i^l ⊗ I_3

2. For each anchor edge (i,0,l):
   form A_i0^l = P_i0^l B_i^l
   form y_i0^l = P_i0^l x_0^l

3. For each neighbor edge (i,j,l):
   form A_ij^(l,i) = -P_ij^l B_i^l
   form A_ij^(l,j) =  P_ij^l B_j^l

4. Initialize control-point states c_i'^(0)
   - from anchor-only BO-EPNP if available
   - or from previous epoch

5. Repeat for s = 0,1,2,... until convergence:
   for each agent i in parallel:
     receive messages m_j→i^(s) = c_j'^(s)
     build local normal matrix H_i
     build local right-hand side g_i^(s)
     solve c_i'^(s+1) = H_i^{-1} g_i^(s)
     apply damping if needed
     broadcast m_i→j^(s+1) = c_i'^(s+1)

6. For each agent i:
   recover {C_ir'} from c_i'
   solve Procrustes between {C_ir} and {C_ir'}
   output (R_i^(0), t_i^(0))
```

## 9. 方法特点与限制

### 优点
- 保留了 BO-EPNP 最核心的控制点参数化思想
- 多机耦合仅发生在控制点层，形式清楚
- 消息定义简单，两节点之间只传 `12` 维控制点向量
- 很适合作为后端 GN-BP 之前的几何初始化器

### 限制
- 该前端是线性化初始化器，不适合直接替代后端非线性优化
- 若锚点边过少、邻接图弱连通或几何近退化，则 `H_i` 会病态
- 图中环较多时，Jacobi 迭代可能收敛慢，需要阻尼或切换到更强的后端

## 10. 与当前工程的对应关系

可直接复用：
- `src/bo_epnp_solver.py`
  - 控制点参数化思路
  - sphere-aware 投影构造
  - Procrustes 恢复位姿
- `docs/ml_refinement_algorithm.md`
  - 后端姿态精化的角度残差定义
- `docs/gn_bp_multi_agent_bearing_solver_design.md`
  - 前端初值后的 GN-BP 后端设计

建议后续实现新增：
- `src/distributed_bo_epnp_frontend.py`
- `src/multi_agent_control_point_graph.py`
- `src/test_multi_agent_bo_epnp_frontend.py`


## 三机Barycentric关系笔记

### 定义
## 三机 Barycentric 权重与矩阵公式

### Barycentric 坐标

- 对节点 B：
$$
P_A^B = \sum_i \alpha_{A,i}^B \, C_{A,i}^B
$$

$$
P_C^B = \sum_i \alpha_{C,i}^B \, C_{C,i}^B
$$
其中$C_{C,i}^B$ 为控制点坐标，$P_A^B$表示A在B的本地坐标系下的坐标，$P_C^B$ 表示C在B的本地坐标系下的坐标
- 对节点 C：

$$
M_A^{B} = \alpha_{A,i}^B \otimes \mathbf{I}_3
$$
$$
M_C^{B} = \alpha_{C,i}^B \otimes \mathbf{I}_3
$$
在B的视角下，A的角度观测为：$b_{AB}$，C的角度观测为:$b_{CB}$，同理定义C视角下的观测


定义控制点堆叠矩阵
$$C_A^{B}=
\begin{bmatrix}C_{A,1}^B
 \\C_{A,2}^B
 \\C_{A,3}^B
 \\C_{A,4}^B
\end{bmatrix}
$$
对于三机问题在同一时刻存在四对观测向量，可以得到八个约束
$$
\begin{aligned}
\\
[b_{AB}]_\times M_A^{B}C_A^{B}=[b_{AB}]_\times P_B^{B} \\
[b_{CB}]_\times M_C^{B}C_C^{B}=[b_{CB}]_\times P_B^{B} \\
[b_{AC}]_\times M_A^{C}C_A^{C}=[b_{AC}]_\times P_C^{C} \\
[b_{BC}]_\times M_B^{C}C_B^{C}=[b_{BC}]_\times P_C^{C}
\end{aligned}
$$
同时，在B本地坐标系下BC的距离在C本地坐标系BC的距离下是相同的，
$$
\left \| M_C^{B}C_C^{B}-P_B^{B}\right \| _{2}=\left \|M_B^{C}C_B^{C}-P_C^{C}\right \|_{2}
$$
在B本地坐标系下，AC的距离与C本地坐标系下AC的距离是相同的
$$
\left \| M_A^{C}C_A^{C}-P_C^{C}\right \| _{2}=\left \|M_A^{B}C_A^{B}-M_C^{B}C_C^{B}\right \|_{2}
$$
在C本地坐标系下，AB的距离与B本地坐标系下AB的距离是相同的
$$
\left \| M_A^{B}C_A^{B}-P_B^{B}\right \| _{2}=\left \|M_B^{C}C_B^{C}-M_A^{C}C_A^{C}\right \|_{2}
$$
将四对观测向量写成线性方程，最小化方程的残差作为目标函数，三个距离方程作为约束，组成一个QCQP问题
### 姿态变换关系
- 节点 C 相对于 B 的旋转和平移：
$$
P_C = R_C^B \, P_B
$$
- 节点 B 相对于 C 的旋转和平移：

$$
P_B = R_B^C \, P_C \quad \Rightarrow \quad P_B = R_B^C \, R_C^B \, P_B
$$



- 定义Barycentric坐标计算
- 边观测：
$$
C_B = (C_{BA_1}^{B_3},\ C_{BC_1}^{B_3}) ,\quad
C_C = (C_{CA_1}^{C_3},\ C_{CB_1}^{C_3})
$$


- 共识矩阵：
$$
M_A^{B_3} = d_{A,i} \otimes \mathbf{1}_3 ,\quad
M_C^{B_3} = d_{C,i} \otimes \mathbf{1}_3
$$


- Barycentric 权重：
$$
P_B = \sum \alpha_{ij} C_{A,i}^{B_3}, \quad
P_C = \sum \alpha_{i} C_{B,i}^{C_3}
$$


### Barycentric坐标计算

$$
\begin{aligned}
M_A^{B_3} \cdot P_C &= [b_{AC}] \times P_C \\
M_C^{B_3} \cdot P_C &= [b_{AC}] \times P_C \\
M_B^{B_3} &= d_{A,i} \otimes \mathbf{1}_3 \\
M_C^{B_3} &= d_{C,i} \otimes \mathbf{1}_3
\end{aligned}
$$

### 姿态与位置关系
$$
P_C = R_C^{B} P_B
$$
$$
P_B = R_B^C P_C \quad \Rightarrow \quad P_B = R_B^C \, R_C^B \, P_B
$$


### 注

- 笔记中提到：多个无人机间观测，通过边共识 $b_{BC}, b_{CB}$ 构建 barycentric 矩阵；
- 利用这些矩阵可计算相对旋转和平移。