 # 传统EPnP算法
 ## 求解控制点在摄像机坐标下的坐标

设 $K$ 是摄像头的内参矩阵，可以通过标定获得。$\{u_i\}_{i=1,\cdots,n}$ 是参考点 $\{p_i\}_{i=1,\cdots,n}$ 的 2D 投影，那么
$$
\forall i,\quad 
w_i
\begin{bmatrix}
u_i\\
1
\end{bmatrix}
=K p_i^{c}
=K \sum_{j=1}^{4}\alpha_{ij} c_j^{c}
$$
用$$ c_j^{c}= \begin{bmatrix}
x_j^{c},
y_j^{c},
z_j^{c}
\end{bmatrix}
^{T}
$$
（回顾：前文已经提到 $c_j^{c}$ 这些坐标是非齐次坐标）代入上式，并且把 $K$ 写成焦距 $f_u,f_v$ 和光心 $(u_c,v_c)$ 的形式：
$$
\forall i,\quad 
w_i
\begin{bmatrix}
u_i\\
v_i\\
1
\end{bmatrix}
=
\begin{bmatrix}
f_u & 0 & u_c\\
0 & f_v & v_c\\
0 & 0 & 1
\end{bmatrix}
\sum_{j=1}^{4}\alpha_{ij}
\begin{bmatrix}
x_j^{c}\\
y_j^{c}\\
z_j^{c}
\end{bmatrix}
$$
从上式可以得到两个线性方程：
$$
\sum_{j=1}^{4}\alpha_{ij} f_u x_j^{c} \;+\; \alpha_{ij}(u_c-u_i)\, z_j^{c}=0
$$
$$
\sum_{j=1}^{4}\alpha_{ij} f_v y_j^{c} \;+\; \alpha_{ij}(v_c-v_i)\, z_j^{c}=0
$$
把所有 $n$ 个点串联起来，我们可以得到一个线性方程组：
$$
Mx=0
$$
其中
$$
x=\begin{bmatrix}
c_1^{cT},\; c_2^{cT},\; c_3^{cT},\; c_4^{cT}
\end{bmatrix}
^{T},
$$
$x$ 就是控制点在摄像头坐标系下的坐标，显然这是一个 $12\times 1$ 的向量，且 $x$ 在 $M$ 的右零空间中，或者 $x\in\ker(M)$：
$$
x=\sum_{i=1}^{N}\beta_i v_i
$$
上述中，$v_i$ 是 $M$ 的 $N$ 个零特征值对应的 $N$ 个特征向量。对于第 $i$ 个控制点：

$$
c_i^c=\sum_{k=1}^{N}\beta_k\,v_k^{[i]}
$$

上述中，$v_k^{[i]}$ 是特征向量 $v_k$ 的第 $i$ 个 $3\times 1$ sub-vector。

我们可以计算 $M^TM$ 的特征向量得到 $v_i$，但还需要求出 $\{\beta_i\}_{i=1,\cdots,N}$。

根据仿真发现 $M^TM$ 为 0 的特征值的个数和摄像头的焦距有关（注意：是和焦距有关，而不是和参考点的位置有关！），EPnP 算法建议只考虑 $N=1,2,3,4$ 的情况。摄像头的外参描述的只是坐标变换，不会改变控制点之间的距离，

$$
\|c_i^c-c_j^c\|^2=\|c_i^w-c_j^w\|^2,
$$

即

$$
\left\|
\sum_{k=1}^{N}\beta_k v_k^{[i]}
-
\sum_{k=1}^{N}\beta_k v_k^{[j]}
\right\|^2
=
\|c_i^w-c_j^w\|^2
$$

这是一个关于 $\{\beta_k\}_{k=1,\cdots,N}$ 的二次方程，这个方程的特点是没有关于 $\{\beta_k\}_{k=1,\cdots,N}$ 的一次项。如果将这些二次项 $\beta_i\beta_j$ 变量替换为 $\beta_{ij}$，那么该方程是 $\{\beta_{ij}\}_{i,j=1,\cdots,N}$ 的线性方程了。对于 4 个控制点，可以得到 $C_4^2=6$ 个这样的方程。

如果不挖掘任何附加条件，$N$ 取不同值的时候，新的线性未知数的个数分别为：

- $N=1$，线性未知数的个数为 1；
- $N=2$，线性未知数的个数为 3；
- $N=3$，线性未知数的个数为 6；
- $N=4$，线性未知数的个数为 10；

$N=4$ 的时候，未知数的个数多于方程的个数。注意到 $\beta_{ab}\beta_{cd}=\beta_{a'b'}\beta_{c'd'}$，其中 $\{a',b',c',d'\}$ 是 $\{a,b,c,d\}$ 的一个排列，我们可以减少未知数的个数。举例，如果我们求出了 $\beta_{11}$、$\beta_{12}$、$\beta_{13}$，那么我们可以得到

$$
\beta_{23}=\frac{\beta_{12}\beta_{13}}{\beta_{11}}.
$$

这样，即使对于 $N=4$，我们也可以求解出 $\{\beta_{ij}\}_{i,j=1,\cdots,N}$。
## 高斯-牛顿最优化

优化的目标函数为：

$$
\mathrm{Error}(\beta)
=
\sum_{(i,j)\ \mathrm{s.t.}\ i<j}
\left(
\|c_i^c-c_j^c\|^2-\|c_i^w-c_j^w\|^2
\right)^2
$$

这是一个无约束的非线性最优化问题，简单。
Mx=0的解表示X属于M的零空间，在控制点的正常解算不退化的情况下，x有解，表示M的秩最大，为11，如果M满秩则无解。若M的零空间增大，即β_i数量增多，则可以认为M的零特征值对应的特征向量增加，表示在这些特征向量方向的约束不足。同时四个特征点可以带来C42=6个距离约束，最多可以保证i=4的情况下可以解算出β，β数量增多则方程组约束不足，会得到一堆通解。也可以理解为在这种情况下控制点的选择不合理，出现了控制点共面的情况，进一步可以认为观测量的可观性较差，无法形成足够的约束

在 EPnP 中，由像平面投影约束构建的齐次线性系统 $Mx=0$ 用于求解相机坐标系下的控制点堆叠向量 $x$。在理想非退化情形下，$\mathrm{rank}(M)=11$，因此 $\mathrm{null}(M)$ 近似为一维，$x$（除尺度外）可唯一确定；当观测几何退化或噪声较大时，$M^TM$ 的近零特征值个数增加，导致零空间维数扩展为 $N>1$，即 $x=\sum_{k=1}^{N}\beta_k v_k$，表明仅由投影方程提供的独立约束不足，需要引入额外的刚体不变约束来确定 $\{\beta_k\}$。EPnP 采用四个控制点并利用两两距离不变性可提供至多 $\binom{4}{2}=6$ 条独立距离约束；在将二次约束线性化（以 $\beta_i\beta_j$ 为新变量）后，未知量数量随 $N$ 增长迅速（$N=1,2,3,4$ 时分别为 1、3、6、10），使得 $N>4$ 情形显著欠定且数值不稳定，而 $N=4$ 亦需结合额外代数关系或非线性最小二乘进一步收敛。因此，EPnP 实践中通常仅枚举并处理 $N\le 4$ 的情况，以在可解性与稳定性之间取得平衡。

## EPnP 距离约束 GN/LM：疑问总结 + 迭代公式整理

### 1) 你的疑问（总结）

你看到推导里写：
$$
r=\sum_{i<j} r_{ij},
\qquad 
\frac{\partial r}{\partial \beta_k}=\sum_{i<j}\frac{\partial r_{ij}}{\partial \beta_k}.
$$

所以你直觉认为：应该把每个距离约束对 $\beta_k$ 的偏导“叠加”起来，得到一个梯度向量 $j$。

但代码里采用的是：
- 把每一个距离约束 $r_{ij}$ 单独作为残差向量的一行；
- 用 `J[row, :] = 2.0 * (dv @ dij)` 生成雅可比矩阵的一整行。

你的疑问是：为什么不是把所有偏导求和？为什么要“每个约束一行”？

---

### 2) 关键区别：标量残差 vs 残差向量（GN/LM 的标准形式）

### 2.1 你写的“叠加导数”对应标量函数
如果你先把所有距离残差相加成一个标量：
$$
\tilde r(\beta)=\sum_{i<j} r_{ij}(\beta),
$$

那么确实有：
$$
\frac{\partial \tilde r}{\partial \beta_k}=\sum_{i<j}\frac{\partial r_{ij}}{\partial \beta_k}.
$$

这是**标量函数**的导数/梯度分量。

### 2.2 GN/LM 通常最小化的是“残差向量的平方和” 重点为 目标函数是平方和形式
EPnP 距离约束的常用目标是：
$$
\min_\beta \ \|\mathbf r(\beta)\|^2
=\sum_{i<j} r_{ij}(\beta)^2,
$$

其中残差向量（4 个控制点共有 6 对）：
$$
\mathbf r(\beta)=
\begin{bmatrix}
r_{01}(\beta)\\
r_{02}(\beta)\\
r_{03}(\beta)\\
r_{12}(\beta)\\
r_{13}(\beta)\\
r_{23}(\beta)
\end{bmatrix}
\in\mathbb R^6.
$$


这时需要的是向量函数 $\mathbf r(\beta)$ 的雅可比矩阵：
$$
J(\beta)=\frac{\partial \mathbf r}{\partial \beta}\in\mathbb R^{6\times N},
$$

即每一行对应一个残差 $r_{ij}$，所以“每个距离约束一行”是 GN/LM 的标准构造方式。

---

### 3) 残差与雅可比（核心公式）

### 3.1 控制点相机坐标表示
令 $v_k$ 为近零空间特征向量（$k=1,\dots,N$），其第 $i$ 个 $3\times1$ 子向量为：
$$
v_k^{[i]}\in\mathbb R^3,\quad i=1,\dots,4.
$$

则第 $i$ 个控制点在相机坐标系：
$$
c_i^c(\beta)=\sum_{k=1}^{N}\beta_k\,v_k^{[i]}.
$$


### 3.2 单个距离残差
对任意 $(i,j)$：
$$
d_{ij}(\beta)=c_i^c(\beta)-c_j^c(\beta),
$$

$$
r_{ij}(\beta)=\|d_{ij}(\beta)\|^2-\|c_i^w-c_j^w\|^2.
$$

展开 $d_{ij}$：
$$
d_{ij}(\beta)=\sum_{k=1}^{N}\beta_k\big(v_k^{[i]}-v_k^{[j]}\big),
$$

并记：
$$
\delta v_k^{[ij]}=v_k^{[i]}-v_k^{[j]}\in\mathbb R^3.
$$


### 3.3 雅可比元素
由
$$
r_{ij}(\beta)=d_{ij}^T d_{ij}-\|c_i^w-c_j^w\|^2
$$

可得：
$$
\frac{\partial r_{ij}}{\partial \beta_k}
=2\,d_{ij}(\beta)^T\,\delta v_k^{[ij]}.
$$

因此：
$$
J_{(ij),k}=2\,d_{ij}(\beta)\cdot \delta v_k^{[ij]}.
$$


---

### 4) 为什么代码是 `J[row, :] = 2.0 * (dv @ dij)`？

对固定 pair $(i,j)$：
- 堆叠
$$
dv=
\begin{bmatrix}
(\delta v_1^{[ij]})^T\\
\vdots\\
(\delta v_N^{[ij]})^T
\end{bmatrix}
\in\mathbb R^{N\times 3},
\quad
dij=d_{ij}(\beta)\in\mathbb R^{3}.
$$

则
$$
dv\cdot dij\in\mathbb R^{N}
$$

第 $k$ 项就是 $\delta v_k^{[ij]}\cdot d_{ij}$。

所以：
$$
J_{(ij),:}=2\,(dv\cdot dij)^T,
$$

对应代码一行写法：`J[row, :] = 2.0 * (dv @ dij)`。

---

### 5) GN/LM 迭代求解过程（可直接对应实现）

#### 5.1 构造残差向量与雅可比矩阵
$$
\mathbf r(\beta)=\begin{bmatrix}r_{01}&r_{02}&r_{03}&r_{12}&r_{13}&r_{23}\end{bmatrix}^T,
\quad
J(\beta)=\frac{\partial \mathbf r}{\partial \beta}.
$$


#### 5.2 线性化（GN 基本形式）
$$
\mathbf r(\beta+\Delta\beta)\approx \mathbf r(\beta)+J(\beta)\Delta\beta.
$$


#### 5.3 Gauss–Newton 步长
$$
(J^T J)\Delta\beta=-J^T\mathbf r.
$$


#### 5.4 Levenberg–Marquardt（更稳健）
$$
(J^T J+\lambda I)\Delta\beta=-J^T\mathbf r,
$$

其中 $\lambda>0$ 为阻尼项。

#### 5.5 更新与停止
更新：
$$
\beta\leftarrow\beta+\Delta\beta.
$$

停止条件（常用任一）：
- $\|\Delta\beta\|<\varepsilon_{\text{step}}$
- $\|\mathbf r(\beta)\|^2<\varepsilon_{\text{cost}}$
- 达到最大迭代次数

#### 5.6 输出控制点相机坐标
$$
c_i^c(\beta)=\sum_{k=1}^{N}\beta_k\,v_k^{[i]},\quad i=1,\dots,4.
$$


---

### 6) 你“叠加导数”的位置（最简对照）

- 你写的 $\sum_{i<j}\frac{\partial r_{ij}}{\partial \beta_k}$ 属于“标量残差/标量函数”的导数。
- GN/LM 用的是残差向量 $\mathbf r$ 的雅可比 $J$，并通过 $J^T\mathbf r$ 隐式完成所有约束的综合。
- 若把目标写成标量
$$
F(\beta)=\sum_{i<j} r_{ij}(\beta)^2,
$$

则梯度为：
$$
\nabla F=2J^T\mathbf r.
$$

这里的“求和叠加”已经包含在矩阵乘法中。

# BO-EPnP算法解析
~~~python
    def solve(p1: np.ndarray, p2: np.ndarray, bearing: np.ndarray, sol_iter: bool = True, enforce_manifold: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        # Validate inputs
        bgpnp.validate_inputs(p1, p2, bearing, min_points=4)
        
        M, b, Alph, Cw = bgpnp.prepare_data(p1, bearing, p2)
        possible_dims = 4
        # Use regularized kernel computation for better noise robustness when enforce_manifold is True
        Km = bgpnp.kernel_noise(M, b, dimker=possible_dims, use_regularization=enforce_manifold)
        R, t, err, _ = bgpnp.KernelPnP(Cw, Km, dims=4, sol_iter=sol_iter, enforce_manifold=enforce_manifold)

        return R, t, err
~~~
## 1.prepare_data
- 计算控制点对应的系数

给定：

- 世界坐标下的 3D 点集  
$$
X_w = \{x_i^w\}_{i=1}^n,\quad x_i^w\in\mathbb{R}^3
$$

- 世界坐标下的 4 个控制点  
$$
C_w = \{c_j^w\}_{j=1}^4,\quad c_j^w\in\mathbb{R}^3
$$


EPnP 中每个 3D 点可以表示为控制点的仿射（重心）组合：
$$
x_i^w=\sum_{j=1}^{4}\alpha_{ij}\,c_j^w,
\qquad
\sum_{j=1}^{4}\alpha_{ij}=1.
$$

对每个点写成齐次形式：
$$
\begin{bmatrix}
x_i^w\\
1
\end{bmatrix}
=
\sum_{j=1}^{4}\alpha_{ij}

\begin{bmatrix}
c_j^w\\
1
\end{bmatrix}
$$


把 4 个控制点堆成一个 $4\times 4$ 矩阵（代码里的 `C`）：
$$
C=
\begin{bmatrix}
c_1^w & c_2^w & c_3^w & c_4^w\\
1 & 1 & 1 & 1
\end{bmatrix}
=
\begin{bmatrix}
(C_w)^T\\
\mathbf{1}^T
\end{bmatrix}
\in\mathbb{R}^{4\times 4}.
$$


把所有点堆成一个 $4\times n$ 矩阵（代码里的 `X`）：
$$
X=
\begin{bmatrix}
x_1^w & x_2^w & \cdots & x_n^w\\
1 & 1 & \cdots & 1
\end{bmatrix}
=
\begin{bmatrix}
(X_w)^T\\
\mathbf{1}^T
\end{bmatrix}
\in\mathbb{R}^{4\times n}.
$$


令
$$
A=
\begin{bmatrix}
\alpha_{11} & \alpha_{21} & \cdots & \alpha_{n1}\\
\alpha_{12} & \alpha_{22} & \cdots & \alpha_{n2}\\
\alpha_{13} & \alpha_{23} & \cdots & \alpha_{n3}\\
\alpha_{14} & \alpha_{24} & \cdots & \alpha_{n4}
\end{bmatrix}
\in\mathbb{R}^{4\times n},
$$

即第 $i$ 列对应点 $x_i^w$ 的 4 个系数。

则有矩阵方程：
$$
C\,A = X.
$$


因此：
$$
A = C^{-1}X,
\quad\text{或用线性方程求解}\quad
A = \mathrm{solve}(C,X).
$$


最后将其转置得到代码返回的 `Alph`（$n\times 4$）：
$$
\mathrm{Alph}=A^T
\begin{bmatrix}
\alpha_{11} & \alpha_{12} & \alpha_{13} & \alpha_{14}\\
\alpha_{21} & \alpha_{22} & \alpha_{23} & \alpha_{24}\\
\vdots & \vdots & \vdots & \vdots\\
\alpha_{n1} & \alpha_{n2} & \alpha_{n3} & \alpha_{n4}
\end{bmatrix}
\in\mathbb{R}^{n\times 4}.
$$
 计算M矩阵和b矩阵

给定第 $i$ 个观测（$i=1,\dots,n$）：

- bearing 向量（单位方向）  
$$
\mathbf u_i \in \mathbb{R}^3
$$

- 目标点（或 3D 点）  
$$
\mathbf p_{2,i}\in\mathbb{R}^3
$$

- 该点相对于 4 个控制点的重心系数  
$$
\boldsymbol{\alpha}_i=
 \begin{bmatrix}
 \alpha_{i1}&\alpha_{i2}&\alpha_{i3}&\alpha_{i4}
 \end{bmatrix}
  \in\mathbb{R}^{1\times 4}
$$
叉乘矩阵（反对称矩阵）

令 $[\mathbf u_i]_\times\in\mathbb{R}^{3\times 3}$ 为 $\mathbf u_i$ 的斜对称矩阵（使得 $[\mathbf u_i]_\times \mathbf x = \mathbf u_i \times \mathbf x$）：

$$
[\mathbf u_i]_\times=
\begin{bmatrix}
0 & -u_{iz} & u_{iy}\\
u_{iz} & 0 & -u_{ix}\\
-u_{iy} & u_{ix} & 0
\end{bmatrix}
$$
代码中的 `S[i] = skew_symmetric_matrix(bearing_i)` 对应的就是 $[\mathbf u_i]_\times$。

---

 控制点按列堆叠成为一个向量

把 4 个控制点在相机坐标系下的坐标堆成一个 $12\times 1$ 向量：
$$
\mathbf x=
\begin{bmatrix}
(\mathbf c_1^c)^T & (\mathbf c_2^c)^T & (\mathbf c_3^c)^T & (\mathbf c_4^c)^T
\end{bmatrix}
^T
\in\mathbb{R}^{12}.
$$

 每个观测的线性方程块

EPnP / bearing-only 形式中，利用“共线约束”（观测方向与向量差平行）可写成叉乘为零：
$$
\mathbf u_i \times \left(\sum_{j=1}^{4}\alpha_{ij}\mathbf c_j^c - \mathbf p_{2,i}\right)=\mathbf 0.
$$
等价为矩阵形式：
$$
[\mathbf u_i]_\times \left(\sum_{j=1}^{4}\alpha_{ij}\mathbf c_j^c\right)
=
[\mathbf u_i]_\times\,\mathbf p_{2,i}.
$$
令
$$
A_i=\boldsymbol{\alpha}_i\otimes I_3 \in\mathbb{R}^{3\times 12},
$$
$$
A_i=
\begin{bmatrix}
\alpha_{11} & 0 & 0 & \alpha_{12} & 0 & 0 & \alpha_{13} & 0 & 0 & \alpha_{14} & 0 & 0 \\
 0 &\alpha_{11} & 0 &  0 &\alpha_{12} & 0 &  0 &\alpha_{13} & 0 &  0 &\alpha_{14} & 0 \\
0 & 0 & \alpha_{11} & 0 & 0 & \alpha_{12} & 0 & 0 & \alpha_{13} & 0 & 0 & \alpha_{14} \\
\end{bmatrix}
$$

其中 $\otimes$ 为 Kronecker 积，$I_3$ 为 $3\times 3$ 单位矩阵。则有
$$
\sum_{j=1}^{4}\alpha_{ij}\mathbf c_j^c = A_i\,\mathbf x.
$$
因此单个观测给出：
$$
[\mathbf u_i]_\times A_i\,\mathbf x
=
[\mathbf u_i]_\times\,\mathbf p_{2,i}.
$$
叠加得到整体线性系统 $M\mathbf x=b$

定义每个观测的块矩阵与块向量：
$$
M_i = [\mathbf u_i]_\times A_i \in\mathbb{R}^{3\times 12},\qquad
b_i = [\mathbf u_i]_\times\,\mathbf p_{2,i}\in\mathbb{R}^{3}.
$$
把 $n$ 个观测按行堆叠：
$$
M=
\begin{bmatrix}
M_1\\
M_2\\
\vdots\\
M_n
\end{bmatrix}
\in\mathbb{R}^{3n\times 12},
\qquad
b=
\begin{bmatrix}
b_1\\
b_2\\
\vdots\\
b_n
\end{bmatrix}
\in\mathbb{R}^{3n}.
$$
最终得到：
$$
M\mathbf x=b.
$$
 与代码逐项对应

- `bgpnp.kron_A_N(Alph, 3)` 生成的就是把所有 $A_i=\boldsymbol{\alpha}_i\otimes I_3$ 纵向堆叠后的矩阵（形状 $3n\times 12$）。
- `S[i]` 是 $[\mathbf u_i]_\times$。
- `M = einsum(...)` 对应逐块左乘：
$$
M_i = [\mathbf u_i]_\times A_i.
$$
- `b = einsum(...)` 对应：
$$
b_i = [\mathbf u_i]_\times\,\mathbf p_{2,i}.
$$
## 2.`kernel_noise(M,b)` 
### 2.1方法流程 
给定线性系统
$$
M x \approx b,\qquad M\in\mathbb{R}^{3n\times 12},\; b\in\mathbb{R}^{3n},
$$

函数返回一个“核噪声矩阵”
$$
K\in\mathbb{R}^{12\times d},\quad d=\texttt{dimker}.
$$
---
SVD 分解与零空间基（前 $d-1$ 列）
对 $M$ 做（截断）奇异值分解：
$$
M = U\,\Sigma\,V^{\mathsf T},
$$
其中
$$
U\in\mathbb{R}^{3n\times r},\;
\Sigma=\mathrm{diag}(\sigma_1,\dots,\sigma_r)\in\mathbb{R}^{r\times r},\;
V\in\mathbb{R}^{12\times r},\;
r=\min(3n,12),
$$

且
$$
\sigma_1\ge \sigma_2\ge \cdots \ge \sigma_r\ge 0.
$$
取 $V$ 的末尾 $d-1$ 个右奇异向量作为近似零空间基（对应代码 `V[:, -dimker+1:]`）：
$$
K_{(:,\,1:d-1)}=\big[\,v_{r-d+2},\; v_{r-d+3},\;\dots,\; v_r\,\big],
$$

其中 $v_j$ 表示 $V$ 的第 $j$ 列（右奇异向量）。

> 直观含义：当 $M$ 近似秩亏时，这些向量对应较小奇异值方向，可视作 $M$ 的“近零空间”。

---
最后一列：最小二乘解（未正则）

最后一列 $K_{(:,d)}$ 默认用最小二乘解：
$$
K_{(:,d)} = x_{\text{LS}}
= \arg\min_{x\in\mathbb{R}^{12}}\|Mx-b\|_2^2.
$$
其正规方程形式为：
$$
M^{\mathsf T}M\,x_{\text{LS}} = M^{\mathsf T}b.
$$
---
可选：Tikhonov（岭）正则化提高抗噪性

先定义条件数（代码用最大/最小奇异值近似）：
$$
\kappa(M)\approx \frac{\sigma_1}{\sigma_r+\varepsilon},
\quad \varepsilon\approx 10^{-15}.
$$
当 $\kappa(M)$ 很大（或用户指定 $\lambda>0$）时，使用 Tikhonov 正则化解：
$$
x_{\lambda}
= \arg\min_{x\in\mathbb{R}^{12}}
\Big(\|Mx-b\|_2^2 + \lambda\|x\|_2^2\Big),
$$

其闭式解满足：
$$
\left(M^{\mathsf T}M+\lambda I\right)x_{\lambda}=M^{\mathsf T}b.
$$
于是
$$
K_{(:,d)} = x_{\lambda}.
$$
3.1 自适应正则系数（代码的默认策略）
当用户未指定 $\lambda$（即 `reg_lambda=0`）时，代码按最小“非零奇异值”比例设定：
$$
\lambda =
10^{-3}\cdot \min\{\sigma_i\mid \sigma_i>10^{-10}\},
$$

若不存在满足阈值的奇异值，则取一个极小常数（代码为 $10^{-8}$）。

---
 输出矩阵的整体结构

最终输出：
$$
K=\begin{bmatrix}
\;| & & | & |\\
k_1 & \cdots & k_{d-1} & k_d\\
\;| & & | & |
\end{bmatrix}\in\mathbb{R}^{12\times d},
$$

其中
- $k_1,\dots,k_{d-1}$ 来自 $M$ 的近零空间（右奇异向量）；
- $k_d$ 是解 $Mx\approx b$ 的最小二乘（或正则化最小二乘）解。

##  3.`KernelPnP`

1. 首先选择特解作为控制点在相机坐标系下的初值
2. 将特解带入Procrustes正交化，即将世界坐标系下的控制点与相机系下的控制点一一对应求解旋转矩阵R与平移向量t
- 补充Procrustes方法求解旋转矩阵，尺度和平移向量的公式
1.  得到R与t之后再利用世界坐标系下的控制点反向求解相机系下的坐标点A，再去解空间中寻找与A最相近的点，解完这个优化问题后再继续套到Procrustes中求解最优的R与t，交替最小化得到最优的R，t
 - 补充交替最小化的公式
---
问题设置与符号约定

- 设世界坐标系下点集（或控制点集）为：
$$
P = [p_1,\dots,p_n] \in \mathbb{R}^{3\times n}
$$

  在代码中对应 `X['P'] = Cw.T`。

- Kernel 矩阵 $K_m$ 描述“解空间/可行子空间”（例如由核空间基向量 + 特解组成），使得某个目标点集（相机系或中间变量）满足：
$$
\mathrm{vec}(V) = K_m \alpha
$$

  其中 $\alpha \in \mathbb{R}^{d}$ 为低维系数向量（代码中 `abcd`），$d=\text{dims}$。

- 目标：求一个位姿变换（以及必要时的尺度）使得世界点与相机系点一致：
$$
p_i \approx R^T v_i + t \quad (\text{或 } p_i \approx b R^T v_i + t)
$$

  其中 $R\in SO(3)$，$t\in\mathbb{R}^3$，$b\in\mathbb{R}$。

---
 Step-1：选择特解作为相机系控制点（初值）

特解的来源
Kernel 矩阵 $K_m$ 的**最后一列**通常存放一个最小二乘得到的特解（particular solution），例如：
$$
c_0 = \arg\min_c \|Mc-b\|^2
$$

或带正则的解。

在 KernelPnP 中的使用
代码：
- `vK = Km[:, -1].reshape(dims, -1).T`

含义：
- 取特解向量并 reshape 成控制点（或点集）形式，作为**相机坐标系下点集的初始估计**：
$$
V^{(0)} \leftarrow \mathrm{unvec}(K_m[:, -1])
$$


---
Step-2：Procrustes 正交化（由一一对应的控制点求 $R,t$ 初值）

在初值点集 $V^{(0)}$ 与世界点集 $P$ 已建立一一对应后，使用 Procrustes（相似变换对齐）求 $R$、尺度 $b$、平移。
 3.1 中心化（消去平移）
$$
\bar p = \frac{1}{n}\sum_{i=1}^n p_i,\qquad
\bar v = \frac{1}{n}\sum_{i=1}^n v_i
$$

$$
P_c = P - \bar p \mathbf{1}^T,\qquad
V_c = V - \bar v \mathbf{1}^T
$$
3.2 求旋转矩阵（Kabsch / Orthogonal Procrustes）
构造相关矩阵：
$$
H = P_c V_c^T
$$

对 $H$ 做 SVD：
$$
H = U\Sigma V^T
$$

旋转解（强制 $SO(3)$）：
$$
R = V
\begin{bmatrix}
1&0&0\\
0&1&0\\
0&0&\mathrm{sign}(\det(VU^T))
\end{bmatrix}

U^T
$$

该修正保证：
$$
R^TR=I,\quad \det(R)=+1 \Rightarrow R\in SO(3)
$$
 3.3 求尺度（相似变换版本，可选）
在允许全局尺度 $b$ 时，固定 $R$ 后的最优尺度为：
$$
b^\* = \frac{\mathrm{tr}(P_c^T R^T V_c)}{\|V_c\|_F^2}
$$

工程实现中常见等价写法（与归一化策略相关）。
 3.4 求平移
$$
t = \bar v - R\,\bar p \quad (\text{若用 } v \approx R p + t)
$$

或在你代码的形式中（对齐关系写成 $p \approx b R^T v + c$）：
$$
c = \bar p - b R^T \bar v
$$


> 在你的实现里，`myProcrustes` 返回 `(R, b, mc)`，其中 `mc` 是将平移向量复制成矩阵以便广播（每列相同）。

---
4. Step-3：交替最小化迭代（投影到解空间 + Procrustes 更新）

得到初值 $(R^{(0)}, mc^{(0)})$ 后，KernelPnP 进入迭代 refine，核心思想是**在两个集合之间来回投影**：

- 集合 A：由当前位姿变换，从本地系转换到全局系的控制点向量 $A(R,mc)$
- 集合 V：由 Kernel 子空间算出的控制点向量 $V(\alpha)$，即控制点的系数$\alpha$

 4.1 由当前 $R,mc$ 反求“相机系控制点堆叠成的向量$A$
代码：
- `A = R @ (-mc + P)`

含义（抽象表达）：
$$
A^{(k)} = A(R^{(k)}, mc^{(k)})
$$
 4.2 在解空间Km（最小奇异值对应的向量加上一个由最小二乘解得的特解）中寻找与 $A$ 最相近的点（Kernel 投影 / 最小二乘），因为控制点堆叠向量A的初值是一个最小二乘解，需要精细化迭代，所以在做完正交化之后，再从映射回子空间，寻找堆叠向量在子空间中最接近的坐标，更新控制点堆叠向量，再由更新后的向量继续计算旋转矩阵
 
代码：

- `abcd = lstsq(Km, vec(A))`
- `newV = unvec(Km @ abcd)`

对应数学问题：
$$
\alpha^{(k+1)}=\arg\min_{\alpha}\ \|\mathrm(A^{(k)})-K_m\alpha\|_2^2
$$

得到：
$$
V^{(k+1)}=\mathrm(K_m \alpha^{(k+1)})
$$


> 这一步的意义：一般情况下 $A^{(k)} \notin \mathrm{span}(K_m)$，因此 `Km@abcd` 得到的是 **$A^{(k)}$ 在子空间 $\mathrm{span}(K_m)$ 上的最佳近似投影**，并不等于 $A^{(k)}$。

 4.3 固定 $V^{(k+1)}$，再做一次 Procrustes 更新 $R,mc$
代码：
- `R, b, mc = myProcrustes(X, newV)`

数学形式：
$$
(R^{(k+1)}, b^{(k+1)}, mc^{(k+1)})
=
\arg\min_{R\in SO(3),\,b,\,mc}\ \|R^T(bV^{(k+1)})+mc-P\|_F^2
$$
 4.4 交替最小化（Block Coordinate Descent）的标准公式
将未知量分成两块：$\alpha$ 与 $(R,mc)$。

- **Step A（解 $\alpha$）**：
$$
\alpha^{(k+1)}=\arg\min_{\alpha}\ J(R^{(k)},mc^{(k)},\alpha)
$$
- **Step B（解 $R,mc$）**：
$$
(R^{(k+1)},mc^{(k+1)})=\arg\min_{R,mc}\ J(R,mc,\alpha^{(k+1)})
$$
其中一个典型目标函数可写为：
$$
J(R,mc,\alpha)=\left\|R^T\,V(\alpha)+mc-P\right\|_F^2,\quad
V(\alpha)=\mathrm{unvec}(K_m\alpha)
$$
> 收敛依据：每一步都对某一块变量做最小化，因此目标值通常单调不增并趋于稳定（收敛到局部最优/不动点）。

 4.5 收敛判据（工程实现）
代码中使用：
- 误差上升（且迭代超过若干次）则停止；
- 或误差小于阈值 `tol` 则停止。
---
 最终输出：由 $R$ 与点集均值恢复平移 $t$

代码末尾：
- `mV = mean(solV, axis=1)`
- `T = mV - R @ X['mP']`

对应质心关系：
$$
t = \bar v - R\,\bar p
$$
---
 可选：SO(3) 流形约束（Manifold Projection）

当噪声较大或数值误差累积时，`myProcrustes` 得到的 $R$ 可能出现轻微非正交或 $\det(R)<0$ 的风险。

做法：
- 每次迭代后将 $R$ 投影回 $SO(3)$，典型做法是：
$$
R \leftarrow U V^T,\quad \text{并强制}\ \det(R)=+1
$$
这能提高鲁棒性并减少发散风险。
---
小结
1. **特解初始化**：取 Kernel 矩阵最后一列特解，得到相机系控制点（或点集）初值 $V^{(0)}$。
2. **Procrustes 初值位姿**：用世界系控制点与相机系控制点的一一对应，通过 Procrustes（Kabsch/Umeyama）求得初始 $R^{(0)},t^{(0)}$（以及尺度 $b$）。
3. **交替最小化 refine**：
   - 用当前 $R,mc$ 将世界点反求得到 $A^{(k)}$；
   - 在解空间 $\mathrm{span}(K_m)$ 中求与 $A^{(k)}$ 最接近的点集 $V^{(k+1)}$（最小二乘投影）；
   - 再用 Procrustes 更新 $R^{(k+1)},t^{(k+1)}$；
   - 重复直到收敛，得到最终最优的 $R,t$。
---
