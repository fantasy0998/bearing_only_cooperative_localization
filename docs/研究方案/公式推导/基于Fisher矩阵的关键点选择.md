# Bearing-only 协同定位中的 Fisher 信息矩阵计算方式

本文参照 Russell 等人的 bearing-only 协同定位建模方式，将角度观测作为量测，将无人机相对坐标系之间的旋转和平移作为待估状态；同时借鉴 Papalia 等人利用 Fisher 信息矩阵衡量协同定位可观性的方法，将每一个 bearing 观测对相对位姿估计的信息贡献累加，构造 bearing-only 条件下的 Fisher 信息矩阵。

## 1. 相对位姿状态建模

设第 $k$ 个候选观测时刻，GPS 可用无人机 A 在全局坐标系 $A_1$ 下的位置为
$$
a_k=p_A^{A_1}(k)\in\mathbb{R}^3
$$
GPS 拒止无人机 B 在自身惯导坐标系 $B_2$ 下的位置为
$$
b_k=p_B^{B_2}(k)\in\mathbb{R}^3
$$
需要估计从 $A_1$ 到 $B_2$ 的相对坐标变换
$$
R\in SO(3),\qquad t\in\mathbb{R}^3
$$
则 A 机在 B 机惯导坐标系 $B_2$ 下的位置为
$$
p_A^{B_2}(k)=R a_k+t
$$
因此，B 机指向 A 机的相对向量为
$$
s_k(R,t)=R a_k+t-b_k
$$
bearing-only 观测只给出该相对向量的方向，不提供距离信息。其单位方向向量为
$$
q_k=\frac{s_k}{\|s_k\|}
$$
若用方位角和俯仰角表示 bearing 观测，则
$$
z_k=
\begin{bmatrix}
\theta_k\\
\phi_k
\end{bmatrix}
=
h(s_k)+n_k
$$
其中

$$
n_k\sim\mathcal{N}(0,R_{z,k})
$$
方位角和俯仰角观测函数为

$$
h(s_k)=
\begin{bmatrix}
\operatorname{atan2}(s_{y,k},s_{x,k})\\
\operatorname{atan2}(s_{z,k},\sqrt{s_{x,k}^2+s_{y,k}^2})
\end{bmatrix}
$$
若方位角和俯仰角噪声相互独立，则
$$
R_{z,k}=
\begin{bmatrix}
\sigma_{\theta,k}^2 & 0\\
0 & \sigma_{\phi,k}^2
\end{bmatrix}
$$

---

## 2. 局部误差状态：李代数扰动

虽然 Russell 论文中可以直接将旋转矩阵元素和平移向量写成 12 维未知量

$$
\psi=
\begin{bmatrix}
\operatorname{vec}(R)^T & t^T
\end{bmatrix}^T
\in\mathbb{R}^{12}
$$

但在 Fisher 信息矩阵和可观性分析中，直接使用旋转矩阵的 9 个元素会引入冗余自由度，因为 $SO(3)$ 的真实自由度只有 3。

因此，本文在当前估计值 $(\hat R,\hat t)$ 附近引入 6 维局部误差状态

$$
\delta \xi=
\begin{bmatrix}
\delta \rho\\
\delta t
\end{bmatrix}
\in\mathbb{R}^6
$$

其中

$$
\delta \rho\in\mathfrak{so}(3)
$$

表示旋转的李代数扰动，

$$
\delta t\in\mathbb{R}^3
$$

表示平移扰动。

采用旋转左扰动和平移加性扰动：

$$
R'=\exp([\delta\rho]_\times)\hat R
\approx
(I+[\delta\rho]_\times)\hat R
$$

$$
t'=\hat t+\delta t
$$

于是相对向量的一阶扰动为

$$
\delta s_k
=
-[\hat R a_k]_\times\delta\rho+\delta t
$$

因此

$$
\frac{\partial s_k}{\partial \delta\xi}
=
G_k
$$

其中

$$
G_k=
\begin{bmatrix}
-[\hat R a_k]_\times & I_3
\end{bmatrix}
\in\mathbb{R}^{3\times 6}
$$

这里 $[\cdot]_\times$ 表示反对称矩阵，即对于向量

$$
v=
\begin{bmatrix}
v_x\\
v_y\\
v_z
\end{bmatrix}
$$

有

$$
[v]_\times=
\begin{bmatrix}
0 & -v_z & v_y\\
v_z & 0 & -v_x\\
-v_y & v_x & 0
\end{bmatrix}
$$

---

## 3. 角度观测对相对向量的雅可比

设

$$
s_k=
\begin{bmatrix}
s_{x,k}\\
s_{y,k}\\
s_{z,k}
\end{bmatrix}
$$

定义

$$
r_k=\sqrt{s_{x,k}^2+s_{y,k}^2}
$$

$$
L_k=\|s_k\|=\sqrt{s_{x,k}^2+s_{y,k}^2+s_{z,k}^2}
$$

角度观测函数为

$$
\theta_k=\operatorname{atan2}(s_{y,k},s_{x,k})
$$

$$
\phi_k=\operatorname{atan2}(s_{z,k},r_k)
$$

则角度观测对相对向量 $s_k$ 的雅可比为

$$
J_h(s_k)
=
\frac{\partial h(s_k)}{\partial s_k}
=
\begin{bmatrix}
-\dfrac{s_{y,k}}{s_{x,k}^2+s_{y,k}^2}
&
\dfrac{s_{x,k}}{s_{x,k}^2+s_{y,k}^2}
&
0
\\[8pt]
-\dfrac{s_{x,k}s_{z,k}}{L_k^2 r_k}
&
-\dfrac{s_{y,k}s_{z,k}}{L_k^2 r_k}
&
\dfrac{r_k}{L_k^2}
\end{bmatrix}
$$

因此，bearing 观测对局部误差状态 $\delta\xi$ 的雅可比为

$$
H_k=
\frac{\partial h(s_k)}{\partial \delta\xi}
=
J_h(s_k)G_k
$$

即

$$
H_k
=
J_h(s_k)
\begin{bmatrix}
-[\hat R a_k]_\times & I_3
\end{bmatrix}
$$

其中

$$
H_k\in\mathbb{R}^{2\times 6}
$$

表示第 $k$ 个 bearing 观测对相对旋转和平移局部误差状态的约束。

---

## 4. 单个 bearing 观测的信息贡献

根据 Fisher 信息矩阵的定义，对于高斯角度噪声模型

$$
z_k=h(s_k)+n_k,\qquad n_k\sim\mathcal{N}(0,R_{z,k})
$$

第 $k$ 个 bearing 观测对状态 $\delta\xi$ 的 Fisher 信息贡献为

$$
F_k
=
H_k^T R_{z,k}^{-1}H_k
$$

代入 $H_k=J_h(s_k)G_k$，可得

$$
F_k
=
G_k^T
J_h(s_k)^T
R_{z,k}^{-1}
J_h(s_k)
G_k
$$

即

$$
\boxed{
F_k
=
G_k^T
J_h(s_k)^T
R_{z,k}^{-1}
J_h(s_k)
G_k
}
$$

其中

$$
G_k=
\begin{bmatrix}
-[\hat R a_k]_\times & I_3
\end{bmatrix}
$$

$$
s_k=\hat R a_k+\hat t-b_k
$$

这个矩阵

$$
F_k\in\mathbb{R}^{6\times 6}
$$

表示第 $k$ 个 bearing-only 观测对相对位姿局部误差状态

$$
\delta\xi=
\begin{bmatrix}
\delta\rho^T & \delta t^T
\end{bmatrix}^T
$$

所提供的信息量。

---

## 5. 单位方向向量形式的等价表达

若不直接使用方位角和俯仰角，而是使用单位 bearing 向量

$$
q_k=\frac{s_k}{\|s_k\|}
$$

则单位向量对相对向量的雅可比为

$$
\frac{\partial q_k}{\partial s_k}
=
\frac{1}{\|s_k\|}
\left(
I_3-q_kq_k^T
\right)
$$

定义

$$
P_k^\perp=I_3-q_kq_k^T
$$

其中 $P_k^\perp$ 表示投影到 bearing 方向法平面的投影矩阵。

则

$$
\frac{\partial q_k}{\partial s_k}
=
\frac{1}{L_k}P_k^\perp
$$

其中

$$
L_k=\|s_k\|
$$

因此，单位 bearing 向量对状态的雅可比为

$$
H_k^q=
\frac{1}{L_k}P_k^\perp G_k
$$

若假设单位 bearing 向量在切平面上的噪声近似各向同性，方差为 $\sigma_q^2$，则单个 bearing 观测的信息贡献可以写成

$$
F_k
=
\frac{1}{\sigma_q^2L_k^2}
G_k^T
P_k^\perp
G_k
$$

即

$$
\boxed{
F_k
=
\frac{1}{\sigma_q^2L_k^2}
G_k^T
\left(I_3-q_kq_k^T\right)
G_k
}
$$

该形式直观表明：

1. bearing-only 观测只约束垂直于视线方向的误差；
2. 沿视线方向的距离尺度不可由单个 bearing 直接确定；
3. 信息强度随距离平方衰减，即

$$
F_k\propto \frac{1}{L_k^2}
$$

---

## 6. 多个 bearing 观测的 Fisher 信息矩阵累加

对于候选关键点集合

$$
\mathcal{K}=\{1,2,\dots,N\}
$$

若选择其中一个子集

$$
\mathcal{S}\subseteq\mathcal{K}
$$

则该子集对应的总 Fisher 信息矩阵为

$$
F(\mathcal{S})
=
F_0+\sum_{k\in\mathcal{S}}F_k
$$

其中 $F_0$ 表示先验信息矩阵，可以来自上一滑动窗口、惯导先验或已有估计结果。如果不考虑先验，则可令

$$
F_0=0
$$

因此，bearing-only 协同定位中的总信息矩阵为

$$
\boxed{
F(\mathcal{S})
=
F_0+
\sum_{k\in\mathcal{S}}
G_k^T
J_h(s_k)^T
R_{z,k}^{-1}
J_h(s_k)
G_k
}
$$

其中

$$
s_k=\hat R a_k+\hat t-b_k
$$

$$
G_k=
\begin{bmatrix}
-[\hat R a_k]_\times & I_3
\end{bmatrix}
$$

---

## 7. Bearing-only 关键点选择目标函数

为了选择对协同定位最有利的关键点，可以将 Papalia 中基于 Fisher 信息矩阵的可定位性评价准则迁移到 bearing-only 场景。

### 7.1 E-optimality：最大化最弱可观方向

E-optimality 关注 Fisher 信息矩阵的最小特征值：

$$
\lambda_{\min}(F)
$$

其物理意义是最大化系统最弱可观方向的信息量。

bearing-only 关键点选择问题可以写成

$$
\boxed{
\mathcal{S}^\star
=
\arg\max_{\mathcal{S}\subseteq\mathcal{K},\ |\mathcal{S}|=M}
\lambda_{\min}
\left(
F_0+
\sum_{k\in\mathcal{S}}
G_k^T
J_h(s_k)^T
R_{z,k}^{-1}
J_h(s_k)
G_k
\right)
}
$$

其中 $M$ 为允许选择的关键点数量。

该目标特别适合 bearing-only 协同定位，因为 bearing-only 问题容易出现几何退化。当 $\lambda_{\min}(F)$ 接近 0 时，说明存在某个相对旋转或平移方向几乎不可观。

---

### 7.2 A-optimality：最小化平均估计方差

A-optimality 关注估计协方差矩阵的迹。由于 Fisher 信息矩阵的逆近似对应估计协方差，因此可以定义

$$
\Phi_A(F)
=
-\operatorname{Tr}(F^{-1})
$$

对应的关键点选择目标为

$$
\boxed{
\mathcal{S}^\star
=
\arg\max_{\mathcal{S}\subseteq\mathcal{K},\ |\mathcal{S}|=M}
-
\operatorname{Tr}
\left[
\left(
F_0+
\sum_{k\in\mathcal{S}}
F_k
\right)^{-1}
\right]
}
$$

A-optimality 的作用是降低整体平均估计不确定性。

---

### 7.3 D-optimality：最大化信息椭球体积

D-optimality 关注 Fisher 信息矩阵行列式：

$$
\Phi_D(F)=\log\det(F)
$$

对应的关键点选择目标为

$$
\boxed{
\mathcal{S}^\star
=
\arg\max_{\mathcal{S}\subseteq\mathcal{K},\ |\mathcal{S}|=M}
\log\det
\left(
F_0+
\sum_{k\in\mathcal{S}}
F_k
+
\epsilon I
\right)
}
$$

其中 $\epsilon I$ 是数值正则项，用于避免 Fisher 信息矩阵奇异。

---

## 8. Localizability constraint 形式

如果不直接最大化信息指标，而是按照 Papalia 的 localizability constraint 思路，也可以将关键点选择写成约束优化问题。

例如，要求关键点集合满足最小可观性阈值：

$$
\lambda_{\min}(F(\mathcal{S}))\ge \beta
$$

同时要求平均估计不确定性满足：

$$
-\operatorname{Tr}(F(\mathcal{S})^{-1})\ge \alpha
$$

则可以构造如下优化问题：

$$
\boxed{
\mathcal{S}^\star
=
\arg\min_{\mathcal{S}\subseteq\mathcal{K}}
\sum_{k\in\mathcal{S}}c_k
}
$$

subject to

$$
\lambda_{\min}
\left(
F_0+\sum_{k\in\mathcal{S}}F_k
\right)
\ge \beta
$$

$$
-\operatorname{Tr}
\left[
\left(
F_0+\sum_{k\in\mathcal{S}}F_k
\right)^{-1}
\right]
\ge \alpha
$$

$$
|\mathcal{S}|\le M
$$

其中 $c_k$ 表示选择第 $k$ 个关键点的代价，可以是通信代价、计算代价、运动代价、图像质量代价或冗余惩罚。

---

## 9. 推荐使用的最终目标函数

对于 bearing-only 协同定位关键点选择，最推荐使用 E-optimality 作为主目标：

$$
\boxed{
\mathcal{S}^\star
=
\arg\max_{\mathcal{S}\subseteq\mathcal{K},\ |\mathcal{S}|=M}
\lambda_{\min}
\left(
F_0+
\sum_{k\in\mathcal{S}}
G_k^T
J_h(s_k)^T
R_{z,k}^{-1}
J_h(s_k)
G_k
\right)
}
$$

其中

$$
s_k=\hat R a_k+\hat t-b_k
$$

$$
G_k=
\begin{bmatrix}
-[\hat R a_k]_\times & I_3
\end{bmatrix}
$$

$$
J_h(s_k)
=
\begin{bmatrix}
-\dfrac{s_{y,k}}{s_{x,k}^2+s_{y,k}^2}
&
\dfrac{s_{x,k}}{s_{x,k}^2+s_{y,k}^2}
&
0
\\[8pt]
-\dfrac{s_{x,k}s_{z,k}}{L_k^2 r_k}
&
-\dfrac{s_{y,k}s_{z,k}}{L_k^2 r_k}
&
\dfrac{r_k}{L_k^2}
\end{bmatrix}
$$

$$
R_{z,k}=
\begin{bmatrix}
\sigma_{\theta,k}^2 & 0\\
0 & \sigma_{\phi,k}^2
\end{bmatrix}
$$

该目标函数的含义是：

> 在候选 bearing 观测中选择 $M$ 个关键点，使相对坐标系旋转和平移参数的 Fisher 信息矩阵最不退化，从而最大化 bearing-only 协同定位中的最弱可观方向。

---

## 10. 计算流程总结

对每一个候选关键点 $k$，依次计算：

$$
s_k=\hat R a_k+\hat t-b_k
$$

$$
J_h(s_k)=\frac{\partial(\theta_k,\phi_k)}{\partial s_k}
$$

$$
G_k=
\begin{bmatrix}
-[\hat R a_k]_\times & I_3
\end{bmatrix}
$$

$$
H_k=J_h(s_k)G_k
$$

$$
F_k=H_k^TR_{z,k}^{-1}H_k
$$

然后对选中的关键点集合 $\mathcal{S}$ 累加：

$$
F(\mathcal{S})=F_0+\sum_{k\in\mathcal{S}}F_k
$$

最后通过如下准则进行关键点选择：

$$
\mathcal{S}^\star
=
\arg\max_{\mathcal{S}\subseteq\mathcal{K},\ |\mathcal{S}|=M}
\lambda_{\min}(F(\mathcal{S}))
$$

这就是 bearing-only 观测下，基于 Russell 相对坐标系变换建模和 Papalia Fisher 信息矩阵思想得到的协同定位关键点选择方法。