# Survey Note: Bearing-Only Multi-Agent Cooperative Localization

## 1. 问题定义与调研范围

本文关注的核心问题是：**bearing-only 观测下的多机协同定位**，即在仅有方位/方向观测（AOA/DOA/bearing vectors）的条件下，估计多机器人/多无人机系统的位姿（至少相对位姿，最好全局位姿）。

对应到你的研究目标（BO-EPNP 扩展）可形式化为：
- 1 个或多个锚点（可全局已知/部分已知）；
- 其余智能体状态 \(X_i=(R_i,t_i)\in SE(3)\)；
- 观测因子来自 agent-agent bearing、anchor-agent bearing、（可选）里程计/IMU；
- 目标是高精度、可扩展、可分布式的全局一致估计。

纳入调研的相关方向：
- bearing-only cooperative localization / multi-agent localization
- distributed state estimation
- factor graph / message passing / loopy BP
- distributed Gauss-Newton
- distributed ADMM
- consensus-based estimation
- network localization / anchor-free localization / pose graph optimization

---

## 2. 文献筛选标准

### 2.1 筛选逻辑（年份-引用量-相关性）

为避免“只罗列”，本文采用三维筛选：

1. **相关性（最高优先）**
- A类：直接研究 bearing-only 多机/网络定位；
- B类：非 bearing-only，但方法可直接作为后端（分布式 FG/BP/GN/ADMM/PGO）；
- C类：联系较弱，仅作方法参考（会明确标注“弱相关”）。

2. **时间新近性**
- 重点保留 **2023–2026** 论文（即便引用低，只要相关性强）。

3. **影响力（引用）**
- 经典高引（>3年）保留；
- >3年且低引论文一般不纳入，除非在细分问题（如 bearing rigidity/localizability）具有代表性。

### 2.2 引用量标记说明

文中“高/中/低”采用**定性分档**（结合可获取的 Scopus/GS/领域公认度）：
- 高：领域经典或长期主流基线；
- 中：较有影响、在子方向常被引用；
- 低：较新或传播范围有限。

> 说明：部分论文仅获取到摘要/题录，无法稳定获取全文细节，已在表中显式标注“内容可得性”。

---

## 3. 主要方法分类

## 3.1 几何/代数初始化路线（BO-EPNP 同类）
- 核心思想：先在局部或短窗口内用闭式/弱迭代方法得到 \(R,t\) 初值；
- 优势：快、易工程化、适合作为前端；
- 劣势：对退化几何和噪声敏感，难单独承担全局一致优化。

## 3.2 集中式因子图后端（Centralized FG / BA / PGO）
- 核心思想：把多机状态统一建图，联合优化所有残差因子；
- 优势：精度高，理论和工程成熟；
- 劣势：中心节点计算和通信压力大，扩展性受限。

## 3.3 分布式因子图 / 消息传递（BP/GBP/GN-BP）
- 核心思想：在图上局部线性化与消息传递，邻域通信求全局一致增量；
- 优势：天然分布式、可扩展、适配图结构协同系统；
- 劣势：图中环路时收敛与阻尼设计关键，初始化质量影响大。

## 3.4 分布式 Gauss-Newton（矩阵分裂或图消息化）
- 核心思想：外层 GN，内层分布式解线性增量系统；
- 优势：保留二阶近似效率；
- 劣势：矩阵分块、内外循环和通信协议较复杂。

## 3.5 ADMM/共识优化路线
- 核心思想：把全局目标分解为局部子问题 + 一致性约束；
- 优势：框架统一、可异步扩展；
- 劣势：非凸 \(SE(3)\) 下参数调节敏感，收敛速度常慢于高质量 GN/BP。

---

## 4. 代表性论文总结

> 字段：标题 / 作者 / 年份 / 期刊会议 / 引用情况 / 核心方法 / 优点 / 局限 / 与 BO-EPNP 多机拓展相关性 / 内容可得性

### 4.1 Bearing-only 核心与基础理论

1. **Cooperative Localization of a GPS-Denied UAV Using Direction-of-Arrival Measurements**  
- 作者：Russell, Ye, Anderson, Hmam, Sarunic  
- 年份：2020（TAES）  
- 引用：中-高（公开页面显示 Scopus 约 57）  
- 方法：单锚+单机 DOA 定位，SDP/Procrustes + ML refinement，讨论三机扩展  
- 优点：与 BO-EPNP 问题设置高度一致；含实飞验证  
- 局限：多机扩展未形成完整分布式后端框架  
- 相关性：**极高**  
- 内容可得性：较完整（摘要/公开版本可得）

2. **Cooperative Localisation of a GPS-Denied UAV in 3D Using Direction of Arrival Measurements**  
- 作者：Russell et al.  
- 年份：2017（IFAC-PapersOnLine）  
- 引用：中  
- 方法：早期 DOA 协同定位框架（双机主）  
- 优点：问题建模清晰，连接后续 TAES 工作  
- 局限：鲁棒与多机后端不足  
- 相关性：高  
- 内容可得性：摘要可得

3. **Localizability and Distributed Protocols for Bearing-Based Network Localization in Arbitrary Dimensions**  
- 作者：Zhao, Zelazo  
- 年份：2016（Automatica）  
- 引用：高（bearing rigidity 经典）  
- 方法：bearing localizability 条件 + 分布式协议  
- 优点：提供“可定位性/可观性”理论地基  
- 局限：偏静态网络与理论层，非直接 BO-EPNP  
- 相关性：高（理论支撑）  
- 内容可得性：摘要可得

4. **Bearing-only Distributed Localization: A Unified Barycentric Approach**  
- 作者：Cao, Han, Lin, Xie  
- 年份：2021（Automatica）  
- 引用：中（页面显示 cited by ~34）  
- 方法：bearing-only 分布式重心坐标定位（2D/3D）  
- 优点：与“分布式 + bearing-only”直接相关  
- 局限：多机机动目标场景与 \(SE(3)\) 姿态估计耦合较弱  
- 相关性：高  
- 内容可得性：摘要与部分段落可得

5. **Bearing-only Distributed Localization for Multi-Agent Systems with Complex Coordinates**  
- 年份：2023（Information Sciences）  
- 引用：低-中（新）  
- 方法：复坐标框架下 bearing-only 分布式定位  
- 优点：近三年、直接相关  
- 局限：与 BO-EPNP 的 3D 刚体姿态耦合需要额外转换  
- 相关性：中-高  
- 内容可得性：摘要可得（全文细节待补）

### 4.2 分布式 GN/BP/因子图（后端候选）

6. **Distributed Gauss–Newton Method for State Estimation Using Belief Propagation (GN-BP)**  
- 作者：Cosovic, Vukobratovic  
- 年份：2019（IEEE TPWRS）  
- 引用：中（公开页面 Scopus 约 30+）  
- 方法：外层 GN + 内层 BP 消息传递  
- 优点：图结构分布式求解，接近集中式精度  
- 局限：论文场景是电力系统，需迁移到机器人测量模型  
- 相关性：**高（方法迁移价值高）**  
- 内容可得性：通过 PDF 图片阅读到算法和公式页

7. **A Distributed Gauss-Newton Method for Power System State Estimation**  
- 作者：Minot, Lu, Li  
- 年份：2016（IEEE TPWRS）  
- 引用：中-高  
- 方法：矩阵分裂实现分布式 GN  
- 优点：收敛分析清楚，工程分区思路明确  
- 局限：依赖区域分块结构，机器人邻接图映射不如 GN-BP 自然  
- 相关性：中-高  
- 内容可得性：通过 PDF 图片阅读到关键公式与算法

8. **A Multi-Vehicle Cooperative Localization Method Based on Belief Propagation in Satellite Denied Environment**  
- 作者：Wang, Wang  
- 年份：2022（JBIT）  
- 引用：低-中  
- 方法：因子图 + BP 车辆协同定位  
- 优点：场景贴近 GNSS-denied 协同  
- 局限：bearing-only 不是唯一观测来源  
- 相关性：中  
- 内容可得性：摘要/部分全文可得

9. **Cooperative Navigation for UAVs in GNSS-denied Area Based on Optimized Belief Propagation**  
- 年份：2022（Measurement）  
- 引用：中  
- 方法：优化 BP 协同导航  
- 优点：UAV 场景、GNSS-denied 实用性强  
- 局限：非纯 bearing-only  
- 相关性：中  
- 内容可得性：摘要可得

10. **A Robot Web for Distributed Many-Device Localization**  
- 作者：Murai et al.  
- 年份：2024（IEEE T-RO）  
- 引用：低-中（新）  
- 方法：基于 Gaussian BP 的大规模分布式定位  
- 优点：大规模、异步通信、分布式工程价值高  
- 局限：非专门 bearing-only，需要自定义 bearing 因子  
- 相关性：中-高  
- 内容可得性：摘要可得（全文细节待补）

### 4.3 ADMM / 分布式 PGO / 共识（后端备选）

11. **Distributed Maximum Likelihood Sensor Network Localization**  
- 作者：Simonetto, Leus  
- 年份：2014（IEEE TSP）  
- 引用：高（经典 ADMM 分布式定位）  
- 方法：ML 松弛 + ADMM 分布式求解  
- 优点：理论完整、异步鲁棒  
- 局限：与 \(SE(3)\) 刚体姿态和强非凸 bearing-only 结合需重构  
- 相关性：中  
- 内容可得性：摘要可得

12. **Majorization Minimization Methods for Distributed Pose Graph Optimization**  
- 作者：Fan, Murphey  
- 年份：2024（IEEE T-RO）  
- 引用：低-中（新）  
- 方法：分布式 PGO（MM / 加速）  
- 优点：机器人后端优化强相关，含无主节点方案  
- 局限：需将 bearing-only 观测转为兼容图因子  
- 相关性：中-高  
- 内容可得性：摘要可得

13. **Distributed Pose Graph Optimization Using Splitting Method Based on ADMM**  
- 年份：2025（J Intell Robot Syst）  
- 引用：低（新）  
- 方法：ADMM + 分裂式 PGO  
- 优点：分布式实现直接  
- 局限：对非凸旋转子问题与参数敏感  
- 相关性：中  
- 内容可得性：摘要可得

14. **Distributed Pose-Graph Optimization with Multi-Level Partitioning for Multi-Robot SLAM**  
- 年份：2024（IEEE）  
- 引用：低（新）  
- 方法：分层分区分布式 PGO  
- 优点：多机器人规模化思路清晰  
- 局限：非 bearing-only 专项  
- 相关性：中  
- 内容可得性：题录可得（全文细节待补）

15. **Distributed localization and circumnavigation with persistent/intermittent bearing measurements**  
- 年份：2020（IEEE）  
- 引用：中（方向相关）  
- 方法：bearing 条件下分布式定位+控制  
- 优点：直接涉及 bearing 测量间歇性问题  
- 局限：偏控制闭环，不是 BO-EPNP 式几何求解框架  
- 相关性：中  
- 内容可得性：题录可得（全文待补）

---

## 5. 不同方法的横向对比

| 方法族 | 核心思想 | 优点 | 局限 | 适用条件 | 与 BO-EPNP 对接性 |
|---|---|---|---|---|---|
| Centralized Factor Graph | 全局图联合优化 | 精度高、成熟 | 中心算力/通信瓶颈 | 中等规模、离线/准实时 | 很高 |
| Distributed FG / BP | 图上消息传递求全局一致 | 分布式天然、扩展好 | 环路收敛与阻尼难点 | 邻接通信网络、局部计算能力 | 很高 |
| Distributed GN（矩阵分裂） | 外层GN+内层线性系统分布式求解 | 保留GN效率、收敛分析可做 | 分块与内外循环复杂 | 有明确分区结构 | 中-高 |
| ADMM 分布式优化 | 局部子问题 + 一致性约束 | 框架通用、可异步 | 非凸场景调参与速度问题 | 约束可分解、通信可控 | 中 |
| 共识/梯度类 | 局部更新 + 邻域平均 | 实现简单 | 收敛慢、精度受限 | 资源受限大规模系统 | 中-低 |

---

## 6. BO-EPNP 拓展到多机协同定位的技术路线分析

### 6.1 BO-EPNP 在系统中的角色

最合理定位：**几何初始化 + 局部求解器 + 前端观测建模模块**  
不建议将 BO-EPNP 单独作为完整后端，因为：
- 多机图中闭环一致性、漂移抑制、全局最优主要由后端承担；
- BO-EPNP 更擅长“快而准的局部位姿起点”。

### 6.2 路线 A：BO-EPNP + Centralized Factor Graph

- 基本思路：每个时窗先用 BO-EPNP 求局部初值，再入图做全局 BA/FG 优化  
- bearing-only 约束匹配：高（可直接作为 bearing 因子残差）  
- 分布式适配：中（需中心节点）  
- 敏感性：对初始化敏感中等；对噪声/可观性可通过鲁棒核与先验缓解  
- 实现复杂度：中  
- 论文潜力：中-高（工程可靠，创新需靠新因子设计/理论）

### 6.3 路线 B：BO-EPNP + Distributed Factor Graph / Message Passing

- 基本思路：BO-EPNP 生成局部初值；后端用 BP/GBP 在图上分布式迭代  
- bearing-only 约束匹配：高  
- 分布式适配：高  
- 敏感性：对图中环路和阻尼策略较敏感；初始化好可显著改善收敛  
- 实现复杂度：中-高  
- 论文潜力：**高**（与你当前 BO-EPNP 基础耦合最自然）

### 6.4 路线 C：BO-EPNP + Distributed Gauss-Newton

- 基本思路：外层 GN 线性化，内层分布式求增量（矩阵分裂或 GN-BP）  
- bearing-only 约束匹配：高（可用你已有解析 Jacobian）  
- 分布式适配：高  
- 敏感性：对线性化点、阻尼、内外循环参数敏感  
- 实现复杂度：高  
- 论文潜力：高（尤其 GN-BP 迁移到 bearing-only UAV 网络）

### 6.5 路线 D：BO-EPNP + ADMM

- 基本思路：将多机图优化拆为局部子问题 + 一致性约束  
- bearing-only 约束匹配：中（非凸旋转子问题较麻烦）  
- 分布式适配：中-高  
- 敏感性：对罚参数、初始化敏感；可能收敛慢  
- 实现复杂度：中-高  
- 论文潜力：中（若有强理论收敛或异步通信创新则可提升）

### 6.6 路线 E：BO-EPNP + Belief Propagation（纯BP/高斯BP）

- 基本思路：把增量估计映射为消息传递，局部信息融合  
- bearing-only 约束匹配：中-高（需良好线性化与高斯近似）  
- 分布式适配：高  
- 敏感性：环路图和非高斯噪声下性能依赖阻尼与重线性化  
- 实现复杂度：中  
- 论文潜力：中-高（与 GN 融合更稳，即 GN-BP）

---

## 7. 推荐方案与理由

## 最推荐路线
**BO-EPNP + Distributed Factor Graph（GN-BP 风格）**

理由：
1. 与 bearing-only 因子图建模最匹配；  
2. 能直接复用你现有 BO-EPNP 与解析 Jacobian；  
3. 兼顾分布式部署与全局一致性；  
4. 研究创新点清晰：收敛控制、鲁棒核、可观性驱动调度、异步消息机制。

## 次优路线
**BO-EPNP + Centralized Factor Graph（先做强基线）**

理由：
- 工程成功率高、便于快速验证 BO-EPNP 的增益；
- 可作为分布式方法的“oracle baseline”。

## 不太推荐路线（当前阶段）
**BO-EPNP + 纯 ADMM 后端（直接上非凸全分布式）**

原因：
- 调参与收敛速度风险高；
- 在 bearing-only 非凸几何下，工程复杂度高于 GN-BP/FG 路线；
- 若无额外理论突破，论文增量不一定优于 GN-BP 路线。

---

## 8. 近三年趋势与研究机会

### 8.1 趋势
- 从“集中式图优化”向“分布式图优化/消息传递”迁移；
- 从单模态观测向多模态（视觉/UWB/IMU）鲁棒协同；
- 强调可扩展、异步通信、边缘计算友好。

### 8.2 结合你课题的机会点

1. **BO-EPNP 作为统一初始化层**  
- 多机图优化里引入“局部几何先验”减少后端收敛难度。

2. **Bearing-only 因子的鲁棒化与可观性自适应**  
- 融合你已有 FIM/条件数分析，动态调节因子权重或消息阻尼。

3. **GN-BP 在 \(SE(3)\) 上的严格实现**  
- 将你已有 `SO(3)` 解析 Jacobian 推广到多节点因子，形成可发表的方法学增量。

4. **异步/掉包通信下的分布式定位稳定性**  
- 面向真实机群网络，工程和论文价值都高。

### 8.3 当前“待补全文”清单（建议后续补充上传）
- 2024/2025 distributed PGO（部分仅获取摘要）  
- 2024/2025 bearing-only 多 UAV 目标定位论文（部分仅题录）  
- 你后续若上传 PDF，可补齐公式与实验细节后更新本综述。

---

## 9. 总结

对“BO-EPNP 扩展到多机 bearing-only 协同定位”这个目标，文献整体结论是：

1. **技术可行**：bearing-only 多机问题可自然建模为图优化；  
2. **方法优选**：BO-EPNP 负责局部几何初始化，后端优先采用 **分布式因子图/GN-BP**；  
3. **研究路径清晰**：先做 centralized FG 强基线，再推进 GN-BP 分布式版；  
4. **论文潜力较高**：尤其在“可观性驱动 + 鲁棒消息传递 + \(SE(3)\) 解析 Jacobian”结合上。

---

## 参考链接（精选）

- Russell et al., TAES 2020: https://doi.org/10.1109/TAES.2019.2942704  
- GN-BP (Cosovic 2019): https://doi.org/10.1109/TPWRS.2018.2866583  
- DGN matrix splitting (Minot 2016): https://doi.org/10.1109/TPWRS.2015.2512023  
- Bearing localizability (Zhao & Zelazo 2016): https://doi.org/10.1016/j.automatica.2016.03.010  
- Bearing-only barycentric distributed localization (Automatica 2021): https://doi.org/10.1016/j.automatica.2021.109834  
- Multi-UAV/vehicle BP cooperative localization (2022): https://doi.org/10.15918/j.jbit1004-0579.2022.029  
- Robot Web (GBP, T-RO 2024): https://doi.org/10.1109/TRO.2023.3324127  
- Distributed PGO (MM, T-RO 2024): https://doi.org/10.1109/TRO.2023.3324818  
- Distributed ADMM-style PGO (2025): https://doi.org/10.1007/s10846-025-02257-w

---

## 附录 A：推荐阅读清单（按价值与优先级）

### A1. 第一优先级（直接服务 BO-EPNP 多机扩展）

1. **Distributed Gauss–Newton Method for State Estimation Using Belief Propagation**  
   - Cosovic, Vukobratovic, 2019, IEEE TPWRS  
   - 价值：GN-BP 主框架，最贴合“分布式后端 + 图消息传递”

2. **A Distributed Gauss-Newton Method for Power System State Estimation**  
   - Minot, Lu, Li, 2016, IEEE TPWRS  
   - 价值：分布式 GN 的矩阵分裂基线，可与 GN-BP 对比

3. **Cooperative Localization of a GPS-Denied UAV Using Direction-of-Arrival Measurements**  
   - Russell et al., 2020, IEEE TAES  
   - 价值：与你当前 bearing-only 问题设定最直接一致

4. **Localizability and Distributed Protocols for Bearing-Based Network Localization in Arbitrary Dimensions**  
   - Zhao, Zelazo, 2016, Automatica  
   - 价值：bearing 可定位性/可观性理论基础

5. **Bearing-only Distributed Localization: A Unified Barycentric Approach**  
   - Cao et al., 2021, Automatica  
   - 价值：bearing-only + distributed 的桥梁工作

### A2. 第二优先级（后端方法扩展与工程实现）

6. **A Robot Web for Distributed Many-Device Localization**  
   - Murai et al., 2024, IEEE T-RO  
   - 价值：大规模 GBP/分布式定位工程参考

7. **Majorization Minimization Methods for Distributed Pose Graph Optimization**  
   - Fan, Murphey, 2024, IEEE T-RO  
   - 价值：分布式 PGO 强后端，可迁移到 bearing 因子图

8. **A Multi-Vehicle Cooperative Localization Method Based on Belief Propagation in Satellite Denied Environment**  
   - Wang, Wang, 2022, Journal of Beijing Institute of Technology  
   - 价值：多车协同 BP 实践参考（场景贴近）

9. **Cooperative Navigation for UAVs in GNSS-denied Area Based on Optimized Belief Propagation**  
   - 2022, Measurement  
   - 价值：UAV 协同 + BP 的工程实现细节

### A3. 第三优先级（ADMM/共识等次优后端）

10. **Distributed Maximum Likelihood Sensor Network Localization**  
    - Simonetto, Leus, 2014, IEEE TSP  
    - 价值：ADMM 分布式定位经典基线

11. **Distributed Pose Graph Optimization Using Splitting Method Based on ADMM**  
    - 2025, Journal of Intelligent & Robotic Systems  
    - 价值：最新 ADMM-PGO，适合做横向对比

12. **Distributed Pose-Graph Optimization with Multi-Level Partitioning for Multi-Robot SLAM**  
    - 2024  
    - 价值：分层分区分布式 PGO 的系统设计参考

### A4. Bearing-only 多机方向补充追踪

13. **Bearing-only Distributed Localization for Multi-Agent Systems with Complex Coordinates**  
    - 2023, Information Sciences  
    - 价值：近三年、bearing-only 直接相关

14. **Distributed localization and circumnavigation with persistent/intermittent bearing measurements**  
    - 2020, IEEE  
    - 价值：bearing 间歇观测鲁棒性参考

15. **Cooperative Localisation of a GPS-Denied UAV in 3D Using Direction of Arrival Measurements**  
    - Russell et al., 2017, IFAC-PapersOnLine  
    - 价值：Russell 2020 的前置版本，便于梳理方法演化

### A5. 建议阅读顺序（最省时间）

1. Russell 2020（问题建模与观测特性）  
2. Cosovic 2019（GN-BP 后端主线）  
3. Minot 2016（分布式 GN 另一主线）  
4. Zhao & Zelazo 2016（可定位性理论）  
5. Cao 2021（bearing-only 分布式桥接）  
6. T-RO 2024 两篇（面向可发表创新的工程扩展）
