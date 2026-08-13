# Derivation Package

## Target

正式化 AnchorScore 的理论线：

1. 公式 1 的定义式（当前呈现为 $n_i^{\text{corr}}/n_i$）；
2. 它与"CLIP per-class accuracy 预测 MLLM per-class accuracy"这一核心经验发现之间，哪些环节是**可推导的**（identity/proposition），哪些是**经验发现**，哪些是**带假设的模型预测**（不能伪装为定理）。

用户的原始诉求是"公式 1 总觉得不专业"。诊断：问题不在记号（已从 indicator → 集合基数 → $\#$ → 计数的逐轮降门槛），而在于公式孤立悬挂在 Method 节，没有与其在证据链中的角色（难度信号的代理量）嵌合。

## Status

COHERENT AFTER REFRAMING

- 公式 1 作为**定义**（identity）是自洽的，无需修改。
- 原目标"公式 1 本身专业化为可推导公式"不成立——准确率定义不能"推导"，只能"陈述"。
- Reframing：将单品公式重构为完整理论线（定义 → 机制 → 期望形式 → 经验桥 → 带假设的共享难度因子模型 → 共识控制预测），公式 1 在其中获得其真正角色：**难度潜因子的代理量（proxy）定义**。

## Invariant Object

**class-level difficulty** $\lambda_c \in \mathbb{R}$：一个类（行为/场景类别）被视觉识别系统正确分类的潜在难易程度。

- AnchorScore($c_i$) 是 $\lambda_c$ 的**代理观察**（通过 CLIP 零样本对齐）。
- MLLM per-class accuracy $M_i$ 是 $\lambda_c$ 的**第二个代理观察**（通过大模型视觉理解）。
- 论文的核心经验主张：两个代理观察秩相关显著（$\rho = 0.769$，$n=13$）——即它们测量同一个潜因子。
- 不变量对象在 SCB5（special case）与 Stanford40（general case）之间保持同一。

## Assumptions

- **A1（机制，identity）**：CLIP 预测为余弦相似度 argmax（公式 2），$T=3$ 模板平均。
- **A2（代理模型，PROPOSITION，强形式未验证）**：存在潜因子 $\lambda_c \in \mathbb{R}$ 与单调链接 $\Phi, \Psi$、噪声项 $\varepsilon^C_c, \varepsilon^M_c$，使得
  $$\text{AnchorScore}(c_i) = \Phi(a\,\lambda_{c_i} + \varepsilon^C_{c_i}), \qquad M_i = \Psi(b\,\lambda_{c_i} + \varepsilon^M_{c_i}).$$
  这是**共享难度因子假设的数学形式**。论文的经验证据（共识控制 partial ≈ 0）与该模型的强形式相容，但误差独立性 $(\varepsilon^C, \varepsilon^M)$ 未被验证（论文 §5.1 明确承认该 confound 未解）。
- **A3（多模型 mean，identity）**：$M_i = \frac{1}{M}\sum_{m=1}^{M} \text{acc}^m_i$，$M=6$（canonical MLLM 集）。
- **A4(controls，identity)**: “consensus” proxy $C_{-m,i}$ = 其余 5 个模型的平均 accuracy（LOO consensus，用于部分相关）。
- **A5（替代控制，经验）**：SigLIP/DINOv2/ResNet-50 的 per-class accuracy 不构成有效的第二代理（$\rho$ 不显著）——经验发现，不推导。

## Notation

- $C$：类别数；$c_i$：第 $i$ 个类；$N_i$：类 $c_i$ 验证图像数；$N = \sum_i N_i$
- $n_i^{\text{corr}}$：类 $c_i$ 中 CLIP 预测正确的图像数；$\hat y_j$：第 $j$ 张图的 CLIP 预测（类索引）；$y_j$：真值（类索引）
- $\mathbf{f}_{\text{img}}, \mathbf{f}_{\text{text}}$：CLIP 视觉/文本编码器；$\bar{\mathbf{f}}_{\text{text}}(c_k)$：$T=3$ 模板平均文本嵌入
- $M_i$：类 $c_i$ 的 mean MLLM accuracy（6 模型平均）
- $\rho$：Spearman 秩相关；$\rho_{\text{partial}}$：偏相关（控制 LOO consensus proxy）
- $T=3$：prompt 模板数

## Derivation Strategy

定义 → 机制 → 期望形式 → 经验桥 → 带假设模型 → 可检验预测。核心原则：**哪些步是 identity、哪些是 proposition、哪些只是经验，逐类标记，绝不合并**。

## Derivation Map

1. 公式 1 是 $\text{AnchorScore}(c_i)$ 的**定义**（identity）——不需要推导，需要被赋予角色。
2. 公式 2 给出 $\hat y_j$ 的机制（identity）。
3. 把公式 1 重写为**期望形式**（identity，概率语言）：$\text{AnchorScore}(c_i) = \hat p_i$，即 $p_i = \Pr(\hat y_j = y_j \mid y_j = c_i)$ 的样本估计。
4. MLLM 侧同理（identity）：$M_i = \hat q_i$，$q_i = \frac{1}{M}\sum_m \Pr_m(\hat y^m_j = y_j \mid y_j = c_i)$。
5. **经验桥（非推导）**：$(\hat p_i, \hat q_i)$ 在 $n=13$/78 上秩相关显著。
6. **模型（A2 下的 proposition）**：若 A2 成立，则 $\rho(p, q) > 0$ 且控制代理后偏相关衰减——给出可检验预测。
7. **经验对照**：partial $= 0.067$（pooled, CI 含 0）与强形式预测相容；class-level partial $= 0.219$ 提示残余信号（弱形式/模型不完美）。

## Main Derivation

### Step 1 — 定义（identity）

$$\text{AnchorScore}(c_i) = \frac{n_i^{\text{corr}}}{n_i}, \qquad n_i = \#\{j : y_j = i\}, \quad n_i^{\text{corr}} = \#\{j : y_j = i,\ \hat y_j = y_j\}.$$

### Step 2 — 预测机制（identity）

$$\hat y_j = \arg\max_{k \in \{1,\dots,C\}} \frac{\mathbf{f}_{\text{img}}(x_j)^\top \bar{\mathbf{f}}_{\text{text}}(c_k)}{\|\mathbf{f}_{\text{img}}(x_j)\| \, \|\bar{\mathbf{f}}_{\text{text}}(c_k)\|}, \qquad \bar{\mathbf{f}}_{\text{text}}(c_k) = \tfrac{1}{T}\textstyle\sum_{t=1}^{T} \mathbf{f}_{\text{text}}(p_{k,t}).$$

（温度缩放不影响 argmax——省略合法。）

### Step 3 — 期望形式（identity，概率语言）

记 $p_i = \Pr\big(\hat y_j = y_j \mid y_j = c_i\big)$（类 $c_i$ 的 CLIP 零样本正确率，总体量）。则

$$\text{AnchorScore}(c_i) = \hat p_i, \qquad \hat p_i = \frac{1}{n_i}\sum_{j: y_j = i} \mathbf{1}[\hat y_j = y_j].$$

即公式 1 是该总体概率 $p_i$ 的经验估计（binomial 计数）。**这是公式 1 的专业化陈述：它不是一个特设指标，而是 $p_i$ 的 MLE。**

### Step 4 — MLLM 侧（identity)

$$q_i = \frac{1}{M}\sum_{m=1}^{M} \Pr_m\big(\hat y^{(m)}_j = y_j \mid y_j = c_i\big), \qquad M_i = \hat q_i.$$

### Step 5 — 经验桥（经验发现，非推导）

数据（canonical evidence）：
- class-level（$n=13$）：$\rho(\hat p, \hat q) = 0.769$，$p = 0.002$；
- pooled（$n=78$，6 MLLM × 13 类）：$\rho = 0.693$；
- 独立复现 Stanford40（$n=40$）：$\rho = 0.817$。

这些数字是**测量结果**，不是由 A1–A4 推导的结论。诚实标注：这一步是理论线的经验锚，不是演绎层。

### Step 6 — 共享难度因子模型（A2 下的 proposition）

若 A2 成立（单调链接 + 潜因子），则对任何两个同单调、异噪声的代理：

- $\rho(p, q) > 0$（单调函数不改变秩结构，噪声使相关<1）；
- 控制任何第三个"近似于 $\lambda_c$ 的噪声估计"$w_i$（如 LOO consensus）后，偏相关
$$\rho_{\text{partial}}(p, q \mid w) \to 0 \quad (\text{若} \ \varepsilon \text{ 相互独立})。$$

**这是模型的预测，不是证明**：它依赖 A2 的强形式（误差独立、单因子）。论文不声称证明了单因子结构。

### Step 7 — 经验对照 (interpretation)

| 量 | 经验值 | 模型强形式预测 | 相容性 |
|---|---|---|---|
| $\rho_{\text{partial}}$ pooled | $0.067$ [CI $-0.036, 0.789$] | $\approx 0$ | 相容（噪声水平高） |
| $\rho_{\text{partial}}$ class-level | $0.219$ | $\approx 0$ | 弱相容（存在残余信号 → 弱形式：$\varepsilon^C, \varepsilon^M$ 相关） |

解读（interpretation，非演绎）：partial 不为 0 的残余提示 CLIP 与 MLLM 共享的不只是通用难度——可能还有语义对齐层面的特定重叠（论文 §5.1 已承认此点）。

## Remarks and Interpretation

1. **公式 1 的专业化路径**：把公式 1 在论文中陈述为 "$p_i$ 的经验估计（Step 3 的期望形式）"，并加一句"$\hat p_i$ 将作为类难度 $\lambda_{c_i}$ 的代理量进入相关分析"——这给予公式 1 以角色，而无需引入任何未经验证的数学。
2. **"不专业"的根源不在记号**：逐轮记号降门槛（$\mathbb{1} \to |\cdot| \to \# \to n^{\text{corr}}/n$）是读者友好性改进；但公式的"专业感"来自它在证据链中的定位（难度代理量），而非复杂度。
3. **不要做的**：不要为"CLIP 难 ⇒ MLLM 难"发明第一性原理推导（如"两者都依赖同一视觉对齐空间"）。该方向没有可验证的机制证明，AGENTS.md 规则禁止包装未验证假设为理论。

## Boundaries and Non-Claims

- 本包**没有**证明"CLIP per-class accuracy 预测 MLLM per-class accuracy"。该相关是经验事实；A2 只是其形式化搭架，非其证明。
- 强形式 A2（误差独立、单调链接具体形式、单因子）**未被验证**；论文明确承认 $\varepsilon^C \perp \varepsilon^M$ 无法确认。
- 秩相关 $\rho$ 不蕴含线性标度关系；不声称 AnchorScore 是 MLLM accuracy 的可校准估计器（ECE=0.152 已证反）。
- 不推导温度缩放、softmax 形式、prompt 平均的增益定量（这些是机械细节或经验测量）。
- 不将 SCB5（13 类教室域）的结果外推到任意域；跨域分析显示医学域信号衰减（经验，见 robustness 章节）。

## Open Risks

1. **A2 强形式未验证**：若 $\varepsilon^C$ 与 $\varepsilon^M$ 强相关，partial≈0 的解释（共享潜因子）与"CLIP/LLM 独立同源困难"的解释不可区分——open problem，诚实保留。
2. **class-level partial = 0.219 的残余**：可能是弱形式信号、也可能是小样本噪声（$n=13$，bootstrap CI 宽）。
3. **pooled 分析中 6 MLLM 非独立**（同一模型的类间相关）——论文用 cluster bootstrap 处理，但残余风险存在。
4. **代理量线性假设**：A2 用单调链接是最弱可用形式；任何更强的参数形式（logistic 等）都没有数据支持，不要声称。