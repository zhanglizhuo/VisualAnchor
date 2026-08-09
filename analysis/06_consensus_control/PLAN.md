# 06_consensus_control — Neurocomputing (2区) 冲刺计划

目标：把现有 SCB5 证据链加固为送审竞争力，转投 Neurocomputing（主攻）；EAAI 仅在加固全过关后考虑。
原则：先打最危险的枪（Exp2），全部 5 项纯 CPU + 现有数据完成（无新 GPU 需求）。

## 实验结果汇总（2026-08-09，全部完成）

### Exp2 跨 MLLM 共识对照（最危险项，decision gate）
`results/06_consensus_control/exp2_consensus_control.json`
| 指标 | 值 |
|---|---|
| ρ(Anchor, MLLM acc) pooled 78 | 0.692 (p~10⁻¹²) |
| ρ(Anchor, 其他5-MLLM 共识代理) | **0.763** (p~10⁻¹⁶) |
| partial ρ(Anchor, MLLM \| 共识) pooled | **0.067** (类级 0.219) |

**判定：CLIP 特异性断言被削弱。** AnchorScore 主要测共享难度因子；共识代理更强。
→ framing 改为 "AnchorScore = 共享难度因子的廉价(cold-start)代理"，删机制专属性断言。
→ 利好：免费前向以 0.76 相关近似"5 个昂贵 MLLM 共识"。
- 类级 cluster bootstrap（类数 n=13）CI 宽 [-0.036, +0.789]——报告保留。
- mllm_image_level jsonl 与 mllm_raw 不一致（不同流水线）→ 排除，只用 mllm_raw。

### Exp1 Δρ 直接检验（coupled bootstrap B=10k）
`results/06_consensus_control/exp1_delta_rho.json`
| predictor | ρ | Δρ vs LAION | CI95 Δρ | sig |
|---|---|---|---|---|
| OpenAI-L/14 | 0.473 | +0.297 | [-0.11,+0.85] | ✗ (n=13 弱) |
| OpenAI-B/32 | 0.242 | +0.527 | [+0.02,+1.12] | ✓ |
| SigLIP | 0.201 | +0.568 | [-0.02,+1.21] | ✗ 边缘 |
| DINOv2 | 0.050 | +0.720 | [+0.06,+1.32] | ✓ |
| BLIP2 | 0.034 | +0.736 | [+0.16,+1.34] | ✓ |
| ResNet-50 | 0.081 | +0.688 | [-0.18,+1.41] | ✗ |
- 结论：LAION 类级 ρ=0.769 显著优于 B/32/DINO/BLIP2；与 OpenAI-L/14 差异在 n=13 无检验力。

### Exp3 baseline 准确率地板表
`results/06_consensus_control/exp3_baseline_floor.json`
- 地板证据：BLIP2 69% 类 <15% 准确率（12/13 类 0 或 100%，二值化地板）；ResNet-50 69% 类 <15%；OpenAI-B/32 46% 类 <15%。
- 但有方差类分布（range 60-100pp 各模型），部分 null 与地板压缩一致、部分非地板（DINO 15% 类 <15，但 σ 31pp 不比 LAION 小）。
- 结论：null baseline 解释需按模型个别处理，不能一刀切"方差压缩"。地板效应真实存在（BLIP2/ResNet）但不是全部解释。

### Exp4 τ 选择留出（全网格 1-99 + pre-registered τ=50 + LOO-class + random/oracle 对照）
`src` exp4_tau_holdout.json，共识 MLLM（6 模型均值）为 fallback：
| ds | clip | τ* → acc* | τ=50 → acc | 节省 | oracle | random |
|---|---|---|---|---|---|---|
| TeacherBehavior | 32.57 | 55 → 54.76 | 44.81 | 52.9% | 54.76 | 54.70±0.0 |
| HandriseReadWrite | 60.92 | 90 → 68.09 | 62.43 | 63.4% | 68.09 | 68.08±0.0 |
| BowTurnHead | 60.79 | 65 → 82.97 | 67.75 | 77.0% | 82.97 | 82.86±0.0 |
- **关键发现：网格最优 τ* 使所有类都路由给 MLLM（sav=0）**——用共识 MLLM 时 CLIP 无竞争力，最优就是全 MLLM。混合增益只存在于"迁移到 MLLM 均值"的场景。
- τ=50（pre-registered）在 TB/HR/BT 反而提供 53-77% 成本节省 + 同量级质量——**preregistered τ=50 是稳健交付选择**，选 τ 的自由度过拟合风险落实为"全走高成本"。
- LOO-class：τ* 区间稳定（TB 51/53/55、HR 81/90、BT 50/65）——在各 leave 类上基本一致。
- random 路由 ≈ oracle（因 CLIP 在多数类劣于 MLLM，随机省钱的边界即 oracle）→ 说明混合增益主要来自成本而不是质量提升，质量提升 = τ* 选择带来的。

### Exp5 backbone 镜像（OpenAI-L/14 当主键）
`src/exp5_backbone_mirror.json`
- OpenAI-L/14 pooled ρ=0.473（LAION 0.769）；per-ds：TB 0.452（ns）、HR 0.5（ns）、BT -1.0（n=2）。
- hybrid τ=50 镜像：TB 55.97%（LA 44.81）、HR 62.21%（LA 62.43）、BT 80.76%（LA 67.75）。
- 结论：OpenAI 主干相关较弱但路由增益在 TB/BT 反而更高（该主干与 MLLM/CLIP 混合边界有利）；主干选择影响 ρ 强弱，不影响路由框架。

## Framing 修改（已完成 2026-08-09）
- ✅ 标题尾缀 "…as a Decision-Support Diagnostic"（删除"CLIP 特异性机制"主张）
- ✅ 核心声明改为：AnchorScore 是共享难度因子的廉价代理（ρ=0.763 vs 5 模型共识，与单模型 0.692 相当），cold-start 场景（MLLM 未跑）下零成本筛选
- ✅ 三应用收敛为"筛查 → 路由（τ=50 pre-registered）→ 消歧 → 复核"管线
- ✅ 摘要删除绝对化表述（0.692→0.763、partial 0.067 写入摘要 + intro）；机制段 SigLIP null 改述为"不同家族代理质量差异"，并新增 Cross-model consensus control 段
- ✅ §5.4 新增限制 (6)：partial≈0（CI 宽 [-0.036, 0.789]）、类数 n=13 检验力、BLIP2/ResNet 地板效应（69% 类 <15%）；baseline 空述

## 修改文件清单（commit pending）
- paper/VisualAnchor.tex（摘要/intro/机制/§5.4/结论 + 标题running head恢复）

## 期刊
- Neurocomputing（2 区）主攻；EAAI 仅当加固全过 + 跨域消歧复现。
- IEEE Access 同一稿互斥。

## 遗留（非此次加固所需）
- R3 部署模拟（MLLM 金标准 + 全量 5416 张路由）——唯一潜在 GPU 项，安排在 5 项之后。
- R3+ 真实标注工：跳过。
- R4 敏感性（单价 ×5/×10、batch、ε）：并入可选项。