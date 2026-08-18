# MyoICL v0.5 — 冻结已发表骨干 ＋ 元训练 ICL 模块

**一句话**:不动任何架构,冻结别人已发表的解码器(有公开权重就直接冻结公开权重),
外挂一个元训练出来的 context 模块,读新用户 30 秒**无标注**信号,零梯度关闭跨用户鸿沟。

零初始化门控 ⇒ 训练第 0 步时复合网络与官方模型**逐位相同**;骨干参数(含 BN 统计)全程不更新
⇒ 任何提升只可能来自我们的模块。冒烟测试第 3b/8 项对此做硬校验。

## 配置一览

| 配置 | 骨干 | 是否需训练骨干 | 用途 |
|---|---|---|---|
| `qwerty_icl_frozen_official.yaml` | **B1 官方 generic.ckpt(冻结)** | 否 | **主表,关键路径,先跑** |
| `qwerty_icl_frozen_M.yaml` / `_L.yaml` | 同上 | 否 | 容量消融(中/大档),后跑 |
| `qwerty_a_v2.yaml` | B2 fairemg Tiny | 是(自训) | 可移植性臂,非关键路径 |
| `qwerty_icl_v2.yaml` | B2(冻结自训权重) | 否 | 可移植性臂第二步 |

模块规模:默认小档 d_ctx=128 / 瓶颈 128 → +0.88M(骨干 5.3M)。先求效果,容量后面再扫。

## 模块地图

| 文件 | 作用 |
|---|---|
| `pretrained.py` | 官方 ckpt 键映射加载 / 冻结骨干 / 锁死 BN 统计 |
| `context.py` | 统计特征(log-RMS+频带功率+log协方差 ⊇ 固定对齐信息集)、序不变 set 编码器、log(1+M) 缩放交叉注意力、低秩 FiLM(均零初始化门控) |
| `model.py` | **B1**:官方 emg2qwerty 架构(前端/TDS 直接 import 官方包)+ ICL 挂点 |
| `model_v2.py` / `featurizer.py` | **B2**:fairemg Transformer Tiny 逐项复刻 + ICL 挂点 |
| `episodes.py` | episodic 采样 + context-type dropout + 跨 session context |
| `synth.py` | episode 一致的合成用户变换(教推断,而非不变性) |
| `align.py` / `eval_ea_official.py` | 统计对齐工具与冻结骨干上的探针评测 |
| `train_qwerty.py` / `eval_qwerty.py` | 训练与 A/B/C 评测 |
| `smoke_test.py` | 8 项自检,含零初始化恒等性与官方 ckpt 往返 |

## 待办(Claude 侧,不阻塞当前实验)

探针 harness(特征统计对齐 / BN-adapt / TENT+边际熵 / 原型头 / k-shot FT,全在冻结 B1 上);
手写(B3)数据适配层;k-shot 窗口 padding 对统计 token 的轻微污染(按长度掩码)。
