<div align="center">

# 🌙 DeepSleep

**睡眠健康领域轻量级 MoE 大语言模型**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**~200M 参数 MoE** | 中英双语 | 睡眠健康领域 | 单卡可训练

[快速开始](#-快速开始) · [模型架构](#-模型架构) · [训练流程](#-训练流程) · [项目结构](#-项目结构)

</div>

---

## 📖 项目简介

DeepSleep 是一个从零开始构建的睡眠健康领域 MoE 大语言模型，灵感来自 [MiniMind](https://github.com/jingyaogong/minimind) 和 [Qwen2.5-MoE](https://qwenlm.github.io/blog/qwen2.5-moe/)。项目完整实现了大模型的全部流程：**Tokenizer 训练 → 预训练 → SFT → DPO → Web 部署**，所有代码开源可复现。

**核心特点：**

- 🧠 **MoE 架构** — 8 层全 MoE，~199M 总参数 / ~65M 活跃参数，softmax 路由
- 🔬 **主流组件** — GQA + RoPE + RMSNorm + SwiGLU + Flash Attention
- 🇨🇳 **中英双语** — 7200 BPE 词表，从 CCI4.0-HQ 语料训练
- 🤖 **小曦人格** — 温暖有趣的睡眠健康伙伴，10000 条 SFT + 1965 对 DPO 数据
- 💻 **单卡可训练** — NVIDIA A10 (24GB) 即可完成全流程
- 📦 **一键启动** — YAML 配置 + Shell 脚本，开箱即用

---

## 🏗 模型架构

```
DeepSleepForCausalLM (~199M params)
├── Embedding (vocab=7200, d_model=768, tied with lm_head)
├── 8 MoE Layers
│   ├── DeepSleepAttention (GQA: 8Q/4KV heads, head_dim=96, RoPE, Flash/SDPA)
│   └── DeepSleepMoE (8 routed experts, top_k=2, SwiGLU, intermediate=1216)
├── Final RMSNorm
└── LM Head (tied, no bias)

Total: ~199M | Active per token: ~65M | Utilization: 32.4%
```

| 超参数 | 值 |
|--------|-----|
| d_model | 768 |
| n_layers | 8 |
| n_heads / n_kv_heads | 8 / 4 (GQA) |
| head_dim | 96 |
| num_routed_experts | 8 |
| top_k | 2 |
| moe_intermediate_size | 1216 |
| vocab_size | 7,200 |
| max_position_embeddings | 8,192 |

**主流组件：** GQA (Grouped Query Attention) · RoPE (Rotary Position Embedding) · RMSNorm · SwiGLU · Flash Attention (SDPA) · Pre-Norm

---

## 🚀 快速开始

### 1. 环境配置

```bash
git clone https://github.com/L-0915/deepsleep.git
cd deepsleep

conda create -n deepsleep python=3.10 -y
conda activate deepsleep
pip install -r requirements.txt
```

### 2. 训练 Tokenizer

```bash
# 从 CCI4.0-HQ 流式加载中英文语料训练 BPE tokenizer
python scripts/train_tokenizer.py --use_cci4 --cci4_max_docs 300000
```

### 3. 训练模型

```bash
# 方式一：一键启动（推荐）
bash scripts/run/run_all.sh          # 全流程：pretrain → SFT → DPO

# 方式二：分阶段运行
bash scripts/run/run_pretrain.sh     # 预训练（流式加载 CCI4.0-HQ）
bash scripts/run/run_sft.sh          # SFT 微调（小曦人格数据）
bash scripts/run/run_dpo.sh          # DPO 对齐

# 方式三：直接命令行
python trainer/train_pretrain.py --config configs/pretrain.yaml --tokenizer_path checkpoints/tokenizer
```

### 4. Web 演示

```bash
python app.py --model out/dpo/final
```

---

## 📊 训练流程

```
Stage 1: Pretrain (流式加载 CCI4.0-HQ, 无需本地下载)
├── 数据: CCI4.0-HQ (中英文混合, HuggingFace streaming)
├── 配置: batch=128 (16×8), seq_len=2048, lr=5e-4, cosine
├── 特性: MoE-aware loss, TensorBoard, 样本生成, checkpoint/resume
└── 目标: ~50K steps

Stage 2: SFT (小曦人格微调)
├── 数据: 10000 条 ChatML 对话 (6类别: 专业诊断/知心安慰/趣味科普/睡前引导/拟人分享/个性化互动)
├── 配置: batch=16, lr=1e-5, 3 epochs
└── 目标: 学会小曦温暖有趣的人格风格

Stage 3: DPO (偏好对齐)
├── 数据: 1965 对偏好对比 (6类别, chosen=小曦风格 vs rejected=通用AI风格)
├── 配置: batch=4, lr=5e-7, beta=0.1, 1 epoch
└── 目标: 强化小曦人格, 拒绝通用 AI 回答
```

### 超参数配置

所有训练超参数均在 YAML 配置文件中，可自由修改：

```bash
configs/
├── pretrain.yaml    # 预训练配置 (模型/数据/训练/日志)
├── sft.yaml         # SFT 配置
└── dpo.yaml         # DPO 配置
```

也可以通过环境变量覆盖：

```bash
LR=3e-4 MAX_STEPS=30000 bash scripts/run/run_pretrain.sh
```

---

## 📁 项目结构

```
deepsleep/
├── model/
│   └── model_deepsleep.py          # 完整模型: Config, Attention, MoE, CausalLM
├── dataset/
│   ├── lm_dataset.py               # PretrainDataset, SFTDataset, DPODataset
│   └── streaming_dataset.py        # CCI4PretrainDataset (流式加载)
├── trainer/
│   ├── train_pretrain.py           # 预训练 (HuggingFace Trainer, MoE-aware)
│   ├── train_sft.py                # SFT 微调
│   ├── train_dpo.py                # DPO 对齐
│   └── trainer_utils.py            # 共享工具函数
├── configs/                        # 训练配置 (YAML)
│   ├── config_utils.py             # YAML → argparse 加载器
│   ├── pretrain.yaml
│   ├── sft.yaml
│   └── dpo.yaml
├── scripts/
│   ├── run/                        # 一键启动脚本
│   │   ├── run_pretrain.sh
│   │   ├── run_sft.sh
│   │   ├── run_dpo.sh
│   │   └── run_all.sh
│   ├── generate_xiaoxi_all.py      # 小曦 SFT 数据生成
│   ├── generate_xiaoxi_dpo.py      # 小曦 DPO 数据生成
│   ├── train_tokenizer.py          # BPE 分词器训练
│   ├── prepare_deepsleep_data.py   # 预训练数据下载工具
│   ├── prepare_sleep_corpus.py     # 睡眠语料筛选
│   └── compare_tracks.py           # 双轨对比评估
├── tests/                          # 单元测试
├── app.py                          # Gradio Web UI
├── Makefile
├── requirements.txt
└── LICENSE
```

---

## 🤖 小曦人格

**星辰曦（小曦）** 是 DeepSleep 的 AI 人格，定位为温暖、有趣的睡眠健康伙伴。

| 能力 | 数据量 | 示例 |
|------|--------|------|
| 专业诊断 (CoT) | 2500 条 | 症状分析 → 推理 → 建议 |
| 知心安慰 | 2500 条 | 共情 + 实用建议 |
| 趣味科普 | 1500 条 | 睡眠冷知识 + 比喻 |
| 睡前引导 | 1000 条 | 呼吸放松、冥想脚本 |
| 拟人分享 | 1000 条 | 小曦的生活小故事 |
| 个性化互动 | 1500 条 | 记住用户、主动回访 |

---

## 💡 参考 & 致谢

- [MiniMind](https://github.com/jingyaogong/minimind) — 轻量级 LLM 训练框架
- [Qwen2.5-MoE](https://qwenlm.github.io/blog/qwen2.5-moe/) — MoE 架构设计灵感
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — softmax routing + aux-loss
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — 模型框架

---

## 📄 License

[MIT License](LICENSE)
