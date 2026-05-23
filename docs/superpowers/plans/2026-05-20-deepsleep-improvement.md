# DeepSleep Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform DeepSleep into **星辰曦（小曦）** — a warm, personality-rich sleep health companion with chain-of-thought reasoning.

**Architecture:** ~199M MoE, 8 layers all-MoE, 8 routed experts (0 shared), top_k=2, `<thinking></thinking>` tokens, ~65M active per token.

**Tokenizer:** BPE, vocab=7200, 中英双语, 从 CCI4.0-HQ + SFT/DPO 数据训练。

**Pretrain Data:** CCI4.0-HQ from HuggingFace, streaming (no local download).

**Tech Stack:** PyTorch 2.9+, transformers 5.3+, HuggingFace Trainer, tokenizers (BPE), OpenAI SDK (DeepSeek API), wandb/tensorboard

---

## Overall Progress

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Architecture (MoE shared experts, think tokens, model config) | ✅ 完成 |
| Phase 2 | 小曦人格 + 双轨训练配置 + 评估脚本 | ✅ 完成 |
| Phase 3 | 数据生成 (SFT + DPO + 预训练语料 + Tokenizer) | ✅ 完成 (SFT 10000条, DPO 1965对, Tokenizer已训练) |
| Phase 4 | 训练 (pretrain → SFT → DPO) | 🔄 预训练脚本已重写(HF Trainer+streaming), 待正式训练 |
| Phase 5 | 评估 + 对比 + 模型选择 + 发布 | ❌ 未开始 |

---

## Phase 1: Architecture — ✅ 全部完成

> Phase 1 代码已合并到 `model/model_deepsleep.py` 和独立脚本中。

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | MoE多共享专家支持 (`nn.ModuleList`) | ✅ 完成 |
| Task 2 | Think tokens `<thinking></thinking>` 加入tokenizer | ✅ 完成 |
| Task 3 | DeepSleep模型配置 (~200M MoE, 10层交替) | ✅ 完成 |
| Task 4 | 睡眠语料筛选脚本 `prepare_sleep_corpus.py` | ✅ 完成 |
| Task 5 | 训练配置 | ✅ 完成 (argparse, 无YAML) |
| Task 6 | 集成测试 (前向传播) | ✅ 完成 |
| Task 7 | 文档更新 | ✅ 完成 |

---

## Phase 2: 小曦人格 + 双轨 — ✅ 全部完成

| Task | Description | Status |
|------|-------------|--------|
| Task 8 | 小曦人格定义 (6类SFT, 3500条总量) | ✅ 完成 (内联到 `generate_xiaoxi_all.py`) |
| Task 9 | 一键小曦SFT生成 `generate_xiaoxi_all.py` | ✅ 完成 (两步分离: prompts → responses) |
| Task 10 | 小曦DPO对比 `generate_xiaoxi_dpo.py` | ✅ 完成 (两步分离: prompts → pairs) |
| Task 11 | 双轨对比评估脚本 `compare_tracks.py` | ✅ 完成 (自包含, 2026-05-22 重写) |

---

## Phase 3: 数据生成 — ✅ 全部完成

### 3.1 SFT数据生成 — ✅ 完成 (10000条)

**脚本**: `scripts/generate_xiaoxi_all.py`
**输出**: `data/sft/xiaoxi/xiaoxi_sft.jsonl` (10000条, 6类别各达标)

| 类别 | 条数 | 目标 | 状态 |
|------|------|------|------|
| 专业诊断(CoT) | 2500 | 2500 | ✅ |
| 知心安慰 | 2500 | 2500 | ✅ |
| 趣味科普 | 1500 | 1500 | ✅ |
| 睡前引导 | 1000 | 1000 | ✅ |
| 拟人分享 | 1000 | 1000 | ✅ |
| 个性化互动 | 1500 | 1500 | ✅ |
| **合计** | **10000** | **10000** | **✅** |

> **Prompt总数**: 10110条 (all_prompts.jsonl)

### 3.2 DPO数据生成 — ✅ 完成

**脚本**: `scripts/generate_xiaoxi_dpo.py`
**输出**: `data/dpo/xiaoxi_dpo.jsonl` (1965/2006对)

| 类别 | Prompt | 已生成 |
|------|--------|--------|
| 专业诊断(CoT) | 800+ | 800 |
| 知心安慰 | 400+ | 403 |
| 趣味科普 | 300+ | 300 |
| 睡前引导 | 160+ | 163 |
| 拟人分享 | 150+ | 150 |
| 个性化互动 | 150+ | 149 |
| **合计** | **2006** | **1965** |

### 3.3 BPE 分词器训练 — ✅ 完成

**脚本**: `scripts/train_tokenizer.py`
**输出**: `checkpoints/tokenizer/tokenizer.json` + `tokenizer_config.json`

| 属性 | 值 |
|------|-----|
| 类型 | BPE (HuggingFace tokenizers) |
| Vocab size | 7200 |
| 特殊 tokens | `<pad>(0)`, `<\|im_start\|>(1)`, `<\|im_end\|>(2)`, `<unk>(3)`, `<s>(4)`, `</s>(5)`, `<thinking>(6)`, `</thinking>(7)`, `<summary>(8)` |
| ChatML 模板 | `<\|im_start\|>role\ncontent<\|im_end\|>\n` |

**训练语料：**

| 数据源 | 语言 | 内容 | 条数 |
|--------|------|------|------|
| `data/sft/xiaoxi/xiaoxi_sft.jsonl` | 中文 | 小曦SFT对话 (6类) | ~10000 |
| `data/dpo/xiaoxi_dpo.jsonl` | 中文 | 小曦DPO偏好对比 | ~1965 |
| `data/sft/xiaoxi/all_prompts.jsonl` | 中文 | SFT prompt (6类) | ~10110 |
| `pubmedqa/data/ori_pqal.json` | 英文 | PubMed生物医学问答 | 1000 |
| **合计** | | | **~20081 segments** |

> **限制**: 网络不可用，未下载额外英文语料。英文覆盖主要靠 PubMedQA。后续可补充。

```bash
python scripts/train_tokenizer.py  # 重新训练
```

### 3.4 预训练数据准备 — ❌ 未开始 (下一任务)

**脚本**: `scripts/prepare_deepsleep_data.py` (2026-05-22 重写)
**数据源**: HuggingFace CCI4.0-HQ 高质量子集
**目标**: 随机采样 ~12B tokens
**输出**: `data/cleaned/pretrain.jsonl`

```bash
# 推荐：用 tokenizer 精确计数
python scripts/prepare_deepsleep_data.py --tokenizer_path checkpoints/tokenizer

# 默认：字符数粗估
python scripts/prepare_deepsleep_data.py
```

---

## Phase 4: 训练 — 🔄 预训练流程已验证通过

> 前提: Phase 3 数据生成 ✅ 全部完成
> 预训练流程验证: 123M params, 75 steps, loss 10.48→0.38
> **当前阻塞**: 预训练数据 (`data/cleaned/pretrain.jsonl`) 未准备

### 4.1 Track B: From Scratch Pretrain

**训练脚本**: `trainer/train_pretrain.py`
**模型**: DeepSleep MoE (~200M params)
**数据**: `data/cleaned/pretrain.jsonl` (CCI4.0-HQ, ~12B tokens)

```bash
python trainer/train_pretrain.py \
    --data_path data/cleaned/pretrain.jsonl \
    --tokenizer_path checkpoints/tokenizer \
    --hidden_size 768 --num_hidden_layers 10 \
    --use_moe 1 --num_experts 8 --num_shared_experts 2 \
    --max_steps 30000 --batch_size 32 --learning_rate 3e-4
```

### 4.2 Track A: Qwen Continual Pretrain

**基座模型**: Qwen2.5-0.5B
**数据**: `data/cleaned/pretrain.jsonl` (CCI4.0-HQ)

### 4.3 Track A & B: SFT训练

**训练脚本**: `trainer/train_sft.py`
**数据**: `data/sft/xiaoxi/xiaoxi_sft.jsonl`

```bash
# Track B SFT
python trainer/train_sft.py \
    --data_path data/sft/xiaoxi/xiaoxi_sft.jsonl \
    --tokenizer_path checkpoints/tokenizer \
    --from_weight out/pretrain_deepsleep.pth

# Track A SFT
python trainer/train_sft.py \
    --data_path data/sft/xiaoxi/xiaoxi_sft.jsonl \
    --tokenizer_path checkpoints/tokenizer \
    --from_weight out/pretrain_qwen.pth
```

### 4.4 Track A & B: DPO训练

**训练脚本**: `trainer/train_dpo.py`
**数据**: `data/dpo/xiaoxi_dpo.jsonl`

```bash
# Track B DPO
python trainer/train_dpo.py \
    --data_path data/dpo/xiaoxi_dpo.jsonl \
    --tokenizer_path checkpoints/tokenizer \
    --sft_checkpoint out/sft_deepsleep.pth

# Track A DPO
python trainer/train_dpo.py \
    --data_path data/dpo/xiaoxi_dpo.jsonl \
    --tokenizer_path checkpoints/tokenizer \
    --sft_checkpoint out/sft_qwen.pth
```

### 4.5 训练所需资源

| 资源 | 要求 |
|------|------|
| GPU | A10 (24GB) 或同等级 |
| 预计训练时间 | Track A: ~5h, Track B: ~10h (合计 ~15h) |
| 磁盘空间 | ~50GB (数据+模型checkpoint) |
| API费用 | SFT+DPO数据生成: ~$5-10 (DeepSeek Flash) |

---

## Phase 5: 评估与发布 — ❌ 未开始

> 前提: Phase 4 训练全部完成

### 5.1 双轨对比评估

**脚本**: `scripts/compare_tracks.py` (自包含, 2026-05-22 重写)

```bash
python scripts/compare_tracks.py \
    --track_a out/sft_a.pth \
    --track_b out/sft_b.pth \
    --tokenizer_path checkpoints/tokenizer \
    --output comparison_results.json
```

评估维度:
1. **MCQ准确率** — 15道睡眠医学知识选择题 (嵌入脚本)
2. **CoT推理率** — `<thinking>` tag出现率和结构化推理质量
3. **人格一致性** — 小曦风格评估 (温暖/有趣/个性化)
4. **推理延迟** — 生成速度对比

### 5.2 模型选择

基于评估结果选择最佳Track。

### 5.3 模型导出与发布

- 导出 HuggingFace 格式
- 创建 Gradio demo (`app.py`)
- 可选: 上传到 HuggingFace Hub

---

## Known Issues

### `train_sft.py` 缺 LoRA 支持

Track A (Qwen base) 的continual pretrain和SFT可能需要LoRA以适应24GB显存。计划添加 `--use_lora` 参数。

### 项目无 `configs/` 目录

实际训练使用 argparse 参数，不依赖YAML配置。若需要YAML配置可后续创建。

---

## Update Log

- **2026-05-22**: SFT数据10000条全部完成 + DPO数据1965对 — SFT 6类别各达目标(专业诊断2500、知心安慰2500、趣味科普1500、睡前引导1000、拟人分享1000、个性化互动1500)。DPO数据扩容至1965对。DPO prompt 2006条。Phase 3 数据生成全部完成，进入 Phase 4 训练准备阶段。下一任务：预训练数据准备 + Track B Pretrain。
- **2026-05-22**: Tokenizer训练完成 — 使用 HuggingFace tokenizers BPE, vocab=7200, 从项目 SFT/DPO/PubMedQA 数据训练 (~15969 segments)。创建 `train_tokenizer.py` 脚本和 `test_pretrain.py` 测试脚本。预训练流程验证通过 (123M params, loss 10.48→0.38, 75 steps)。模型默认 vocab_size 从 32000 改为 7200。
- **2026-05-22**: SFT数据扩容至10000条 — `ALL_CATEGORIES` 目标从 3500 提升至 10000。prompt 生成改为全并发模式（`_generate_one_batch` + ThreadPoolExecutor 20 workers），与 response 生成一致。断点续生：prompt 去重追加，response 跳过已有。当前 4643/10000。
- **2026-05-22**: 修复SFT thinking不一致 — 修改 `generate_xiaoxi_all.py` 所有类别加入 thinking 格式，新增 `--step supplement` 补充模式。
- **2026-05-22**: 修复脚本 — 删除废弃的 `generate_cot_data.py` 和 `generate_dpo_data.py`；重写 `compare_tracks.py` 为自包含脚本；重写 `prepare_deepsleep_data.py` 改用 CCI4.0-HQ；更新文档。
- **2026-05-21**: 更新计划文档 — 完整记录Phase 1-5状态。Phase 3 SFT数据生成中 (448/3500)。CLAUDE.md同步更新。
- **2026-05-21**: 架构重构 — 合并模型文件, 废弃YAML, 修复review bug, 创建两步分离数据生成脚本, 创建预训练数据下载脚本, 配置API key。
- **2026-05-20**: Phase 1+2 代码完成 — MoE共享专家, think tokens, 模型配置, 小曦人格, 双轨配置, 评估。
- **2026-04-07**: 项目初始化
