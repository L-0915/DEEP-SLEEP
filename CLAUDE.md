# DeepSleep: Medical Sleep Health Domain LLM

## Project Overview

DeepSleep is a ~201.6M parameter (~64M active per token) Mixture-of-Experts (MoE) language model purpose-built for the medical sleep health domain. It supports Chinese-English bilingual input and is designed for sleep health consultation, diagnosis assistance, and sleep medicine knowledge services.

**Architecture:** Qwen2.5-MoE inspired | **Framework:** HuggingFace | **Vocab:** 32K BPE | **Context:** 8K tokens (extendable to 32K via NTK-RoPE)

Design spec: [docs/superpowers/specs/2026-04-07-deepsleep-design.md](docs/superpowers/specs/2026-04-07-deepsleep-design.md)

---

## Project Directory Structure

```
deepsleep/
├── configs/            # YAML configs (model, data, train, eval, deploy)
├── src/
│   ├── model/          # Model architecture (config, attention, MoE, embedding, layers)
│   ├── data/           # Crawlers, processing, datasets, synthetic data, tokenizer
│   ├── training/       # Pretrain, SFT, DPO training loops
│   ├── evaluation/     # Benchmarks, judge, safety eval, inference bench
│   ├── inference/      # API server, chat, quantization
│   └── utils/          # Distributed, logging, FSDP, checkpointing
├── scripts/            # Operational scripts (crawling, cleaning, training, eval)
├── data/
│   ├── cleaned/        # Cleaned unified pretrain dataset (3.2M docs, 15GB)
│   ├── keywords/       # Sleep domain keyword lists (en/zh)
│   ├── tokenizer_corpus/ # Tokenizer training corpus (~10GB, en+zh)
│   └── IndustryCorpus_medicine/ # Source medical corpus (169GB, en+zh jsonl.gz)
├── checkpoints/        # Model checkpoints (empty - training not started)
├── output/             # Final model outputs (empty)
├── tests/              # Unit & integration tests
└── data_cleanse_pipeline.py  # Standalone data cleaning pipeline
```

---

## Current Progress

### Completed

| Phase | Status | Details |
|-------|--------|---------|
| **Model Architecture** | Done | DeepSleepConfig, DeepSleepForCausalLM, DeepSleepAttention (GQA + FlashAttn), DeepSleepMoE (DeepSeek-style softmax routing, aux/z-loss), SwiGLU MLP, RoPE, RMSNorm, weight tying |
| **Tokenizer Integration** | Done | DeepSleepTokenizer (PreTrainedTokenizerFast), ChatML template, HuggingFace auto-registration |
| **Data Crawlers** | Done | PubMed, PMC OA, arXiv, Wikipedia (en/zh), ClinicalTrials.gov, medical websites, Zhihu, general crawlers |
| **Data Processing** | Done | Dedup (MinHash + exact), quality scoring (perplexity + heuristics), language detection (fasttext), format cleaning (HTML/PDF), safety filtering (PII + harmful content), domain scoring (BM25 + keywords) |
| **Data Cleaning Pipeline** | Done | `data_cleanse_pipeline.py` - 4-stage pipeline: quality filter, text cleaning, privacy scrubbing, dedup. Processes IndustryCorpus data |
| **Scripts** | Done | 30+ operational scripts for crawling, cleaning, merging, tokenizer corpus preparation |
| **Unified Pretrain Dataset** | Done | `data/cleaned/deepsleep_pretrain_unified.jsonl` - 3,237,797 documents, 12.6GB text (62% en, 38% zh) |
| **Tokenizer Corpus** | Done | `data/tokenizer_corpus/{en,zh}_corpus.txt` - ~5GB each, ready for BPE training |
| **Dataset Classes** | Done | PretrainDataset (memory-mapped), SFTDataset (ChatML), DPODataset (preference pairs) |
| **Training Loops** | Done | Pretrain (FSDP, BF16, gradient accumulation), SFT (LoRA + full), DPO (reference model) |
| **Training Infrastructure** | Done | Callbacks (checkpoint, eval, wandb), loss functions (LM + DPO), cosine scheduler, FSDP config, distributed utils, checkpoint management |
| **Evaluation Framework** | Done | Benchmark runner, LLM judge, safety evaluator, sleep-med QA, inference benchmark |
| **Inference/Deploy** | Done | FastAPI server (OpenAI-compatible), chat interface, quantization (GGUF export), vLLM config |
| **Synthetic Data** | Done | Generator, augmenter, quality filter, seed prompts for sleep domain conversations |
| **Config System** | Done | Full YAML configs for all pipeline stages (model, data, train, eval, deploy) |
| **Tests** | Done | test_model.py, test_tokenizer.py, test_data_pipeline.py, test_training.py |
| **Pretrain v4** | Done | 11,718 steps, 3 epochs, loss 3.00→3.00, ~1.5h on RTX 3090. All-MoE 8层, 6 routed experts, top_k=2, d_model=768 |
| **SFT v4** | Done | 32,625 steps, 3 epochs, train loss 2.60, eval loss 1.84. Data: industry_instruction_fixed_医疗 (中文医疗问答SFT数据) |
| **DPO v2** | Done | 320 steps, 1 epoch, train loss 0.624, eval loss 0.628, accuracy 63.6%, margin 0.153. 5,405 pairs (5,351 general medical + 54 high-quality sleep domain) |

### Not Yet Started

| Phase | Status | What Needs to Be Done |
|-------|--------|----------------------|
| **Full Evaluation** | Not started | 用多样化 prompt 全面测试 DPO vs SFT 模型质量对比，运行 benchmark 评估 |
| **Model Export** | Not started | Export to GGUF (edge) and vLLM format (server). scripts/export_gguf.py exists |
| **HuggingFace Hub** | Not started | Upload final model to deepsleep-ai/deepsleep-2b |
| **More DPO Data** | Optional | 目前仅54条高质量睡眠DPO数据，可继续补充更多（当前数据以通用医学DPO为主） |
| **Multi-epoch DPO** | Optional | 当前 accuracy 63.6%，可尝试2-3个epoch提升对齐效果 |

---

## Key Commands

```bash
# Install
pip install -e ".[dev]"

# Data
make crawl          # Run all crawlers
make process        # Run data processing pipeline

# Tokenizer
bash scripts/train_tokenizer.sh

# Training (single RTX 3090 or any CUDA GPU with >=16GB VRAM)
python src/training/pretrain.py --config configs/train/pretrain.yaml
python src/training/sft.py --config configs/train/sft.yaml
python src/training/dpo.py --config configs/train/dpo.yaml

# Evaluation
make eval           # Run benchmarks

# Serving
make serve          # Start API server

# Quality
make format         # black + ruff format
make lint           # ruff + mypy
make test           # pytest
```

---

## Unified Dataset Statistics

Source: `data/cleaned/unified_dataset_report.json`

| Metric | Value |
|--------|-------|
| Total documents | 3,237,797 |
| Total characters | 12,608,462,592 (~12.6 GB) |
| English docs | 2,023,984 (62.5%) |
| Chinese docs | 1,213,736 (37.5%) |
| Avg doc length | 3,894 chars |
| Median doc length | 2,764 chars |

**Sources:**
- IndustryCorpus EN: 2,000,000 docs
- IndustryCorpus ZH (cleaned): 714,293 docs
- IndustryCorpus ZH (raw): 500,000 docs
- Crawled data: 23,504 docs

---

## Model Architecture Quick Reference

```
DeepSleepForCausalLM (v4 - 实际训练版本)
├── Embedding (vocab=32K, dim=768, tied with lm_head)
├── 8 Decoder Layers (all_moe)
│   ├── DeepSleepAttention (GQA: 8Q/4KV heads, head_dim=96, RoPE)
│   ├── RMSNorm (pre-norm)
│   └── DeepSleepMoE - 6 routed + 0 shared experts, top_k=2
│       └── DeepSeek-style softmax routing (aux_loss=0.1, z_loss=0.01)
│       └── SwiGLU expert FFN (moe_intermediate=1472)
├── Final RMSNorm
└── LM Head (tied, no bias)

Total: ~201.6M params | Active per token: ~64M
Layer pattern: all_moe (全部8层都是MoE)
```

---

## Data Pipeline Flow

```
IndustryCorpus (169GB gzipped) + Web Crawlers
  → scripts/process_industry_corpus.py / data_cleanse_pipeline.py
  → Quality Filter → Text Clean → Privacy Scrub → Dedup
  → data/cleaned/deepsleep_pretrain_unified.jsonl (15GB, 3.2M docs)

Tokenizer Corpus (10GB en+zh) → train_tokenizer.sh → BPE 32K vocab

Pretrain v4 (11,718 steps, loss 3.00)
  → SFT v4 (32,625 steps, train loss 2.60, eval loss 1.84)
  → DPO v2 (320 steps, train loss 0.624, eval accuracy 63.6%)
  → Final Model: /root/autodl-tmp/data/deepsleep_model_dpo_v4_r2/final_model/
```

---

## Dependencies

Core: PyTorch >=2.1, transformers >=4.40, datasets >=2.18, trl >=0.8, accelerate >=0.29, deepspeed >=0.14, flash-attn >=2.5

Data: biopython, arxiv, scrapy, trafilatura, wikipedia-api, pymupdf, fasttext, datasketch

Training: sentencepiece, tiktoken, einops, wandb, hydra-core

Serving: fastapi, uvicorn, vllm

Dev: pytest, black, ruff, mypy

---

## Known Issues / Bugs

1. ~~**modeling_deepsleep.py line 580**: References undefined variable `hidden_states_out`~~ - **FIXED** 2026-04-22: 已改为 `all_hidden_states`
2. ~~**feedforward.py duplicate**: Both `src/model/feedforward.py` and `src/model/layers.py` define `DeepSleepMLP`~~ - **FIXED** 2026-04-22: 已删除无用的 feedforward.py
3. **Crawled data volume low**: Only 23,504 crawled docs vs 3.2M total - most data comes from IndustryCorpus, not domain-specific crawlers. Sleep domain density may be insufficient

---

## Trained Model Artifacts

| Model | Path | Key Metrics |
|-------|------|-------------|
| **Pretrain v4** | `/root/autodl-tmp/data/deepsleep_model_v4/final_model/` | 11,718 steps, loss 3.00, 3 epochs |
| **SFT v4** | `/root/autodl-tmp/data/deepsleep_model_sft_v4/final_model/` | 32,625 steps, train 2.60, eval 1.84, 3 epochs |
| **DPO v2** | `/root/autodl-tmp/data/deepsleep_model_dpo_v4_r2/final_model/` | 320 steps, train 0.624, eval 0.628, acc 63.6% |
| **DPO v1 (deprecated)** | `/root/autodl-tmp/data/deepsleep_model_dpo_v4/` | 生成严重退化（模板数据导致"以下"重复循环），已废弃 |

### Training Data

| Data | Path | Details |
|------|------|---------|
| **SFT data** | `/root/autodl-tmp/data/industry_instruction_fixed_医疗_{train,eval}.jsonl` | 中文医疗问答SFT数据 |
| **DPO data (merged)** | `/root/autodl-tmp/data/medical_evidence_DPO/merged_dpo_v2.jsonl` | 5,405条 (5,351 general + 54 sleep domain) |
| **DPO data (original)** | `/root/autodl-tmp/data/medical_evidence_DPO/merged_dpo_dataset.jsonl` | 5,351条 (deepseek/dpo_answer/clinical/filter) |
| **Sleep DPO (Claude-generated)** | `/root/autodl-tmp/data/medical_evidence_DPO/sleep_dpo_direct_{1..7}.jsonl` | 54条高质量睡眠领域DPO数据，Claude直接生成 |

### Training Scripts

| Script | Location |
|--------|----------|
| **Pretrain** | `/root/autodl-tmp/data/train_deepsleep_native.py` |
| **SFT** | `/root/autodl-tmp/data/sft_deepsleep_native.py` |
| **DPO** | `/root/autodl-tmp/data/train_deepsleep_dpo.py` |
| **Sleep DPO generator** | `/root/autodl-tmp/data/medical_evidence_DPO/generate_sleep_dpo_v2.py` (script-generated, 未使用) |

### Training Config (DPO)

- Base model: SFT v4
- Batch size: 4 × 4 (gradient accumulation) = 16
- Seq length: 768
- LR: 5e-7, beta: 0.1
- 1 epoch, cosine schedule
- Reference model: frozen copy of policy model

---

## Next Milestone: Evaluation & Deployment

训练三阶段已全部完成。下一步：

1. **全面评估** — 多样化 prompt 测试，对比 SFT vs DPO 生成质量
2. **模型导出** — GGUF 格式（边缘部署）、vLLM 配置（服务端部署）
3. **发布** — 上传 HuggingFace Hub

---

## Important Notes for Development

- **实际模型规模**: ~201.6M 总参数（非最初设计的1.9B），8层全MoE，6个routed expert + 0个shared expert，top_k=2，d_model=768，vocab_size=32K。每token激活约64M参数。
- **单卡即可训练**: 单张 RTX 3090 (24GB) 完成全部训练（Pretrain ~1.5h + SFT ~2h + DPO ~4min）。DPO阶段需同时加载policy+reference两个模型，约需1.6GB显存。
- **训练完成**: Pretrain v4 → SFT v4 → DPO v2 三阶段全部完成。最终模型在 `/root/autodl-tmp/data/deepsleep_model_dpo_v4_r2/final_model/`。
- **Data is gitignored**: `data/raw/`, `data/processed/`, `data/tokenized/`, `data/sft/`, `data/dpo/` are in `.gitignore`. Only `data/cleaned/`, `data/keywords/`, and `data/tokenizer_corpus/` are tracked.
- **Model files gitignored**: `*.bin`, `*.pt`, `*.safetensors`, `*.gguf` are excluded from git.
- **Cost estimate**: 单卡 3090 云租用约 2-3 元/小时，完整训练管线预估几十元级别。

---

## Update Log

- **2026-04-07**: Project initialized, design spec approved
- **2026-04-07 ~ 04-09**: All source code modules written (model, data, training, eval, inference, utils)
- **2026-04-09 ~ 04-15**: Data crawling campaigns (PubMed, PMC, arXiv, Wikipedia, medical sites, Zhihu)
- **2026-04-15 ~ 04-17**: Data cleaning and merging (IndustryCorpus + crawled data → unified dataset)
- **2026-04-19**: Unified pretrain dataset finalized: 3.2M docs, 15GB
- **2026-04-20**: Tokenizer corpus prepared: ~10GB (5GB en + 5GB zh)
- **2026-04-22**: CLAUDE.md created - project status: code complete, data collected, training not started
- **2026-04-22**: Updated training hardware requirement - 参考 MiniMind，单卡 RTX 3090 即可训练，无需 A100/H100 集群
- **2026-04-25**: Pretrain v4 完成 - 11,718 steps, loss 3.00, all-MoE 8层, DeepSeek-style routing
- **2026-04-26**: SFT v4 完成 - 32,625 steps, train loss 2.60, eval loss 1.84, 中文医疗问答
- **2026-04-27**: DPO v1 完成（废弃）- 模板化DPO数据导致生成严重退化，"以下"重复循环
- **2026-04-27**: DPO v2 完成 - 去掉模板数据，使用5,405条优质数据，生成质量正常，train loss 0.624, eval accuracy 63.6%
