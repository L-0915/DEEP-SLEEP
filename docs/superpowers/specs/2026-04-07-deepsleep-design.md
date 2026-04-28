# DeepSleep-2B: Medical Sleep Health Domain LLM

## Design Specification

**Date:** 2026-04-07
**Status:** Approved
**Author:** DeepSleep Team

---

## 1. Overview

DeepSleep is a lightweight (~2B params, ~0.7B active) Mixture-of-Experts language model purpose-built for the medical sleep health domain. It supports Chinese-English bilingual input and is designed for sleep health consultation, diagnosis assistance, and sleep medicine knowledge services.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Qwen2.5-MoE style | Best Chinese support, mature ecosystem |
| Training | From scratch | Full control over domain knowledge |
| Data strategy | Public + Crawled + Synthetic | Maximum coverage of sleep domain |
| Framework | HuggingFace + FSDP | Mature tooling, good HF integration |
| Deployment | Server (vLLM) + Edge (GGUF) | Max reach, flexible usage |
| Hardware | Cloud GPU rental (A100/H100) | Cost-effective for bursty training |

---

## 2. Model Architecture

### 2.1 Core Architecture

```
Model: DeepSleep-2B (Qwen2.5-MoE Inspired)
Total Parameters:  ~1.92B
Active Parameters: ~0.72B
```

### 2.2 Detailed Configuration

| Hyperparameter | Value |
|----------------|-------|
| `d_model` | 2048 |
| `n_layers` | 24 (12 Dense + 12 MoE, alternating) |
| `n_heads` (Q) | 16 |
| `n_kv_heads` (KV) | 4 (GQA ratio 4:1) |
| `head_dim` | 128 (2048 / 16) |
| `vocab_size` | 64,000 |
| `max_position_embeddings` | 8,192 |
| `hidden_act` | SiLU (Swish) |
| `rms_norm_eps` | 1e-6 |
| `tie_word_embeddings` | True |
| `use_bias` | False |

### 2.3 Dense Layer Configuration (12 layers)

| Component | Shape | Params per layer |
|-----------|-------|-----------------|
| Q projection | (2048, 2048) | 4.2M |
| K projection | (2048, 512) | 1.0M |
| V projection | (2048, 512) | 1.0M |
| O projection | (2048, 2048) | 4.2M |
| SwiGLU gate | (2048, 5632) | 11.5M |
| SwiGLU up | (2048, 5632) | 11.5M |
| SwiGLU down | (5632, 2048) | 11.5M |
| **Subtotal per Dense layer** | | **44.9M** |
| **12 Dense layers total** | | **539M** |

### 2.4 MoE Layer Configuration (12 layers)

| Component | Shape | Params per layer |
|-----------|-------|-----------------|
| Q/K/V/O projections | same as Dense | 10.5M |
| Routed experts (6) | SwiGLU (2048 -> 1408) | 6 x 11.5M = 69M |
| Shared expert (1) | SwiGLU (2048 -> 5632) | 23M |
| Router | (2048 -> 6) | 0.012M |
| **Subtotal per MoE layer** | | **102.5M** |
| **12 MoE layers total** | | **1,230M** |

### 2.5 MoE Router Details

- **Top-K**: 2 (select 2 of 6 routed experts)
- **Auxiliary loss coefficient**: 0.01 (load balancing)
- **Z-loss coefficient**: 0.001 (numerical stability)
- **Jitter noise**: 0.1 (during training only)
- **Expert capacity factor**: 1.25
- **Drop tokens**: True (when expert capacity exceeded)
- **Noisy gating**: True

### 2.6 Parameter Summary

| Component | Params |
|-----------|--------|
| Embedding (tied) | 131M |
| Dense layers (12) | 539M |
| MoE layers (12) | 1,230M |
| Router params (12) | 0.15M |
| RMSNorm (25) | 0.1M |
| **Total** | **~1.9B** |
| **Active (per token)** | **~0.72B** |

### 2.7 Positional Encoding

- **Type**: Rotary Position Embedding (RoPE)
- **Base frequency**: 10,000
- **NTK-aware scaling**: Enabled for extended context
- **Max context**: 8,192 tokens (configurable to 32K with NTK)

---

## 3. Tokenizer

### 3.1 Specification

| Property | Value |
|----------|-------|
| Type | BPE (Byte-Pair Encoding) |
| Vocab size | 64,000 |
| Normalizer | NFKC Unicode normalization |
| Pre-tokenizer | Byte-level (Qwen2 style regex) |

### 3.2 Special Tokens

```
<pad> <eos> <unk> <|endoftext|>
<s> </s>
<|im_start|> <|im_end|>
<|user|> <|assistant| <|system|>
<|tool_call|> <|tool_response|>
<|thinking|> <|summary|>
```

### 3.3 Training Data Distribution

| Source | Proportion |
|--------|-----------|
| Chinese sleep medicine literature | 40% |
| English sleep medicine literature | 25% |
| General Chinese text | 25% |
| General English text | 10% |

### 3.4 Training Procedure

- **Training corpus**: ~10GB raw text, sampled from all data sources
- **Tool**: tiktoken (fast BPE implementation)
- **Evaluation**: Compression ratio on held-out sleep domain text

---

## 4. Data Pipeline

### 4.1 Data Sources

| Source | Type | Est. Tokens | Access Method | Relevance |
|--------|------|------------|---------------|-----------|
| PubMed/MEDLINE | Paper abstracts | ~500M | E-utilities API | High |
| PMC Open Access | Full-text papers | ~2B | FTP bulk download | High |
| CNKI/Wanfang (open) | Chinese papers | ~200M | Open access | High |
| Wikipedia (zh+en) | Encyclopedia | ~5B | Wikimedia dumps | Medium |
| C4-zh (filtered) | Chinese web | ~50B | HuggingFace | Low-Medium |
| CC-100-CN (filtered) | Chinese web | ~20B | HuggingFace | Low-Medium |
| AASM/NSF/Sleep.org | Medical websites | ~50M | Scrapy | High |
| Zhihu (sleep topic) | Q&A community | ~100M | Crawler | High |
| DXY (sleep related) | Medical community | ~50M | Crawler | High |
| Chinese med textbooks | Books | ~500M | Public PDFs | High |
| arXiv (sleep/neuro) | Preprints | ~300M | arXiv API | Medium-High |
| OpenWebText / Pile | English general | ~100B | HuggingFace | Low |
| Synthetic dialogues | AI-generated | ~5B | GPT-4/Claude | High |
| **Total** | | **~200B+** | | |

### 4.2 Processing Pipeline

#### Step 1: Deduplication

- **MinHash LSH**: Document-level dedup (Jaccard similarity > 0.7)
- **Exact dedup**: Paragraph-level, n-gram hash (5-gram)
- **URL dedup**: Remove duplicate crawled URLs

#### Step 2: Quality Filtering

- **Language detection**: fasttext lid.176.bin (Chinese/English)
- **Perplexity filter**: Train kenlm 5-gram model, reject > threshold
- **Toxicity filter**: Classifier + keyword blocklist
- **Medical relevance score**: BM25 against medical terminology dictionary
- **Format cleaning**: HTML->text (trafilatura), PDF->text (marker/pymupdf)

#### Step 3: Sleep Domain Scoring

- **Coarse filter**: 1000+ sleep medicine keyword matching
- **Fine ranking**: Trained relevance classifier (text -> sleep probability)
- **Buckets**:
  - High relevance (>0.8): Sleep-specific papers, guidelines, textbooks
  - Medium relevance (0.5-0.8): General medical, physiology
  - Low relevance (0.2-0.5): General health, wellness
  - Irrelevant (<0.2): General text

#### Step 4: Data Mixing

| Category | Proportion | Purpose |
|----------|-----------|---------|
| Sleep domain data | 40% | Core domain knowledge |
| General medical data | 15% | Medical context |
| Chinese general corpus | 25% | Language fluency |
| English general corpus | 15% | Bilingual capability |
| Code/structured data | 5% | Structured reasoning |

### 4.3 Data Format

- Tokenized data stored as memory-mapped `.bin` files
- Shard size: ~10B tokens per file
- Format: uint16 token IDs, sequential, no padding

---

## 5. Training Pipeline

### 5.1 Stage 1: Pre-training

| Parameter | Value |
|-----------|-------|
| Total tokens | 200-500B |
| Hardware | 4x A100 80GB or 2x H100 80GB |
| Parallelism | FSDP (Fully Sharded Data Parallel) |
| Sequence length | 8,192 tokens |
| Global batch size | 512 |
| Micro batch per GPU | 16 |
| Peak LR | 3e-4 |
| LR schedule | Cosine decay with 2000 warmup steps |
| Weight decay | 0.1 |
| Adam (beta1, beta2, eps) | (0.9, 0.95, 1e-8) |
| Gradient clipping | 1.0 |
| Precision | BF16 mixed (with loss scaling) |
| Checkpoint interval | Every 5,000 steps |
| Eval interval | Every 10,000 steps |
| Est. duration | 14-35 days (A100) / 7-17 days (H100) |
| Est. cost | $3,000-12,000 |

### 5.2 Stage 2: SFT (Supervised Fine-Tuning)

| Parameter | Value |
|-----------|-------|
| Training data | 50K-200K high-quality sleep consultation dialogues |
| Hardware | 2x A100 80GB |
| Strategy | LoRA r=64, alpha=128 (initial) -> Full fine-tune (final) |
| Sequence length | 4,096 tokens |
| LR | 2e-5 (cosine, 100 warmup) |
| Epochs | 3-5 |
| Batch size | 128 |
| Est. duration | 1-3 days |
| Est. cost | $200-600 |

#### SFT Data Distribution

| Category | Proportion |
|----------|-----------|
| Sleep disorder diagnosis consultation | 30% |
| Sleep hygiene guidance | 20% |
| Sleep study report interpretation | 15% |
| Medication/treatment knowledge | 10% |
| Pediatric/special population sleep | 10% |
| Sleep research methods / PSG interpretation | 8% |
| Psychology/anxiety-related sleep | 7% |

### 5.3 Stage 3: DPO (Direct Preference Optimization)

| Parameter | Value |
|-----------|-------|
| Preference data | 10K-30K pairs |
| Hardware | 2x A100 80GB |
| Strategy | Full fine-tune + DPO loss |
| LR | 5e-7 |
| Beta (DPO) | 0.1 |
| Epochs | 1-3 |
| Est. duration | 0.5-1 day |
| Est. cost | $100-300 |

---

## 6. Evaluation Framework

### 6.1 Automated Benchmarks

| Benchmark | Domain | Purpose |
|-----------|--------|---------|
| SleepMedQA (custom, 500+) | Sleep medicine | Domain knowledge accuracy |
| C-Eval (medical subset) | Chinese general | Chinese language + knowledge |
| MMLU (clinical) | English clinical | English medical knowledge |
| CMMLU (medical) | Chinese medical | Chinese medical knowledge |
| HellaSwag, ARC, WinoGrande | General | General reasoning capability |

### 6.2 Generation Quality

- **GPT-4-as-Judge**: Accuracy, safety, helpfulness (1-5 scale)
- **Medical expert review**: Random sample of 200 generated responses
- **Hallucination detection**: Fact consistency against authoritative references
- **Safety audit**: Rate of rejecting inappropriate medical advice

### 6.3 Inference Efficiency

- Time-to-first-token (TTFT)
- Throughput (tokens/second)
- Memory usage across batch sizes
- Quantized performance comparison (FP16 / Q8 / Q4)

---

## 7. Deployment

### 7.1 Server Deployment (vLLM)

- FastAPI + vLLM backend
- OpenAI-compatible API endpoint
- Support: Chat, Completion, Embedding
- Quantization: FP16 / AWQ / GPTQ
- Hardware requirement: 1x A100 40GB (FP16)

### 7.2 Edge Deployment (llama.cpp / GGUF)

- Quantization levels: Q4_K_M / Q5_K_M / Q8_0
- Memory: Q4 ~1.2GB, Q8 ~2.2GB
- Backend support: CPU / CUDA / Metal
- Target hardware: RTX 3060+ / Apple M1+ / 8GB RAM

### 7.3 HuggingFace Hub

- Repository: `deepsleep-ai/deepsleep-2b`
- Contents: Model weights + tokenizer + sample code
- License: Apache 2.0 with medical use disclaimer
- Model Card: Usage guidelines + safety declarations

---

## 8. Project Structure

```
deepsleep/
├── README.md
├── pyproject.toml
├── Makefile
├── configs/
│   ├── model/          # Model & tokenizer configs
│   ├── data/           # Data source, preprocessing, mixing configs
│   ├── train/          # Pretrain, SFT, DPO hyperparams
│   ├── eval/           # Benchmark & judge configs
│   └── deploy/         # vLLM & GGUF configs
├── src/
│   ├── model/          # Model implementation (config, modeling, attention, moe, etc.)
│   ├── data/           # Data engineering (crawling, synthetic, processing, dataset)
│   ├── training/       # Training logic (pretrain, sft, dpo, callbacks, schedulers)
│   ├── evaluation/     # Evaluation (benchmarks, judge, safety, inference bench)
│   ├── inference/      # Inference (chat, API server, quantization)
│   └── utils/          # Distributed, logging, FSDP, checkpoint
├── scripts/            # Shell scripts for pipeline steps
├── data/               # Data directory (gitignored)
├── checkpoints/        # Model checkpoints (gitignored)
├── output/             # Final outputs (gitignored)
├── tests/              # Unit & integration tests
├── docs/               # Documentation
└── notebooks/          # Experiment notebooks
```

---

## 9. Safety & Compliance

- **Medical disclaimer**: Model outputs are for informational purposes only, not medical advice
- **Refusal training**: Model should refuse to provide specific dosage instructions or diagnoses
- **Evaluation for safety**: Every release must pass safety evaluation before publishing
- **License restriction**: Include medical AI use limitations in license
