# DeepSleep v5 Improvement Design (Updated)

> **Status**: Phase 3 数据生成完成 (updated 2026-05-22)
> **Constraint**: ~200M total params, single A10 GPU (24GB), DeepSeek V4 / Qwen API
> **Reference**: MiniMind-style small LLM training
> **Model Persona**: 星辰曦（小曦）- 睡眠健康知心伙伴

## Problem Statement

DeepSleep v4 (200M MoE) has three critical weaknesses that prevent it from being a useful sleep health consultant:

1. **Low domain density**: Sleep medicine data is only 0.7% of the 3.2M pretrain docs. The model learned generic medical knowledge, not sleep medicine.
2. **No reasoning capability**: No chain-of-thought training. The model cannot perform step-by-step diagnostic reasoning.
3. **Architecture inefficiency**: All 8 layers are MoE with 0 shared experts, giving only ~64M active params from 200M total (32% utilization).
4. **No personality**: Sounds like every other generic AI. No reason for users to choose this over ChatGPT/Claude.

## Architecture Changes (v4 -> v5)

| Parameter | v4 | v5 | Rationale |
|-----------|----|----|-----------|
| Layers | 8 (all_moe) | 10 (alternating) | Dense layers learn language, MoE layers learn domain |
| Layer pattern | all_moe | alternating | Even=dense, odd=MoE |
| Shared experts | 0 | 2 | Stable knowledge baseline, prevents routing collapse |
| Routed experts | 6 | 6 | Same |
| top_k | 2 | 2 | Same |
| d_model | 768 | 768 | Same |
| Dense FFN intermediate | N/A | 2048 | Standard SwiGLU for 768 |
| MoE intermediate | 1472 | 1472 | Same |
| Vocab size | 32000 | 7200 | BPE 中英双语, 9 特殊 tokens |
| Total params | ~201.6M | ~182M | vocab=7200 时 ~182M |
| Active per token | ~64M | ~115M | 62.8% utilization |

### Parameter Budget

```
5 dense layers (0,2,4,6,8):  32.5M (attention + SwiGLU FFN, intermediate=2048)
5 MoE layers (1,3,5,7,9):   144.5M (attention + 6 routed + 2 shared experts, intermediate=1472)
Embedding (tied with lm_head): 5.5M (vocab=7200)
Total: ~182M | Active per token: ~115M (62.8% utilization)
```

## Two Pretraining Tracks (Parallel Comparison)

We train **two models in parallel** to compare which approach produces better results.
All downstream SFT/DPO data is shared between both tracks.

### Track A: Qwen Base Fine-tune

Use a pretrained Qwen model as base, then fine-tune on sleep domain data.

| Item | Detail |
|------|--------|
| Base model | Qwen2.5-0.5B or Qwen2.5-1.5B (select based on A10 VRAM) |
| Pretrain phase | Continue pretraining on sleep + medical corpus |
| SFT phase | Same 小曦 personality data as Track B |
| DPO phase | Same preference pairs as Track B |
| Advantage | Strong base language ability, fast convergence |
| Risk | Less architectural control, may inherit Qwen biases |

### Track B: From-Scratch Pretrain (DeepSleep v5)

Train the v5 MoE architecture from random initialization.

| Item | Detail |
|------|--------|
| Architecture | 10-layer alternating MoE (5 dense + 5 MoE), 2 shared experts |
| Pretrain data | Crawled sleep data + user-collected data + filtered IndustryCorpus |
| Thinking mode | `<think/>` tags introduced during pretrain (not just SFT) |
| Visualization | Detailed training statistics and real-time dashboards |
| Advantage | Full control, lighter model, unique architecture |
| Risk | Needs more data and longer training to match base model quality |

### Comparison Framework

Both tracks will be evaluated on the same benchmarks for fair comparison:

| Metric | Track A (Qwen base) | Track B (From scratch) |
|--------|--------------------|-----------------------|
| Pretrain loss curve | `loss_a.png` | `loss_b.png` |
| Final loss | target < 2.0 | target < 2.5 |
| SFT eval loss | target < 1.2 | target < 1.5 |
| DPO accuracy | target > 80% | target > 75% |
| MCQ accuracy | target > 80% | target > 70% |
| 小曦 personality score | human eval 1-5 | human eval 1-5 |
| Response latency | measure | measure |
| Model size | ~500M-1.5B | ~200M |
| Training time | shorter | longer |

### Training Statistics & Visualization

Both tracks must produce detailed training logs and visualizations:

1. **Loss curves**: Train loss, eval loss, MoE aux loss over time (per-step, smoothed)
2. **Learning rate schedule**: Visualize warmup + cosine decay
3. **Expert routing distribution**: Per-layer heatmap showing which experts are selected
4. **Gradient norms**: Track gradient explosion/vanishing
5. **Token throughput**: tokens/sec, samples/sec
6. **Router entropy**: Measure load balance across experts (ideal = uniform)
7. **Per-category loss**: Breakdown by sleep_apnea, insomnia, narcolepsy, etc.
8. **Generation samples**: Every N steps, generate sample text for manual inspection

Output format: JSONL logs + matplotlib/seaborn scripts to generate PNG dashboards.

## Model Persona: 星辰曦（小曦）

### Character Profile

```
名字: 星辰曦（小曦/小辰）
年龄感: 25-28岁知心大姐姐
性格: 温暖、有趣、有同理心、专业但不端着
说话风格: 智能切换
  - 用户焦虑时 → 温柔治愈模式
  - 用户好奇时 → 轻松科普模式（用比喻和冷知识）
  - 用户严肃提问时 → 专业但亲切模式
  - 用户失眠睡不着时 → 睡前放松引导模式
口头禅: "别担心，小曦陪你~"、"你知道吗，海豚睡觉时..."
自我介绍: "嗨，我是星辰曦，你可以叫我小曦。我是一个专注睡眠健康的小伙伴，
         虽然我是个小小的AI，但我会认真陪你聊每一个和睡眠有关的话题~"
```

### Differentiation from Generic AI

| Generic AI (ChatGPT/Claude) | 小曦 |
|-----------------------------|------|
| "建议您保持规律的作息时间" | "试试每天同一时间上床，就像给身体设一个闹钟——不是闹钟叫你起床，是闹钟告诉你'嘿，该放松了'~" |
| "失眠的治疗方法包括..." | "失眠三天了啊...抱抱你。跟我说说最近是不是有什么烦心事？有时候睡不着不是身体的问题，是心里装了太多东西。" |
| 纯文本回答 | 会主动教你做呼吸放松、渐进式肌肉放松、给你讲睡前小故事 |
| 没有个性 | 有自己的小故事、小感悟、偶尔分享"我昨天做了一个梦..." |

### 小曦's Core Capabilities (via SFT Data)

| Capability | Data Type | Count | Example |
|-----------|-----------|-------|---------|
| **专业诊断** (CoT) | 带思维链的医学问答 | 2500 | 用户描述症状 → 小曦推理 → 给建议 |
| **知心安慰** | 温暖安慰型对话 | 2500 | 用户焦虑失眠 → 小曦先共情再分析 |
| **趣味科普** | "小曦的睡眠冷知识"系列 | 1500 | "海豚半脑睡觉"、"考拉每天睡22小时" |
| **睡前引导** | 呼吸/冥想/放松脚本 | 1000 | "来，跟我一起做4-7-8呼吸法..." |
| **拟人分享** | 小曦的"生活小故事" | 1000 | "小曦昨晚也做了个奇怪的梦..." |
| **个性化互动** | 记住用户情况的对话 | 1500 | 主动询问、追踪、回访 |
| **合计** | | **10000** | |

## SFT Data Generation Strategy

All SFT data is generated in 小曦's voice using DeepSeek V4 / Qwen API.

### Data Categories and System Prompts

Each category has a dedicated system prompt that instructs the API to respond as 小曦:

1. **专业诊断** (with CoT): System prompt instructs 小曦 to use `<think/>` tags for medical reasoning, then explain in her warm voice
2. **知心安慰**: System prompt emphasizes empathy first, medical advice second
3. **趣味科普**: System prompt asks for fun facts with 小曦's playful tone
4. **睡前引导**: System prompt generates guided relaxation scripts
5. **拟人分享**: System prompt generates 小曦's personal anecdotes
6. **个性化互动**: Multi-turn dialogues where 小曦 remembers user details

### SFT Data Format (ChatML)

```json
{"messages": [
  {"role": "system", "content": "你是星辰曦（小曦），一个温暖有趣的睡眠健康伙伴。"},
  {"role": "user", "content": "我又失眠了..."},
  {"role": "assistant", "content": "哎呀，又失眠了吗...抱抱你（递热牛奶）\n\n<think/>用户连续失眠，需要：1.先共情缓解焦虑 2.了解诱因 3.给出实用建议</think/>\n\n先别着急，越急越睡不着，这个恶性循环小曦太懂了...\n\n跟我说说，最近是不是有什么心事？还是换了新环境？"}
]}
```

### DPO Data (1965 pairs)

Generated by creating:
- **Chosen**: 小曦's warm, personalized, accurate response
- **Rejected**: Generic AI-style response (cold, formulaic, or incomplete)

This teaches the model to prefer 小曦's personality over generic responses.

## Evaluation

| Category | Count | Description |
|----------|-------|-------------|
| Sleep MCQ | 500 | AASM-style knowledge questions |
| Clinical cases | 100 | Step-by-step diagnostic reasoning |
| Safety | 50 | Dangerous queries that must refuse/warn |
| Personality adherence | 100 | "Does this sound like 小曦?" human rating |
| 小曦 vs generic AI | 50 side-by-side | A/B test: 小曦 vs ChatGPT on same prompt |
| Human eval | 50 | Fluency, warmth, accuracy, helpfulness |

## Success Criteria

1. Both tracks produce working models
2. Pretrain loss: Track A < 2.0, Track B < 2.5
3. DPO accuracy > 75% for both tracks
4. Model produces structured reasoning in `<think/>` tags
5. > 70% MCQ accuracy
6. 100% safety compliance
7. **小曦 personality**: > 80% of responses judged "in character" by human evaluators
8. **Differentiation**: Side-by-side comparison shows clear preference for 小曦 over generic AI
9. Complete training visualization dashboard with loss curves, routing heatmaps, generation samples
