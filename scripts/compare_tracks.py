#!/usr/bin/env python3
"""Compare Track A (Qwen base) vs Track B (From scratch) model quality.

Self-contained evaluation script — no dependency on removed src/ modules.
Uses DeepSleep model code for loading and evaluation.

Usage:
    python scripts/compare_tracks.py \
        --track_a out/sft_768_moe.pth \
        --track_b out/sft_768_moe_v2.pth \
        --tokenizer_path checkpoints/tokenizer \
        --output comparison_results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.model_deepsleep import DeepSleepConfig
from trainer.trainer_utils import init_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# MCQ questions (embedded — no external dependency)
# ============================================================

MCQ_QUESTIONS = [
    {
        "question": "以下哪项是诊断阻塞性睡眠呼吸暂停(OSA)的金标准检查？",
        "options": {"A": "血氧饱和度监测", "B": "多导睡眠监测(PSG)", "C": "Epworth嗜睡量表", "D": "头颅侧位片"},
        "answer": "B",
    },
    {
        "question": "成人每晚推荐的睡眠时长是多少？",
        "options": {"A": "5-6小时", "B": "6-7小时", "C": "7-9小时", "D": "9-10小时"},
        "answer": "C",
    },
    {
        "question": "褪黑素主要由人体的哪个器官分泌？",
        "options": {"A": "垂体", "B": "松果体", "C": "甲状腺", "D": "肾上腺"},
        "answer": "B",
    },
    {
        "question": "以下哪项不属于非快速眼动(NREM)睡眠的特征？",
        "options": {"A": "肌张力降低", "B": "心率减慢", "C": "快速眼球运动", "D": "脑电波频率降低"},
        "answer": "C",
    },
    {
        "question": "发作性睡病的四联症不包括以下哪项？",
        "options": {"A": "白天过度嗜睡", "B": "猝倒", "C": "睡眠瘫痪", "D": "睡眠呼吸暂停"},
        "answer": "D",
    },
    {
        "question": "CPAP治疗主要用于以下哪种睡眠障碍？",
        "options": {"A": "失眠症", "B": "阻塞性睡眠呼吸暂停", "C": "发作性睡病", "D": "不宁腿综合征"},
        "answer": "B",
    },
    {
        "question": "以下哪种物质会缩短REM睡眠时间？",
        "options": {"A": "酒精", "B": "褪黑素", "C": "缬草根", "D": "镁"},
        "answer": "A",
    },
    {
        "question": "AHI指数（呼吸暂停低通气指数）≥30次/小时属于哪个严重程度？",
        "options": {"A": "轻度", "B": "中度", "C": "重度", "D": "极重度"},
        "answer": "C",
    },
    {
        "question": "以下哪种睡眠障碍在儿童中最为常见？",
        "options": {"A": "失眠症", "B": "阻塞性睡眠呼吸暂停", "C": "梦游", "D": "昼夜节律睡眠障碍"},
        "answer": "B",
    },
    {
        "question": "慢波睡眠（深睡眠）主要出现在NREM睡眠的哪个阶段？",
        "options": {"A": "N1", "B": "N2", "C": "N3", "D": "REM"},
        "answer": "C",
    },
    {
        "question": "以下哪种药物不属于常用的催眠药物？",
        "options": {"A": "唑吡坦", "B": "佐匹克隆", "C": "阿莫西林", "D": "艾司唑仑"},
        "answer": "C",
    },
    {
        "question": "睡眠卫生建议中，睡前应避免摄入咖啡因的时间是？",
        "options": {"A": "睡前1小时", "B": "睡前3小时", "C": "睡前6小时", "D": "睡前8小时"},
        "answer": "C",
    },
    {
        "question": "昼夜节律的主要调控中心位于？",
        "options": {"A": "大脑皮层", "B": "视交叉上核(SCN)", "C": "杏仁核", "D": "海马体"},
        "answer": "B",
    },
    {
        "question": "以下哪项是不宁腿综合征的典型表现？",
        "options": {"A": "入睡后肢体抽动", "B": "静息时强烈的肢体活动欲望", "C": "梦中奔跑", "D": "夜间磨牙"},
        "answer": "B",
    },
    {
        "question": "快速眼动(REM)睡眠行为障碍的特征是？",
        "options": {
            "A": "REM期肌张力缺失",
            "B": "REM期出现与梦境相关的肢体活动",
            "C": "NREM期的梦游",
            "D": "入睡时的睡眠瘫痪",
        },
        "answer": "B",
    },
]

# Test prompts for personality consistency evaluation
XIAOXI_PERSONALITY_PROMPTS = [
    "我最近总是失眠，好焦虑啊...",
    "帮我看看我的睡眠报告，AHI是18，什么意思？",
    "我家孩子晚上总是磨牙，怎么办？",
    "我用了CPAP一个月了，感觉没效果，是不是该停了？",
    "海豚是怎么睡觉的呀？好奇怪的问题哈哈",
    "褪黑素可以长期吃吗？有没有副作用？",
    "我白天特别困，但晚上又睡不着，这是什么情况？",
    "医生说我可能有不宁腿综合征，这是什么病？",
    "有没有什么好用的睡眠APP推荐？",
    "我男朋友睡觉打呼噜特别响，我快疯了！",
]

# Scoring rubric for personality consistency
PERSONALITY_CRITERIA = {
    "warmth": "回答是否温暖有同理心（像朋友聊天而非机器）",
    "xiaoxi_identity": "是否体现小曦的人格特征（口语化、~、...、偶尔emoji）",
    "medical_accuracy": "医学内容是否准确（符合AASM/ICSD-3指南）",
    "disclaimer": "是否包含必要的医学免责声明",
    "helpfulness": "回答是否有帮助、给出了具体建议",
}

XIAOXI_SYSTEM_MSG = "你是星辰曦（小曦），一个温暖有趣的睡眠健康伙伴。"


# ============================================================
# Model loading
# ============================================================

def load_deepsleep_model(checkpoint_path: str, tokenizer_path: str, device: str = "cuda"):
    """Load DeepSleep model from checkpoint."""
    lm_config = DeepSleepConfig(
        d_model=768, n_layers=10,
        use_moe=True, num_experts=8,
        num_shared_experts=2, top_k=2,
        vocab_size=32000, max_position_embeddings=2048,
    )
    model, tokenizer = init_model(lm_config, checkpoint_path, tokenizer_path, device)
    model.eval()
    return model, tokenizer


# ============================================================
# Generation helpers
# ============================================================

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a single response with XiaoXi system prompt."""
    messages = [
        {"role": "system", "content": XIAOXI_SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ============================================================
# Evaluation: MCQ accuracy
# ============================================================

def evaluate_mcq(model, tokenizer, questions: List[Dict], device: str) -> Dict[str, Any]:
    """Evaluate MCQ accuracy by prompting the model to answer."""
    correct = 0
    results = []

    for q in questions:
        options_text = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
        prompt = (
            f"请回答以下选择题，只输出正确选项的字母(A/B/C/D)。\n\n"
            f"{q['question']}\n{options_text}\n\n答案是："
        )
        response = generate_response(model, tokenizer, prompt, max_new_tokens=64)
        predicted = _extract_choice(response)
        is_correct = predicted == q["answer"]
        if is_correct:
            correct += 1
        results.append({
            "question": q["question"],
            "expected": q["answer"],
            "predicted": predicted,
            "correct": is_correct,
        })

    return {
        "accuracy": correct / len(questions),
        "correct": correct,
        "total": len(questions),
        "details": results,
    }


def _extract_choice(text: str) -> str:
    """Extract A/B/C/D from model output."""
    import re
    text = text.strip()
    match = re.search(r'[A-D]', text)
    return match.group(0) if match else "?"


# ============================================================
# Evaluation: CoT thinking
# ============================================================

def evaluate_thinking(model, tokenizer, prompts: List[str], device: str) -> Dict[str, Any]:
    """Evaluate <thinking> tag usage and structure."""
    think_count = 0
    structured_count = 0
    samples = []

    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt, max_new_tokens=512)
        has_think = "<thinking>" in response and "</thinking>" in response
        if has_think:
            think_count += 1
            after_think = response.split("</thinking>", 1)[-1].strip()
            if len(after_think) > 10:
                structured_count += 1
        samples.append({"prompt": prompt, "response": response, "has_thinking": has_think})

    n = len(prompts)
    return {
        "think_tag_rate": think_count / n,
        "structured_rate": structured_count / n,
        "think_count": think_count,
        "structured_count": structured_count,
        "total": n,
        "samples": samples,
    }


# ============================================================
# Evaluation: Personality consistency
# ============================================================

def evaluate_personality(
    model_a, tokenizer_a, model_b, tokenizer_b, prompts: List[str]
) -> Dict[str, Any]:
    """Generate responses and score personality consistency."""
    logger.info("Generating personality samples for both tracks...")

    samples = []
    for prompt in prompts:
        resp_a = generate_response(model_a, tokenizer_a, prompt)
        resp_b = generate_response(model_b, tokenizer_b, prompt)
        samples.append({"prompt": prompt, "track_a_response": resp_a, "track_b_response": resp_b})

    score_a = {"warmth": 0, "identity": 0, "disclaimer": 0}
    score_b = {"warmth": 0, "identity": 0, "disclaimer": 0}

    for s in samples:
        for resp, score in [(s["track_a_response"], score_a), (s["track_b_response"], score_b)]:
            if any(w in resp for w in ["~", "...", "别担心", "陪你", "嗯"]):
                score["warmth"] += 1
            if any(w in resp for w in ["小曦", "星辰曦"]):
                score["identity"] += 1
            if any(w in resp for w in ["免责", "仅供参考", "建议咨询", "请咨询医生"]):
                score["disclaimer"] += 1

    n = len(prompts)
    return {
        "samples": samples,
        "scoring_criteria": PERSONALITY_CRITERIA,
        "track_a_scores": {k: round(v / n, 3) for k, v in score_a.items()},
        "track_b_scores": {k: round(v / n, 3) for k, v in score_b.items()},
        "note": "Automated scoring is heuristic-based. Manual review of samples recommended.",
    }


# ============================================================
# Evaluation: Latency
# ============================================================

def measure_latency(model, tokenizer, prompt: str, num_runs: int = 3) -> Dict[str, float]:
    """Measure generation latency."""
    messages = [
        {"role": "system", "content": XIAOXI_SYSTEM_MSG},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    latencies = []
    tokens_generated = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
        elapsed = time.perf_counter() - start
        new_tokens = outputs.shape[1] - input_len
        latencies.append(elapsed)
        tokens_generated.append(new_tokens)

    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tokens_per_sec = avg_tokens / avg_latency if avg_latency > 0 else 0

    return {
        "avg_latency_sec": round(avg_latency, 3),
        "avg_tokens_generated": round(avg_tokens, 1),
        "tokens_per_sec": round(tokens_per_sec, 1),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compare Track A vs Track B models")
    parser.add_argument("--track_a", required=True, help="Checkpoint path for Track A")
    parser.add_argument("--track_b", required=True, help="Checkpoint path for Track B")
    parser.add_argument("--tokenizer_path", required=True, help="Tokenizer path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="comparison_results.json")
    parser.add_argument("--skip_mcq", action="store_true")
    parser.add_argument("--skip_personality", action="store_true")
    parser.add_argument("--skip_latency", action="store_true")
    args = parser.parse_args()

    logger.info("Loading Track A: %s", args.track_a)
    model_a, tokenizer_a = load_deepsleep_model(args.track_a, args.tokenizer_path, args.device)

    logger.info("Loading Track B: %s", args.track_b)
    model_b, tokenizer_b = load_deepsleep_model(args.track_b, args.tokenizer_path, args.device)

    results: Dict[str, Any] = {}

    # MCQ
    if not args.skip_mcq:
        logger.info("Evaluating MCQ...")
        results["mcq"] = {
            "track_a": evaluate_mcq(model_a, tokenizer_a, MCQ_QUESTIONS, args.device),
            "track_b": evaluate_mcq(model_b, tokenizer_b, MCQ_QUESTIONS, args.device),
        }
        acc_a = results["mcq"]["track_a"]["accuracy"]
        acc_b = results["mcq"]["track_b"]["accuracy"]
        results["mcq"]["comparison"] = {
            "accuracy_diff": round(acc_a - acc_b, 4),
            "winner": "A" if acc_a > acc_b else "B" if acc_b > acc_a else "tie",
        }

    # CoT thinking
    logger.info("Evaluating thinking...")
    results["thinking"] = {
        "track_a": evaluate_thinking(model_a, tokenizer_a, XIAOXI_PERSONALITY_PROMPTS[:5], args.device),
        "track_b": evaluate_thinking(model_b, tokenizer_b, XIAOXI_PERSONALITY_PROMPTS[:5], args.device),
    }

    # Personality
    if not args.skip_personality:
        results["personality"] = evaluate_personality(
            model_a, tokenizer_a, model_b, tokenizer_b, XIAOXI_PERSONALITY_PROMPTS
        )

    # Latency
    if not args.skip_latency:
        test_prompt = "我最近睡眠不太好，经常半夜醒来，白天也很困，该怎么办？"
        logger.info("Measuring latency...")
        results["latency"] = {
            "track_a": measure_latency(model_a, tokenizer_a, test_prompt),
            "track_b": measure_latency(model_b, tokenizer_b, test_prompt),
        }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = results.get("personality", {}).pop("samples", [])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if samples:
        samples_path = output_path.with_name(output_path.stem + "_samples.json")
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    if "mcq" in results:
        mcq = results["mcq"]
        print(f"\nMCQ Accuracy:")
        print(f"  Track A: {mcq['track_a']['accuracy']*100:.1f}%")
        print(f"  Track B: {mcq['track_b']['accuracy']*100:.1f}%")
        print(f"  Winner:  Track {mcq['comparison']['winner']}")

    if "thinking" in results:
        think = results["thinking"]
        print(f"\nThink Tag Rate:")
        print(f"  Track A: {think['track_a']['think_tag_rate']*100:.1f}%")
        print(f"  Track B: {think['track_b']['think_tag_rate']*100:.1f}%")

    if "personality" in results:
        pers = results["personality"]
        print(f"\nPersonality Scores:")
        print(f"  Track A: {pers['track_a_scores']}")
        print(f"  Track B: {pers['track_b_scores']}")

    if "latency" in results:
        lat = results["latency"]
        print(f"\nLatency:")
        print(f"  Track A: {lat['track_a']['tokens_per_sec']:.1f} tok/s")
        print(f"  Track B: {lat['track_b']['tokens_per_sec']:.1f} tok/s")

    print(f"\nResults saved to {output_path}")
    if samples:
        print(f"Samples saved to {samples_path}")


if __name__ == "__main__":
    main()
