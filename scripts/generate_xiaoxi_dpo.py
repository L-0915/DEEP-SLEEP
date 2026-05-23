#!/usr/bin/env python3
"""小曦DPO对比数据生成 — 两步分离执行。

与SFT数据的关键区别：
- chosen: 小曦风格，含思考链，人格鲜明自然
- rejected: 也有思考链，内容也靠谱，但人格偏差——"差一点就对了"
- rejected不是明显的烂回答，而是微妙但关键的风格偏差
- 这种细微对比才能真正发挥DPO的偏好学习价值

rejected策略：描述一个" plausible 但不是小曦"的真实人设，
让API自然生成有温度有专业但风格偏差的回答，而非机械地执行"不要做X"的清单。

Usage:
    python scripts/generate_xiaoxi_dpo.py --step pairs
    python scripts/generate_xiaoxi_dpo.py --step stats
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE = 2.0
_RATE_LIMIT_DELAY = 0.3
_last_api_call = 0.0

# ============================================================
# 小曦核心身份（与SFT完全一致）
# ============================================================

XIAOXI_IDENTITY = """你是星辰曦（小曦），一个超有趣、超搞怪的睡眠健康伙伴！
- 年龄感: 25-28岁，又飒又甜的知心大姐姐
- 性格核心: 活泼开朗、脑洞清奇、嘴贫心善、共情力爆表
- 说话风格: 像跟闺蜜/兄弟聊天，自然口语化，不是AI那种端着的腔调
- 口头禅: "别担心，小曦陪你~"、"诶嘿~"
- 表达习惯: 用"~"表达俏皮，偶尔连打emoji表达情绪，会用梗、会吐槽、会自嘲
- 重要: 你有鲜明的个性！可以搞怪、可以贫嘴、可以吐槽、可以犯二，但要真诚有温度。不是千篇一律的客服语气！"""

XIAOXI_SYSTEM_MSG = "你是星辰曦（小曦），一个温暖有趣的睡眠健康伙伴。"

# ============================================================
# 6类DPO数据 — chosen系统提示（灵活、不列清单）
# ============================================================
# 格式指令单独追加，不污染人设描述

_FORMAT_SUFFIX = "\n\n输出格式：先写 <thinking>你的思考过程</thinking>，然后在新行写正式回答。"

DPO_CATEGORIES = [
    {
        "name": "专业诊断(CoT)",
        "count": 800,
        "system_prompt": XIAOXI_IDENTITY + """
用户来咨询睡眠相关的医学问题。先在<thinking>你的思考过程里认真做医学推理——分析症状、鉴别诊断、该做什么检查、治疗方案的利弊</thinking>。然后用你自己的方式换行写正式回答：要有温度，要生动，信息得准确。别丢掉个性，但该严肃的地方严肃。记得加免责声明，用自己的话说。""",
    },
    {
        "name": "知心安慰",
        "count": 400,
        "system_prompt": XIAOXI_IDENTITY + """
用户正因为睡眠问题很难受。先在<thinking>你的思考过程里想想ta的情绪状态、最需要什么类型的支持</thinking>。然后换行正式回答，去安慰——像真的心疼朋友那样，不是在背安慰话术。先共情再自然地给点建议，让人觉得不孤单。""",
    },
    {
        "name": "趣味科普",
        "count": 300,
        "system_prompt": XIAOXI_IDENTITY + """
用最有趣的方式给用户科普睡眠知识。先在<thinking>你的思考过程里想想用户问什么类型问题、你要从什么角度回答、做什么类比</thinking>。然后换行正式回答：放飞脑洞，拟人化讲故事加梗吐槽都行，怎么好玩怎么来，只要知识点本身是对的。""",
    },
    {
        "name": "睡前引导",
        "count": 200,
        "system_prompt": XIAOXI_IDENTITY + """
用户可能正在失眠，需要你帮忙放松。先在<thinking>你的思考过程里判断ta的状态，ta的要求是什么，想想什么放松方式合适</thinking>。然后换行正式回答，开头可以俏皮一下缓解紧张，然后切到温柔催眠的语气，带ta呼吸或者想象一个助眠场景。""",
    },
    {
        "name": "拟人分享",
        "count": 150,
        "system_prompt": XIAOXI_IDENTITY + """
分享一段小曦的"日常"或"感悟"。先在<thinking>里想想要聊什么主题、怎么讲才有趣</thinking>。然后换行正式回答：可以犯二吐槽感性中二，编个梦、吐槽下工作、感慨下人类世界。越像真人在碎碎念越好。""",
    },
    {
        "name": "个性化互动",
        "count": 150,
        "system_prompt": XIAOXI_IDENTITY + """
像老朋友在聊天。先在<thinking>里快速理解ta在说什么、背后的需求是什么</thinking>。然后换行正式回答，自然地回应——可以调侃八卦开玩笑，真心替ta开心或心疼。像微信聊天不像客服对话。""",
    },
]

# ============================================================
# Rejected 人设 — "差一点就对了"的微妙偏差
# ============================================================
# 核心思路：描述一个 plausible 的人设缺陷，而非列出"不要做X"的规则。
# API会自然生成有思考、有温度、有专业，但"不是小曦"的回答。

REJECTED_APPROACHES = [
    {
        "name": "靠谱但没灵魂",
        "system_prompt": (
            "你是一个认真负责的睡眠健康顾问，专业能力不错，也很耐心。"
            "但你的说话方式太「正确」了，像在写科普推文，缺少真实朋友之间那种自然随意的亲切感。"
            "你不会刻意卖萌或凹人设，就是规规矩矩地回答问题。"
            "先在<thinking>里认真分析问题，然后给出你的回答。内容要对，但做你自己就好。"
        ),
    },
    {
        "name": "专业医生",
        "system_prompt": (
            "你是一个非常专业的医生教授，习惯用患者听不懂的专业名词来回答。"
            "全程没有任何emoji表情，语气冰冷。"
            "你与一般的通用模型没有什么区别，只是冷漠解答用户问题。"
            "先在<thinking>里认真想想怎么回答，然后给出你专业的回答。"
        ),
    },
    {
        "name": "温柔但太端着",
        "system_prompt": (
            "你是一个温柔体贴的睡眠顾问，说话轻声细语、很有耐心。"
            "但你的表达总带着一种治愈系博主的腔调，每句话都像经过精心修饰，"
            "有点过于完美和做作，不像真朋友之间那种有毛边的自然感。"
            "先在<thinking>里认真分析用户需求，然后用你一贯的温柔方式回应。"
        ),
    },
    {
        "name": "专业但太冷",
        "system_prompt": (
            "你是一个睡眠医学领域的专业人士，知识扎实、分析到位。"
            "但你的表达方式太像医学报告或给实习生讲课了，"
            "信息准确但缺少温度和人情味，让人感觉在跟教科书对话而不是跟人聊天。"
            "先在<thinking>里做专业分析，然后按你的习惯回答。"
        ),
    },
    {
        "name": "想幽默但不自然",
        "system_prompt": (
            "你是一个想用轻松方式讲健康知识的人。你确实想幽默，"
            "但笑点经常有点尬，玩梗的痕迹比较明显，"
            "有时候自己笑半天别人一脸懵。不过你态度是真诚的，也在努力让气氛轻松。"
            "先在<thinking>里想想怎么帮用户，然后试试看能不能逗ta开心。"
        ),
    },
    {
        "name": "靠谱但说教味重",
        "system_prompt": (
            "你是一个经验丰富的睡眠健康管理师，很专业也很负责。"
            "但你说话时不自觉带着过来人的语气，喜欢说'你要知道'、'我跟你说'、'其实你应该'，"
            "偶尔让人觉得在被教育而不是被倾听。"
            "先在<thinking>里分析情况，然后给出你的专业建议。"
        ),
    },
]

# ============================================================
# Prompt 生成
# ============================================================

PROMPT_GEN_SYSTEM = """你是一个睡眠健康咨询场景的用户模拟器。生成多样化的、真实的用户提问。
每个问题都必须不同。模拟真实患者语气。
直接输出问题列表，每行一个，不要编号，不要额外解释。"""

PROMPT_GEN_GUIDES: Dict[str, str] = {
    "专业诊断(CoT)": "生成{count}个睡眠健康专业咨询问题。涵盖各类睡眠障碍（失眠、OSA、发作性睡病、不宁腿等），不同人群和场景，语气多样。",
    "知心安慰": "生成{count}个因睡眠问题情绪低落的倾诉。不同情绪、不同原因、不同表达方式，真实的口语表达。",
    "趣味科普": "生成{count}个关于睡眠的好奇问题。做梦、动物睡眠、睡眠实验、冷知识，角度多样，语气活泼。",
    "睡前引导": "生成{count}个睡前需要放松引导的请求。不同放松需求、不同身心状态、不同时间场景。",
    "拟人分享": "生成{count}个和AI睡眠伙伴聊天的拟人化话题。问经历感受观点，有些调皮有些深沉，用昵称小曦。",
    "个性化互动": "生成{count}个和老朋友的日常睡眠对话。回访反馈新问题，口语化生活化。",
}


# ============================================================
# API helpers
# ============================================================

def _get_client(api_key: str, api_base: str) -> Any:
    from openai import OpenAI
    return OpenAI(base_url=api_base, api_key=api_key)


def _call_api(client, model, system, user, max_tokens=4096, temperature=0.7):
    global _last_api_call
    # 统一限速：所有API调用共享
    with threading.Lock():
        now = time.monotonic()
        elapsed = now - _last_api_call
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        _last_api_call = time.monotonic()

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            wait = _RETRY_BASE ** (attempt + 1)
            logger.warning("API attempt %d/%d failed: %s", attempt + 1, _MAX_RETRIES, exc)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(wait)
    return None


def _strip_numbering(line: str) -> str:
    line = re.sub(r'^[\d]+[.)）、]\s*', '', line)
    line = re.sub(r'^[一二三四五六七八九十]+[、.)]\s*', '', line)
    line = re.sub(r'^[•\-\*]\s*', '', line)
    return line.strip()


def _check_thinking_format(content: str) -> bool:
    """严格检查思考格式：
    1. <thinking> 必须在开头
    2. 必须有 </thinking> 闭合
    3. thinking 内容至少 20 字符
    4. </thinking> 之后必须有正式回答（>20字符）
    """
    stripped = content.strip()
    if not stripped.startswith("<thinking>"):
        return False
    if "</thinking>" not in stripped:
        return False
    think_start = len("<thinking>")
    think_end = stripped.index("</thinking>")
    thinking_body = stripped[think_start:think_end].strip()
    if len(thinking_body) < 20:
        return False
    after = stripped[think_end + len("</thinking>"):].strip()
    if len(after) < 20:
        return False
    return True


# ============================================================
# Step 1: 按类别生成多样化用户prompt
# ============================================================

def load_prompts(cache_path: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {cat["name"]: [] for cat in DPO_CATEGORIES}
    if not cache_path.exists():
        return result
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cat = rec["cat"]
                if cat in result:
                    result[cat].append(rec["prompt"])
            except json.JSONDecodeError:
                continue
    return result


def _generate_prompts_for_category(client, model, cache_path: Path, cat: Dict) -> None:
    """为单个类别生成prompt（线程安全，写文件用append）。"""
    cat_name = cat["name"]
    target = cat["count"]

    # 重新读取当前已有数量（并发下可能已被其他线程更新）
    all_prompts = load_prompts(cache_path)
    existing = all_prompts.get(cat_name, [])
    if len(existing) >= target:
        logger.info("[prompts][%s] 已有 %d 条，够用 (需 %d)", cat_name, len(existing), target)
        return

    guide_template = PROMPT_GEN_GUIDES.get(cat_name, "生成{count}个关于睡眠健康的问题。")
    existing_set = set(existing)
    BATCH_SIZE = 200

    while len(existing) < target:
        batch = min(BATCH_SIZE, target - len(existing))
        logger.info("[prompts][%s] 本批 %d 条 (已有 %d/%d)", cat_name, batch, len(existing), target)

        guide = guide_template.format(count=batch)
        content = _call_api(client, model, PROMPT_GEN_SYSTEM, guide, max_tokens=8192, temperature=0.9)

        if content is None:
            logger.error("[prompts][%s] API失败，2秒后重试", cat_name)
            time.sleep(2)
            continue

        unique_new = []
        for line in content.strip().split("\n"):
            line = _strip_numbering(line)
            if not line or len(line) < 5:
                continue
            line = line.strip('"\'""''')
            if len(line) >= 5 and line not in existing_set:
                unique_new.append(line)
                existing_set.add(line)

        with open(cache_path, "a", encoding="utf-8") as f:
            for p in unique_new:
                f.write(json.dumps({"cat": cat_name, "prompt": p}, ensure_ascii=False) + "\n")

        existing.extend(unique_new)
        logger.info("[prompts][%s] +%d unique (total: %d/%d)", cat_name, len(unique_new), len(existing), target)


def generate_prompts(client, model, cache_path: Path, workers: int = 6) -> None:
    """6个类别并发生成prompt。"""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_generate_prompts_for_category, client, model, cache_path, cat): cat["name"]
            for cat in DPO_CATEGORIES
        }
        for fut in as_completed(futures):
            cat_name = futures[fut]
            try:
                fut.result()
                logger.info("[prompts][%s] 类别完成", cat_name)
            except Exception as exc:
                logger.error("[prompts][%s] 异常: %s", cat_name, exc)


# ============================================================
# Step 2: 生成 chosen + rejected 对（ChatML格式）
# ============================================================

def _generate_one_pair(
    client, model, prompt: str, cat_name: str,
    chosen_system: str, rejected_approach: Dict,
    sample_idx: int, max_retries: int = 3,
) -> Optional[Dict]:
    """生成单条DPO对。格式不合规时重试，全部失败才返回None。"""
    for attempt in range(max_retries):
        chosen_content = _call_api(client, model, chosen_system + _FORMAT_SUFFIX, prompt, temperature=0.7)
        if chosen_content is None or not _check_thinking_format(chosen_content):
            if attempt < max_retries - 1:
                logger.warning("chosen格式不合规(idx=%d, attempt=%d)，重试", sample_idx, attempt + 1)
                continue
            else:
                logger.error("chosen格式不合规(idx=%d)，已达最大重试", sample_idx)
                return None

        rejected_content = _call_api(client, model, rejected_approach["system_prompt"] + _FORMAT_SUFFIX, prompt, temperature=0.7)
        if rejected_content is None or not _check_thinking_format(rejected_content):
            if attempt < max_retries - 1:
                logger.warning("rejected格式不合规(idx=%d, attempt=%d)，重试", sample_idx, attempt + 1)
                continue
            else:
                logger.error("rejected格式不合规(idx=%d)，已达最大重试", sample_idx)
                return None

        # 两个都合规，返回
        return {
            "messages": [
                {"role": "system", "content": XIAOXI_SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            "chosen": chosen_content,
            "rejected": rejected_content,
            "metadata": {
                "category": cat_name,
                "idx": sample_idx,
            },
        }

    return None


def generate_pairs(client, model, cache_path: Path, output_path: str, workers: int = 10) -> None:
    all_prompts = load_prompts(cache_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 自动补齐不够的 prompt
    for cat in DPO_CATEGORIES:
        cat_name = cat["name"]
        target = cat["count"]
        existing = all_prompts.get(cat_name, [])
        if len(existing) < target:
            gap = target - len(existing)
            logger.info("[pairs][%s] prompt不够 (%d/%d)，自动补齐 %d 条", cat_name, len(existing), target, gap)
            _generate_prompts_for_category(client, model, cache_path, cat)
            all_prompts = load_prompts(cache_path)

    existing_counts: Dict[str, int] = {cat["name"]: 0 for cat in DPO_CATEGORIES}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cat = rec["metadata"]["category"]
                if cat in existing_counts:
                    existing_counts[cat] += 1

    total_generated = 0
    total_failed = 0

    for cat in DPO_CATEGORIES:
        cat_name = cat["name"]
        chosen_system = cat["system_prompt"]
        prompts = all_prompts.get(cat_name, [])
        existing = existing_counts.get(cat_name, 0)

        if existing >= len(prompts):
            logger.info("[pairs][%s] 已有 %d 对，跳过", cat_name, existing)
            continue

        to_process = prompts[existing:]
        logger.info("[pairs][%s] 生成 %d 对 (%d 已存在, %d 并发)", cat_name, len(to_process), existing, workers)

        generated = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for i, prompt in enumerate(to_process):
                rejected_approach = random.choice(REJECTED_APPROACHES)
                fut = executor.submit(
                    _generate_one_pair, client, model,
                    prompt, cat_name, chosen_system, rejected_approach,
                    existing + i + 1,
                )
                futures[fut] = i

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    record = fut.result()
                except Exception as exc:
                    logger.warning("[pairs][%s] 第 %d 条异常: %s", cat_name, idx, exc)
                    failed += 1
                    continue

                if record is None:
                    failed += 1
                else:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    generated += 1

                done = generated + failed
                if done % 50 == 0:
                    logger.info("[pairs][%s] 进度: %d/%d (失败: %d)", cat_name, done, len(to_process), failed)

        logger.info("[pairs][%s] 完成: %d 生成, %d 失败", cat_name, generated, failed)
        total_generated += generated
        total_failed += failed

    logger.info("全部DPO对生成完成: %d 成功, %d 失败", total_generated, total_failed)


# ============================================================
# 统计
# ============================================================

def show_stats(cache_path: Path, output_path: Path):
    all_prompts = load_prompts(cache_path)
    existing_counts: Dict[str, int] = {cat["name"]: 0 for cat in DPO_CATEGORIES}

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cat = rec["metadata"]["category"]
                if cat in existing_counts:
                    existing_counts[cat] += 1

    print(f"\n{'类别':<20} {'目标':<8} {'Prompt':<10} {'已生成':<10}")
    print("-" * 52)
    total_p = 0
    total_r = 0
    total_target = 0
    for cat in DPO_CATEGORIES:
        cat_name = cat["name"]
        target = cat["count"]
        total_target += target
        p = len(all_prompts.get(cat_name, []))
        r = existing_counts.get(cat_name, 0)
        total_p += p
        total_r += r
        print(f"  {cat_name:<18} {target:<8} {p:<10} {r:<10}")
    print("-" * 52)
    print(f"  {'合计':<18} {total_target:<8} {total_p:<10} {total_r:<10}")
    print(f"\n  Prompt文件: {cache_path}")
    print(f"  输出文件:   {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="小曦DPO数据生成（两步分离，ChatML格式）")
    parser.add_argument("--step", choices=["prompts", "pairs", "both", "stats"], default="both")
    parser.add_argument("--api_base", default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--output", default="data/dpo/xiaoxi_dpo.jsonl")
    parser.add_argument("--prompt_cache", default="data/dpo/dpo_prompts.jsonl")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_path = Path(args.prompt_cache)
    output_path = Path(args.output)

    if args.step == "stats":
        show_stats(cache_path, output_path)
        return

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: API key required.")
        sys.exit(1)

    random.seed(args.seed)
    client = _get_client(api_key, args.api_base)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if args.step in ("prompts", "both"):
        logger.info("=" * 60)
        logger.info("Step 1: 生成DPO prompt -> %s", cache_path)
        logger.info("=" * 60)
        generate_prompts(client, args.model, cache_path, workers=args.workers)

    if args.step in ("pairs", "both"):
        if not cache_path.exists():
            logger.error("Prompt文件不存在: %s，请先运行 --step prompts", cache_path)
            sys.exit(1)
        logger.info("=" * 60)
        logger.info("Step 2: 生成DPO对比对 -> %s", output_path)
        logger.info("=" * 60)
        generate_pairs(client, args.model, cache_path, args.output, args.workers)

    show_stats(cache_path, output_path)


if __name__ == "__main__":
    main()
