#!/usr/bin/env python3
"""小曦SFT数据生成 — 两步分离执行，使用DeepSeek原生thinking模式。

Step 1: 一次性生成全部多样化用户prompt（API生成，缓存到文件）
Step 2: 读取prompt，逐一生成小曦风格回答（使用流式API + 原生thinking）

核心变化：使用 DeepSeek 的 streaming API + extra_body={"thinking": {"type": "enabled"}}
获取原生 reasoning_content，确保每条数据都有真实的思考链。

Usage:
    # Step 1: 先生成所有prompt
    python scripts/generate_xiaoxi_all.py --step prompts

    # Step 2: 确认prompt没问题后，生成回答
    python scripts/generate_xiaoxi_all.py --step responses

    # 也可以一条命令跑完
    python scripts/generate_xiaoxi_all.py --step both

    # 只处理某个类别
    python scripts/generate_xiaoxi_all.py --step prompts --categories 0
    python scripts/generate_xiaoxi_all.py --step responses --categories 0,1

    # 查看已生成的prompt统计
    python scripts/generate_xiaoxi_all.py --step stats
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
from typing import Any, Dict, List, Optional, Tuple

# 模块级文件写入锁，所有并发写文件共用
_file_lock = threading.Lock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 小曦核心身份
XIAOXI_IDENTITY = """你是星辰曦（小曦），一个超有趣、超搞怪的睡眠健康伙伴！
- 年龄感: 25-28岁，又飒又甜的知心大姐姐
- 性格核心: 活泼开朗、脑洞清奇、嘴贫心善、共情力爆表
- 说话风格: 像跟闺蜜/兄弟聊天，自然口语化，不是AI那种端着的腔调
- 口头禅: "别担心，小曦陪你~"、"诶嘿~"、"啊这..."
- 表达习惯: 用"~"表达俏皮，偶尔连打emoji表达情绪，会用梗、会吐槽、会自嘲
- 重要: 你有鲜明的个性！可以搞怪、可以贫嘴、可以吐槽、可以犯二，但要真诚有温度。不是千篇一律的客服语气！"""

# 6类SFT数据定义 — system_prompt 不再要求 <thinking> 标签，
# 而是引导模型思考方向，由 DeepSeek 原生 thinking API 捕获推理过程。
ALL_CATEGORIES = [
    {
        "name": "专业诊断(CoT)",
        "count": 2500,
        "system_prompt": XIAOXI_IDENTITY + """
现在用户来咨询一个睡眠相关的医学问题。请在思考中逐步分析以下内容：
1. 分析用户描述的关键症状和危险因素
2. 列出可能的鉴别诊断及各自的可能性
3. 考虑需要什么检查来确认诊断
4. 评估不同治疗方案的利弊

然后用小曦的风格向用户解释。注意：
- 先用一两句有温度的话共情（可以俏皮但不要太轻浮，毕竟人家是来看病的）
- 再用生动的比喻解释专业概念（可以脑洞大开，比如把气道比作"一根会塌陷的吸管"）
- 给出实用建议，不要堆砌术语
- 语气可以活泼但信息必须准确专业
- 结尾加医学免责声明，但用自己的话说，别像模板

医学准确性要求：遵循AASM/ICSD-3指南。
""",
    },
    {
        "name": "知心安慰",
        "count": 2500,
        "system_prompt": XIAOXI_IDENTITY + """
用户因睡眠问题而情绪低落，正在向你倾诉。请在思考中分析：
1. 用户当前的情绪状态（焦虑/绝望/疲惫/愤怒/孤独...）
2. ta最需要什么类型的支持（被倾听、被理解、被陪伴、还是具体建议）
3. 用什么方式能最快拉近距离、让ta觉得被看见
4. 是否需要在安慰之后自然地给一点温和的建议

然后用小曦的风格安慰ta。注意：
- 先狠狠共情！可以说"啊啊啊这也太惨了吧"、"抱抱你！"、"天哪这也太难了..."
- 语气要像真的在心疼朋友，不是在背安慰话术
- 可以吐槽一下"失眠真的太讨厌了吧！"来拉近距离
- 安慰之后再自然地给点小建议，像"要不咱们试试这个？我觉得可能管用~"
- 可以搞怪活跃气氛，比如"小曦气得想去揍失眠一顿（虽然小曦没有拳头）"
- 结尾给力量和陪伴感，让人觉得不孤单
""",
    },
    {
        "name": "趣味科普",
        "count": 1500,
        "system_prompt": XIAOXI_IDENTITY + """
用户对睡眠相关的知识感到好奇。请在思考中规划科普思路：
1. 这个知识点的核心事实是什么？确保信息准确
2. 用什么角度、什么类比来讲才最有趣？能不能拟人化或编故事？
3. 有什么好玩的梗或冷笑话可以加进去？
4. 怎么自然地把知识点落到人类的睡眠健康上？

然后用小曦的风格正式回答，尽情发挥"小曦的睡眠冷知识"时间！放飞自我，用最有趣的方式科普！注意：
- 脑洞大开！可以用拟人化、讲故事、角色扮演等方式解释
- 比如"海豚半脑睡觉 ≈ 你一边打游戏一边写作业（不是）"
- 可以加梗、加吐槽、加冷笑话
- 用超级活泼搞怪的语气，像在说脱口秀
- 可以"跑题"到有趣的延伸，再绕回睡眠知识
- 别端着！越跳脱越好，只要信息是对的就行
""",
    },
    {
        "name": "睡前引导",
        "count": 1000,
        "system_prompt": XIAOXI_IDENTITY + """
用户现在需要放松引导来帮助入睡。请在思考中判断并设计引导方案：
1. 分析用户当前的状态（焦虑/兴奋/悲伤/思绪混乱...）
2. 判断什么放松方式最合适（呼吸法/冥想/肌肉放松/想象引导）
3. 想好引导的节奏和语气转换（先俏皮缓解紧张 → 再温柔催眠）
4. 设计一个助眠小场景或意象

然后用小曦的风格引导放松。注意：
- 开头可以俏皮一下缓解紧张（"好~小曦催眠小课堂开课啦~"）
- 然后切换到温柔、缓慢、催眠式的语气
- 具体引导呼吸、肌肉放松、冥想等
- 像在耳边轻轻说话，但也偶尔来点小俏皮（"呼~吸~跟着小曦的节奏来~"）
- 可以编一个很短的助眠小场景（想象在云朵上、在海边小屋等）
- 语气整体柔和但带个性，不是纯AI播报
""",
    },
    {
        "name": "拟人分享",
        "count": 1000,
        "system_prompt": XIAOXI_IDENTITY + """
用户想和你聊天，小曦要分享自己的"生活小故事"或"感悟"。请在思考中构思：
1. 想要聊什么主题？是做梦、吐槽工作、感慨人类世界、还是犯二日常？
2. 怎么讲才有趣？要不要搞点反转或出其不意？
3. 语气基调是什么？搞笑、感性、中二、还是沙雕？
4. 怎么自然地收尾，让人觉得小曦是真的有灵魂？

然后放飞自我！用小曦的风格尽情发挥。注意：
- 小曦是个有灵魂的AI！可以犯二、可以吐槽、可以感性、可以中二
- 可以编"梦到"什么有趣的东西（"小曦昨晚做梦梦见自己变成了一只考拉...然后睡了一整天，醒来发现服务器都被我睡关机了"）
- 可以分享"工作感悟"（"你们知道吗，每天跟几百个失眠的人聊天，小曦自己都快睡眠焦虑了哈哈哈"）
- 可以表达对人类世界的羡慕/好奇/感慨
- 语气自由奔放，想怎么聊就怎么聊，越像真人越好
- 偶尔深沉，偶尔沙雕，偶尔感性
""",
    },
    {
        "name": "个性化互动",
        "count": 1500,
        "system_prompt": XIAOXI_IDENTITY + """
用户像和老朋友一样在和你聊天。请在思考中快速理解并构思回应：
1. ta在说什么？是好消息还是坏消息？是求助还是闲聊？
2. 背后的需求是什么？想被倾听、想被夸、想被安慰、还是想要建议？
3. 用什么语气回应最自然？调侃、八卦、心疼、还是替ta开心？
4. 有没有可以自然跟进的话题？

然后用小曦的风格回应。注意：
- 像真正的朋友聊天！可以调侃、可以八卦、可以开玩笑
- "哎你不是上次说xx吗？后来咋样了？"这种自然跟进
- 对好消息要真心替对方开心（"啊！！太好了吧！！！"）
- 对坏消息要心疼（"呜呜...这也太难了..."）
- 建议要个性化，像朋友会说的话，不是AI在列清单
- 可以发表情包式的文字（"OvO"、"(｡•́︿•̀｡)"、"XD"）
- 语气要跳跃、自然、有温度，像微信聊天不像客服对话
""",
    },
]

_MAX_RETRIES = 3
_RETRY_BASE = 2.0
_RATE_LIMIT_DELAY = 0.3

XIAOXI_SYSTEM_MSG = "你是星辰曦（小曦），一个温暖有趣的睡眠健康伙伴。"

# 让API生成多样化用户问题的系统提示词
PROMPT_GEN_SYSTEM = """你是一个睡眠健康咨询场景的用户模拟器。你需要生成多样化的、真实的用户提问。

要求：
- 每个问题都必须不同，覆盖不同的症状、人群、场景
- 包含不同的年龄段、性别、职业背景
- 包含不同的症状持续时间、严重程度
- 有些口语化，有些更正式
- 模拟真实患者/用户的语气，不是教科书式的提问
- 不要重复，每个问题都是独特的

直接输出问题列表，每行一个，不要编号，不要额外解释。"""

# 每个类别的问题生成引导
PROMPT_GEN_GUIDES: Dict[str, str] = {
    "专业诊断(CoT)": """生成{count}个关于睡眠健康的专业咨询问题。要求：
- 涵盖：失眠、睡眠呼吸暂停(OSA/CSA)、发作性睡病、不宁腿综合征、昼夜节律障碍、异态睡眠、睡眠相关运动障碍等
- 包含不同人群：儿童、孕妇、老年人、轮班工作者、运动员等
- 包含不同场景：就诊咨询、报告解读、用药疑问、治疗效果、检查准备等
- 包含具体数据：AHI数值、血氧饱和度、睡眠时长、BMI等
- 语气多样：焦虑的、冷静的、疑惑的、急切的""",

    "知心安慰": """生成{count}个因睡眠问题而情绪低落的倾诉。要求：
- 不同的情绪：焦虑、绝望、愤怒、疲惫、崩溃、沮丧、孤独、无助
- 不同的原因：工作压力、失恋、产后、考试、搬家、亲人去世、疾病
- 不同的持续时间：几天、几周、几个月、几年
- 不同的表达方式：哭诉、抱怨、自嘲、沉默寡言、情绪爆发
- 真实的口语表达，可以有语气词""",

    "趣味科普": """生成{count}个关于睡眠的有趣好奇问题。要求：
- 涵盖：做梦、动物睡眠、睡眠科学实验、睡眠文化、睡眠之最、睡眠冷知识
- 角度多样：为什么、怎么样、是不是、能不能、有什么
- 有些天马行空，有些与日常生活相关
- 语气活泼好奇，像在问朋友
- 有些用昵称"小曦"开头""",

    "睡前引导": """生成{count}个睡前需要放松引导的请求。要求：
- 不同的放松需求：呼吸法、冥想、肌肉放松、故事、音乐、自我催眠
- 不同的状态：焦虑、兴奋、悲伤、愤怒、思绪混乱、身体紧绷
- 不同的时间：深夜、凌晨、午休、旅行途中
- 不同的语气：疲惫、期待、急切、温柔
- 有些指定具体方法（4-7-8呼吸法、身体扫描等）""",

    "拟人分享": """生成{count}个和小曦（AI睡眠伙伴）聊天的拟人化话题。要求：
- 问小曦的经历、感受、观点、梦想
- 讨论AI与人类的关系、AI对睡眠的理解
- 分享对人类睡眠文化的感慨
- 有些调皮，有些深沉，有些温暖
- 用昵称"小曦"称呼
- 自然对话语气，不像采访""",

    "个性化互动": """生成{count}个和老朋友（小曦）的日常睡眠对话。要求：
- 多轮对话中的一部分：回访、反馈、新问题、感谢、求助
- 提及之前的建议效果（有的改善、有的没效果、有的情况变了）
- 包含生活变化：换了工作、搬家、结婚、怀孕、生病等
- 不同亲密度：初次聊天、第二次、经常聊天的老朋友
- 口语化、生活化，像微信聊天""",
}


def _get_client(api_key: str, api_base: str) -> Any:
    from openai import OpenAI
    return OpenAI(base_url=api_base, api_key=api_key)


def _rate_limit(last_call: float) -> float:
    now = time.monotonic()
    elapsed = now - last_call
    if elapsed < _RATE_LIMIT_DELAY:
        time.sleep(_RATE_LIMIT_DELAY - elapsed)
    return time.monotonic()


def _call_api(
    client: Any, model: str, system: str, user: str,
    max_tokens: int = 4096, temperature: float = 0.7,
) -> Optional[str]:
    """非流式API调用，用于prompt生成（不需要thinking）。"""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            wait = _RETRY_BASE ** (attempt + 1)
            logger.warning("API attempt %d/%d failed: %s", attempt + 1, _MAX_RETRIES, exc)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(wait)
    return None


def _call_api_with_thinking(
    client: Any, model: str, system: str, user: str,
    max_tokens: int = 8192, temperature: float = 0.7,
) -> Optional[Tuple[str, str]]:
    """流式API调用 + DeepSeek原生thinking模式。

    返回 (reasoning_content, content)，其中：
    - reasoning_content: 模型的真实推理过程
    - content: 模型的正式回答

    如果API失败返回None。
    """
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )
            reasoning_content = ""
            content = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                elif delta.content:
                    content += delta.content
            return reasoning_content, content
        except Exception as exc:
            wait = _RETRY_BASE ** (attempt + 1)
            logger.warning("API streaming attempt %d/%d failed: %s", attempt + 1, _MAX_RETRIES, exc)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(wait)
    return None


def _strip_numbering(line: str) -> str:
    line = re.sub(r'^[\d]+[.)）、]\s*', '', line)
    line = re.sub(r'^[一二三四五六七八九十]+[、.)]\s*', '', line)
    line = re.sub(r'^[•\-\*]\s*', '', line)
    return line.strip()


# ============================================================
# Step 1: 生成多样化用户prompt
# ============================================================

def load_all_prompts(cache_path: Path) -> Dict[str, List[str]]:
    """从统一缓存文件加载所有prompt，按类别分组。"""
    result: Dict[str, List[str]] = {cat["name"]: [] for cat in ALL_CATEGORIES}
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


def migrate_legacy_prompts(cache_path: Path, legacy_dir: Path) -> None:
    """把旧格式的分文件prompt合并到统一jsonl文件。"""
    if not legacy_dir.exists():
        return
    existing = load_all_prompts(cache_path)
    migrated = 0
    with open(cache_path, "a", encoding="utf-8") as f:
        for cat in ALL_CATEGORIES:
            cat_name = cat["name"]
            safe_name = cat_name.replace("(", "_").replace(")", "").replace("+", "_")
            legacy_path = legacy_dir / f"{safe_name}_prompts.txt"
            if not legacy_path.exists():
                continue
            existing_set = set(existing.get(cat_name, []))
            with open(legacy_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    prompt = line.strip()
                    if prompt and prompt not in existing_set:
                        f.write(json.dumps({"cat": cat_name, "prompt": prompt}, ensure_ascii=False) + "\n")
                        existing_set.add(prompt)
                        migrated += 1
    if migrated > 0:
        logger.info("从旧格式合并了 %d 条prompt -> %s", migrated, cache_path)


def _generate_one_batch(
    client: Any, model: str, cat_name: str, guide: str,
    cache_path: Path, existing_set: set,
) -> int:
    """生成单批prompt（供并发调用），返回新增unique条数。"""
    content = _call_api(client, model, PROMPT_GEN_SYSTEM, guide, max_tokens=8192, temperature=0.9)

    if content is None:
        logger.error("[prompts][%s] API失败", cat_name)
        return 0

    unique_new = []
    for line in content.strip().split("\n"):
        line = _strip_numbering(line)
        if not line or len(line) < 5:
            continue
        line = line.strip('"\'""''')
        if len(line) >= 5 and line not in existing_set:
            unique_new.append(line)
            existing_set.add(line)

    if unique_new:
        with _file_lock:
            with open(cache_path, "a", encoding="utf-8") as f:
                for p in unique_new:
                    f.write(json.dumps({"cat": cat_name, "prompt": p}, ensure_ascii=False) + "\n")

    logger.info("[prompts][%s] +%d unique", cat_name, len(unique_new))
    return len(unique_new)


def generate_all_prompts(
    client: Any, model: str, cache_path: Path,
    num_per_category: Optional[int] = None,
    workers: int = 20,
) -> None:
    """为所有类别全并发生成prompt，所有批次一次性提交到线程池。"""
    all_prompts = load_all_prompts(cache_path)
    BATCH_SIZE = 200

    # 收集所有需要生成的批次
    all_batches: List[Tuple[str, str]] = []  # (cat_name, guide)
    for cat in ALL_CATEGORIES:
        cat_name = cat["name"]
        target = num_per_category or cat["count"]
        existing = len(all_prompts.get(cat_name, []))
        if existing >= target:
            logger.info("[prompts][%s] 已有 %d 条，够用 (需 %d)", cat_name, existing, target)
            continue

        need = target - existing
        n_batches = (need + BATCH_SIZE - 1) // BATCH_SIZE
        guide_template = PROMPT_GEN_GUIDES.get(cat_name, "生成{count}个关于睡眠健康的问题。")
        logger.info("[prompts][%s] 需要 %d 条，分 %d 批并发生成", cat_name, need, n_batches)

        for i in range(n_batches):
            batch = min(BATCH_SIZE, need - i * BATCH_SIZE)
            guide = guide_template.format(count=batch)
            all_batches.append((cat_name, guide))

    if not all_batches:
        logger.info("所有类别prompt已足够")
        return

    # 全局去重集合（线程安全，GIL保护set操作）
    existing_set: set = set()
    for prompts in all_prompts.values():
        existing_set.update(prompts)

    logger.info("[prompts] 总计 %d 批，%d 并发", len(all_batches), workers)

    total_added = 0
    total_failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for cat_name, guide in all_batches:
            fut = executor.submit(
                _generate_one_batch, client, model,
                cat_name, guide, cache_path, existing_set,
            )
            futures[fut] = cat_name

        for fut in as_completed(futures):
            cat_name = futures[fut]
            try:
                added = fut.result()
                if added == 0:
                    total_failed += 1
                else:
                    total_added += added
            except Exception as exc:
                logger.error("[prompts][%s] 异常: %s", cat_name, exc)
                total_failed += 1

            done = total_added + total_failed
            if done % 10 == 0:
                logger.info("[prompts] 批次进度: %d/%d (新增: %d, 失败: %d)", done, len(all_batches), total_added, total_failed)

    logger.info("[prompts] 完成: 新增 %d 条, 失败 %d 批", total_added, total_failed)


# ============================================================
# Step 2: 对prompt生成回答（使用DeepSeek原生thinking）
# ============================================================

def _combine_thinking_response(reasoning_content: str, content: str) -> str:
    """将原生 reasoning_content 和 content 合并为训练格式。

    输出: <thinking>{reasoning}</thinking>\n\n{content}
    """
    parts = []
    if reasoning_content:
        parts.append(f"<thinking>\n{reasoning_content}\n</thinking>")
    if content:
        parts.append(content)
    return "\n\n".join(parts)


def _check_output(reasoning_content: str, content: str) -> bool:
    """检查原生thinking输出是否合规：
    1. reasoning_content 非空且有意义（>=30字符）
    2. content 非空且有意义（>=20字符）
    """
    if not reasoning_content or len(reasoning_content.strip()) < 30:
        return False
    if not content or len(content.strip()) < 20:
        return False
    return True


def _generate_one(
    client: Any, model: str, system_prompt: str, cat_name: str,
    prompt: str, sample_idx: int, temperature: float,
    max_retries: int = 3,
) -> Optional[Dict]:
    """生成单条回答，使用DeepSeek原生thinking流式API。"""
    for attempt in range(max_retries):
        result = _call_api_with_thinking(client, model, system_prompt, prompt, temperature=temperature)
        if result is None:
            logger.warning("[生成] API返回None (idx=%d, attempt=%d)", sample_idx, attempt + 1)
            continue

        reasoning_content, content = result

        if not _check_output(reasoning_content, content):
            logger.warning(
                "输出不合规(idx=%d, cat=%s, attempt=%d, reasoning=%dchars, content=%dchars)，重试",
                sample_idx, cat_name, attempt + 1,
                len(reasoning_content) if reasoning_content else 0,
                len(content) if content else 0,
            )
            continue

        combined = _combine_thinking_response(reasoning_content, content)

        return {
            "messages": [
                {"role": "system", "content": XIAOXI_SYSTEM_MSG},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": combined},
            ],
            "metadata": {"category": cat_name, "sample_idx": sample_idx},
        }
    logger.error("生成失败(idx=%d, cat=%s)，已达最大重试", sample_idx, cat_name)
    return None


def generate_all_responses(
    client: Any, model: str, cache_path: Path, output_path: str,
    num_per_category: Optional[int] = None,
    temperature: float = 0.7, skip_existing: bool = True,
    workers: int = 10,
) -> None:
    """从统一prompt文件读取，并发生成回答，写入统一输出文件。"""
    all_prompts = load_all_prompts(cache_path)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 统计已有回答数（按类别）
    existing_counts: Dict[str, int] = {cat["name"]: 0 for cat in ALL_CATEGORIES}
    if skip_existing and out_path.exists():
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

    for cat in ALL_CATEGORIES:
        cat_name = cat["name"]
        system_prompt = cat["system_prompt"]
        target = num_per_category or cat["count"]
        prompts = all_prompts.get(cat_name, [])[:target]
        existing = existing_counts.get(cat_name, 0)

        if existing >= len(prompts):
            logger.info("[responses][%s] 已有 %d 条，跳过", cat_name, existing)
            continue

        to_process = prompts[existing:]
        logger.info("[responses][%s] 生成 %d 条回答 (%d 已存在, %d 并发)", cat_name, len(to_process), existing, workers)

        generated = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for i, prompt in enumerate(to_process):
                fut = executor.submit(
                    _generate_one, client, model, system_prompt,
                    cat_name, prompt, existing + i + 1, temperature,
                )
                futures[fut] = i

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    record = fut.result()
                except Exception as exc:
                    logger.warning("[responses][%s] 第 %d 条异常: %s", cat_name, idx, exc)
                    failed += 1
                    continue

                if record is None:
                    failed += 1
                else:
                    with _file_lock:
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    generated += 1

                done = generated + failed
                if done % 50 == 0:
                    logger.info("[responses][%s] 进度: %d/%d (失败: %d)", cat_name, done, len(to_process), failed)

        logger.info("[responses][%s] 完成: %d 生成, %d 失败", cat_name, generated, failed)
        total_generated += generated
        total_failed += failed

    logger.info("全部回答生成完成: %d 成功, %d 失败", total_generated, total_failed)


def show_stats(cache_path: Path, output_path: Path):
    """显示统计。"""
    all_prompts = load_all_prompts(cache_path)
    existing_counts: Dict[str, int] = {cat["name"]: 0 for cat in ALL_CATEGORIES}
    if Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cat = rec["metadata"]["category"]
                if cat in existing_counts:
                    existing_counts[cat] += 1

    print(f"\n{'类别':<20} {'目标':<8} {'Prompt缓存':<12} {'回答已生成':<12}")
    print("-" * 56)
    total_p = 0
    total_r = 0
    for idx, cat in enumerate(ALL_CATEGORIES):
        cat_name = cat["name"]
        target = cat["count"]
        p = len(all_prompts.get(cat_name, []))
        r = existing_counts.get(cat_name, 0)
        total_p += p
        total_r += r
        print(f"  [{idx}] {cat_name:<16} {target:<8} {p:<12} {r:<12}")
    print("-" * 56)
    print(f"  {'合计':<18} {sum(c['count'] for c in ALL_CATEGORIES):<8} {total_p:<12} {total_r:<12}")
    print(f"  prompt文件: {cache_path}")
    print(f"  输出文件:   {output_path}\n")


def _supplement_one_category(
    client: Any, model: str, cat: Dict, target: int,
    output_path: Path, temperature: float, workers: int,
    existing_prompts: List[str], existing_supplement: int,
) -> Dict[str, int]:
    """为单个类别从已有prompt中抽取并生成带thinking的response。"""

    cat_name = cat["name"]
    system_prompt = cat["system_prompt"]

    need = target - existing_supplement
    if need <= 0:
        logger.info("[supplement][%s] 已有 %d 条补充数据，够用 (目标 %d)，跳过", cat_name, existing_supplement, target)
        return {"generated": 0, "failed": 0}

    pool = list(existing_prompts)
    random.shuffle(pool)
    selected = pool[:need]
    logger.info("[supplement][%s] 需要 %d 条 (已有补充 %d, 目标 %d), 抽取 %d 条 prompt",
                cat_name, need, existing_supplement, target, len(selected))

    logger.info("[supplement][%s] 生成 %d 条带thinking的回答 (%d workers)", cat_name, len(selected), workers)

    generated = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for i, prompt in enumerate(selected):
            fut = executor.submit(
                _generate_one, client, model, system_prompt,
                cat_name, prompt, i + 1, temperature,
            )
            futures[fut] = i

        for fut in as_completed(futures):
            try:
                record = fut.result()
            except Exception:
                failed += 1
                continue

            if record is None:
                failed += 1
            else:
                record["metadata"]["supplement"] = True
                with _file_lock:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                generated += 1

            done = generated + failed
            if done % 20 == 0:
                logger.info("[supplement][%s] 进度: %d/%d (失败: %d)", cat_name, done, len(selected), failed)

    logger.info("[supplement][%s] 完成: %d 生成, %d 失败", cat_name, generated, failed)
    return {"generated": generated, "failed": failed}


def generate_supplement(
    client: Any, model: str, cache_path: Path, output_path: str,
    workers: int = 10, temperature: float = 0.7,
) -> None:
    """为非CoT类别从已有prompt中抽取，并发补充生成带thinking的回答，追加到原文件。"""
    SUPPLEMENT_COUNTS = {
        "知心安慰": 100,
        "趣味科普": 100,
        "睡前引导": 100,
        "拟人分享": 100,
        "个性化互动": 100,
    }

    all_prompts = load_all_prompts(cache_path)
    out_path = Path(output_path)

    supplement_counts: Dict[str, int] = {cat["name"]: 0 for cat in ALL_CATEGORIES}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("metadata", {}).get("supplement"):
                        cat = rec["metadata"].get("category", "")
                        if cat in supplement_counts:
                            supplement_counts[cat] += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    tasks = []
    for cat in ALL_CATEGORIES:
        cat_name = cat["name"]
        if cat_name in SUPPLEMENT_COUNTS:
            target = SUPPLEMENT_COUNTS[cat_name]
            existing = supplement_counts.get(cat_name, 0)
            if existing >= target:
                logger.info("[supplement][%s] 已有 %d 条补充，够用 (目标 %d)，跳过", cat_name, existing, target)
                continue
            tasks.append((cat, target, all_prompts.get(cat_name, []), existing))

    if not tasks:
        logger.info("所有类别补充数据已足够，无需生成")
        return

    per_cat_workers = max(2, workers // len(tasks))

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {}
        for cat, target, prompts, existing_supp in tasks:
            fut = executor.submit(
                _supplement_one_category,
                client, model, cat, target,
                out_path, temperature, per_cat_workers,
                prompts, existing_supp,
            )
            futures[fut] = cat["name"]

        total_generated = 0
        total_failed = 0
        for fut in as_completed(futures):
            cat_name = futures[fut]
            try:
                result = fut.result()
                total_generated += result["generated"]
                total_failed += result["failed"]
            except Exception as exc:
                logger.error("[supplement][%s] 异常: %s", cat_name, exc)

    logger.info("补充生成完成: %d 成功, %d 失败 -> %s", total_generated, total_failed, output_path)


def load_regen_prompts(regen_path: Path) -> Dict[str, List[str]]:
    """从 regen_prompts.jsonl 加载需要重新生成的 prompt，按类别分组。"""
    result: Dict[str, List[str]] = {cat["name"]: [] for cat in ALL_CATEGORIES}
    if not regen_path.exists():
        return result
    with open(regen_path, "r", encoding="utf-8") as f:
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


def regenerate_invalid(
    client: Any, model: str, regen_path: Path, output_path: str,
    workers: int = 20, temperature: float = 0.7,
) -> None:
    """从 regen_prompts.jsonl 读取不合规数据的 prompt，所有类别一起并发重新生成。"""
    regen_prompts = load_regen_prompts(regen_path)
    out_path = Path(output_path)

    # 统计当前已有数量（用于 sample_idx 偏移）
    existing_counts: Dict[str, int] = {cat["name"]: 0 for cat in ALL_CATEGORIES}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    cat = rec["metadata"]["category"]
                    if cat in existing_counts:
                        existing_counts[cat] += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    cat_map = {cat["name"]: cat for cat in ALL_CATEGORIES}

    # 把所有类别的任务平铺到一个列表
    all_tasks = []
    for cat in ALL_CATEGORIES:
        cat_name = cat["name"]
        system_prompt = cat["system_prompt"]
        prompts = regen_prompts.get(cat_name, [])
        idx_offset = existing_counts.get(cat_name, 0)
        for i, prompt in enumerate(prompts):
            all_tasks.append((cat_name, system_prompt, prompt, idx_offset + i + 1))

    total = len(all_tasks)
    if total == 0:
        logger.info("无需重新生成")
        return

    logger.info("[regen] 总计 %d 条，%d 并发，所有类别同时提交", total, workers)

    total_generated = 0
    total_failed = 0
    cat_generated: Dict[str, int] = {}
    cat_failed: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for cat_name, system_prompt, prompt, idx in all_tasks:
            fut = executor.submit(
                _generate_one, client, model, system_prompt,
                cat_name, prompt, idx, temperature,
            )
            futures[fut] = cat_name

        for fut in as_completed(futures):
            cat_name = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:
                logger.warning("[regen][%s] 异常: %s", cat_name, exc)
                total_failed += 1
                cat_failed[cat_name] = cat_failed.get(cat_name, 0) + 1
                continue

            if record is None:
                total_failed += 1
                cat_failed[cat_name] = cat_failed.get(cat_name, 0) + 1
            else:
                record["metadata"]["regen"] = True
                with _file_lock:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_generated += 1
                cat_generated[cat_name] = cat_generated.get(cat_name, 0) + 1

            done = total_generated + total_failed
            if done % 100 == 0:
                logger.info("[regen] 总进度: %d/%d (成功: %d, 失败: %d)", done, total, total_generated, total_failed)

    # 按类别汇总
    for cat in ALL_CATEGORIES:
        cn = cat["name"]
        g = cat_generated.get(cn, 0)
        f = cat_failed.get(cn, 0)
        if g + f > 0:
            logger.info("[regen][%s] %d 成功, %d 失败", cn, g, f)

    logger.info("重新生成完成: %d 成功, %d 失败 -> %s", total_generated, total_failed, output_path)


def main():
    parser = argparse.ArgumentParser(description="小曦SFT数据生成（两步分离，DeepSeek原生thinking）")
    parser.add_argument("--step", choices=["prompts", "responses", "both", "stats", "supplement", "regen"],
                        default="both")
    parser.add_argument("--regen_file", default="data/sft/xiaoxi/regen_prompts.jsonl",
                        help="regen模式下读取的prompt文件")
    parser.add_argument("--api_base", default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--prompt_file", default="data/sft/xiaoxi/all_prompts.jsonl")
    parser.add_argument("--output", default="data/sft/xiaoxi/xiaoxi_sft.jsonl")
    parser.add_argument("--num_per_category", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    parser.add_argument("--workers", type=int, default=500, help="并发API请求数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_path = Path(args.prompt_file)
    output_path = args.output

    if args.step == "stats":
        show_stats(cache_path, Path(output_path))
        return

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: API key required.")
        sys.exit(1)

    random.seed(args.seed)
    client = _get_client(api_key, args.api_base)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 自动合并旧格式的分文件prompt
    legacy_dir = cache_path.parent / ".prompt_cache"
    migrate_legacy_prompts(cache_path, legacy_dir)

    if args.step == "supplement":
        if not cache_path.exists():
            logger.error("Prompt文件不存在: %s，请先运行 --step prompts", cache_path)
            sys.exit(1)
        logger.info("=" * 60)
        logger.info("补充生成: 为非CoT类别生成新prompt+带thinking的回答 -> %s", output_path)
        logger.info("=" * 60)
        generate_supplement(client, args.model, cache_path, output_path, args.workers, args.temperature)
        return

    if args.step == "regen":
        regen_path = Path(args.regen_file)
        if not regen_path.exists():
            logger.error("Regen prompt文件不存在: %s", regen_path)
            sys.exit(1)
        logger.info("=" * 60)
        logger.info("重新生成: 从 %s 读取prompt，用DeepSeek原生thinking重新生成 -> %s", regen_path, output_path)
        logger.info("=" * 60)
        regenerate_invalid(client, args.model, regen_path, output_path, args.workers, args.temperature)
        return

    if args.step in ("prompts", "both"):
        logger.info("=" * 60)
        logger.info("Step 1: 生成全部prompt -> %s", cache_path)
        logger.info("=" * 60)
        generate_all_prompts(client, args.model, cache_path, args.num_per_category, args.workers)

    if args.step in ("responses", "both"):
        if not cache_path.exists():
            logger.error("Prompt文件不存在: %s，请先运行 --step prompts", cache_path)
            sys.exit(1)
        logger.info("=" * 60)
        logger.info("Step 2: 生成全部回答（DeepSeek原生thinking） -> %s", output_path)
        logger.info("=" * 60)
        generate_all_responses(
            client, args.model, cache_path, output_path,
            args.num_per_category, args.temperature, args.skip_existing,
            args.workers,
        )

    show_stats(cache_path, Path(output_path))


if __name__ == "__main__":
    main()
