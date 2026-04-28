"""Seed prompts for synthetic sleep medicine data generation.

Provides categorized prompt templates covering major sleep medicine domains
in both Chinese and English, with difficulty-level annotations.  Used as
input to ``SyntheticDataGenerator`` to produce training conversations.
"""

from __future__ import annotations

import random
from typing import Any

# ---------------------------------------------------------------------------
# Consultation prompts
# ---------------------------------------------------------------------------

_CONSULTATION_PROMPTS_ZH: list[dict[str, str]] = [
    {"category": "insomnia", "prompt": "我最近一个月每天都很难入睡，躺在床上至少要两三个小时才能睡着。白天很疲惫，无法集中精力工作。请问这可能是什么问题？有什么方法可以改善？", "difficulty": "simple", "language": "zh"},
    {"category": "insomnia", "prompt": "我经常在半夜醒来后就再也睡不着了，这种情况已经持续了三个月。我已经尝试过喝热牛奶和泡脚，但效果不明显。请问除了这些还有什么有效的非药物方法？", "difficulty": "medium", "language": "zh"},
    {"category": "insomnia", "prompt": "我被诊断为慢性失眠障碍，CBT-I治疗效果不佳，医生建议药物治疗。请问目前指南推荐的药物有哪些？各自的优缺点是什么？长期使用有什么注意事项？", "difficulty": "complex", "language": "zh"},
    {"category": "sleep_apnea", "prompt": "我妻子说我睡觉时打呼噜很大声，有时还会憋气。白天总觉得困，开车的时候差点睡着。这可能是什么病？需要做哪些检查？", "difficulty": "simple", "language": "zh"},
    {"category": "sleep_apnea", "prompt": "我做了睡眠监测，结果显示AHI为28次/小时，最低血氧85%。医生说是中度阻塞性睡眠呼吸暂停。请问CPAP治疗和口腔矫正器哪个更适合我？手术能根治吗？", "difficulty": "medium", "language": "zh"},
    {"category": "sleep_apnea", "prompt": "我患有重度OSA（AHI 45），使用CPAP治疗半年后AHI降到5以下，但最近出现了中枢性呼吸暂停事件。请问什么是治疗性中枢性睡眠呼吸暂停（CSA）？该如何处理？", "difficulty": "complex", "language": "zh"},
    {"category": "narcolepsy", "prompt": "我白天会突然不受控制地睡着，有时在吃饭或开会的时候也会睡着。还经常做噩梦，刚醒来的时候身体动不了。这是什么病？", "difficulty": "simple", "language": "zh"},
    {"category": "narcolepsy", "prompt": "我被诊断为1型嗜睡症（伴有猝倒），目前服用莫达非尼，但效果不太理想。请问有哪些其他治疗选择？新型药物如 pitolisant 和 solriamfetol 的效果如何？", "difficulty": "complex", "language": "zh"},
    {"category": "narcolepsy", "prompt": "我的孩子今年8岁，上课总是打瞌睡，成绩下降明显。老师说他在课堂上会突然低头睡着。我们需要做哪些检查来排除嗜睡症？儿童嗜睡症和成人有什么不同？", "difficulty": "medium", "language": "zh"},
    {"category": "rls", "prompt": "每天晚上准备睡觉的时候，我的腿就觉得很不舒服，有一种说不出来的虫爬感，必须不停地动腿才能缓解。这已经影响到了我的睡眠，请问这是什么病？", "difficulty": "simple", "language": "zh"},
    {"category": "rls", "prompt": "我确诊不宁腿综合征两年了，血清铁蛋白检测为35 ng/mL。医生建议补铁。请问铁剂补充的目标值是多少？补铁的同时可以服用多巴胺受体激动剂吗？需要注意哪些副作用？", "difficulty": "medium", "language": "zh"},
    {"category": "rls", "prompt": "我使用普拉克索治疗不宁腿综合征已经三年，最近出现了症状恶化（augmentation），药量越加越大但效果越来越差。请问什么是不宁腿综合征的augmentation？如何处理？可以换用加巴喷丁类药物治疗吗？", "difficulty": "complex", "language": "zh"},
    {"category": "children_sleep", "prompt": "我的宝宝6个月大，晚上总是频繁醒来，每次醒来都要喂奶才能重新入睡。请问这是正常的吗？如何培养宝宝自主入睡的能力？", "difficulty": "simple", "language": "zh"},
    {"category": "children_sleep", "prompt": "我5岁的孩子每天晚上都要我陪在身边才能入睡，如果我不在他就哭闹不止。他还经常做噩梦，半夜跑到我房间。请问如何帮助孩子克服睡眠焦虑？什么年龄应该开始分床睡？", "difficulty": "medium", "language": "zh"},
    {"category": "children_sleep", "prompt": "我12岁的孩子上初中后睡眠时间严重不足，每天只睡5-6个小时，说作业太多写不完。周末会补觉到中午。请问青少年每天应该睡多久？长期睡眠不足会影响发育吗？如何帮助他改善睡眠？", "difficulty": "medium", "language": "zh"},
    {"category": "elderly_sleep", "prompt": "我今年72岁，最近几年睡眠越来越浅，每天凌晨三四点就醒了，白天也没什么精神。请问老年人睡眠变差是正常的吗？有什么办法可以改善？", "difficulty": "simple", "language": "zh"},
    {"category": "elderly_sleep", "prompt": "我父亲80岁，患有帕金森病，晚上睡觉时经常大声喊叫、手脚乱动，有时还会从床上摔下来。请问这是什么问题？和帕金森病有关吗？需要怎么治疗？", "difficulty": "medium", "language": "zh"},
    {"category": "sleep_hygiene", "prompt": "请问有哪些好的睡眠习惯可以帮助改善睡眠质量？我平时经常熬夜刷手机，早上起不来。", "difficulty": "simple", "language": "zh"},
    {"category": "sleep_hygiene", "prompt": "我上夜班已经有两年了，白天总是睡不好，感觉很疲惫。请问倒班工作者应该如何调整作息？褪黑素有用吗？", "difficulty": "medium", "language": "zh"},
    {"category": "medication", "prompt": "我服用艾司佐匹克隆已经半年了，现在想停药。可以直接停吗？还是需要逐渐减量？", "difficulty": "simple", "language": "zh"},
    {"category": "medication", "prompt": "请问褪黑素可以长期服用吗？有什么副作用？和安眠药相比有什么优缺点？", "difficulty": "medium", "language": "zh"},
    {"category": "medication", "prompt": "我对多种催眠药物产生了耐受性，包括佐匹克隆、右佐匹克隆和曲唑酮。目前残留的入睡潜伏期仍有60-90分钟。请问是否有其他药物治疗方案？联合用药的安全性和证据级别如何？", "difficulty": "complex", "language": "zh"},
    {"category": "circadian", "prompt": "我每天凌晨三四点才睡，下午才起床。试过设定闹钟但坚持不了。请问这是睡眠相位延迟障碍吗？", "difficulty": "medium", "language": "zh"},
    {"category": "circadian", "prompt": "我有一个视力严重受损的盲人朋友，他的睡眠非常不规律，有时候白天睡觉晚上清醒。请问这是什么问题？有没有什么有效的治疗方法？", "difficulty": "complex", "language": "zh"},
    {"category": "psg", "prompt": "医生让我做睡眠监测（PSG），请问这个检查是怎么做的？需要住院吗？检查前有什么需要注意的？", "difficulty": "simple", "language": "zh"},
    {"category": "psg", "prompt": "我的睡眠监测报告显示N3睡眠只有8%，REM睡眠15%，觉醒指数25次/小时。请问这些指标正常吗？代表什么？", "difficulty": "medium", "language": "zh"},
]

_CONSULTATION_PROMPTS_EN: list[dict[str, str]] = [
    {"category": "insomnia", "prompt": "I have been struggling to fall asleep for the past month. It takes me 2-3 hours in bed before I finally drift off. I am exhausted during the day and cannot focus at work. What could be causing this and what can I do about it?", "difficulty": "simple", "language": "en"},
    {"category": "insomnia", "prompt": "I wake up frequently during the night and cannot get back to sleep. This has been going on for three months. I have tried warm milk and foot baths without much improvement. What evidence-based non-pharmacological treatments are available?", "difficulty": "medium", "language": "en"},
    {"category": "insomnia", "prompt": "I was diagnosed with chronic insomnia disorder. CBT-I was partially effective but I still have significant residual symptoms. My physician is considering pharmacotherapy. What are the current guideline-recommended medications, their comparative efficacy, and long-term safety considerations?", "difficulty": "complex", "language": "en"},
    {"category": "sleep_apnea", "prompt": "My partner tells me I snore loudly and sometimes stop breathing in my sleep. I am always tired during the day and nearly fell asleep while driving. What condition could this be and what tests do I need?", "difficulty": "simple", "language": "en"},
    {"category": "sleep_apnea", "prompt": "My sleep study showed an AHI of 28/hour with nadir SpO2 of 85%. I was told I have moderate OSA. Should I use CPAP or an oral appliance? Is surgery a viable option for a permanent cure?", "difficulty": "medium", "language": "en"},
    {"category": "sleep_apnea", "prompt": "I have severe OSA (AHI 45) and have been on CPAP for 6 months with good control (residual AHI < 5). However, my recent follow-up PSG showed emerging central sleep apnea events. What is treatment-emergent CSA and how should it be managed?", "difficulty": "complex", "language": "en"},
    {"category": "narcolepsy", "prompt": "I suddenly fall asleep during the day without warning, sometimes while eating or in meetings. I also experience vivid dreams and paralysis when waking up. What could be wrong with me?", "difficulty": "simple", "language": "en"},
    {"category": "narcolepsy", "prompt": "I have type 1 narcolepsy with cataplexy currently managed with modafinil, but my symptoms are not well controlled. What are the alternatives? How do newer agents like pitolisant and solriamfetol compare in terms of efficacy and side effects?", "difficulty": "complex", "language": "en"},
    {"category": "rls", "prompt": "Every evening when I lie down to sleep, I get an uncomfortable crawling sensation in my legs and have to keep moving them. This is disrupting my sleep. What condition could this be?", "difficulty": "simple", "language": "en"},
    {"category": "rls", "prompt": "I was diagnosed with RLS 2 years ago. My serum ferritin is 35 ng/mL. My doctor recommended iron supplementation. What is the target ferritin level for RLS treatment? Can I take a dopamine agonist alongside iron therapy?", "difficulty": "medium", "language": "en"},
    {"category": "rls", "prompt": "I have been on pramipexole for RLS for 3 years and recently developed augmentation -- the dose keeps increasing but symptoms worsen. What is the current treatment algorithm for DA-augmentation? Should I switch to a gabapentinoid?", "difficulty": "complex", "language": "en"},
    {"category": "sleep_hygiene", "prompt": "What are the most effective sleep hygiene practices for improving sleep quality? I habitually scroll on my phone late at night and struggle to wake up in the morning.", "difficulty": "simple", "language": "en"},
    {"category": "sleep_hygiene", "prompt": "I have been working night shifts for two years and cannot sleep well during the day. I feel chronically fatigued. How should shift workers manage their sleep schedule? Is melatonin helpful for shift work disorder?", "difficulty": "medium", "language": "en"},
    {"category": "medication", "prompt": "I have been taking eszopiclone for 6 months and want to stop. Can I stop abruptly or do I need to taper? What is the recommended tapering schedule?", "difficulty": "simple", "language": "en"},
    {"category": "medication", "prompt": "Can melatonin be used long-term? What are the side effects? How does it compare to prescription hypnotics in terms of efficacy and safety?", "difficulty": "medium", "language": "en"},
    {"category": "circadian", "prompt": "I naturally fall asleep around 3-4 AM and wake up in the afternoon. I have tried setting alarms but cannot maintain a normal schedule. Could I have delayed sleep-wake phase disorder?", "difficulty": "medium", "language": "en"},
    {"category": "circadian", "prompt": "My friend is totally blind and has very irregular sleep patterns, sometimes sleeping during the day and being awake at night. What is the underlying mechanism and what treatment options are available?", "difficulty": "complex", "language": "en"},
    {"category": "psg", "prompt": "My doctor ordered a polysomnography (PSG). What does the test involve? Do I need to stay overnight in a lab? How should I prepare?", "difficulty": "simple", "language": "en"},
    {"category": "psg", "prompt": "My PSG report shows N3 sleep at 8%, REM at 15%, and an arousal index of 25/hour. Are these values normal? What do they indicate?", "difficulty": "medium", "language": "en"},
]

# ---------------------------------------------------------------------------
# Report generation prompts
# ---------------------------------------------------------------------------

_REPORT_PROMPTS_ZH: list[dict[str, str]] = [
    {"category": "report", "prompt": "请根据以下数据生成一份睡眠监测报告：患者男性，45岁，BMI 28。总睡眠时间380分钟，睡眠效率79%，N1 15%，N2 55%，N3 10%，REM 20%。AHI 32次/小时，最低血氧82%。请给出诊断建议。", "difficulty": "medium", "language": "zh"},
    {"category": "report", "prompt": "请解释以下PSG报告：入睡潜伏期15分钟，REM潜伏期90分钟，N2睡眠中观察到3次阵发性4-6Hz活动（额叶为主），期间伴有频繁体动觉醒。觉醒后心率从65升至95 bpm。", "difficulty": "complex", "language": "zh"},
]

_REPORT_PROMPTS_EN: list[dict[str, str]] = [
    {"category": "report", "prompt": "Generate a sleep study report based on the following data: Male patient, 45 years old, BMI 28. TST 380 min, sleep efficiency 79%, N1 15%, N2 55%, N3 10%, REM 20%. AHI 32/h, nadir SpO2 82%. Provide diagnostic impressions.", "difficulty": "medium", "language": "en"},
    {"category": "report", "prompt": "Interpret this PSG report: sleep latency 15 min, REM latency 90 min, 3 episodes of paroxysmal 4-6 Hz activity (frontal predominance) during N2 sleep with frequent movement arousals. Heart rate increased from 65 to 95 bpm upon arousal.", "difficulty": "complex", "language": "en"},
]

# ---------------------------------------------------------------------------
# QA prompts
# ---------------------------------------------------------------------------

_QA_PROMPTS_ZH: list[dict[str, str]] = [
    {"category": "qa", "prompt": "什么是多导睡眠图（PSG）？它能检测哪些睡眠参数？", "difficulty": "simple", "language": "zh"},
    {"category": "qa", "prompt": "OSA的AHI评分标准是什么？轻度、中度和重度分别对应什么范围？", "difficulty": "simple", "language": "zh"},
    {"category": "qa", "prompt": "什么是CBT-I？它包含哪些核心组成部分？治疗效果如何？", "difficulty": "medium", "language": "zh"},
    {"category": "qa", "prompt": "请比较佐匹克隆、右佐匹克隆和唑吡坦在药代动力学、疗效和副作用方面的差异。", "difficulty": "complex", "language": "zh"},
    {"category": "qa", "prompt": "REM睡眠行为障碍（RBD）的病理生理机制是什么？它和帕金森病、路易体痴呆之间有什么关系？", "difficulty": "complex", "language": "zh"},
]

_QA_PROMPTS_EN: list[dict[str, str]] = [
    {"category": "qa", "prompt": "What is polysomnography (PSG)? What sleep parameters can it measure?", "difficulty": "simple", "language": "en"},
    {"category": "qa", "prompt": "What are the AHI cutoff values for mild, moderate, and severe OSA?", "difficulty": "simple", "language": "en"},
    {"category": "qa", "prompt": "What is CBT-I? What are its core components and what is the evidence for its efficacy?", "difficulty": "medium", "language": "en"},
    {"category": "qa", "prompt": "Compare zopiclone, eszopiclone, and zolpidem in terms of pharmacokinetics, efficacy, and side effect profiles.", "difficulty": "complex", "language": "en"},
    {"category": "qa", "prompt": "What is the pathophysiology of REM sleep behavior disorder (RBD)? What is its relationship with Parkinson disease and Lewy body dementia?", "difficulty": "complex", "language": "en"},
]


class SleepSeedPrompts:
    """Manager for seed prompts used in synthetic data generation.

    Provides categorized prompt templates in Chinese and English for
    sleep medicine conversations, report generation, and Q&A.
    """

    def __init__(self) -> None:
        """Initialize with the built-in prompt database."""
        self._consultation_zh = list(_CONSULTATION_PROMPTS_ZH)
        self._consultation_en = list(_CONSULTATION_PROMPTS_EN)
        self._report_zh = list(_REPORT_PROMPTS_ZH)
        self._report_en = list(_REPORT_PROMPTS_EN)
        self._qa_zh = list(_QA_PROMPTS_ZH)
        self._qa_en = list(_QA_PROMPTS_EN)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_consultation_prompts(self) -> list[dict[str, str]]:
        """Return all consultation-style prompts (Chinese and English)."""
        return list(self._consultation_zh) + list(self._consultation_en)

    def get_report_prompts(self) -> list[dict[str, str]]:
        """Return all sleep-study report generation prompts."""
        return list(self._report_zh) + list(self._report_en)

    def get_qa_prompts(self) -> list[dict[str, str]]:
        """Return all Q&A format prompts."""
        return list(self._qa_zh) + list(self._qa_en)

    def get_diverse_prompts(self, num_prompts: int = 1000) -> list[dict[str, str]]:
        """Generate a diverse set of seed prompts with variations.

        Takes the built-in prompts and applies textual variations (prefixes,
        rephrasing, persona changes) to expand the pool up to *num_prompts*.

        Args:
            num_prompts: Target number of prompts to generate.

        Returns:
            List of prompt dictionaries.
        """
        base_prompts = (
            self._consultation_zh
            + self._consultation_en
            + self._report_zh
            + self._report_en
            + self._qa_zh
            + self._qa_en
        )

        if num_prompts <= len(base_prompts):
            return base_prompts[:num_prompts]

        rng = random.Random(42)
        expanded: list[dict[str, str]] = list(base_prompts)

        zh_prefixes = [
            "医生您好，",
            "请问，",
            "我想咨询一下，",
            "我最近有个问题想请教，",
            "专家您好，我需要一些专业建议。",
        ]
        en_prefixes = [
            "Hello doctor, ",
            "I have a question: ",
            "I would like to ask about ",
            "I need some advice regarding ",
            "Could you help me understand ",
        ]

        zh_personas = [
            "作为一名30岁的上班族，",
            "我今年50岁，",
            "作为一个新手妈妈，",
            "作为一名医务工作者，我想代患者咨询：",
        ]
        en_personas = [
            "As a 30-year-old office worker, ",
            "I am 50 years old. ",
            "As a new mother, ",
            "On behalf of my patient, I would like to ask: ",
        ]

        while len(expanded) < num_prompts:
            base = rng.choice(base_prompts)
            variation = dict(base)
            lang = base["language"]

            variation_type = rng.choice(["prefix", "persona", "rephrase"])

            if variation_type == "prefix":
                if lang == "zh":
                    prefix = rng.choice(zh_prefixes)
                else:
                    prefix = rng.choice(en_prefixes)
                variation["prompt"] = prefix + base["prompt"]

            elif variation_type == "persona":
                if lang == "zh":
                    persona = rng.choice(zh_personas)
                else:
                    persona = rng.choice(en_personas)
                variation["prompt"] = persona + base["prompt"]

            else:
                # Simple rephrase markers (just tag it; actual rephrasing
                # should be done by the generator LLM)
                variation["prompt"] = base["prompt"] + (
                    " 请详细解答。" if lang == "zh" else " Please provide a detailed answer."
                )

            expanded.append(variation)

        return expanded[:num_prompts]
