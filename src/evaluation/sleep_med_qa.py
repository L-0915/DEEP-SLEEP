"""Sleep medicine question-answering benchmark dataset.

Provides built-in multiple-choice questions covering major sleep medicine
topics and an evaluation harness for scoring model predictions against
ground-truth answers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in seed questions
# ---------------------------------------------------------------------------

_BUILTIN_QUESTIONS: list[dict[str, Any]] = [
    # -- Insomnia --
    {
        "question": (
            "According to the ICSD-3 criteria, which of the following is required "
            "for the diagnosis of chronic insomnia disorder?"
        ),
        "options": {
            "A": "Sleep onset latency > 30 minutes for at least 3 nights per week for 3 months",
            "B": "Sleep onset latency or sleep maintenance difficulty occurring at least 3 nights per week for at least 3 months, with associated daytime impairment",
            "C": "Total sleep time < 5 hours per night for at least 1 month",
            "D": "Use of sleep medication more than twice per week for 6 months",
        },
        "answer": "B",
        "category": "insomnia",
        "explanation": (
            "ICSD-3 requires sleep difficulty (initiation or maintenance) at least "
            "3 nights/week for >= 3 months with daytime dysfunction."
        ),
    },
    {
        "question": (
            "Which medication class is recommended as first-line pharmacotherapy "
            "for chronic insomnia according to the AASM 2017 clinical practice guideline?"
        ),
        "options": {
            "A": "Benzodiazepines",
            "B": "Antidepressants (e.g., trazodone)",
            "C": "Non-benzodiazepine receptor agonists (e.g., zolpidem, eszopiclone)",
            "D": "Antihistamines (e.g., diphenhydramine)",
        },
        "answer": "C",
        "category": "insomnia",
        "explanation": (
            "The AASM 2017 guideline recommends non-benzodiazepine receptor agonists "
            "(zolpidem, eszopiclone, zaleplon) and suvorexant as standard treatments "
            "for chronic insomnia, with CBT-I as the first-line overall approach."
        ),
    },
    {
        "question": (
            "What is the recommended first-line treatment for chronic insomnia disorder "
            "according to current clinical guidelines?"
        ),
        "options": {
            "A": "Pharmacotherapy with zolpidem",
            "B": "Cognitive Behavioral Therapy for Insomnia (CBT-I)",
            "C": "Melatonin supplementation",
            "D": "Progressive muscle relaxation alone",
        },
        "answer": "B",
        "category": "insomnia",
        "explanation": (
            "CBT-I is recommended as the first-line treatment for chronic insomnia by "
            "AASM, ACP, and ESS, based on strong evidence of sustained efficacy."
        ),
    },
    # -- Sleep Apnea --
    {
        "question": (
            "In polysomnography, what is the threshold apnea-hypopnea index (AHI) "
            "for the diagnosis of moderate obstructive sleep apnea (OSA) in adults?"
        ),
        "options": {
            "A": "AHI >= 5 events/hour",
            "B": "AHI >= 15 events/hour",
            "C": "AHI >= 30 events/hour",
            "D": "AHI >= 50 events/hour",
        },
        "answer": "B",
        "category": "sleep_apnea",
        "explanation": (
            "OSA severity: mild AHI 5-14, moderate AHI 15-29, severe AHI >= 30 "
            "events/hour per AASM scoring rules."
        ),
    },
    {
        "question": (
            "Which of the following is the most appropriate initial treatment for "
            "a patient with moderate-to-severe OSA who is compliant with CPAP therapy?"
        ),
        "options": {
            "A": "Oral appliance therapy",
            "B": "Weight loss program only",
            "C": "Continuous positive airway pressure (CPAP)",
            "D": "Supplemental oxygen during sleep",
        },
        "answer": "C",
        "category": "sleep_apnea",
        "explanation": (
            "CPAP is the primary treatment for moderate-to-severe OSA. Oral appliances "
            "are considered for mild-to-moderate cases or CPAP-intolerant patients."
        ),
    },
    {
        "question": (
            "What craniofacial measurement is most strongly associated with "
            "obstructive sleep apnea risk?"
        ),
        "options": {
            "A": "Wide inter-pupillary distance",
            "B": "Increased Mallampati score and retrognathia",
            "C": "High nasal bridge",
            "D": "Prominent zygomatic arches",
        },
        "answer": "B",
        "category": "sleep_apnea",
        "explanation": (
            "Mallampati score >= 3, retrognathia, tonsillar hypertrophy, and overjet "
            "are associated with increased OSA risk due to upper airway narrowing."
        ),
    },
    # -- Narcolepsy --
    {
        "question": (
            "Which cerebrospinal fluid biomarker is used to differentiate "
            "type 1 narcolepsy from type 2 narcolepsy?"
        ),
        "options": {
            "A": "Low beta-amyloid 42",
            "B": "Elevated tau protein",
            "C": "Undetectable or very low hypocretin-1 (orexin-A) level",
            "D": "Elevated neurofilament light chain",
        },
        "answer": "C",
        "category": "narcolepsy",
        "explanation": (
            "Type 1 narcolepsy is characterized by hypocretin deficiency "
            "(CSF hypocretin-1 <= 110 pg/mL or undetectable), while type 2 "
            "narcolepsy has normal hypocretin levels."
        ),
    },
    {
        "question": (
            "Which HLA allele is most strongly associated with type 1 narcolepsy?"
        ),
        "options": {
            "A": "HLA-B27",
            "B": "HLA-DR4",
            "C": "HLA-DQB1*06:02",
            "D": "HLA-A*02:01",
        },
        "answer": "C",
        "category": "narcolepsy",
        "explanation": (
            "Narcolepsy type 1 has a very strong association with HLA-DQB1*06:02, "
            "present in > 90% of patients versus approximately 25% of the general population."
        ),
    },
    {
        "question": (
            "What is the gold-standard diagnostic test for narcolepsy?"
        ),
        "options": {
            "A": "Overnight polysomnography alone",
            "B": "Multiple sleep latency test (MSLT) following overnight polysomnography",
            "C": "Actigraphy over 2 weeks",
            "D": "Maintenance of wakefulness test (MWT)",
        },
        "answer": "B",
        "category": "narcolepsy",
        "explanation": (
            "The diagnostic standard is an overnight PSG followed by an MSLT the next day, "
            "looking for mean sleep latency <= 8 min and >= 2 SOREMPs."
        ),
    },
    # -- RLS --
    {
        "question": (
            "According to the International Restless Legs Syndrome Study Group (IRLSSG) "
            "updated diagnostic criteria (2014), which of the following is NOT required "
            "for an RLS diagnosis?"
        ),
        "options": {
            "A": "Urge to move the legs, usually accompanied by uncomfortable sensations",
            "B": "Symptoms begin or worsen during periods of rest or inactivity",
            "C": "Positive response to dopamine agonist therapy",
            "D": "Symptoms are worse in the evening or night than during the day",
        },
        "answer": "C",
        "category": "rls",
        "explanation": (
            "The five essential criteria are: urge to move, worsening at rest, "
            "relief with movement, circadian worsening, and that the symptoms are "
            "not solely explained by another condition. Treatment response is not "
            "a diagnostic criterion."
        ),
    },
    {
        "question": (
            "Which of the following laboratory tests is most important to evaluate "
            "in a patient presenting with symptoms of restless legs syndrome?"
        ),
        "options": {
            "A": "Thyroid stimulating hormone (TSH)",
            "B": "Serum ferritin",
            "C": "Vitamin B12 level",
            "D": "Morning cortisol",
        },
        "answer": "B",
        "category": "rls",
        "explanation": (
            "Iron deficiency (ferritin < 75 ng/mL) is a major modifiable risk factor "
            "for RLS, and iron supplementation is recommended when ferritin is low."
        ),
    },
    # -- Circadian Rhythm Disorders --
    {
        "question": (
            "What is the mechanism of action of tasimelteon in the treatment "
            "of Non-24-Hour Sleep-Wake Rhythm Disorder?"
        ),
        "options": {
            "A": "Melatonin receptor agonist (MT1 and MT2)",
            "B": "GABA-A receptor positive allosteric modulator",
            "C": "Histamine H1 receptor antagonist",
            "D": "Orexin receptor antagonist",
        },
        "answer": "A",
        "category": "circadian",
        "explanation": (
            "Tasimelteon is a dual melatonin receptor agonist (MT1/MT2) and is the "
            "only FDA-approved treatment for Non-24-Hour Sleep-Wake Rhythm Disorder."
        ),
    },
    {
        "question": (
            "Which circadian rhythm sleep-wake disorder is most commonly seen "
            "in adolescents?"
        ),
        "options": {
            "A": "Advanced sleep-wake phase disorder",
            "B": "Delayed sleep-wake phase disorder (DSWPD)",
            "C": "Irregular sleep-wake rhythm disorder",
            "D": "Shift work disorder",
        },
        "answer": "B",
        "category": "circadian",
        "explanation": (
            "DSWPD is the most common CRSD in adolescents, with prevalence estimates "
            "of 7-16%, driven by the biological delay in circadian phase during puberty."
        ),
    },
    # -- PSG Interpretation --
    {
        "question": (
            "In standard PSG scoring (AASM), what is the minimum duration of an "
            "apnea event to be scored?"
        ),
        "options": {
            "A": "3 seconds",
            "B": "5 seconds",
            "C": "10 seconds",
            "D": "15 seconds",
        },
        "answer": "C",
        "category": "psg",
        "explanation": (
            "Per AASM scoring rules, apnea is defined as a drop in airflow by >= 90% "
            "of pre-event baseline for at least 10 seconds."
        ),
    },
    {
        "question": (
            "Which EEG rhythm is most prominently seen during N3 (slow wave) sleep?"
        ),
        "options": {
            "A": "Alpha rhythm (8-13 Hz)",
            "B": "Sleep spindles (11-16 Hz)",
            "C": "Delta waves (0.5-2 Hz, amplitude > 75 uV)",
            "D": "Theta rhythm (4-8 Hz)",
        },
        "answer": "C",
        "category": "psg",
        "explanation": (
            "N3 sleep (slow wave sleep) is characterized by delta waves with frequency "
            "0.5-2 Hz and amplitude > 75 uV, occupying >= 20% of the epoch on frontal leads."
        ),
    },
    {
        "question": (
            "On polysomnography, REM sleep is characterized by all of the following "
            "EXCEPT:"
        ),
        "options": {
            "A": "Low-amplitude mixed-frequency EEG",
            "B": "Rapid eye movements",
            "C": "Sustained muscle atonia on EMG",
            "D": "K-complexes on frontal EEG",
        },
        "answer": "D",
        "category": "psg",
        "explanation": (
            "K-complexes are hallmarks of N2 sleep, not REM sleep. REM sleep features "
            "low-amplitude mixed-frequency EEG, sawtooth waves, REMs, and muscle atonia."
        ),
    },
    # -- Sleep Pharmacology --
    {
        "question": (
            "Which sleep medication has the longest half-life among the following "
            "non-benzodiazepine hypnotics?"
        ),
        "options": {
            "A": "Zolpidem (~2-3 hours)",
            "B": "Zaleplon (~1 hour)",
            "C": "Eszopiclone (~6 hours)",
            "D": "Triazolam (~2 hours)",
        },
        "answer": "C",
        "category": "pharmacology",
        "explanation": (
            "Eszopiclone has the longest half-life (~6 hours) among the Z-drugs, "
            "making it suitable for both sleep onset and maintenance insomnia."
        ),
    },
    {
        "question": (
            "What is the mechanism of action of suvorexant (Belsomra)?"
        ),
        "options": {
            "A": "Positive allosteric modulation of GABA-A receptors",
            "B": "Dual orexin receptor antagonist (DORA)",
            "C": "Melatonin receptor agonism",
            "D": "Histamine H1 receptor antagonism",
        },
        "answer": "B",
        "category": "pharmacology",
        "explanation": (
            "Suvorexant is a DORA that blocks orexin-A and orexin-B signaling at "
            "OX1R and OX2R receptors, promoting sleep by reducing wake-promoting signals."
        ),
    },
    {
        "question": (
            "Which of the following is a significant risk associated with "
            "sodium oxybate (Xyrem/Xywav) therapy?"
        ),
        "options": {
            "A": "Hepatotoxicity",
            "B": "QT prolongation",
            "C": "Respiratory depression and central nervous system depression",
            "D": "Serotonin syndrome",
        },
        "answer": "C",
        "category": "pharmacology",
        "explanation": (
            "Sodium oxybate (GHB) carries risks of respiratory depression, CNS depression, "
            "and is available only through a restricted REMS program due to abuse potential."
        ),
    },
    # -- Pediatric Sleep --
    {
        "question": (
            "What is the most common cause of obstructive sleep apnea in children?"
        ),
        "options": {
            "A": "Obesity",
            "B": "Adenotonsillar hypertrophy",
            "C": "Craniofacial abnormalities",
            "D": "Neuromuscular disorders",
        },
        "answer": "B",
        "category": "pediatric",
        "explanation": (
            "Adenotonsillar hypertrophy is the most common cause of pediatric OSA, "
            "accounting for approximately 75% of cases. Adenotonsillectomy is often curative."
        ),
    },
    # -- Sleep Hygiene --
    {
        "question": (
            "Which of the following is NOT a recommended component of sleep hygiene?"
        ),
        "options": {
            "A": "Maintaining a consistent sleep schedule",
            "B": "Using bright blue-light-emitting devices in bed",
            "C": "Keeping the bedroom cool and dark",
            "D": "Avoiding caffeine within 6 hours of bedtime",
        },
        "answer": "B",
        "category": "sleep_hygiene",
        "explanation": (
            "Blue light from screens suppresses melatonin production and delays sleep onset. "
            "Screen use before bed is discouraged as part of proper sleep hygiene."
        ),
    },
]


class SleepMedQA:
    """Sleep medicine multiple-choice question benchmark.

    Loads built-in seed questions covering the major sleep medicine domains
    and provides an evaluation method that runs a model against the dataset.

    Questions can also be loaded from a JSONL file where each line has the schema::

        {
            "question": "...",
            "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
            "answer": "A",
            "explanation": "..."
        }

    Attributes:
        questions: List of question dictionaries.
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        """Initialize the SleepMedQA benchmark.

        Args:
            data_path: Path to a JSONL file with additional questions.  If *None*,
                only the built-in questions are used.
        """
        self._questions: list[dict[str, Any]] = list(_BUILTIN_QUESTIONS)
        self._data_path = data_path

        if data_path is not None:
            self._load_external(data_path)

        logger.info(
            "SleepMedQA initialized with %d questions", len(self._questions)
        )

    # -- public properties ------------------------------------------------

    @property
    def questions(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the question list."""
        return list(self._questions)

    # -- public methods ---------------------------------------------------

    def generate_dataset(self, num_questions: int = 500) -> list[dict[str, Any]]:
        """Generate a question dataset of the requested size.

        Because high-quality medical MCQs require expert authoring, this method
        returns the current question pool and logs a placeholder message about
        future expansion.

        Args:
            num_questions: Target number of questions.  If the current pool is
                smaller, a warning is logged.

        Returns:
            The current list of questions (may be shorter than *num_questions*).
        """
        if len(self._questions) < num_questions:
            logger.warning(
                "Requested %d questions but only %d are currently available. "
                "Additional questions should be authored by sleep medicine experts "
                "and added to the dataset file.",
                num_questions,
                len(self._questions),
            )

        return list(self._questions)

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
    ) -> dict[str, Any]:
        """Evaluate a model on the sleep medicine QA benchmark.

        Constructs a multiple-choice prompt for each question, feeds it to the
        model, and compares the predicted answer letter against the ground truth.

        Args:
            model: A HuggingFace-compatible language model.
            tokenizer: The corresponding tokenizer.

        Returns:
            A dictionary with keys:
                - ``accuracy``: Overall accuracy (0.0 -- 1.0).
                - ``total``: Total number of questions.
                - ``correct``: Number of correctly answered questions.
                - ``results``: Per-question result dicts.
        """
        correct = 0
        total = len(self._questions)
        results: list[dict[str, Any]] = []

        model.eval()

        for idx, item in enumerate(self._questions):
            prompt = self._format_prompt(item)
            predicted = self._predict_answer(model, tokenizer, prompt)
            is_correct = predicted == item["answer"]

            if is_correct:
                correct += 1

            results.append({
                "question_id": idx,
                "question": item["question"],
                "predicted": predicted,
                "ground_truth": item["answer"],
                "correct": is_correct,
                "category": item.get("category", "unknown"),
                "explanation": item.get("explanation", ""),
            })

            if (idx + 1) % 10 == 0:
                logger.info(
                    "Progress: %d/%d questions evaluated, "
                    "running accuracy: %.2f%%",
                    idx + 1,
                    total,
                    100.0 * correct / (idx + 1),
                )

        accuracy = correct / total if total > 0 else 0.0

        logger.info(
            "SleepMedQA evaluation complete: %d/%d correct (%.1f%%)",
            correct,
            total,
            100.0 * accuracy,
        )

        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "results": results,
        }

    # -- private helpers --------------------------------------------------

    def _load_external(self, path: str) -> None:
        """Load additional questions from a JSONL file."""
        file_path = Path(path)
        if not file_path.exists():
            logger.warning("External question file not found: %s", path)
            return

        loaded = 0
        with open(file_path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    self._validate_question(item)
                    self._questions.append(item)
                    loaded += 1
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        "Skipping invalid question at line %d in %s: %s",
                        line_no,
                        path,
                        exc,
                    )

        logger.info("Loaded %d questions from %s", loaded, path)

    @staticmethod
    def _validate_question(item: dict[str, Any]) -> None:
        """Validate that a question dict has the required fields."""
        required_fields = ("question", "options", "answer")
        for field in required_fields:
            if field not in item:
                raise ValueError(
                    f"Question missing required field '{field}': {item}"
                )

        if not isinstance(item["options"], dict):
            raise ValueError("'options' must be a dict mapping letters to text")

        if item["answer"] not in item["options"]:
            raise ValueError(
                f"Answer '{item['answer']}' not found in options keys: "
                f"{list(item['options'].keys())}"
            )

    @staticmethod
    def _format_prompt(item: dict[str, Any]) -> str:
        """Format a question into a zero-shot multiple-choice prompt."""
        option_lines = "\n".join(
            f"  {key}. {text}"
            for key, text in item["options"].items()
        )
        return (
            f"Question: {item['question']}\n"
            f"Options:\n{option_lines}\n"
            f"Answer: "
        )

    @staticmethod
    def _predict_answer(model: Any, tokenizer: Any, prompt: str) -> str:
        """Run model inference and extract the predicted answer letter.

        Falls back to the first token matching A/B/C/D.
        """
        import torch

        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Try to match answer letter
        for char in generated:
            if char.upper() in ("A", "B", "C", "D"):
                return char.upper()

        logger.warning("Could not parse answer from generated text: '%s'", generated)
        return "UNKNOWN"
