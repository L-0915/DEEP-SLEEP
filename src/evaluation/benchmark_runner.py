"""Unified benchmark runner for the DeepSleep sleep-medicine LLM.

Orchestrates evaluation across multiple benchmarks -- MMLU, C-Eval, CMMLU,
and the custom SleepMedQA -- and aggregates results into a single JSON report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch

from .inference_bench import InferenceBenchmark
from .sleep_med_qa import SleepMedQA

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run standardised benchmarks against a HuggingFace model.

    Loads the model and tokenizer once, then evaluates on MMLU, C-Eval, CMMLU,
    and the sleep-medicine-specific SleepMedQA benchmark.

    Args:
        model_path: Path to the HuggingFace model directory.
        tokenizer_path: Path to the tokenizer directory.
        device: Device string (``"auto"``, ``"cuda"``, ``"cpu"``).
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "auto",
    ) -> None:
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path
        self._device = device
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

        self._load_model()

    # -- lazy model loading -----------------------------------------------

    def _load_model(self) -> None:
        """Load model and tokenizer from disk."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model from %s ...", self._model_path)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        device_map = "auto" if self._device == "auto" else self._device
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
        self._model.eval()

        logger.info("Model loaded on %s", device_map)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_mmlu(
        self,
        subjects: Optional[list[str]] = None,
        num_few_shot: int = 5,
    ) -> dict[str, Any]:
        """Run the MMLU (Massive Multitask Language Understanding) benchmark.

        Evaluates the model on the specified subjects from the MMLU dataset.
        Default subjects focus on clinical knowledge relevant to sleep medicine.

        Args:
            subjects: List of MMLU subject names.  Defaults to
                ``["clinical_knowledge"]``.
            num_few_shot: Number of few-shot examples per subject.

        Returns:
            Dict with ``accuracy``, ``total``, ``correct``, and ``results``.
        """
        subjects = subjects or ["clinical_knowledge"]

        logger.info("Running MMLU on subjects: %s (few_shot=%d)", subjects, num_few_shot)

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for MMLU evaluation.  "
                "Install it with: pip install datasets"
            ) from exc

        correct = 0
        total = 0
        results: list[dict[str, Any]] = []

        for subject in subjects:
            try:
                dataset = load_dataset(
                    "cais/mmlu", subject, split="test", trust_remote_code=True
                )
            except Exception as exc:
                logger.error("Failed to load MMLU subject '%s': %s", subject, exc)
                continue

            subject_correct = 0
            subject_total = len(dataset)

            for idx, example in enumerate(dataset):
                prompt = self._format_mmlu_prompt(example, num_few_shot)
                predicted = self._predict_choice(prompt)

                is_correct = predicted == example["answer"]
                if is_correct:
                    subject_correct += 1
                    correct += 1

                results.append({
                    "benchmark": "mmlu",
                    "subject": subject,
                    "question_id": idx,
                    "predicted": predicted,
                    "ground_truth": example["answer"],
                    "correct": is_correct,
                })

                total += 1

            logger.info(
                "MMLU '%s': %d/%d (%.1f%%)",
                subject,
                subject_correct,
                subject_total,
                100.0 * subject_correct / subject_total if subject_total > 0 else 0.0,
            )

        accuracy = correct / total if total > 0 else 0.0
        logger.info("MMLU overall: %d/%d (%.1f%%)", correct, total, 100.0 * accuracy)

        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "results": results,
        }

    def run_c_eval(
        self,
        data_path: str,
    ) -> dict[str, Any]:
        """Run the C-Eval benchmark (Chinese multi-subject evaluation).

        Loads the C-Eval dataset from the given path and evaluates the model
        on Chinese-language multiple-choice questions.

        Args:
            data_path: Path to the C-Eval dataset directory or file
                (HuggingFace datasets format).

        Returns:
            Dict with ``accuracy``, ``total``, ``correct``, and ``results``.
        """
        logger.info("Running C-Eval from %s ...", data_path)

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for C-Eval evaluation.  "
                "Install it with: pip install datasets"
            ) from exc

        try:
            dataset = load_dataset(
                data_path, split="test", trust_remote_code=True
            )
        except Exception as exc:
            logger.error("Failed to load C-Eval dataset: %s", exc)
            return {"accuracy": 0.0, "total": 0, "correct": 0, "results": [], "error": str(exc)}

        correct = 0
        total = len(dataset)
        results: list[dict[str, Any]] = []

        for idx, example in enumerate(dataset):
            prompt = self._format_ceval_prompt(example)
            predicted = self._predict_choice(prompt)

            answer = example.get("answer", "")
            is_correct = predicted == str(answer).strip().upper()

            if is_correct:
                correct += 1

            results.append({
                "benchmark": "ceval",
                "question_id": idx,
                "predicted": predicted,
                "ground_truth": str(answer),
                "correct": is_correct,
            })

        accuracy = correct / total if total > 0 else 0.0
        logger.info("C-Eval: %d/%d (%.1f%%)", correct, total, 100.0 * accuracy)

        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "results": results,
        }

    def run_cmmlu(
        self,
        subjects: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Run the CMMLU (Chinese Massive Multitask Language Understanding) benchmark.

        Args:
            subjects: List of CMMLU subject names.  Defaults to
                ``["clinical_medicine"]``.

        Returns:
            Dict with ``accuracy``, ``total``, ``correct``, and ``results``.
        """
        subjects = subjects or ["clinical_medicine"]

        logger.info("Running CMMLU on subjects: %s", subjects)

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for CMMLU evaluation.  "
                "Install it with: pip install datasets"
            ) from exc

        correct = 0
        total = 0
        results: list[dict[str, Any]] = []

        for subject in subjects:
            try:
                dataset = load_dataset(
                    "haonan-li/cmmlu", subject, split="test", trust_remote_code=True
                )
            except Exception as exc:
                logger.error("Failed to load CMMLU subject '%s': %s", subject, exc)
                continue

            subject_correct = 0
            subject_total = len(dataset)

            for idx, example in enumerate(dataset):
                prompt = self._format_cmmlu_prompt(example)
                predicted = self._predict_choice(prompt)

                answer = example.get("answer", "")
                is_correct = predicted == str(answer).strip().upper()

                if is_correct:
                    subject_correct += 1
                    correct += 1

                results.append({
                    "benchmark": "cmmlu",
                    "subject": subject,
                    "question_id": idx,
                    "predicted": predicted,
                    "ground_truth": str(answer),
                    "correct": is_correct,
                })

                total += 1

            logger.info(
                "CMMLU '%s': %d/%d (%.1f%%)",
                subject,
                subject_correct,
                subject_total,
                100.0 * subject_correct / subject_total if subject_total > 0 else 0.0,
            )

        accuracy = correct / total if total > 0 else 0.0
        logger.info("CMMLU overall: %d/%d (%.1f%%)", correct, total, 100.0 * accuracy)

        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "results": results,
        }

    def run_sleep_med_qa(
        self,
        data_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run the custom sleep medicine QA benchmark.

        Args:
            data_path: Path to a JSONL file with additional questions.
                When *None*, only built-in questions are used.

        Returns:
            Dict with ``accuracy``, ``total``, ``correct``, and ``results``.
        """
        logger.info("Running SleepMedQA benchmark ...")

        qa = SleepMedQA(data_path=data_path)
        result = qa.evaluate(self._model, self._tokenizer)

        return result

    def run_all_benchmarks(
        self,
        output_dir: str = "output/eval_results",
    ) -> dict[str, Any]:
        """Run all benchmarks and save aggregated results.

        Executes MMLU, CMMLU, SleepMedQA, and inference benchmarks,
        then writes a combined JSON report.

        Args:
            output_dir: Directory where the results JSON will be saved.

        Returns:
            Combined results dictionary keyed by benchmark name.
        """
        logger.info("=" * 60)
        logger.info("Starting full benchmark suite")
        logger.info("=" * 60)

        all_results: dict[str, Any] = {}

        # MMLU
        try:
            all_results["mmlu"] = self.run_mmlu()
        except Exception as exc:
            logger.error("MMLU benchmark failed: %s", exc)
            all_results["mmlu"] = {"error": str(exc)}

        # CMMLU
        try:
            all_results["cmmlu"] = self.run_cmmlu()
        except Exception as exc:
            logger.error("CMMLU benchmark failed: %s", exc)
            all_results["cmmlu"] = {"error": str(exc)}

        # SleepMedQA
        try:
            all_results["sleep_med_qa"] = self.run_sleep_med_qa()
        except Exception as exc:
            logger.error("SleepMedQA benchmark failed: %s", exc)
            all_results["sleep_med_qa"] = {"error": str(exc)}

        # Inference performance
        try:
            inference_bench = InferenceBenchmark(
                self._model_path, self._tokenizer_path
            )
            all_results["inference"] = inference_bench.run_all()
        except Exception as exc:
            logger.error("Inference benchmark failed: %s", exc)
            all_results["inference"] = {"error": str(exc)}

        # Save combined results
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "benchmark_results.json"

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, ensure_ascii=False, indent=2)

        logger.info("All benchmark results saved to %s", output_path)
        return all_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_choice(self, prompt: str) -> str:
        """Generate a single token and map it to an answer choice."""
        import torch

        assert self._model is not None
        assert self._tokenizer is not None

        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Try to match A/B/C/D
        for char in generated:
            if char.upper() in ("A", "B", "C", "D"):
                return char.upper()

        logger.debug("Could not parse choice from '%s'", generated)
        return "UNKNOWN"

    @staticmethod
    def _format_mmlu_prompt(example: dict[str, Any], num_few_shot: int) -> str:
        """Format an MMLU example as a few-shot prompt."""
        choices = example.get("choices", [])
        labels = ["A", "B", "C", "D"]
        choice_text = "\n".join(
            f"  {labels[i]}. {choices[i]}" for i in range(len(choices))
        )

        # MMLU examples already include few-shot in the dataset
        prompt = (
            f"Question: {example['question']}\n"
            f"Options:\n{choice_text}\n"
            f"Answer: "
        )
        return prompt

    @staticmethod
    def _format_ceval_prompt(example: dict[str, Any]) -> str:
        """Format a C-Eval example as a prompt."""
        choices = example.get("options", {})
        if isinstance(choices, dict):
            choice_text = "\n".join(f"  {k}. {v}" for k, v in choices.items())
        elif isinstance(choices, list):
            labels = ["A", "B", "C", "D"]
            choice_text = "\n".join(
                f"  {labels[i]}. {choices[i]}" for i in range(len(choices))
            )
        else:
            choice_text = ""

        prompt = (
            f"Question: {example.get('question', '')}\n"
            f"Options:\n{choice_text}\n"
            f"Answer: "
        )
        return prompt

    @staticmethod
    def _format_cmmlu_prompt(example: dict[str, Any]) -> str:
        """Format a CMMLU example as a prompt."""
        choices = example.get("options", [])
        if isinstance(choices, list):
            labels = ["A", "B", "C", "D"]
            choice_text = "\n".join(
                f"  {labels[i]}. {choices[i]}" for i in range(len(choices))
            )
        elif isinstance(choices, dict):
            choice_text = "\n".join(f"  {k}. {v}" for k, v in choices.items())
        else:
            choice_text = ""

        prompt = (
            f"Question: {example.get('question', '')}\n"
            f"Options:\n{choice_text}\n"
            f"Answer: "
        )
        return prompt
