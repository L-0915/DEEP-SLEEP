"""Inference performance benchmarking utilities.

Measures time-to-first-token (TTFT), throughput (tokens/second), and GPU
memory consumption under various batch sizes and sequence lengths.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# Default prompt used for benchmarking (detokenizes to a reasonable length).
_DEFAULT_PROMPT = (
    "Please explain the difference between obstructive sleep apnea and "
    "central sleep apnea, including their pathophysiology, diagnostic criteria, "
    "and treatment approaches. Detail the role of polysomnography in "
    "differentiating these conditions."
)


class InferenceBenchmark:
    """Measure inference performance metrics for a HuggingFace model.

    Args:
        model_path: Path to the model directory.
        tokenizer_path: Path to the tokenizer directory.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
    ) -> None:
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    def _load(self) -> tuple[Any, Any]:
        """Load model and tokenizer lazily."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model from %s for benchmarking ...", self._model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        return self._model, self._tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def measure_ttft(
        self,
        prompt: Optional[str] = None,
        num_runs: int = 10,
    ) -> dict[str, Any]:
        """Measure time-to-first-token (TTFT).

        Runs the prompt through the model multiple times and records the
        wall-clock time between submitting the prompt and receiving the
        very first generated token.

        Args:
            prompt: Input text.  Defaults to ``_DEFAULT_PROMPT``.
            num_runs: Number of repeated measurements.

        Returns:
            Dict with ``ttft_ms`` (mean), ``ttft_std_ms``, ``ttft_min_ms``,
            ``ttft_max_ms``, and ``num_runs``.
        """
        prompt = prompt or _DEFAULT_PROMPT
        model, tokenizer = self._load()
        device = next(model.parameters()).device

        # Warm-up run
        self._single_forward(model, tokenizer, prompt, device)

        timings_ms: list[float] = []
        for run_idx in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            self._single_forward(model, tokenizer, prompt, device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings_ms.append(elapsed_ms)

            if (run_idx + 1) % 5 == 0:
                logger.info("TTFT run %d/%d", run_idx + 1, num_runs)

        return self._summarize_timings(timings_ms, metric="ttft_ms", num_runs=num_runs)

    def measure_throughput(
        self,
        prompt: Optional[str] = None,
        max_tokens: int = 512,
        batch_sizes: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Measure tokens-per-second throughput at different batch sizes.

        Args:
            prompt: Input text.  Repeated across batch dimension.
            max_tokens: Maximum number of tokens to generate per sequence.
            batch_sizes: List of batch sizes to benchmark.

        Returns:
            Dict mapping each batch size to throughput stats.
        """
        prompt = prompt or _DEFAULT_PROMPT
        batch_sizes = batch_sizes or [1, 4, 8, 16]
        model, tokenizer = self._load()
        device = next(model.parameters()).device

        results: dict[str, Any] = {}
        for bs in batch_sizes:
            logger.info("Measuring throughput: batch_size=%d, max_tokens=%d", bs, max_tokens)
            try:
                tokens_per_sec = self._benchmark_batch(
                    model, tokenizer, prompt, device, batch_size=bs, max_tokens=max_tokens
                )
                results[str(bs)] = {
                    "tokens_per_second": tokens_per_sec,
                    "batch_size": bs,
                    "max_tokens": max_tokens,
                }
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM at batch_size=%d, skipping.", bs)
                results[str(bs)] = {
                    "tokens_per_second": 0.0,
                    "batch_size": bs,
                    "max_tokens": max_tokens,
                    "error": "out_of_memory",
                }

        return {"throughput_results": results}

    def measure_memory(
        self,
        batch_size: int = 1,
        seq_length: int = 2048,
    ) -> dict[str, Any]:
        """Measure peak GPU memory consumption.

        Creates a dummy input of the specified batch size and sequence length,
        runs a single forward pass, and reports peak allocated GPU memory.

        Args:
            batch_size: Number of sequences in the batch.
            seq_length: Token count per sequence.

        Returns:
            Dict with ``peak_memory_mb``, ``allocated_memory_mb``,
            ``reserved_memory_mb``, ``batch_size``, and ``seq_length``.
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available; memory benchmark requires a GPU.")
            return {
                "peak_memory_mb": 0,
                "allocated_memory_mb": 0,
                "reserved_memory_mb": 0,
                "batch_size": batch_size,
                "seq_length": seq_length,
                "error": "no_cuda",
            }

        model, tokenizer = self._load()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        device = next(model.parameters()).device
        dummy_input_ids = torch.randint(
            0, tokenizer.vocab_size, (batch_size, seq_length), device=device
        )

        attention_mask = torch.ones_like(dummy_input_ids)

        with torch.no_grad():
            _ = model(dummy_input_ids, attention_mask=attention_mask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)

        logger.info(
            "Memory benchmark: peak=%.1f MB, allocated=%.1f MB, reserved=%.1f MB "
            "(batch=%d, seq=%d)",
            peak_mb, allocated_mb, reserved_mb, batch_size, seq_length,
        )

        return {
            "peak_memory_mb": round(peak_mb, 2),
            "allocated_memory_mb": round(allocated_mb, 2),
            "reserved_memory_mb": round(reserved_mb, 2),
            "batch_size": batch_size,
            "seq_length": seq_length,
        }

    def run_all(
        self,
        output_path: str = "output/eval_results/inference_bench.json",
    ) -> dict[str, Any]:
        """Run the full inference benchmark suite and save results.

        Args:
            output_path: Path to save the JSON report.

        Returns:
            Combined results dictionary.
        """
        logger.info("Starting full inference benchmark suite ...")
        all_results: dict[str, Any] = {}

        all_results["ttft"] = self.measure_ttft()
        all_results["throughput"] = self.measure_throughput()
        all_results["memory"] = self.measure_memory()

        from pathlib import Path

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, ensure_ascii=False)

        logger.info("Inference benchmark results saved to %s", output_path)
        return all_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _single_forward(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        device: torch.device,
        max_new_tokens: int = 1,
    ) -> None:
        """Run a forward pass that generates exactly one token."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    def _benchmark_batch(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        device: torch.device,
        batch_size: int,
        max_tokens: int,
        warmup: int = 2,
    ) -> float:
        """Measure tokens/second for a specific batch size."""
        # Tokenize once and repeat
        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )["input_ids"]

        batch_ids = input_ids.repeat(batch_size, 1).to(device)
        attention_mask = torch.ones_like(batch_ids)

        # Warm-up
        for _ in range(warmup):
            with torch.no_grad():
                model.generate(
                    batch_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

        # Timed run
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                batch_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        total_generated = (outputs.shape[1] - batch_ids.shape[1]) * batch_size
        tokens_per_sec = total_generated / elapsed if elapsed > 0 else 0.0

        logger.info(
            "  batch_size=%d -> %.1f tokens/s (%d tokens in %.2f s)",
            batch_size,
            tokens_per_sec,
            total_generated,
            elapsed,
        )
        return round(tokens_per_sec, 2)

    @staticmethod
    def _summarize_timings(
        timings_ms: list[float],
        metric: str,
        num_runs: int,
    ) -> dict[str, Any]:
        """Compute mean, std, min, max for a list of timing measurements."""
        import statistics

        mean = statistics.mean(timings_ms)
        std = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0

        return {
            metric: round(mean, 2),
            f"{metric}_std": round(std, 2),
            f"{metric}_min": round(min(timings_ms), 2),
            f"{metric}_max": round(max(timings_ms), 2),
            "num_runs": num_runs,
        }
