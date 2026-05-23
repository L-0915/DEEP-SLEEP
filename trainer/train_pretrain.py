"""DeepSleep Pretraining with HuggingFace Trainer.

Industrial-grade pretraining script that streams CCI4.0-HQ (bilingual)
directly from HuggingFace, no local data download needed.

=== Industrial Features ===

Optimizer & Schedule:
  - AdamW (PyTorch), betas=(0.9, 0.95), weight_decay=0.1
    - betas=(0.9, 0.95) follows LLaMA/GPT practice (not default 0.999)
    - HF Trainer auto-excludes bias & LayerNorm from weight decay
  - Cosine LR schedule with linear warmup (first ~2% of steps)
  - Gradient clipping at max_norm=1.0
  - BF16 mixed precision

Tokens Budget (default config):
  - effective_batch=128 * seq_len=2048 * max_steps=50000 = ~13.1B tokens
  - Chinchilla optimal for ~200M params: ~4B tokens
  - MoE 64.5M active params: 13.1B / 64.5M ~ 203 tokens/param (reasonable)
  - Streaming data has no fixed "epoch"; tokens budget is the control knob

Checkpoint & Resume:
  - Periodic checkpoint saving every `save_steps` steps (HF Trainer format)
  - Only keeps `save_total_limit` most recent checkpoints to save disk
  - Full resume: restores model weights, optimizer, lr scheduler, RNG state,
    global step — pass --resume_from_checkpoint <path> to continue training
  - Final model saved in both HF format and .pth (for SFT/DPO compatibility)

Training Logging:
  - Console: loss, lr, tokens/sec via Python logging every `logging_steps` steps
  - TensorBoard: train_loss, eval_loss, learning_rate, tokens_seen (default)
  - WandB: set --report_to wandb --wandb_project <name>
  - Eval perplexity computed and logged after each evaluation
  - Periodic sample text generation for visual quality inspection

Training Curves & Report:
  - TrainingReportCallback auto-collects all metrics during training
  - On completion: saves training_curves.png (loss + lr + perplexity)
  - On completion: saves report.json (hyperparams, best/final metrics, timing)
  - On completion: saves train_log.jsonl + eval_log.jsonl (raw per-step data)

Data & Performance:
  - Sequence packing (default): zero padding, full Flash Attention throughput
  - Streaming from CCI4.0-HQ: no local download needed
  - MoE-aware loss: CE + router load balance auxiliary loss
  - BF16 mixed precision on single GPU

=== Usage ===
    # Full training (from YAML config)
    bash scripts/run/run_pretrain.sh

    # Resume from checkpoint
    RESUME=out/pretrain/checkpoint-10000 bash scripts/run/run_pretrain.sh

    # Quick smoke test
    python trainer/train_pretrain.py --tokenizer_path checkpoints/tokenizer \
        --max_steps 10 --eval_steps 5 --save_steps 5

    # View training curves during training
    tensorboard --logdir out/pretrain/runs
    tensorboard --logdir out/pretrain/runs --port 6006
    断点重续

    RESUME=out/pretrain/checkpoint-20000 bash scripts/run/run_pretrain.sh
"""

import os
import sys
import math
import json
import time

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoTokenizer,
)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
from dataset.streaming_dataset import CCI4PretrainDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeepSleep-Pretrain")


# =============================================================================
# MoE-aware Trainer
# =============================================================================


class MoETrainer(Trainer):
    """Custom Trainer that combines CE loss + MoE auxiliary loss for backprop.

    DeepSleepForCausalLM returns MoeCausalLMOutputWithPast with separate
    `loss` (cross-entropy) and `aux_loss` (router load balance). The default
    Trainer only uses `loss` for backprop, ignoring `aux_loss`. This subclass
    combines them so gradients flow through both.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, labels=labels)

        loss = outputs.loss
        if outputs.aux_loss is not None and outputs.aux_loss.requires_grad:
            loss = loss + outputs.aux_loss

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# Callbacks
# =============================================================================

DEFAULT_GENERATE_PROMPTS = [
    "睡眠是",
    "研究表明，",
    "Sleep is",
    "The most common sleep",
]


class SampleGenerationCallback(TrainerCallback):
    """Generate sample text after each evaluation for visual inspection."""

    def __init__(self, tokenizer, prompts=None, max_new_tokens=100):
        self.tokenizer = tokenizer
        self.prompts = prompts or DEFAULT_GENERATE_PROMPTS
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return

        device = next(model.parameters()).device
        model.eval()
        logger.info("=" * 60)
        logger.info("Sample Generation (step %d)", state.global_step)
        logger.info("=" * 60)

        for prompt in self.prompts:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                logger.info("  [%s] -> %s", prompt, text)
            except Exception as e:
                logger.warning("  Generation failed for '%s': %s", prompt, e)

        model.train()


class PerplexityCallback(TrainerCallback):
    """Log eval perplexity alongside eval loss."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            try:
                ppl = math.exp(metrics["eval_loss"])
                metrics["eval_perplexity"] = ppl
                logger.info("Eval perplexity: %.2f", ppl)
            except OverflowError:
                metrics["eval_perplexity"] = float("inf")
                logger.info("Eval perplexity: inf")


class TrainingReportCallback(TrainerCallback):
    """Collect metrics during training and generate report + curves on exit.

    Saves to output_dir:
      - training_curves.png: loss, eval_loss, learning_rate, perplexity plots
      - report.json: hyperparams, best/final metrics, timing, model info
    """

    def __init__(self):
        self.train_log = []   # [{step, loss, lr, ...}]
        self.eval_log = []    # [{step, eval_loss, eval_perplexity, ...}]
        self.start_time = None
        self.best_eval_loss = float("inf")
        self.best_eval_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        entry = {"step": step}
        if "loss" in logs:
            entry["loss"] = logs["loss"]
        if "learning_rate" in logs:
            entry["lr"] = logs["learning_rate"]
        if "total_flos" in logs:
            entry["total_flos"] = logs["total_flos"]
        if entry.keys() != {"step"}:
            self.train_log.append(entry)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        step = state.global_step
        entry = {"step": step}
        if "eval_loss" in metrics:
            entry["eval_loss"] = metrics["eval_loss"]
            if metrics["eval_loss"] < self.best_eval_loss:
                self.best_eval_loss = metrics["eval_loss"]
                self.best_eval_step = step
        if "eval_perplexity" in metrics:
            entry["eval_perplexity"] = metrics["eval_perplexity"]
        if "eval_runtime" in metrics:
            entry["eval_runtime_s"] = metrics["eval_runtime"]
        self.eval_log.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        output_dir = args.output_dir
        total_time = time.time() - self.start_time if self.start_time else 0
        final_step = state.global_step

        # --- Save train_log.jsonl and eval_log.jsonl ---
        self._save_jsonl(os.path.join(output_dir, "train_log.jsonl"), self.train_log)
        self._save_jsonl(os.path.join(output_dir, "eval_log.jsonl"), self.eval_log)

        # --- Save training_curves.png ---
        self._save_training_curves(output_dir)

        # --- Save report.json ---
        self._save_report(output_dir, final_step, total_time, args)

    def _save_jsonl(self, path, records):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _save_training_curves(self, output_dir):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping training_curves.png")
            return

        if not self.train_log and not self.eval_log:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("DeepSleep Pretrain Training Curves", fontsize=14, fontweight="bold")

        # (0,0) Train Loss
        ax = axes[0, 0]
        if self.train_log:
            steps = [e["step"] for e in self.train_log if "loss" in e]
            losses = [e["loss"] for e in self.train_log if "loss" in e]
            ax.plot(steps, losses, alpha=0.6, linewidth=0.8)
            ax.set_xlabel("Step")
            ax.set_ylabel("Train Loss")
            ax.set_title("Training Loss")
            ax.grid(True, alpha=0.3)

        # (0,1) Eval Loss
        ax = axes[0, 1]
        if self.eval_log:
            steps = [e["step"] for e in self.eval_log if "eval_loss" in e]
            losses = [e["eval_loss"] for e in self.eval_log if "eval_loss" in e]
            ax.plot(steps, losses, "o-", color="orange", linewidth=1.5, markersize=4)
            ax.set_xlabel("Step")
            ax.set_ylabel("Eval Loss")
            ax.set_title("Eval Loss")
            ax.grid(True, alpha=0.3)

        # (1,0) Learning Rate
        ax = axes[1, 0]
        if self.train_log:
            steps = [e["step"] for e in self.train_log if "lr" in e]
            lrs = [e["lr"] for e in self.train_log if "lr" in e]
            ax.plot(steps, lrs, color="green", linewidth=1.2)
            ax.set_xlabel("Step")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.grid(True, alpha=0.3)

        # (1,1) Eval Perplexity
        ax = axes[1, 1]
        if self.eval_log:
            steps = [e["step"] for e in self.eval_log if "eval_perplexity" in e]
            ppls = [e["eval_perplexity"] for e in self.eval_log if "eval_perplexity" in e]
            ax.plot(steps, ppls, "s-", color="red", linewidth=1.5, markersize=4)
            ax.set_xlabel("Step")
            ax.set_ylabel("Perplexity")
            ax.set_title("Eval Perplexity")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "training_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training curves saved to %s", path)

    def _save_report(self, output_dir, final_step, total_time, args):
        final_train_loss = None
        if self.train_log:
            for entry in reversed(self.train_log):
                if "loss" in entry:
                    final_train_loss = entry["loss"]
                    break

        final_eval_loss = None
        final_eval_ppl = None
        if self.eval_log:
            last = self.eval_log[-1]
            final_eval_loss = last.get("eval_loss")
            final_eval_ppl = last.get("eval_perplexity")

        report = {
            "final_step": final_step,
            "total_time_hours": round(total_time / 3600, 2),
            "best_eval_loss": round(self.best_eval_loss, 6) if self.best_eval_loss < float("inf") else None,
            "best_eval_step": self.best_eval_step,
            "final_train_loss": round(final_train_loss, 6) if final_train_loss else None,
            "final_eval_loss": round(final_eval_loss, 6) if final_eval_loss else None,
            "final_eval_perplexity": round(final_eval_ppl, 2) if final_eval_ppl else None,
            "config": {
                "output_dir": args.output_dir,
                "max_steps": args.max_steps,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "lr_scheduler_type": args.lr_scheduler_type,
                "bf16": args.bf16,
                "save_steps": args.save_steps,
                "eval_steps": args.eval_steps,
            },
        }

        path = os.path.join(output_dir, "report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Training report saved to %s", path)


# =============================================================================
# Data Collator
# =============================================================================


def pretrain_data_collator(features):
    """Stack pre-tokenized samples into a batch.

    Handles both packed sequences (input_ids + labels) and padded sequences
    (input_ids + labels + attention_mask).
    """
    input_ids = torch.stack([f["input_ids"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    batch = {"input_ids": input_ids, "labels": labels}
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
    return batch


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSleep Pretraining (HuggingFace Trainer)")

    # Model
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=1, type=int, choices=[0, 1])
    parser.add_argument("--num_experts", default=8, type=int)
    parser.add_argument("--num_shared_experts", default=0, type=int)
    parser.add_argument("--num_experts_per_tok", default=2, type=int)
    parser.add_argument("--vocab_size", default=7200, type=int)
    parser.add_argument("--max_seq_len", default=2048, type=int)

    # Data
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Single HF dataset (default: None = bilingual SkyPile+OpenWebText)")
    parser.add_argument("--max_eval_samples", type=int, default=10_000)
    parser.add_argument("--min_text_length", type=int, default=50)
    parser.add_argument("--pack_sequences", default=1, type=int, choices=[0, 1],
                        help="Pack multiple docs into fixed-length sequences (recommended)")
    parser.add_argument("--zh_ratio", type=float, default=0.7,
                        help="Chinese vs English ratio in bilingual mode (default: 0.7)")
    parser.add_argument("--bilingual", default=1, type=int, choices=[0, 1],
                        help="Use bilingual zh+en mix (default: on when dataset_name is None)")

    # Training
    parser.add_argument("--output_dir", type=str, default="out/pretrain")
    parser.add_argument("--max_steps", type=int, default=50_000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer: adamw_torch (default), adamw_torch_fused, adafactor")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16")

    # Eval & Save
    parser.add_argument("--eval_steps", type=int, default=1_000)
    parser.add_argument("--save_steps", type=int, default=2_000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=50)

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Logging
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help="Comma-separated: tensorboard, wandb")
    parser.add_argument("--wandb_project", type=str, default="DeepSleep-Pretrain")
    parser.add_argument("--run_name", type=str, default=None)

    # Generation
    parser.add_argument("--generate_prompts", type=str, default=None,
                        help="Comma-separated prompts for sample generation")

    # Config
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (overrides defaults)")

    args = parser.parse_args()

    # Load YAML config if provided
    if args.config:
        from configs.config_utils import load_yaml_config
        args = load_yaml_config(args)
        logger.info("Loaded config from %s", args.config)

    return args


def main():
    args = parse_args()
    pack = bool(args.pack_sequences)

    # --- Performance optimizations ---
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # --- Tokenizer ---
    logger.info("Loading tokenizer from %s", args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    logger.info("  Vocab size: %d", tokenizer.vocab_size)

    # --- Model Config ---
    config = DeepSleepConfig(
        d_model=args.hidden_size,
        n_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        num_experts=args.num_experts,
        num_shared_experts=args.num_shared_experts,
        top_k=args.num_experts_per_tok,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_len,
    )

    # --- Model ---
    model = DeepSleepForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info("Model: %.2fM total params, %.2fM trainable", total_params, trainable_params)

    # --- Datasets ---
    mode = "bilingual (zh %.0f%% + en %.0f%%)" % (args.zh_ratio * 100, (1 - args.zh_ratio) * 100) if args.dataset_name is None else args.dataset_name
    logger.info("Creating streaming datasets: %s (pack=%s)", mode, pack)
    train_dataset = CCI4PretrainDataset(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        dataset_name=args.dataset_name,
        seed=42,
        num_samples=None,
        min_length=args.min_text_length,
        pack_sequences=pack,
        zh_ratio=args.zh_ratio,
        bilingual=bool(args.bilingual),
    )
    eval_dataset = CCI4PretrainDataset(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        dataset_name=args.dataset_name,
        seed=123,
        num_samples=args.max_eval_samples,
        min_length=args.min_text_length,
        pack_sequences=pack,
        zh_ratio=args.zh_ratio,
        bilingual=bool(args.bilingual),
    )
    logger.info("  Train: streaming (infinite), Eval: %d samples (held out)", args.max_eval_samples)

    # --- Callbacks ---
    gen_prompts = (
        args.generate_prompts.split(",") if args.generate_prompts
        else DEFAULT_GENERATE_PROMPTS
    )
    report_callback = TrainingReportCallback()
    callbacks = [
        SampleGenerationCallback(
            tokenizer=tokenizer,
            prompts=gen_prompts,
            max_new_tokens=100,
        ),
        PerplexityCallback(),
        report_callback,
    ]

    # --- Training Arguments ---
    run_name = args.run_name or f"pretrain-d{args.hidden_size}-moe"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # Training
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        # Optimizer
        optim=args.optim,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        # Precision
        bf16=args.bf16,
        bf16_full_eval=True,
        # Memory
        gradient_checkpointing=args.gradient_checkpointing,
        # Speed
        torch_compile=True,
        torch_compile_backend="inductor",
        # Eval & Save
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        # Data loading
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        # Logging
        report_to=args.report_to.split(",") if args.report_to else [],
        run_name=run_name,
        # Misc
        remove_unused_columns=False,
        include_num_input_tokens_seen=True,
    )
    logger.info("TrainingArguments:")
    for key, val in sorted(training_args.to_dict().items()):
        if val is not None and val is not False:
            logger.info("  %s: %s", key, val)

    # --- Tokens budget ---
    eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_tokens = eff_batch * args.max_seq_len * args.max_steps
    logger.info("Tokens budget: %d (effective_batch=%d, seq_len=%d, steps=%d)",
                total_tokens, eff_batch, args.max_seq_len, args.max_steps)
    logger.info("  = %.1fB tokens | %.1f tokens/param (model %dM)",
                total_tokens / 1e9,
                total_tokens / (total_params * 1e6),
                int(total_params))

    # --- Trainer ---
    trainer = MoETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=pretrain_data_collator,
        callbacks=callbacks,
    )

    # --- Train ---
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --- Save Final Model (HF format) ---
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Final model saved to %s (HF format)", final_dir)

    # --- Save .pth for SFT/DPO compatibility ---
    pth_path = os.path.join(final_dir, "model.pth")
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), pth_path)
    logger.info("Final model saved to %s (.pth)", pth_path)

    # --- Final Eval ---
    metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", metrics)


if __name__ == "__main__":
    main()
