"""DeepSleep DPO (Direct Preference Optimization) Training."""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_deepsleep import DeepSleepConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler,
)

warnings.filterwarnings('ignore')


def get_batch_logps(logits, labels, mask=None):
    """Compute log probabilities for the response tokens."""
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        per_token_logps = per_token_logps * mask
    return per_token_logps.sum(dim=-1)


def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                      ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Compute DPO loss (sigmoid variant)."""
    logits_diff = (
        policy_chosen_logps - ref_chosen_logps
        - policy_rejected_logps + ref_rejected_logps
    )
    loss = -F.logsigmoid(beta * logits_diff)
    return loss.mean()


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch['x_chosen'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        with autocast_ctx:
            # Policy forward (gradients needed for BOTH chosen and rejected)
            policy_chosen_logits = model(x_chosen).logits
            policy_chosen_logps = get_batch_logps(policy_chosen_logits, y_chosen, mask_chosen)
            policy_rejected_logits = model(x_rejected).logits
            policy_rejected_logps = get_batch_logps(policy_rejected_logits, y_rejected, mask_rejected)

            # Reference forward (no gradients)
            with torch.no_grad():
                ref_chosen_logits = ref_model(x_chosen).logits
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, y_chosen, mask_chosen)
                ref_rejected_logits = ref_model(x_rejected).logits
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, y_rejected, mask_rejected)

            loss = compute_dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=args.dpo_beta,
            )
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend = time.time() - start_time
            cur_loss = loss.item() * args.accumulation_steps
            eta = spend / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {cur_loss:.4f}, eta: {eta:.1f}min')
            if wandb:
                wandb.log({"dpo_loss": cur_loss, "lr": optimizer.param_groups[-1]['lr']})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model,
                optimizer=optimizer, scaler=scaler, epoch=epoch, step=step,
                wandb=wandb, save_dir=args.save_dir,
            )
            model.train()

        del batch, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSleep DPO")
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--save_weight", default="dpo", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=200)
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
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--sft_checkpoint", type=str, required=True, help="SFT checkpoint path (used for both policy and reference)")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="DeepSleep-DPO")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    args = parser.parse_args()
    if args.config:
        from configs.config_utils import load_yaml_config
        args = load_yaml_config(args)

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42)

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = DeepSleepConfig(
        d_model=args.hidden_size, n_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe), num_experts=args.num_experts,
        num_shared_experts=args.num_shared_experts, top_k=args.num_experts_per_tok,
        vocab_size=args.vocab_size, max_position_embeddings=args.max_seq_len,
    )

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if "cpu" in args.device else torch.cuda.amp.autocast(dtype=dtype)

    # Policy model (trainable)
    model, tokenizer = init_model(lm_config, args.sft_checkpoint, args.tokenizer_path, args.device)
    # Reference model (frozen copy)
    ref_model, _ = init_model(lm_config, args.sft_checkpoint, args.tokenizer_path, args.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.save_dir) if args.from_resume else None
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        if ckp_data.get('optimizer'):
            optimizer.load_state_dict(ckp_data['optimizer'])
        start_epoch = ckp_data.get('epoch', 0)
        start_step = ckp_data.get('step', 0)

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    wandb = None
    if args.use_wandb and is_main_process():
        import wandb as wb
        wb.init(project=args.wandb_project, name=f"dpo-d{args.hidden_size}")
        wandb = wb

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch and start_step > 0 else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        train_epoch(epoch, loader, len(loader) + skip, start_step if epoch == start_epoch else 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()