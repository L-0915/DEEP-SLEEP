"""Shared training utilities for DeepSleep."""

import os
import math
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer


def Logger(content):
    if is_main_process():
        print(content)


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_lr(current_step, total_steps, lr):
    """Cosine learning rate with warmup (first 10% of steps)."""
    warmup = total_steps * 0.1
    if current_step < warmup:
        return lr * (current_step / warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (current_step - warmup) / (total_steps - warmup))))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_params(model, config):
    """Print total and active parameter counts."""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'num_routed_experts', 0)
    n_active = config.top_k if config.use_moe else 0
    n_shared = getattr(config, 'num_shared_experts', 0)
    if n_routed > 0 and n_active > 0:
        expert_params = sum(
            p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n
        ) / 1e6
        shared_params = sum(
            p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n
        ) / 1e6 if n_shared > 0 else 0
        base = total - (expert_params * n_routed) - (shared_params * n_shared)
        active = base + (expert_params * n_active) + (shared_params * n_shared)
        Logger(f'Model Params: {total:.2f}M (Active: {active:.2f}M)')
    else:
        Logger(f'Model Params: {total:.2f}M')


def init_model(config, from_weight='none', tokenizer_path=None, device='cuda'):
    """Initialize model and tokenizer."""
    from model.model_deepsleep import DeepSleepForCausalLM

    tokenizer = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    model = DeepSleepForCausalLM(config)

    if from_weight != 'none' and os.path.exists(from_weight):
        weights = torch.load(from_weight, map_location=device)
        model.load_state_dict(weights, strict=False)
        Logger(f'Loaded weights from {from_weight}')

    get_model_params(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f'Trainable Params: {trainable:.3f}M')
    return model.to(device), tokenizer


def lm_checkpoint(config, model=None, optimizer=None, scaler=None, epoch=0, step=0,
                  save_dir='out', weight='pretrain', wandb=None, **kwargs):
    """Save or load training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    moe_suffix = '_moe' if config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{config.d_model}{moe_suffix}.pth'
    resume_path = f'{save_dir}/{weight}_{config.d_model}{moe_suffix}_resume.pth'

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}

        tmp = ckp_path + '.tmp'
        torch.save(state_dict, tmp)
        os.replace(tmp, ckp_path)

        wandb_id = None
        if wandb:
            run = getattr(wandb, 'get_run', lambda: None)()
            if run:
                wandb_id = getattr(run, 'id', None)
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id,
        }
        if scaler:
            resume_data['scaler'] = scaler.state_dict()

        tmp = resume_path + '.tmp'
        torch.save(resume_data, tmp)
        os.replace(tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            data = torch.load(resume_path, map_location='cpu')
            saved_ws = data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                data['step'] = data['step'] * saved_ws // current_ws
                Logger(f'GPU count changed ({saved_ws}->{current_ws}), step adjusted to {data["step"]}')
            return data
        return None


class SkipBatchSampler(Sampler):
    """Batch sampler that skips already-trained batches for resuming."""

    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if batch and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total - self.skip_batches)