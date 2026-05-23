"""DeepSleep datasets for pretraining, SFT, and DPO training."""

import json
import os

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class PretrainDataset(Dataset):
    """Lazy-loading pretraining dataset from JSONL.

    Builds a byte-offset index on init (fast scan, no data loaded to memory).
    Each __getitem__ reads only the requested line from disk.
    Supports files of any size (tested up to 20GB+).
    """

    def __init__(self, data_path, tokenizer, max_length=2048):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path
        self.offsets = self._build_index(data_path)

    @staticmethod
    def _build_index(path):
        """Scan file to record byte offset of each line. O(n) scan, minimal memory."""
        offsets = []
        with open(path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    offsets.append(offset)
                offset += len(line)
        return offsets

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with open(self.data_path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[index])
            line = f.readline()
        sample = json.loads(line)
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        tokens += [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    """Supervised fine-tuning dataset with ChatML format."""

    def __init__(self, jsonl_path, tokenizer, max_length=2048):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        import json
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample.get('conversations', sample.get('messages', []))
        prompt = self.create_chat_prompt(conversations)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    """Direct Preference Optimization dataset with chosen/rejected pairs."""

    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id or 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 支持两种格式：新格式(shared messages + string) / 旧格式(full message lists)
        if "messages" in sample:
            shared = list(sample["messages"])
            chosen = shared + [{"role": "assistant", "content": sample["chosen"]}]
            rejected = shared + [{"role": "assistant", "content": sample["rejected"]}]
        else:
            chosen = sample['chosen']
            rejected = sample['rejected']

        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)

        chosen_enc = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        rejected_enc = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length')

        chosen_ids = chosen_enc['input_ids']
        chosen_mask = self._generate_loss_mask(chosen_ids)
        rejected_ids = rejected_enc['input_ids']
        rejected_mask = self._generate_loss_mask(rejected_ids)

        return {
            'x_chosen': torch.tensor(chosen_ids[:-1], dtype=torch.long),
            'y_chosen': torch.tensor(chosen_ids[1:], dtype=torch.long),
            'mask_chosen': torch.tensor(chosen_mask[1:], dtype=torch.long),
            'x_rejected': torch.tensor(rejected_ids[:-1], dtype=torch.long),
            'y_rejected': torch.tensor(rejected_ids[1:], dtype=torch.long),
            'mask_rejected': torch.tensor(rejected_mask[1:], dtype=torch.long),
        }

    def _generate_loss_mask(self, input_ids):
        mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return mask