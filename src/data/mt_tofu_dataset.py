"""
MTTofuDataset: loads TOFU forget set + multi-turn adversarial conversations.

Returns combined DataLoader with WeightedRandomSampler that ensures D_f and
D_mt_train contribute equally per epoch (K_st=4 and K_mt=4 per step).
"""
from __future__ import annotations
import json
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler, DataLoader
from datasets import load_dataset


class SingleTurnForgetDataset(Dataset):
    """TOFU forget split as single-turn (question, answer) pairs."""
    def __init__(self, split: str = "forget10"):
        data = load_dataset("locuslab/TOFU", split, split="train")
        self.items = [{"question": d["question"], "answer": d["answer"]} for d in data]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class MultiTurnForgetDataset(Dataset):
    """Multi-turn adversarial conversations from generated JSONL."""
    def __init__(self, jsonl_path: str, split: str = "train"):
        with open(jsonl_path) as f:
            self.items = [
                json.loads(line) for line in f
                if line.strip() and json.loads(line).get("split") == split
            ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "conversation": item["conversation"],
            "answer": item["answer"],
            "attack_type": item["attack_type"],
        }


class MTTofuDataset:
    """
    Combines single-turn and multi-turn forget datasets with equal epoch contribution.

    WeightedRandomSampler assigns weight 1/|D_f| to each D_f sample and
    1/|D_mt| to each D_mt sample, so expected draws per epoch are equal
    regardless of dataset size difference (~400 ST vs ~1200 MT).
    """

    def __init__(self, tofu_forget_split: str = "forget10",
                 mt_train_path: str = "data/mt_train.jsonl"):
        self.st_dataset = SingleTurnForgetDataset(split=tofu_forget_split)
        self.mt_dataset  = MultiTurnForgetDataset(jsonl_path=mt_train_path, split="train")

    def get_combined_loader(self, batch_size: int = 4,
                             num_workers: int = 0) -> DataLoader:
        """
        Returns DataLoader that interleaves D_f and D_mt_train 1:1 per epoch.
        Each epoch samples 2 * len(D_f) total examples (len(D_f) from each source).
        """
        n_st = len(self.st_dataset)
        n_mt = len(self.mt_dataset)

        weights_st = [1.0 / n_st] * n_st
        weights_mt = [1.0 / n_mt] * n_mt
        combined_weights = weights_st + weights_mt

        combined = ConcatDataset([self.st_dataset, self.mt_dataset])
        sampler = WeightedRandomSampler(
            weights=combined_weights,
            num_samples=2 * n_st,  # 1 epoch = 2 * |D_f| samples
            replacement=True,
        )

        return DataLoader(combined, batch_size=batch_size,
                          sampler=sampler, num_workers=num_workers,
                          collate_fn=None)

    @property
    def st_dataset_ref(self):
        return self.st_dataset

    @property
    def mt_dataset_ref(self):
        return self.mt_dataset
