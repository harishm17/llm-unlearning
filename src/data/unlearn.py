import torch
from torch.utils.data import Dataset


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, anchor="forget"):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
        """
        self.forget = forget
        self.retain = retain
        self.anchor = anchor

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.anchor == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when anchor=forget"
            )
            return len(self.forget)
        elif self.anchor == "retain":
            assert self.retain is not None, ValueError(
                "retain dataset can't be None when anchor=retain"
            )
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            item["forget"] = self.forget[idx]
            if self.retain:
                retain_idx = torch.randint(0, len(self.retain), (1,)).item()
                item["retain"] = self.retain[retain_idx]
        elif self.anchor == "retain":
            item["retain"] = self.retain[idx]
            if self.forget:
                forget_idx = torch.randint(0, len(self.forget), (1,)).item()
                item["forget"] = self.forget[forget_idx]
        return item


class MTForgetRetainDataset(ForgetRetainDataset):
    """
    Extends ForgetRetainDataset to also yield pre-tokenized multi-turn forget examples.

    mt_forget items are tokenized at construction time using MultiTurnForgetCollator
    so that DataCollatorForSupervisedDataset can handle the 'mt_forget' key via its
    standard recursive path (which expects dicts with input_ids/labels tensors).
    """

    def __init__(self, forget, retain, mt_forget, tokenizer, max_length: int = 2048, anchor="forget"):
        super().__init__(forget, retain, anchor=anchor)
        from data.mt_collator import MultiTurnForgetCollator
        collator = MultiTurnForgetCollator(tokenizer, max_length=max_length)

        # Pre-tokenize: store list of {"input_ids": 1D tensor, "attention_mask": 1D tensor, "labels": 1D tensor}
        self._mt_tokenized = []
        for item in mt_forget:
            conv = item.get("conversation")
            if conv:
                input_ids, labels = collator._process_conversation(conv)
            else:
                input_ids, labels = collator._process_single_turn(
                    item["question"], item["answer"]
                )
            self._mt_tokenized.append({
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            })

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        mt_idx = idx % len(self._mt_tokenized)
        item["mt_forget"] = self._mt_tokenized[mt_idx]
        return item
