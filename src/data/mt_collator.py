"""
MultiTurnForgetCollator: processes single-turn and multi-turn forget examples.

For multi-turn: applies the model's chat template to the full conversation,
then masks ALL tokens before the final assistant response with -100 in labels.
This ensures |y| in the SimNPO loss = only the final assistant turn's token count.

Accepts batches of dicts with either:
  - {"question": str, "answer": str}  — single-turn
  - {"conversation": list[dict]}      — multi-turn (list of role/content dicts)
"""
from __future__ import annotations
import torch
from typing import List, Dict, Any
from transformers import PreTrainedTokenizer


class MultiTurnForgetCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _prefix_length(self, conversation: List[Dict]) -> int:
        """
        Number of tokens in everything before the final assistant response.
        This is the length that will be masked with -100.
        """
        prefix_conv = conversation[:-1]  # all turns except the final assistant turn
        prefix_text = self.tokenizer.apply_chat_template(
            prefix_conv, tokenize=False, add_generation_prompt=True
        )
        return len(self.tokenizer(
            prefix_text, add_special_tokens=False
        )["input_ids"])

    def _process_conversation(self, messages: List[Dict]) -> tuple:
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            add_special_tokens=False
        )
        input_ids = torch.tensor(encoded["input_ids"])
        labels = input_ids.clone()

        prefix_len = self._prefix_length(messages)
        # Clamp to sequence length in case truncation shortened it
        prefix_len = min(prefix_len, len(input_ids) - 1)
        labels[:prefix_len] = -100  # mask prefix, expose only final response

        return input_ids, labels

    def _process_single_turn(self, question: str, answer: str) -> tuple:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": answer},
        ]
        return self._process_conversation(messages)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        all_input_ids, all_labels, all_attn_masks = [], [], []

        for item in batch:
            if "conversation" in item:
                input_ids, labels = self._process_conversation(item["conversation"])
            else:
                input_ids, labels = self._process_single_turn(
                    item["question"], item["answer"]
                )
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attn_masks.append(torch.ones_like(input_ids))

        pad_id = (self.tokenizer.pad_token_id
                  if self.tokenizer.pad_token_id is not None
                  else self.tokenizer.eos_token_id)

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                all_input_ids, batch_first=True, padding_value=pad_id),
            "labels": torch.nn.utils.rnn.pad_sequence(
                all_labels, batch_first=True, padding_value=-100),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                all_attn_masks, batch_first=True, padding_value=0),
        }
