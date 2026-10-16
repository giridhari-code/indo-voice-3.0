"""Minimal tokenizer for first demo (no external deps).

Replace later with real AERIS / HF tokenizer.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from torch import Tensor


class DemoTokenizer:
    """
    Character-level + simple Hindi-friendly tokenizer for demo.
    vocab_size defaults to match ModelConfig (32000) with room for specials.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        # printable ASCII + common Devanagari block mapped into range
        self._special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.eos_token_id = self.EOS
        self.bos_token_id = self.BOS
        self.pad_token_id = self.PAD

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
    ) -> Union[List[int], Tensor]:
        ids: List[int] = []
        if add_special_tokens:
            ids.append(self.BOS)
        for ch in text:
            o = ord(ch)
            # map into [4, vocab_size-1]
            tid = 4 + (o % (self.vocab_size - 4))
            ids.append(tid)
        if add_special_tokens:
            ids.append(self.EOS)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids: Union[List[int], Tensor], skip_special_tokens: bool = True) -> str:
        if isinstance(ids, Tensor):
            ids = ids.view(-1).tolist()
        chars = []
        for i in ids:
            if skip_special_tokens and i in (self.PAD, self.BOS, self.EOS):
                continue
            if i == self.UNK:
                chars.append("?")
                continue
            # reverse map (lossy for demo)
            o = (i - 4) % 128
            if 32 <= o < 127:
                chars.append(chr(o))
            elif 0x0900 <= (i - 4) <= 0x097F:
                chars.append(chr(i - 4))
            else:
                chars.append("·")
        return "".join(chars)

    def __len__(self) -> int:
        return self.vocab_size
