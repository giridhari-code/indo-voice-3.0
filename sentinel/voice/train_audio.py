"""Training script for Indo-Voice-3 Audio Encoder + Modality Projector.

Trains the custom audio stack while keeping SENTINEL core frozen (or optionally unfrozen).
No hard-coded paths — all via CLI / env.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sentinel.config.model_config import IndoVoiceConfig, AudioConfig, ModelConfig
from sentinel.modules.audio_encoder import AudioEncoder
from sentinel.modules.modality_projector import ModalityProjector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("indo_voice.train")


class DummyAudioTextDataset(Dataset):
    """
    Placeholder dataset.
    Replace with real Hindi/Hinglish speech-text pairs.
    Expected item: (waveform: FloatTensor[T], text_ids: LongTensor[L], audio_len: int)
    """

    def __init__(self, size: int = 100, sr: int = 16000, max_sec: float = 5.0, vocab_size: int = 32000):
        self.size = size
        self.sr = sr
        self.max_samples = int(sr * max_sec)
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        T = torch.randint(self.sr, self.max_samples, (1,)).item()
        waveform = torch.randn(T) * 0.1
        L = torch.randint(5, 40, (1,)).item()
        text_ids = torch.randint(0, self.vocab_size, (L,))
        return waveform, text_ids, T


def collate_audio_text(batch):
    waveforms, text_ids, lengths = zip(*batch)
    max_t = max(w.size(0) for w in waveforms)
    max_l = max(t.size(0) for t in text_ids)
    B = len(batch)

    wav_pad = torch.zeros(B, max_t)
    txt_pad = torch.full((B, max_l), -100, dtype=torch.long)  # ignore_index
    lens = torch.tensor(lengths, dtype=torch.long)

    for i, (w, t) in enumerate(zip(waveforms, text_ids)):
        wav_pad[i, : w.size(0)] = w
        txt_pad[i, : t.size(0)] = t

    return wav_pad, txt_pad, lens


class AudioOnlyTrainer:
    """
    Trains AudioEncoder + ModalityProjector.
    Optionally freezes SENTINEL and only trains the audio path.
    """

    def __init__(
        self,
        cfg: IndoVoiceConfig,
        device: str = "cpu",
        freeze_sentinel: bool = True,
        sentinel_model: Optional[nn.Module] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.encoder = AudioEncoder(cfg.audio).to(self.device)
        self.projector = ModalityProjector(cfg.audio, cfg.model).to(self.device)
        self.sentinel = sentinel_model
        self.freeze_sentinel = freeze_sentinel

        if self.sentinel is not None and freeze_sentinel:
            for p in self.sentinel.parameters():
                p.requires_grad_(False)
            self.sentinel.eval()

        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        self.optim = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)

    def train_step(self, waveform: torch.Tensor, text_ids: torch.Tensor, lengths: torch.Tensor) -> dict:
        self.encoder.train()
        self.projector.train()
        waveform = waveform.to(self.device)
        text_ids = text_ids.to(self.device)
        lengths = lengths.to(self.device)

        features, frame_lens = self.encoder(waveform, lengths)
        embeds = self.projector(features)  # (B, T_a, d_model)

        loss = torch.tensor(0.0, device=self.device)
        metrics = {"audio_frames": float(features.size(1))}

        if self.sentinel is not None:
            # Teacher-forcing path: use audio embeds as prefix, predict text tokens
            # For simplicity we use mean-pooled audio as conditioning bias
            # Full sequence injection can be enabled later
            B = embeds.size(0)
            # Take first few audio frames as prefix embeds
            prefix_len = min(8, embeds.size(1))
            prefix = embeds[:, :prefix_len, :]

            # Build input_ids from text (shift for LM)
            # Here we just run a reconstruction-style loss on projector output norm
            # Real training should use paired speech-text data + LM loss
            targets = text_ids[:, 1:].contiguous()
            input_ids = text_ids[:, :-1].contiguous()
            # Mask ignore_index
            valid = (input_ids != -100) & (targets != -100)
            if valid.any():
                out = self.sentinel(
                    input_ids=input_ids.clamp(min=0),
                    targets=targets,
                    inputs_embeds=None,  # pure text path for now
                    collect_stats=False,
                )
                if out.loss is not None:
                    loss = out.loss
                    metrics["lm_loss"] = float(loss.detach().item())
        else:
            # No sentinel: contrastive / reconstruction placeholder
            # Encourage non-zero, stable features
            loss = (features.pow(2).mean() - 1.0).pow(2) + embeds.pow(2).mean() * 0.01
            metrics["proxy_loss"] = float(loss.detach().item())

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.projector.parameters()), 1.0
        )
        self.optim.step()

        metrics["loss"] = float(loss.detach().item())
        return metrics

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "projector": self.projector.state_dict(),
                "audio_cfg": self.cfg.audio.__dict__,
                "model_cfg": self.cfg.model.__dict__,
            },
            path,
        )
        logger.info("Saved audio stack → %s", path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.projector.load_state_dict(ckpt["projector"])
        logger.info("Loaded audio stack ← %s", path)


def main():
    parser = argparse.ArgumentParser(description="Train Indo-Voice-3 Audio Encoder")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="checkpoints/indo_voice_audio.pt")
    parser.add_argument("--data-size", type=int, default=64, help="Dummy dataset size")
    args = parser.parse_args()

    cfg = IndoVoiceConfig.from_env()
    trainer = AudioOnlyTrainer(cfg, device=args.device, freeze_sentinel=True)

    ds = DummyAudioTextDataset(size=args.data_size, vocab_size=cfg.model.vocab_size)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_audio_text, shuffle=True)

    step = 0
    while step < args.steps:
        for batch in loader:
            metrics = trainer.train_step(*batch)
            step += 1
            if step % 10 == 0:
                logger.info("step=%d %s", step, metrics)
            if step >= args.steps:
                break

    trainer.save(args.output)
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
