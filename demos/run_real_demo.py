#!/usr/bin/env python3
"""Indo-Voice-3 — Real checkpoint demo (AERIS_main weights)."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.checkpoint.resolve import load_model_from_checkpoint
from sentinel.config.model_config import IndoVoiceConfig
from sentinel.voice.adapter import VoiceAdapter
from sentinel.voice.stt_tts import create_stt, create_tts, waveform_to_tensor, AudioChunk


def save_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    print(f"  saved: {path}")


def make_synthetic_speech(seconds: float = 1.2, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), dtype=np.float32)
    f0 = 130.0 + 20 * np.sin(2 * np.pi * 3 * t)
    sig = 0.25 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * f0 * 2 * t)
    env = np.minimum(1.0, t * 10) * np.minimum(1.0, (seconds - t) * 10)
    return (sig * env).astype(np.float32)


class TokBridge:
    """Adapter between CharTokenizer and VoiceAdapter / generate API."""

    def __init__(self, tok):
        self.tok = tok
        self.eos_token_id = None
        stops = getattr(tok, "stop_ids", set()) or set()
        if stops:
            self.eos_token_id = next(iter(stops))

    def encode(self, text, return_tensors=None, **kw):
        ids = self.tok.encode(text)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.view(-1).tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self.tok.decode(ids)


def generate(model, tok: TokBridge, prompt: str, max_new: int = 80, device: str = "cpu") -> str:
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    generated = []
    state = None
    print("  generating: ", end="", flush=True)
    with torch.inference_mode():
        for step in range(max_new):
            cur = ids if step == 0 else ids[:, -1:]
            out = model(input_ids=cur, state=state, collect_stats=False)
            state = out.state
            next_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
            generated.append(next_id)
            print(tok.decode([next_id]), end="", flush=True)
            ids = torch.tensor([[next_id]], device=device)
            if tok.eos_token_id is not None and next_id == tok.eos_token_id:
                break
    print()
    return tok.decode(generated)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/AERIS_main")
    p.add_argument("--device", default="cpu")
    p.add_argument("--text", default="Hello")
    p.add_argument("--out", default="demos/output_real")
    p.add_argument("--mode", choices=["text", "audio", "both"], default="both")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Indo-Voice-3 REAL Demo")
    print("=" * 40)
    model, tokenizer, full_cfg, meta = load_model_from_checkpoint(
        args.ckpt, device=args.device, strict=False
    )
    print(f"  meta: {meta.get('model_id') or meta.get('product')} steps={meta.get('steps')}")
    print(f"  d_model={full_cfg.model.d_model} vocab={full_cfg.model.vocab_size}")
    print(f"  params={sum(x.numel() for x in model.parameters()):,}")

    tok = TokBridge(tokenizer)
    vcfg = IndoVoiceConfig.from_env()
    vcfg.model.d_model = full_cfg.model.d_model
    vcfg.model.vocab_size = full_cfg.model.vocab_size
    vcfg.audio.encoder_dim = min(512, max(128, full_cfg.model.d_model * 2))

    adapter = VoiceAdapter(vcfg)
    adapter.set_sentinel(model)
    adapter.set_tokenizer(tok)
    # rebuild projector to match d_model
    from sentinel.modules.modality_projector import ModalityProjector
    from sentinel.modules.audio_encoder import AudioEncoder
    adapter.audio_encoder = AudioEncoder(vcfg.audio).to(args.device)
    adapter.projector = ModalityProjector(vcfg.audio, vcfg.model).to(args.device)
    adapter.to(args.device)

    stt, tts = create_stt("dummy"), create_tts("dummy")

    if args.mode in ("text", "both"):
        print("\n=== TEXT (real SENTINEL weights) ===")
        print(f"  prompt: {args.text!r}")
        reply = generate(model, tok, args.text, max_new=64, device=args.device)
        print(f"  reply: {reply!r}")
        save_wav(out_dir / "real_text_reply.wav", tts.synthesize(reply or "hi").samples)

    if args.mode in ("audio", "both"):
        print("\n=== AUDIO path (encoder random + SENTINEL real) ===")
        samples = make_synthetic_speech()
        save_wav(out_dir / "real_input_synth.wav", samples)
        waveform = torch.from_numpy(samples).unsqueeze(0).to(args.device)
        with torch.inference_mode():
            out = adapter(waveform=waveform, collect_stats=False)
        print(f"  audio inject logits: {tuple(out.logits.shape)}")
        reply = generate(model, tok, args.text, max_new=64, device=args.device)
        print(f"  reply: {reply!r}")
        save_wav(out_dir / "real_audio_reply.wav", tts.synthesize(reply or "hi").samples)

    print("\n" + "=" * 40)
    print("DONE — real checkpoint loaded and running.")
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
