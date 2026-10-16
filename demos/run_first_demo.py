#!/usr/bin/env python3
"""Indo-Voice-3 — First Audio Demo

Flow:
  1. Synthetic / file audio  →  AudioEncoder  →  Projector
  2. Inject into SENTINEL (inputs_embeds) OR text path
  3. Stream tokens
  4. Dummy TTS → save wav

Baad mein: real checkpoint, real tokenizer, Whisper, Coqui replace kar lena.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import torch

# project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.config import IndoVoiceConfig, ModelConfig, SafetyConfig, AblationConfig
from sentinel.model import SENTINEL
from sentinel.voice.adapter import VoiceAdapter, VoiceSession
from sentinel.voice.demo_tokenizer import DemoTokenizer
from sentinel.voice.stt_tts import create_stt, create_tts, AudioChunk, waveform_to_tensor


def make_synthetic_speech(seconds: float = 1.5, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """Simple synthetic 'speech-like' waveform for demo (no real recording needed)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sr * seconds), dtype=np.float32)
    # mix of formant-like tones + noise
    f0 = 120.0 + 30.0 * np.sin(2 * np.pi * 2.5 * t)
    sig = 0.3 * np.sin(2 * np.pi * f0 * t)
    sig += 0.15 * np.sin(2 * np.pi * (f0 * 2) * t)
    sig += 0.08 * np.sin(2 * np.pi * (f0 * 3) * t)
    sig += 0.05 * rng.standard_normal(len(t)).astype(np.float32)
    # envelope
    env = np.minimum(1.0, t * 8) * np.minimum(1.0, (seconds - t) * 8)
    return (sig * env).astype(np.float32)


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


def build_system(device: str = "cpu"):
    cfg = IndoVoiceConfig.from_env()
    # smaller k for faster demo
    mc = ModelConfig(
        d_model=cfg.model.d_model,
        d_k=cfg.model.d_k,
        d_v=cfg.model.d_v,
        n_skills=cfg.model.n_skills,
        top_k=cfg.model.top_k,
        k_max=2,
        dendrite_window=cfg.model.dendrite_window,
        vocab_size=cfg.model.vocab_size,
        max_seq_len=cfg.model.max_seq_len,
        dropout=0.0,
        tie_embeddings=True,
    )
    model = SENTINEL(mc, SafetyConfig(), AblationConfig())
    model.to(device)
    model.eval()

    tokenizer = DemoTokenizer(vocab_size=mc.vocab_size)
    adapter = VoiceAdapter(cfg)
    adapter.set_sentinel(model)
    adapter.set_tokenizer(tokenizer)
    adapter.to(device)

    stt = create_stt("dummy")
    tts = create_tts("dummy")
    return cfg, model, adapter, tokenizer, stt, tts


def run_text_demo(adapter, tokenizer, tts, text: str, out_dir: Path, device: str):
    print("\n=== TEXT → SENTINEL → TTS ===")
    print(f"  input: {text!r}")
    session = adapter.create_session("demo-text", language="hi")
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    tokens = []
    print("  generating: ", end="", flush=True)
    with torch.inference_mode():
        for tid in adapter.generate_stream(input_ids, session, max_new_tokens=32):
            tokens.append(tid)
            piece = tokenizer.decode([tid])
            print(piece, end="", flush=True)
    print()
    reply = tokenizer.decode(tokens)
    print(f"  reply (decoded): {reply!r}")

    audio = tts.synthesize(reply or "Namaste", language="hi")
    save_wav(out_dir / "demo_text_reply.wav", audio.samples, audio.sample_rate)
    return reply


def run_audio_demo(adapter, model, tokenizer, stt, tts, out_dir: Path, device: str, wav_path: Path | None):
    print("\n=== AUDIO → Encoder → SENTINEL → TTS ===")
    sr = 16000
    if wav_path and wav_path.exists():
        import wave as wv
        with wv.open(str(wav_path), "rb") as w:
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        print(f"  loaded file: {wav_path} ({len(samples)/sr:.2f}s @ {sr}Hz)")
    else:
        samples = make_synthetic_speech(1.5, sr=16000)
        save_wav(out_dir / "demo_input_synthetic.wav", samples, 16000)
        print("  using synthetic speech (1.5s)")

    chunk = AudioChunk(samples=samples, sample_rate=sr, duration_sec=len(samples) / sr)
    # STT (dummy returns empty — we still run audio injection path)
    transcript = stt.transcribe(chunk, language="hi")
    print(f"  STT transcript: {transcript!r} (dummy is empty — expected)")

    waveform = waveform_to_tensor(chunk, target_sr=16000).to(device)

    # Path A: full audio injection (no text tokens)
    print("  running AudioEncoder → Projector → SENTINEL(inputs_embeds)...")
    with torch.inference_mode():
        out = adapter(waveform=waveform, collect_stats=False)
    print(f"  logits shape: {tuple(out.logits.shape)}")
    # greedy one-step sample for demo display
    next_ids = out.logits[:, -1, :].argmax(dim=-1)
    print(f"  next-token ids: {next_ids.tolist()}")

    # Path B: text prompt + continue (more natural for demo without trained audio)
    prompt = "Namaste, main Indo-Voice hoon."
    if transcript.strip():
        prompt = transcript
    print(f"  text-gen prompt: {prompt!r}")
    session = adapter.create_session("demo-audio", language="hi")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    tokens = []
    print("  generating: ", end="", flush=True)
    with torch.inference_mode():
        for tid in adapter.generate_stream(input_ids, session, max_new_tokens=40):
            tokens.append(tid)
            print(tokenizer.decode([tid]), end="", flush=True)
    print()
    reply = tokenizer.decode(tokens)
    print(f"  reply: {reply!r}")

    audio_out = tts.synthesize(reply or "Namaste, kaise madad karun?", language="hi")
    save_wav(out_dir / "demo_audio_reply.wav", audio_out.samples, audio_out.sample_rate)
    return reply


def main():
    parser = argparse.ArgumentParser(description="Indo-Voice-3 first audio demo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wav", type=str, default=None, help="Optional input wav (16k mono)")
    parser.add_argument("--text", type=str, default="Namaste, aap kaise hain?")
    parser.add_argument("--out", type=str, default="demos/output")
    parser.add_argument("--mode", choices=["text", "audio", "both"], default="both")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Indo-Voice-3 First Demo")
    print("=" * 40)
    print("Building system (random weights — replace with checkpoint later)...")
    cfg, model, adapter, tokenizer, stt, tts = build_system(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SENTINEL params: {n_params:,}")
    print(f"  AudioEncoder params: {sum(p.numel() for p in adapter.audio_encoder.parameters()):,}")
    print(f"  STT={type(stt).__name__}  TTS={type(tts).__name__}")

    if args.mode in ("text", "both"):
        run_text_demo(adapter, tokenizer, tts, args.text, out_dir, args.device)

    if args.mode in ("audio", "both"):
        wav_path = Path(args.wav) if args.wav else None
        run_audio_demo(adapter, model, tokenizer, stt, tts, out_dir, args.device, wav_path)

    print("\n" + "=" * 40)
    print("Demo complete.")
    print(f"Outputs in: {out_dir.resolve()}")
    print("\nNext (real):")
    print("  1. Load AERIS checkpoint into SENTINEL")
    print("  2. Replace DemoTokenizer with real vocab")
    print("  3. pip install faster-whisper TTS  → real STT/TTS")
    print("  4. Train audio encoder on Hindi speech")


if __name__ == "__main__":
    main()
