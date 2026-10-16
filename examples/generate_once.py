#!/usr/bin/env python3
"""Example: load production stack and generate once.

  export INDO_CHECKPOINT=checkpoints/AERIS_main
  python examples/generate_once.py --text "Hello"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.voice.factory import build_voice_system
from sentinel.config.settings import get_settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--ckpt", default=None)
    args = p.parse_args()
    settings = get_settings()
    adapter, model, tok, stt, tts, settings = build_voice_system(
        settings, checkpoint=args.ckpt, require_checkpoint=True
    )
    session = adapter.create_session("example", language=settings.primary_language)
    ids = tok.encode(args.text, return_tensors="pt").to(settings.device)
    for tid in adapter.generate_stream(ids, session):
        sys.stdout.write(tok.decode([tid]))
        sys.stdout.flush()
    print()


if __name__ == "__main__":
    main()
