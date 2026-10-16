#!/usr/bin/env python3
"""Load any SENTINEL/AERIS checkpoint folder and print summary.

Supports small example (AERIS_main) and large production folders:

  my_big_model/
    model.safetensors
    config.json
    vocab.json | tokenizer.json
    meta.json

Usage:
  python scripts/load_checkpoint.py checkpoints/AERIS_main
  python scripts/load_checkpoint.py /path/to/big_indo_ckpt --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinel.checkpoint.resolve import load_model_from_checkpoint


def main():
    p = argparse.ArgumentParser(description="Load SENTINEL checkpoint")
    p.add_argument("ckpt", type=str, help="Checkpoint folder or .safetensors")
    p.add_argument("--device", default="cpu")
    p.add_argument("--prompt", default=None, help="Optional generate test prompt")
    p.add_argument("--max-new", type=int, default=40)
    args = p.parse_args()

    path = Path(args.ckpt)
    if not path.exists():
        print(f"ERROR: {path} not found")
        print("Place a full checkpoint folder here, e.g.:")
        print("  checkpoints/indo_large/")
        print("    model.safetensors")
        print("    config.json")
        print("    vocab.json")
        sys.exit(1)

    model, tok, cfg, meta = load_model_from_checkpoint(path, device=args.device, strict=False)
    n = sum(x.numel() for x in model.parameters())
    print("=" * 50)
    print(f"Loaded: {path}")
    print(f"  model_id : {meta.get('model_id') or meta.get('product')}")
    print(f"  steps    : {meta.get('steps')}")
    print(f"  d_model  : {cfg.model.d_model}")
    print(f"  vocab    : {cfg.model.vocab_size}")
    print(f"  params   : {n:,}")
    print(f"  device   : {args.device}")
    print("=" * 50)

    if args.prompt:
        import torch
        ids = tok.encode(args.prompt)
        t = torch.tensor([ids], device=args.device)
        state = None
        out_ids = []
        print(f"Prompt: {args.prompt!r}")
        print("Gen: ", end="", flush=True)
        with torch.inference_mode():
            for i in range(args.max_new):
                cur = t if i == 0 else t[:, -1:]
                out = model(input_ids=cur, state=state, collect_stats=False)
                state = out.state
                nid = int(out.logits[:, -1].argmax(-1).item())
                out_ids.append(nid)
                ch = tok.decode([nid])
                print(ch, end="", flush=True)
                t = torch.tensor([[nid]], device=args.device)
        print()
        print("Full:", repr(tok.decode(out_ids)))


if __name__ == "__main__":
    main()
