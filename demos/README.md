# Deprecated

Use production entrypoints instead:

```bash
export INDO_CHECKPOINT=checkpoints/AERIS_main
python -m sentinel.voice.cli health
python -m sentinel.voice.cli generate --text "Hello"
python -m sentinel.voice.cli serve
```

See `examples/generate_once.py`.
