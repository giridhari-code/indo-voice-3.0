# Indo-Voice-3

Production speech-to-speech stack on **SENTINEL** (two-speed memory architecture).

- Pure convolutional + SENTINEL-style **AudioEncoder** (no Transformer LM)
- **ModalityProjector** → `inputs_embeds` injection into SENTINEL
- Hindi / Hinglish **SkillRouter**
- Config and secrets via **environment only**
- HTTP + WebSocket service, CLI, training entrypoints

## Requirements

```bash
pip install torch numpy safetensors
pip install fastapi "uvicorn[standard]" websockets python-dotenv pydantic
# optional production I/O
pip install faster-whisper
# pip install TTS
```

## Checkpoint layout

```
checkpoints/<name>/
  model.safetensors
  config.json
  vocab.json          # or tokenizer.json
  meta.json           # optional
```

```bash
export INDO_CHECKPOINT=checkpoints/AERIS_main
```

## CLI

```bash
python -m sentinel.voice.cli health
python -m sentinel.voice.cli generate --text "Hello"
python -m sentinel.voice.cli serve --host 0.0.0.0 --port 8000
```

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + backend info |
| POST | `/v1/session` | Create session |
| WS | `/v1/voice` | Realtime voice |

WebSocket: first message `{"type":"session.init","language":"hi"}`, then text or PCM16 audio frames.

## Train audio encoder

```bash
# manifest.jsonl: {"wav": "...", "text": "..."}
python -m sentinel.voice.train_audio \
  --data data/hindi_speech/manifest.jsonl \
  --ckpt $INDO_CHECKPOINT \
  --steps 2000 \
  --output checkpoints/indo_audio.pt

export INDO_AUDIO_WEIGHTS=checkpoints/indo_audio.pt
```

## Architecture

```
Audio → STT (optional) → text
Audio → AudioEncoder → Projector → SENTINEL(inputs_embeds)
Text  → Tokenizer → SENTINEL(input_ids)
SENTINEL → tokens → TTS → audio
```

Core language model remains **SENTINEL**, not a Transformer LM.

## License

See upstream AERIS / SENTINEL licensing for model weights you load.
