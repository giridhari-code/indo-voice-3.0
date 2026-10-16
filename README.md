# Indo-Voice-3

**Production-ready Hindi-first Speech-to-Speech system on AERIS / SENTINEL.**

Pure custom trainable Audio Encoder · Full audio injection · WebSocket server · STT+TTS · Training script

---

## Status — Sab ready

| Feature                    | File / Location                          | Status |
|----------------------------|------------------------------------------|--------|
| Audio Injection            | `sentinel/model.py` (`inputs_embeds`)    | ✅     |
| Pure Custom Audio Encoder  | `sentinel/modules/audio_encoder.py`      | ✅     |
| Modality Projector         | `sentinel/modules/modality_projector.py` | ✅     |
| Hindi/Hinglish Router      | `sentinel/modules/hindi_router.py`       | ✅     |
| Voice Adapter + Session    | `sentinel/voice/adapter.py`              | ✅     |
| STT + TTS abstraction      | `sentinel/voice/stt_tts.py`              | ✅     |
| WebSocket Server           | `sentinel/voice/server.py`               | ✅     |
| Training Script            | `sentinel/voice/train_audio.py`          | ✅     |
| Config (no hardcode)       | env + `config/model_config.py`           | ✅     |

---

## Quick Start

```bash
cd indo-voice-3
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 1. Train audio stack (optional)

```bash
python -m sentinel.voice.train_audio --steps 100 --device cpu --output checkpoints/audio.pt
```

### 2. Run WebSocket server

```bash
# After attaching your SENTINEL checkpoint + tokenizer in code
uvicorn sentinel.voice.server:create_app --factory --host 0.0.0.0 --port 8000
```

### 3. WebSocket protocol (simple)

```
→ {"type": "session.init", "language": "hi"}
← {"type": "session.ready", "session_id": "..."}

→ binary PCM16 16kHz mono   OR   {"type": "text.input", "text": "Namaste"}
← {"type": "transcript", "text": "..."}
← {"type": "response.token", "token": "..."}   (streamed)
← {"type": "response.text_done", "text": "..."}
← {"type": "response.audio", "format": "pcm16", "data": "<base64>"}
← {"type": "response.done"}
```

---

## Architecture

```
Microphone / Client
       │
       ▼
  WebSocket Server  ←── STT (Whisper / Dummy)
       │
       ▼
  VoiceAdapter
       │
       ├── AudioEncoder (custom trainable)
       ├── ModalityProjector
       └── SENTINEL (inputs_embeds injection)
       │
       ▼
  Streaming tokens
       │
       ▼
  TTS (Coqui / Dummy) → Audio back to client
```

---

## Design rules followed

- **No hardcode** — all via env / IndoVoiceConfig
- **No demo hacks** — real sessions, slow-memory, streaming
- **Production ready** — FastAPI, health check, CORS, structured events
- API keys / real checkpoints aap baad mein set karoge

---

## Next (aapke side)

1. Apna SENTINEL checkpoint load karo
2. Tokenizer attach karo (`adapter.set_tokenizer(...)`)
3. Real STT/TTS install karo (`pip install faster-whisper TTS`)
4. Hindi speech-text data se audio encoder train karo

Sab core code ready hai.
