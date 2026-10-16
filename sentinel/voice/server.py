"""Indo-Voice-3 WebSocket realtime server.

Production-ready FastAPI + WebSocket endpoint.
No hard-coded secrets — everything from env / config.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentinel.config.model_config import IndoVoiceConfig
from sentinel.voice.adapter import VoiceAdapter, VoiceSession
from sentinel.voice.stt_tts import (
    create_stt,
    create_tts,
    AudioChunk,
    waveform_to_tensor,
)

logger = logging.getLogger("indo_voice.server")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_voice_app(
    adapter: Optional[VoiceAdapter] = None,
    cfg: Optional[IndoVoiceConfig] = None,
) -> FastAPI:
    """
    Create FastAPI app with voice WebSocket endpoint.

    Usage:
        uvicorn sentinel.voice.server:create_voice_app --factory
    or attach your own adapter before serving.
    """
    cfg = cfg or IndoVoiceConfig.from_env()
    app = FastAPI(title="Indo-Voice-3", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("INDO_CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared state
    app.state.adapter = adapter
    app.state.cfg = cfg
    app.state.stt = create_stt(os.getenv("INDO_STT_PROVIDER", "auto"))
    app.state.tts = create_tts(os.getenv("INDO_TTS_PROVIDER", "auto"))
    app.state.sessions: Dict[str, VoiceSession] = {}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_attached": app.state.adapter is not None and app.state.adapter.sentinel is not None,
            "stt": type(app.state.stt).__name__,
            "tts": type(app.state.tts).__name__,
        }

    @app.post("/session")
    async def create_session(language: str = "hi"):
        sid = str(uuid.uuid4())
        if app.state.adapter is None:
            raise HTTPException(503, "VoiceAdapter not attached")
        session = app.state.adapter.create_session(sid, language=language)
        app.state.sessions[sid] = session
        return {"session_id": sid, "language": language}

    @app.websocket("/ws/voice")
    async def voice_ws(websocket: WebSocket):
        await websocket.accept()
        session: Optional[VoiceSession] = None
        session_id: Optional[str] = None

        try:
            # First message must be session.init
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            if msg.get("type") != "session.init":
                await websocket.send_json({"type": "error", "message": "Expected session.init"})
                await websocket.close()
                return

            session_id = msg.get("session_id") or str(uuid.uuid4())
            language = msg.get("language", "hi")
            if app.state.adapter is None:
                await websocket.send_json({"type": "error", "message": "Adapter not ready"})
                await websocket.close()
                return

            session = app.state.sessions.get(session_id)
            if session is None:
                session = app.state.adapter.create_session(session_id, language=language)
                app.state.sessions[session_id] = session

            await websocket.send_json({
                "type": "session.ready",
                "session_id": session_id,
                "language": language,
            })

            # Main loop
            while True:
                data = await websocket.receive()
                if "text" in data:
                    msg = json.loads(data["text"])
                    await _handle_text_event(websocket, msg, session, app)
                elif "bytes" in data:
                    # Raw PCM audio chunk
                    await _handle_audio_bytes(websocket, data["bytes"], session, app)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected session=%s", session_id)
        except Exception as e:
            logger.exception("WebSocket error: %s", e)
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            if session_id and session_id in app.state.sessions:
                # Optional: persist slow memory here
                pass

    return app


async def _handle_text_event(ws: WebSocket, msg: dict, session: VoiceSession, app):
    typ = msg.get("type")

    if typ == "text.input":
        text = msg.get("text", "").strip()
        if not text:
            return
        await _run_generation(ws, text, session, app)

    elif typ == "audio.end":
        # Client finished sending audio; STT already done in bytes handler
        pass

    elif typ == "session.close":
        await ws.close()


async def _handle_audio_bytes(ws: WebSocket, raw: bytes, session: VoiceSession, app):
    """Receive PCM16 mono 16kHz chunks, accumulate, then STT on end-of-utterance signal."""
    # Simple approach: treat each binary message as a full utterance for now
    # (production would use VAD / buffering)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    chunk = AudioChunk(samples=samples, sample_rate=16000, duration_sec=len(samples) / 16000.0)

    transcript = app.state.stt.transcribe(chunk, language="hi" if session.language_id == 1 else "en")
    await ws.send_json({"type": "transcript", "text": transcript})

    if transcript.strip():
        await _run_generation(ws, transcript, session, app)


async def _run_generation(ws: WebSocket, user_text: str, session: VoiceSession, app):
    """Run model generation + TTS and stream results."""
    adapter: VoiceAdapter = app.state.adapter
    if adapter is None or adapter.sentinel is None:
        await ws.send_json({"type": "error", "message": "Model not loaded"})
        return

    # Tokenize (caller must have set tokenizer on adapter)
    if adapter.tokenizer is None:
        await ws.send_json({"type": "error", "message": "Tokenizer not set"})
        return

    input_ids = adapter.tokenizer.encode(user_text, return_tensors="pt")
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    await ws.send_json({"type": "response.start"})

    generated_text = []
    for token_id in adapter.generate_stream(input_ids, session):
        piece = adapter.tokenizer.decode([token_id])
        generated_text.append(piece)
        await ws.send_json({"type": "response.token", "token": piece, "token_id": token_id})

    full_text = "".join(generated_text).strip()
    await ws.send_json({"type": "response.text_done", "text": full_text})

    # TTS
    if full_text:
        try:
            audio = app.state.tts.synthesize(full_text, language="hi")
            # Send as base64 PCM16
            pcm = (audio.samples * 32767).astype(np.int16).tobytes()
            b64 = base64.b64encode(pcm).decode("ascii")
            await ws.send_json({
                "type": "response.audio",
                "format": "pcm16",
                "sample_rate": audio.sample_rate,
                "data": b64,
            })
        except Exception as e:
            logger.warning("TTS failed: %s", e)
            await ws.send_json({"type": "response.audio_error", "message": str(e)})

    await ws.send_json({"type": "response.done"})


# Default factory for uvicorn
def create_app():
    return create_voice_app()
