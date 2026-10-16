"""STT + TTS abstraction layer for Indo-Voice-3.

Design:
- Pure interfaces, no hard-coded providers
- Config-driven backend selection
- External STT/TTS by default (production practical)
- Hook points for future end-to-end audio heads
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger("indo_voice.stt_tts")


@dataclass
class AudioChunk:
    """Normalized audio payload."""
    samples: np.ndarray          # float32, mono, [-1, 1]
    sample_rate: int
    duration_sec: float


# ---------------------------------------------------------------------------
# STT Interface
# ---------------------------------------------------------------------------

class STTBackend(ABC):
    @abstractmethod
    def transcribe(self, audio: AudioChunk, language: str = "hi") -> str:
        """Return text transcript."""
        ...

    @abstractmethod
    def transcribe_streaming_start(self, language: str = "hi") -> Any:
        """Optional: return a streaming session handle."""
        ...


class WhisperSTT(STTBackend):
    """OpenAI Whisper / faster-whisper backend (optional dependency)."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self.device)
            logger.info("Loaded faster-whisper model=%s", self.model_size)
        except ImportError:
            try:
                import whisper
                self._model = whisper.load_model(self.model_size, device=self.device)
                self._is_openai = True
                logger.info("Loaded openai-whisper model=%s", self.model_size)
            except ImportError as e:
                raise RuntimeError(
                    "Neither faster-whisper nor openai-whisper installed. "
                    "pip install faster-whisper"
                ) from e

    def transcribe(self, audio: AudioChunk, language: str = "hi") -> str:
        self._load()
        # faster-whisper path
        if hasattr(self._model, "transcribe") and not getattr(self, "_is_openai", False):
            segments, _ = self._model.transcribe(
                audio.samples,
                language=language if language != "hinglish" else "hi",
                beam_size=1,
            )
            return " ".join(s.text.strip() for s in segments).strip()
        # openai-whisper path
        result = self._model.transcribe(
            audio.samples,
            language=language if language != "hinglish" else "hi",
            fp16=False,
        )
        return result.get("text", "").strip()

    def transcribe_streaming_start(self, language: str = "hi") -> Any:
        return None  # not implemented for base Whisper


class DummySTT(STTBackend):
    """Fallback when no STT backend is available (dev only)."""

    def transcribe(self, audio: AudioChunk, language: str = "hi") -> str:
        logger.warning("DummySTT used — returning empty transcript")
        return ""

    def transcribe_streaming_start(self, language: str = "hi") -> Any:
        return None


# ---------------------------------------------------------------------------
# TTS Interface
# ---------------------------------------------------------------------------

class TTSBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, language: str = "hi", voice: str = "default") -> AudioChunk:
        """Return waveform for the given text."""
        ...


class CoquiTTS(TTSBackend):
    """Coqui TTS backend (optional)."""

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.model_name = model_name
        self._tts = None

    def _load(self):
        if self._tts is not None:
            return
        try:
            from TTS.api import TTS
            self._tts = TTS(self.model_name)
            logger.info("Loaded Coqui TTS: %s", self.model_name)
        except ImportError as e:
            raise RuntimeError("Coqui TTS not installed. pip install TTS") from e

    def synthesize(self, text: str, language: str = "hi", voice: str = "default") -> AudioChunk:
        self._load()
        # XTTS path
        wav = self._tts.tts(text=text, language=language)
        samples = np.asarray(wav, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        sr = getattr(self._tts, "synthesizer", None)
        sample_rate = 22050
        if sr and hasattr(sr, "output_sample_rate"):
            sample_rate = sr.output_sample_rate
        return AudioChunk(
            samples=samples,
            sample_rate=sample_rate,
            duration_sec=len(samples) / sample_rate,
        )


class DummyTTS(TTSBackend):
    """Silence generator (dev fallback)."""

    def synthesize(self, text: str, language: str = "hi", voice: str = "default") -> AudioChunk:
        sr = 16000
        duration = max(0.5, min(10.0, len(text) * 0.06))
        samples = np.zeros(int(sr * duration), dtype=np.float32)
        return AudioChunk(samples=samples, sample_rate=sr, duration_sec=duration)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_stt(provider: str = "auto", **kwargs) -> STTBackend:
    provider = (provider or "auto").lower()
    if provider in ("whisper", "faster-whisper", "auto"):
        try:
            return WhisperSTT(**kwargs)
        except Exception as e:
            logger.warning("WhisperSTT unavailable: %s — falling back to DummySTT", e)
            return DummySTT()
    if provider == "dummy":
        return DummySTT()
    raise ValueError(f"Unknown STT provider: {provider}")


def create_tts(provider: str = "auto", **kwargs) -> TTSBackend:
    provider = (provider or "auto").lower()
    if provider in ("coqui", "xtts", "auto"):
        try:
            return CoquiTTS(**kwargs)
        except Exception as e:
            logger.warning("CoquiTTS unavailable: %s — falling back to DummyTTS", e)
            return DummyTTS()
    if provider == "dummy":
        return DummyTTS()
    raise ValueError(f"Unknown TTS provider: {provider}")


def waveform_to_tensor(chunk: AudioChunk, target_sr: int = 16000) -> Tensor:
    """Convert AudioChunk → (1, T) float tensor at target_sr."""
    samples = chunk.samples
    if chunk.sample_rate != target_sr:
        # simple linear resample (replace with torchaudio if available)
        ratio = target_sr / chunk.sample_rate
        new_len = int(len(samples) * ratio)
        x = np.linspace(0, len(samples) - 1, new_len)
        samples = np.interp(x, np.arange(len(samples)), samples).astype(np.float32)
    return torch.from_numpy(samples).unsqueeze(0)
