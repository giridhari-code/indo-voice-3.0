from sentinel.voice.adapter import VoiceAdapter, VoiceSession
from sentinel.voice.stt_tts import create_stt, create_tts, AudioChunk

__all__ = [
    "VoiceAdapter",
    "VoiceSession",
    "create_stt",
    "create_tts",
    "AudioChunk",
]

def __getattr__(name):
    if name == "create_voice_app":
        from sentinel.voice.server import create_voice_app
        return create_voice_app
    raise AttributeError(name)
