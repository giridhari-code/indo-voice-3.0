"""SENTINEL v2 + Indo-Voice-3 extensions.

Core: two-speed memory architecture (AERIS_64).
Voice: AudioEncoder, ModalityProjector, HindiRouter, VoiceAdapter.
"""

from .config import (
    AblationConfig,
    ModelConfig,
    SENTINELConfig,
    NeuroFieldConfig,
    SafetyConfig,
    load_config,
    IndoVoiceConfig,
    AudioConfig,
    VoiceConfig,
)
from .model import ModelOutput, SENTINEL, NeuroField

__version__ = "2.5.1-indo-voice"

__all__ = [
    "SENTINEL",
    "NeuroField",
    "ModelOutput",
    "SENTINELConfig",
    "NeuroFieldConfig",
    "ModelConfig",
    "SafetyConfig",
    "AblationConfig",
    "load_config",
    "IndoVoiceConfig",
    "AudioConfig",
    "VoiceConfig",
    "__version__",
]

# Optional training (may not be present in slim package)
try:
    from .training import Trainer  # type: ignore
    __all__.append("Trainer")
except ImportError:
    pass
