"""Local SraVaani-1.0 transcription using the benchmark's VAD/window pipeline.

Configuration:
  SRAVAANI_MODEL_PATH  Local model snapshot (default /home/models/SraVaani-1.0).
  SRAVAANI_BATCH_SIZE  Windows per batch (default 8).
  STT_DEVICE          cpu (default), cuda, or auto. CUDA requires an explicit pin.
  STT_DEVICE_INDEX    Logical CUDA index, then legacy WHISPER_DEVICE_INDEX,
                      then GPU_DEDICATED_INDEX. One visible GPU implies index 0.

The language argument is retained for API/Telegram compatibility. SraVaani
recognizes speech without a language hint. No Whisper ASR model is loaded;
faster-whisper supplies only the PyAV decoder and bundled Silero-v6 VAD.
"""
import asyncio
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from utils import get_logger

logger = get_logger("stt")
_MODEL_PATH = Path(os.getenv("SRAVAANI_MODEL_PATH", "/home/models/SraVaani-1.0"))
_BATCH_SIZE = int(os.getenv("SRAVAANI_BATCH_SIZE", "8"))
_DEVICE = os.getenv("STT_DEVICE", "cpu").strip().lower()
_DEVICE_INDEX = os.getenv(
    "STT_DEVICE_INDEX",
    os.getenv("WHISPER_DEVICE_INDEX", os.getenv("GPU_DEDICATED_INDEX", "")),
).strip()
_EXTENSIONS = {".ogg", ".oga", ".opus", ".webm", ".mp3", ".m4a", ".mp4", ".wav", ".flac"}

# Both the model cache and inference are protected. VAD/ASR objects are reused
# across executor threads, and concurrent cold requests must load only once.
_lock = threading.Lock()
_pipeline = None


def _resolve_device() -> str:
    if _DEVICE not in {"cpu", "cuda", "auto"}:
        raise ValueError("STT_DEVICE must be cpu, cuda, or auto")
    if _DEVICE == "cpu":
        return "cpu"
    index = _DEVICE_INDEX
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not index and visible and visible != "-1" and "," not in visible:
        index = "0"
    if not index:
        logger.warning("[STT] No GPU pin configured; using CPU for SraVaani")
        return "cpu"
    if not index.isdecimal():
        raise ValueError("STT_DEVICE_INDEX must be a nonnegative logical CUDA index")
    import torch

    if not torch.cuda.is_available():
        if _DEVICE == "auto":
            return "cpu"
        raise RuntimeError("STT_DEVICE=cuda but CUDA is unavailable")
    if int(index) >= torch.cuda.device_count():
        raise ValueError(f"STT_DEVICE_INDEX={index} is outside the visible CUDA devices")
    return f"cuda:{int(index)}"


def _get_pipeline():
    """Called only with _lock held; defer ASR weights until speech is present."""
    global _pipeline
    if _pipeline is None:
        from sravaani_stt import Pipeline

        _pipeline = Pipeline(_MODEL_PATH, _resolve_device(), _BATCH_SIZE)
    return _pipeline


def _transcribe_sync(audio_path: str, language: Optional[str] = None) -> Optional[str]:
    from faster_whisper.audio import decode_audio

    # PyAV handles browser/Telegram containers and resamples stereo or other
    # sample rates to the benchmark's mono 16 kHz input.
    wav = decode_audio(audio_path, sampling_rate=16000)
    with _lock:
        return _get_pipeline().transcribe(wav)


def _transcribe_bytes(file_bytes: bytes, filename: Optional[str]) -> Optional[str]:
    suffix = Path(filename).suffix.lower() if filename else ""
    if suffix not in _EXTENSIONS:
        suffix = ".ogg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        return _transcribe_sync(tmp_path)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def transcribe_voice(
    file_bytes: bytes,
    language: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[str]:
    """Audio bytes -> transcript; language is metadata, not an ASR hint.

    Preserve the existing bot contract: return None for silence or failure.
    Temporary-file ownership stays in the worker, so cancelling an HTTP request
    cannot delete a file while the decoder is still reading it.
    """
    if not file_bytes:
        return None
    try:
        return await asyncio.to_thread(_transcribe_bytes, bytes(file_bytes), filename)
    except Exception:
        logger.exception("[STT] SraVaani transcription failed")
        return None


def _warmup_sync():
    with _lock:
        _get_pipeline().load()


async def warmup_stt() -> None:
    """Load the same cached SraVaani pipeline used by both voice transports."""
    await asyncio.to_thread(_warmup_sync)
