"""SraVaani-1.0 with the vad-asr-bench Silero-v6/29cd002f pipeline.

The segmenter and stitcher are copied from benchmark commit 0bcbb1e9f14922c8.
The backend keeps its batched greedy decoding, with explicit device selection.
There is no language conditioning or fallback ASR model.
"""
import logging
from pathlib import Path

import numpy as np

from .backend import SraVaani
from .segment import Policy, segment
from .stitch import stitch

logger = logging.getLogger("stt")
SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 512
POLICY = Policy(
    theta_on=0.5, min_silence_s=0.2, pad_s=0.25,
    max_gap_s=0.4, max_window_s=10.0,
)


class Pipeline:
    """One cached VAD and ASR model; the caller serializes access."""

    def __init__(self, model_path, device="cpu", batch_size=8):
        from faster_whisper.vad import get_vad_model

        self.model_path = Path(model_path).expanduser()
        self.device = device
        self.batch_size = batch_size
        self.vad = get_vad_model()
        self.backend = None

    def load(self):
        if self.backend is None:
            required = (
                "config.json", "configuration_sravaani.py", "modeling_sravaani.py",
                "model-asr.fp16.ts", "preproc.pt", "tokenizer.model",
            )
            missing = [name for name in required if not (self.model_path / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Incomplete local SraVaani model at {self.model_path}: {', '.join(missing)}"
                )
            backend = SraVaani(self.model_path, self.device, self.batch_size)
            backend.load()
            self.backend = backend
            logger.info(
                "[STT] Loaded SraVaani-1.0 path=%s device=%s dtype=%s policy=%s",
                self.model_path, self.device, backend.dtype, POLICY.key(),
            )
        return self.backend

    def transcribe(self, wav):
        """Decode mono float32 audio at 16 kHz; return None for no speech."""
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError("SraVaani expects mono audio")
        if not len(wav):
            return None
        padded = np.pad(wav, (0, (-len(wav)) % VAD_FRAME_SAMPLES))
        probabilities = np.asarray(self.vad(padded), dtype=np.float32).reshape(-1)
        windows, _ = segment(
            probabilities, VAD_FRAME_SAMPLES / SAMPLE_RATE,
            POLICY, len(wav) / SAMPLE_RATE,
        )
        if not windows:
            return None
        rows = self.load().transcribe_windows(
            wav, [(w.t0, w.t1) for w in windows], iso=None,
        )
        text, _ = stitch(windows, [row.text for row in rows])
        return text or None
