"""ASRBackend protocol. Every backend decodes windows INDEPENDENTLY (no cross-window
context) and returns one row per window in order."""
from dataclasses import dataclass, field


@dataclass
class Row:
    text: str
    decode_s: float
    avg_logprob: float = None
    no_speech_prob: float = None
    extra: dict = field(default_factory=dict)


class ASRBackend:
    key = "base"
    hard_cap_s = None       # model's own audio limit in seconds; asserted per window
    needs_gpu = True
    need_gb = 8.0           # VRAM to ask pick_gpu for
    batch = 1

    def load(self):
        raise NotImplementedError

    def transcribe_windows(self, wav, windows, iso):
        """wav: float32 16 kHz; windows: list of (t0, t1); iso: language code. -> list[Row]"""
        raise NotImplementedError

    def check_caps(self, windows):
        if self.hard_cap_s is not None:
            for t0, t1 in windows:
                assert t1 - t0 <= self.hard_cap_s + 1e-6, f"window {t1-t0:.2f}s exceeds {self.key} cap {self.hard_cap_s}s"
