"""SraVaani-1.0 (ARTPARK-IISc) FastConformer-TDT TorchScript through the repo's HF wrapper.
fp16 on an explicitly selected GPU, fp32 on CPU (the wrapper upcasts to fp32 unless float16 is requested explicitly; bf16 is
unsupported), batch 8 sorted by duration, windows shorter than 0.1 s zero-padded (a 1-frame
input divides by zero), 16 kHz arrays so the built-in FFT resampler never runs. No language
argument exists: script-match downstream is its diagnostic. Encoder (batched TorchScript) and
greedy decoder (Python loop) are timed separately (D36)."""
import time
import numpy as np
from .base import ASRBackend, Row

SR = 16000
MIN_S = 0.1


class SraVaani(ASRBackend):
    needs_gpu = True
    need_gb = 6.0
    hard_cap_s = 40.0

    def __init__(self, path, device="cpu", batch_size=8):
        self.key, self.path = "sravaani", path
        self.device, self.batch = device, batch_size
        self.dtype = "float16" if device.startswith("cuda") else "float32"
        if self.batch < 1:
            raise ValueError("SRAVAANI_BATCH_SIZE must be positive")

    def _synchronize(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def load(self):
        import torch
        from transformers import AutoModel
        self.torch = torch
        t = time.time()
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model = AutoModel.from_pretrained(
            str(self.path), trust_remote_code=True, local_files_only=True, torch_dtype=dtype,
        )
        if getattr(model, "_req_dtype", None) != dtype:      # transformers 5 may rename the kwarg
            model._req_dtype = dtype
            model = model.to(dtype=dtype)
        model = model.to(self.device).eval()
        model._ensure_loaded()
        assert model._io_dtype == dtype, f"io dtype {model._io_dtype}: fp16 request was ignored"
        assert model._anchor.device == torch.device(self.device)
        self.model = model
        self.sp = model._get_tokenizer()
        self.pad_value = float(model._p["pad_value"])
        assert int(model.config.sample_rate) == SR
        self.load_s = time.time() - t
        print(f"[sravaani] loaded {self.dtype} on {self.device} in {self.load_s:.0f}s; blank {model.config.blank_id}, durations {model.config.durations}", flush=True)

    def transcribe_windows(self, wav, windows, iso):
        self.check_caps(windows)
        torch = self.torch
        items = []
        for i, (t0, t1) in enumerate(windows):
            a, b = int(round(t0 * SR)), int(round(t1 * SR))
            x = np.ascontiguousarray(wav[a:b], dtype=np.float32)
            if len(x) < int(MIN_S * SR):
                x = np.pad(x, (0, int(MIN_S * SR) - len(x)))
            items.append((i, x))
        order = sorted(items, key=lambda it: -len(it[1]))
        rows = [None] * len(windows)
        for s in range(0, len(order), self.batch):
            chunk = order[s:s + self.batch]
            with torch.no_grad():
                feats, lens = [], []
                for _, x in chunk:
                    w = torch.from_numpy(x).unsqueeze(0)
                    f, sl = self.model.extract_features(w, torch.tensor([w.shape[1]]))
                    feats.append(f[0]); lens.append(int(sl[0]))
                Fdim, Tmax = feats[0].shape[0], max(f.shape[1] for f in feats)
                xb = torch.full((len(feats), Fdim, Tmax), self.pad_value, dtype=feats[0].dtype, device=feats[0].device)
                for i, f in enumerate(feats):
                    xb[i, :, :f.shape[1]] = f
                flen = torch.tensor(lens, dtype=torch.long, device=xb.device)
                self._synchronize(); t = time.time()
                out = self.model.forward(xb, flen)
                self._synchronize(); enc_s = time.time() - t
                enc, enc_len = out.last_hidden_state, out.encoder_lengths
                dec = []
                for i in range(enc.size(0)):
                    t = time.time()
                    toks, _frames = self.model._greedy_one(enc[i:i + 1], int(enc_len[i].item()))
                    self._synchronize(); dec.append((toks, time.time() - t))
            total = sum(len(x) for _, x in chunk) or 1
            for (idx, x), (toks, dec_s) in zip(chunk, dec):
                text = self.sp.decode([int(k) for k in toks]) if toks else ""
                enc_share = enc_s * len(x) / total
                rows[idx] = Row(text=text.strip(), decode_s=enc_share + dec_s,
                                extra=dict(encoder_s=round(enc_share, 4), decoder_s=round(dec_s, 4), n_tokens=len(toks),
                                           batch_n=len(chunk), batch_elapsed_s=round(enc_s + sum(d for _, d in dec), 4)))
        return rows
