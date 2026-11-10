"""SegmentPolicy: per-frame speech series -> speech regions -> ASR windows.

    hysteresis -> drop short -> merge close -> pad -> pack -> force-split -> min-window

Windows are contiguous wall-clock slices [t0, t1). They keep short internal
silences (honest timestamps) but never span a pause longer than max_gap_s and
never exceed max_window_s. A region longer than max_window_s is force-split at
its quietest instant with a small overlap; stitch.py removes the duplicate words.
Naive pipelines skip force-split and silently truncate at the model's limit.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class Policy:
    theta_on: float = 0.5
    theta_off: float = None          # None -> theta_on - 0.15
    min_speech_s: float = 0.15
    min_silence_s: float = 0.4
    pad_s: float = 0.1
    max_window_s: float = 28.0
    max_gap_s: float = 0.8
    min_window_s: float = 1.0
    split_overlap_s: float = 0.5

    @property
    def off(self):
        return self.theta_on - 0.15 if self.theta_off is None else self.theta_off

    def key(self):
        d = asdict(self); d["theta_off"] = self.off
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]

    @staticmethod
    def fixed(max_window_s=28.0):
        """The no-VAD control: contiguous max_window_s slices, no overlap, no padding."""
        return Policy(theta_on=0.5, theta_off=0.5, min_speech_s=0.0, min_silence_s=0.0, pad_s=0.0,
                      max_window_s=max_window_s, max_gap_s=float("inf"), min_window_s=0.0, split_overlap_s=0.0)

    @staticmethod
    def binary(**kw):
        """For 0/1 backends: both thresholds at 0.5."""
        return Policy(theta_on=0.5, theta_off=0.5, **kw)


@dataclass
class Window:
    t0: float
    t1: float
    forced: bool = False
    n_regions: int = 1
    silence_inside_s: float = 0.0


def hysteresis(p, hop_s, theta_on, theta_off, min_silence_s):
    """Two-threshold state machine -> [(t0, t1)]. Between the thresholds the state is kept.
    A low run must last >= min_silence_s to close a region; a frame >= theta_on cancels it."""
    regions, speaking, start, tentative = [], False, 0.0, None
    for i, pi in enumerate(p):
        t = i * hop_s
        if not speaking:
            if pi >= theta_on:
                speaking, start, tentative = True, t, None
        else:
            if pi < theta_off:
                if tentative is None:
                    tentative = t
                elif t - tentative >= min_silence_s:
                    regions.append((start, tentative)); speaking, tentative = False, None
            elif pi >= theta_on:
                tentative = None
    if speaking:
        regions.append((start, tentative if tentative is not None else len(p) * hop_s))
    return regions


def drop_short(regions, min_speech_s):
    return [(a, b) for a, b in regions if b - a >= min_speech_s]


def merge_close(regions, min_silence_s):
    out = []
    for a, b in regions:
        if out and a - out[-1][1] < min_silence_s:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def pad(regions, pad_s, total_s):
    """Widen each region by pad_s, clipped to the file; split the gap evenly if two would touch."""
    out = []
    for i, (a, b) in enumerate(regions):
        a2, b2 = max(0.0, a - pad_s), min(total_s, b + pad_s)
        if i > 0 and a2 < out[-1][1]:
            mid = (regions[i - 1][1] + a) / 2.0
            out[-1] = (out[-1][0], mid); a2 = mid
        out.append((a2, b2))
    return out


def pack(regions, max_window_s, max_gap_s):
    groups, cur = [], []
    for r in regions:
        if cur and (r[1] - cur[0][0] <= max_window_s) and (r[0] - cur[-1][1] <= max_gap_s):
            cur.append(r)
        else:
            if cur:
                groups.append(cur)
            cur = [r]
    if cur:
        groups.append(cur)
    return groups


def _quietest(p, hop_s, lo, hi, probabilistic):
    """Cut point in [lo, hi): argmin p for probabilistic series; midpoint of the longest
    low run for 0/1 series; hi if no low frame exists."""
    i0, i1 = int(lo / hop_s), max(int(lo / hop_s) + 1, int(hi / hop_s))
    seg = p[i0:i1]
    if len(seg) == 0:
        return hi
    if probabilistic:
        # the quietest frame, and among near-ties (within 0.05) the LATEST one, so flat
        # speech is cut at the limit (full windows) while a real dip is still preferred
        thr = float(seg.min()) + 0.05
        return (i0 + int(np.where(seg <= thr)[0][-1])) * hop_s
    low = seg < 0.5
    if not low.any():
        return hi
    best, cur_start, best_len = None, None, 0
    for j, v in enumerate(list(low) + [False]):
        if v and cur_start is None:
            cur_start = j
        elif not v and cur_start is not None:
            if j - cur_start > best_len:
                best_len, best = j - cur_start, (cur_start + j) / 2.0
            cur_start = None
    return (i0 + best) * hop_s


def force_split(groups, p, hop_s, pol, probabilistic):
    out = []
    for g in groups:
        t0, t1 = g[0][0], g[-1][1]
        n_reg = len(g)
        speech = sum(b - a for a, b in g)
        if t1 - t0 <= pol.max_window_s + 1e-9:
            out.append(Window(t0, t1, False, n_reg, max(0.0, (t1 - t0) - speech)))
            continue
        start = t0
        while t1 - start > pol.max_window_s + 1e-9:
            lo = start + 0.6 * pol.max_window_s
            hi = start + pol.max_window_s - pol.split_overlap_s
            cut = _quietest(p, hop_s, lo, hi, probabilistic)
            cut = min(max(cut, lo), hi)
            out.append(Window(start, min(cut + pol.split_overlap_s, t1), True, n_reg, 0.0))
            start = cut
        out.append(Window(start, t1, True, n_reg, 0.0))
    return out


def min_window_pass(windows, pol):
    out = []
    for w in windows:
        if out and (w.t1 - w.t0) < pol.min_window_s and not w.forced and not out[-1].forced \
                and (w.t1 - out[-1].t0) <= pol.max_window_s and (w.t0 - out[-1].t1) <= pol.max_gap_s:
            prev = out[-1]
            out[-1] = Window(prev.t0, w.t1, False, prev.n_regions + w.n_regions,
                             prev.silence_inside_s + (w.t0 - prev.t1) + w.silence_inside_s)
        else:
            out.append(w)
    return out


def segment(p, hop_s, pol, total_s, probabilistic=True):
    """Full policy. Returns (windows, regions)."""
    p = np.asarray(p, dtype=np.float32)
    regions = hysteresis(p, hop_s, pol.theta_on, pol.off, pol.min_silence_s)
    regions = [(a, min(b, total_s)) for a, b in regions if a < total_s]
    regions = drop_short(regions, pol.min_speech_s)
    regions = merge_close(regions, pol.min_silence_s)
    regions = pad(regions, pol.pad_s, total_s)
    groups = pack(regions, pol.max_window_s, pol.max_gap_s)
    windows = force_split(groups, p, hop_s, pol, probabilistic)
    windows = min_window_pass(windows, pol)
    # invariants
    for w in windows:
        assert w.t1 - w.t0 <= pol.max_window_s + 1e-6, f"window {w} exceeds max_window_s"
        assert w.t0 >= 0 and w.t1 <= total_s + 1e-6 and w.t1 > w.t0
    for a, b in zip(windows, windows[1:]):
        assert b.t0 >= a.t0
        if not a.forced:
            assert b.t0 >= a.t1 - 1e-6, "non-forced windows must not overlap"
    return windows, regions


def stats(windows, regions, total_s):
    d = [w.t1 - w.t0 for w in windows]
    covered = sum(d)
    return dict(n_windows=len(windows), n_regions=len(regions),
                mean_window_s=float(np.mean(d)) if d else 0.0, max_window_s=max(d) if d else 0.0,
                pct_forced=100.0 * sum(w.forced for w in windows) / max(1, len(windows)),
                pct_over_30s=100.0 * sum(x > 30 for x in d) / max(1, len(d)),
                silence_removed_ratio=1.0 - covered / total_s if total_s else 0.0,
                speech_s=sum(b - a for a, b in regions))
