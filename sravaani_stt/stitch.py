"""Join window texts into one transcript. Consecutive windows overlap only at forced
splits (split_overlap_s); the duplicated words at the seam are found with an LCS over
the last K tokens of window i and the first K of window i+1 and removed once."""
import re

K = 8


def _norm(tok):
    return re.sub(r"[^\w]", "", tok.lower())


def _lcs_trim(prev_tail, next_head):
    """Return how many leading tokens of next_head duplicate the end of prev_tail."""
    a = [_norm(t) for t in prev_tail]
    b = [_norm(t) for t in next_head]
    best = 0
    # longest suffix of a that equals a prefix of b, allowing one mismatch inside
    for n in range(min(len(a), len(b)), 0, -1):
        sa, sb = a[-n:], b[:n]
        mism = sum(x != y for x, y in zip(sa, sb))
        if mism <= (0 if n < 3 else 1):
            best = n
            break
    return best


def stitch(windows, texts):
    """windows: list of Window (with .forced); texts: list of str. Returns (transcript, trimmed_texts)."""
    out = []
    for i, (w, t) in enumerate(zip(windows, texts)):
        toks = t.split()
        if i > 0 and windows[i - 1].forced and out and toks:
            prev = out[-1].split()
            k = _lcs_trim(prev[-K:], toks[:K])
            toks = toks[k:]
        out.append(" ".join(toks))
    return " ".join(x for x in out if x).strip(), out
