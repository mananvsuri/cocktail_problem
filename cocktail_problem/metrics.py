"""Objective metrics for two-speaker separation.

Everything here works on plain torch tensors shaped [n_sources, time] and is
permutation invariant, because a separation model has no idea which output
channel "should" be which speaker.

Metrics provided:
  * SI-SNR      scale-invariant signal-to-noise ratio (dB, higher is better)
  * SI-SNRi     improvement of SI-SNR over the unprocessed mixture
  * per-chunk permutation analysis, which is how you catch a long recording
    where the model silently swaps the two speakers halfway through.
"""
from itertools import permutations

import torch

EPS = 1e-8


def _zero_mean(x):
    return x - x.mean(dim=-1, keepdim=True)


def si_snr(est, ref):
    """Scale-invariant SNR in dB for a single pair of 1-D signals."""
    est = _zero_mean(est)
    ref = _zero_mean(ref)
    alpha = torch.sum(est * ref, dim=-1, keepdim=True) / (
        torch.sum(ref * ref, dim=-1, keepdim=True) + EPS
    )
    target = alpha * ref
    noise = est - target
    ratio = torch.sum(target**2, dim=-1) / (torch.sum(noise**2, dim=-1) + EPS)
    return 10 * torch.log10(ratio + EPS)


def best_permutation(est, ref):
    """Find the output->reference assignment maximising mean SI-SNR.

    Args:
        est: [n_sources, time] estimated sources.
        ref: [n_sources, time] reference (ground-truth) sources.

    Returns:
        (perm, mean_si_snr, per_source_si_snr) where perm[i] is the index of the
        estimate assigned to reference i.
    """
    n = ref.shape[0]
    best = (None, -float("inf"), None)
    for perm in permutations(range(est.shape[0]), n):
        scores = torch.stack([si_snr(est[perm[i]], ref[i]) for i in range(n)])
        mean = float(scores.mean())
        if mean > best[1]:
            best = (perm, mean, scores)
    return best


def si_snr_improvement(est, ref, mixture):
    """SI-SNR of the best assignment, the mixture baseline, and the improvement.

    The baseline is the mixture scored against each reference: that is what you
    get for free by doing nothing at all, so it is the number to beat.
    """
    perm, est_score, per_source = best_permutation(est, ref)
    base = torch.stack([si_snr(mixture, ref[i]) for i in range(ref.shape[0])])
    base_score = float(base.mean())
    return {
        "perm": perm,
        "si_snr": est_score,
        "si_snr_per_source": [float(v) for v in per_source],
        "mixture_si_snr": base_score,
        "si_snri": est_score - base_score,
    }


def chunkwise_permutation(est, ref, sr, window_seconds=4.0):
    """Check whether the speaker assignment stays stable along the recording.

    Scores each window independently and reports which permutation wins locally.
    If a long file starts with speaker A on output 1 and ends with speaker A on
    output 2, the global SI-SNR collapses while every individual window still
    looks fine -- this is the metric that exposes it.

    Returns a dict with the dominant permutation, the fraction of windows that
    agree with it, and the per-window scores.
    """
    win = int(window_seconds * sr)
    if win <= 0 or est.shape[-1] < win:
        win = est.shape[-1]

    windows = []
    for start in range(0, est.shape[-1] - win + 1, win):
        sl = slice(start, start + win)
        e, r = est[:, sl], ref[:, sl]
        if float(r.abs().max()) < EPS:
            continue  # silent window, nothing to attribute
        perm, score, _ = best_permutation(e, r)
        windows.append({"start_s": start / sr, "perm": perm, "si_snr": score})

    if not windows:
        return {"n_windows": 0, "dominant_perm": None, "consistency": float("nan"),
                "n_flips": 0, "windows": []}

    counts = {}
    for w in windows:
        counts[w["perm"]] = counts.get(w["perm"], 0) + 1
    dominant = max(counts, key=counts.get)
    agree = counts[dominant] / len(windows)
    flips = sum(1 for a, b in zip(windows, windows[1:]) if a["perm"] != b["perm"])

    return {
        "n_windows": len(windows),
        "dominant_perm": dominant,
        "consistency": agree,
        "n_flips": flips,
        "mean_window_si_snr": sum(w["si_snr"] for w in windows) / len(windows),
        "windows": windows,
    }
