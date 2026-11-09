"""Score separated tracks against the ground-truth sources.

Reports permutation-invariant SI-SNR, the improvement over the untouched
mixture (SI-SNRi), and a window-by-window check that the speaker assignment
stays put for the whole recording.

Usage:
    python evaluate.py --mixture mixture.wav --refs refs --est output
"""
import argparse
import glob
import json
import os

import soundfile as sf
import torch
import torchaudio

from metrics import chunkwise_permutation, si_snr_improvement


def load(path, target_sr=None):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    sig = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    if target_sr and sr != target_sr:
        sig = torchaudio.functional.resample(sig, sr, target_sr)
        sr = target_sr
    return sig, sr


def stack_dir(pattern, target_sr):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched {pattern}")
    sigs, sr = [], target_sr
    for p in paths:
        sig, sr = load(p, target_sr)
        sigs.append(sig)
    length = min(s.shape[1] for s in sigs)
    return torch.cat([s[:, :length] for s in sigs], dim=0), sr, paths


def score_tensors(est, refs, mix, sr, window_seconds=4.0):
    """Score already-loaded tensors. est/refs are [n_sources, time], mix is [1, time].

    All three are trimmed to a common length first, so a model that returns a
    slightly different number of samples than the input does not break scoring.
    """
    length = min(est.shape[1], refs.shape[1], mix.shape[1])
    est, refs, mix = est[:, :length], refs[:, :length], mix[:, :length]

    result = si_snr_improvement(est, refs, mix[0])
    result["chunk_analysis"] = chunkwise_permutation(est, refs, sr, window_seconds)
    result["duration_s"] = length / sr
    result["sample_rate"] = sr
    return result


def evaluate(mixture_path, refs_dir, est_dir, window_seconds=4.0):
    est, sr, est_paths = stack_dir(os.path.join(est_dir, "speaker*.wav"), None)
    refs, _, ref_paths = stack_dir(os.path.join(refs_dir, "ref*.wav"), sr)
    mix, _ = load(mixture_path, sr)

    result = score_tensors(est, refs, mix, sr, window_seconds)
    result["est_files"] = est_paths
    result["ref_files"] = ref_paths
    return result


def main():
    parser = argparse.ArgumentParser(description="Score separated tracks against references.")
    parser.add_argument("--mixture", required=True, help="The mixture WAV that was separated.")
    parser.add_argument("--refs", required=True, help="Directory with ref1.wav / ref2.wav.")
    parser.add_argument("--est", required=True, help="Directory with speaker1.wav / speaker2.wav.")
    parser.add_argument("--window-seconds", type=float, default=4.0,
                        help="Window length for the identity-stability check.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Names for the references, in order, e.g. --labels near far")
    parser.add_argument("--json", default=None, help="Optional path to dump full results.")
    args = parser.parse_args()

    r = evaluate(args.mixture, args.refs, args.est, args.window_seconds)
    ca = r["chunk_analysis"]
    labels = args.labels or [f"ref{i + 1}" for i in range(len(r["si_snr_per_source"]))]
    if args.labels:
        r["labels"] = args.labels

    print(f"Duration            {r['duration_s']:.1f}s at {r['sample_rate']} Hz")
    print(f"Assignment          est->ref {r['perm']}")
    print(f"Mixture SI-SNR      {r['mixture_si_snr']:+.2f} dB  (do-nothing baseline)")
    print(f"Separated SI-SNR    {r['si_snr']:+.2f} dB")
    for name, val in zip(labels, r["si_snr_per_source"]):
        print(f"   {name:<16} {val:+.2f} dB")
    print(f"Improvement         {r['si_snri']:+.2f} dB")
    print(f"Identity stability  {ca['consistency'] * 100:.0f}% of {ca['n_windows']} windows agree, "
          f"{ca['n_flips']} flips")
    if ca["n_windows"]:
        print(f"Mean window SI-SNR  {ca['mean_window_si_snr']:+.2f} dB")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
