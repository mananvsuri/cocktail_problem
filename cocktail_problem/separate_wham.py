"""Enhanced separation using SepFormer-WHAM model.

The WHAM (WSJ0 Hipster Ambient Mixtures) model is trained on data with
environmental noise and reverberation, making it more robust to real-world
recordings than the standard WSJ0-2mix model.

Usage:
    python separate_wham.py --input recording.wav --outdir output
"""
import argparse
import json
import os
import resource
import time

import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation

MODEL_SOURCE = "speechbrain/sepformer-wham"
MODEL_SR = 8000  # WHAM model operates at 8 kHz
EPS = 1e-8


def load_mono(path):
    """Load an audio file and return a mono waveform tensor [1, time] and its sample rate."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [time, channels]
    mono = data.mean(axis=1)  # mix down to mono -> [time]
    sig = torch.from_numpy(mono).unsqueeze(0)  # [1, time]
    return sig, sr


def load_model(device):
    print(f"Loading {MODEL_SOURCE} model (trained with noise & reverberation)...")
    return SepformerSeparation.from_hparams(
        source=MODEL_SOURCE,
        savedir="pretrained_models/sepformer-wham",
        run_opts={"device": device},
    )


def separate_whole(model, sig):
    """Run the model over the entire signal at once. Returns [n_sources, time]."""
    with torch.no_grad():
        est = model.separate_batch(sig)  # [batch, time, n_sources]
    return est[0].permute(1, 0).cpu()  # -> [n_sources, time]


def _match_permutation(prev_tail, cur_head):
    """Pick the ordering of cur_head that best lines up with prev_tail."""
    n = prev_tail.shape[0]
    p = prev_tail - prev_tail.mean(dim=-1, keepdim=True)
    c = cur_head - cur_head.mean(dim=-1, keepdim=True)
    p = p / (p.norm(dim=-1, keepdim=True) + EPS)
    c = c / (c.norm(dim=-1, keepdim=True) + EPS)
    corr = p @ c.T  # [n_prev, n_cur]

    if n == 2:
        keep = float(corr[0, 0] + corr[1, 1])
        swap = float(corr[0, 1] + corr[1, 0])
        return ((1, 0), True) if swap > keep else ((0, 1), False)

    perm = [-1] * n
    used = set()
    order = sorted(((float(corr[i, j]), i, j) for i in range(n) for j in range(n)),
                   reverse=True)
    for _, i, j in order:
        if perm[i] == -1 and j not in used:
            perm[i], _ = j, used.add(j)
    return tuple(perm), tuple(perm) != tuple(range(n))


def separate_chunked(model, sig, sr, chunk_seconds, overlap_seconds):
    """Separate a long signal in overlapping windows, stitched with a cross-fade."""
    total = sig.shape[1]
    chunk = int(chunk_seconds * sr)
    overlap = int(overlap_seconds * sr)
    if overlap >= chunk:
        raise ValueError("overlap-seconds must be smaller than chunk-seconds")
    hop = chunk - overlap

    if total <= chunk:
        return separate_whole(model, sig), {"n_windows": 1, "n_swaps_corrected": 0}

    starts = list(range(0, max(total - overlap, 1), hop))
    out = None
    weight = torch.zeros(total)
    prev_tail = None
    swaps = 0

    for idx, start in enumerate(starts):
        end = min(start + chunk, total)
        piece = sig[:, start:end]
        pad = chunk - piece.shape[1]
        if pad > 0:
            piece = torch.cat([piece, torch.zeros(1, pad, device=piece.device)], dim=1)

        est = separate_whole(model, piece)
        est = est[:, : end - start]

        if out is None:
            out = torch.zeros(est.shape[0], total)

        if prev_tail is not None:
            head = est[:, : prev_tail.shape[1]]
            perm, swapped = _match_permutation(prev_tail, head)
            est = est[list(perm)]
            swaps += int(swapped)

        win = torch.ones(est.shape[1])
        if idx > 0 and overlap > 0:
            n = min(overlap, est.shape[1])
            win[:n] = torch.linspace(0, 1, n)
        if idx < len(starts) - 1 and overlap > 0:
            n = min(overlap, est.shape[1])
            win[-n:] = torch.minimum(win[-n:], torch.linspace(1, 0, n))

        out[:, start:end] += est * win
        weight[start:end] += win
        prev_tail = est[:, -overlap:] if est.shape[1] >= overlap else est

    out = out / torch.clamp(weight, min=EPS)
    return out, {"n_windows": len(starts), "n_swaps_corrected": swaps}


def main():
    parser = argparse.ArgumentParser(description="Separate with WHAM model (noise-robust).")
    parser.add_argument("--input", required=True, help="Input mixture WAV file.")
    parser.add_argument("--outdir", default="output_wham", help="Directory for separated tracks.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                        help="Compute device.")
    parser.add_argument("--chunk-seconds", type=float, default=10.0,
                        help="Process in windows of this length. 0 means whole file at once.")
    parser.add_argument("--overlap-seconds", type=float, default=2.0,
                        help="Overlap between windows.")
    parser.add_argument("--report", default=None,
                        help="Optional JSON file for timing and window statistics.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model = load_model(args.device)

    sig, sr = load_mono(args.input)
    duration = sig.shape[1] / sr
    if sr != MODEL_SR:
        sig = torchaudio.functional.resample(sig, sr, MODEL_SR)
    sig = sig.to(args.device)

    print(f"Separating {duration:.1f}s of audio"
          + (f" in {args.chunk_seconds:.0f}s windows..." if args.chunk_seconds else " in one pass..."))
    t0 = time.perf_counter()
    if args.chunk_seconds and args.chunk_seconds > 0:
        est_sources, info = separate_chunked(model, sig, MODEL_SR,
                                             args.chunk_seconds, args.overlap_seconds)
    else:
        est_sources, info = separate_whole(model, sig), {"n_windows": 1, "n_swaps_corrected": 0}
    elapsed = time.perf_counter() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    paths = []
    for i in range(est_sources.shape[0]):
        track = est_sources[i].unsqueeze(0)
        peak = track.abs().max()
        if peak > 0:
            track = track / peak
        out_path = os.path.join(args.outdir, f"speaker{i + 1}.wav")
        sf.write(out_path, track.squeeze(0).numpy(), MODEL_SR)
        paths.append(out_path)
        print(f"Saved {out_path}")

    stats = {
        "model": MODEL_SOURCE,
        "input": args.input,
        "duration_s": duration,
        "device": args.device,
        "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds if args.chunk_seconds else 0.0,
        "elapsed_s": elapsed,
        "real_time_factor": elapsed / duration if duration else float("nan"),
        "peak_rss_mb": peak_mb,
        "outputs": paths,
        **info,
    }
    print(f"Took {elapsed:.1f}s ({stats['real_time_factor']:.2f}x real time), "
          f"peak RSS {peak_mb:.0f} MB, windows={info['n_windows']}, "
          f"swaps corrected={info['n_swaps_corrected']}")

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"Wrote report to {args.report}")

    print("Done. Listen to the files in", args.outdir)


if __name__ == "__main__":
    main()
