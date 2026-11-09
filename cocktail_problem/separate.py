"""Separate two overlapping voices from a single mono recording.

Uses the pre-trained SpeechBrain SepFormer model (trained on WSJ0-2mix, 8 kHz)
to split a single-channel mixture into two speaker tracks.

For long recordings, running the whole file through the transformer at once gets
expensive fast (attention cost grows super-linearly with length). Pass
--chunk-seconds to process the file in overlapping windows instead. Windows are
stitched with a cross-fade, and each new window's outputs are re-ordered to match
the previous one so a speaker does not jump between tracks mid-file.

Usage:
    python separate.py --input mixture.wav --outdir output
    python separate.py --input long_mixture.wav --outdir output \
        --chunk-seconds 10 --overlap-seconds 2
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

MODEL_SOURCE = "speechbrain/sepformer-wsj02mix"
MODEL_SR = 8000  # the SepFormer model operates at 8 kHz
EPS = 1e-8


def load_mono(path):
    """Load an audio file and return a mono waveform tensor [1, time] and its sample rate."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [time, channels]
    mono = data.mean(axis=1)  # mix down to mono -> [time]
    sig = torch.from_numpy(mono).unsqueeze(0)  # [1, time]
    return sig, sr


def load_model(device):
    return SepformerSeparation.from_hparams(
        source=MODEL_SOURCE,
        savedir="pretrained_models/sepformer-wsj02mix",
        run_opts={"device": device},
    )


def separate_whole(model, sig):
    """Run the model over the entire signal at once. Returns [n_sources, time]."""
    with torch.no_grad():
        est = model.separate_batch(sig)  # [batch, time, n_sources]
    return est[0].permute(1, 0).cpu()  # -> [n_sources, time]


def _match_permutation(prev_tail, cur_head):
    """Pick the ordering of cur_head that best lines up with prev_tail.

    Both are [n_sources, overlap]. Correlation is done on the shared region, so
    if output 1 of this window is the speaker that was on output 2 of the last
    window, we detect the swap and undo it. Returns (perm, swapped_flag).
    """
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

    # Generic fallback: greedy assignment on descending correlation.
    perm = [-1] * n
    used = set()
    order = sorted(((float(corr[i, j]), i, j) for i in range(n) for j in range(n)),
                   reverse=True)
    for _, i, j in order:
        if perm[i] == -1 and j not in used:
            perm[i], _ = j, used.add(j)
    return tuple(perm), tuple(perm) != tuple(range(n))


def separate_chunked(model, sig, sr, chunk_seconds, overlap_seconds):
    """Separate a long signal in overlapping windows, stitched with a cross-fade.

    Returns ([n_sources, time], info dict).
    """
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
        if pad > 0:  # zero-pad the final short window for consistent model behaviour
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

        # Cross-fade window: ramp in where we overlap the previous window,
        # ramp out where the next window will overlap us.
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
    parser = argparse.ArgumentParser(description="Separate two voices from a mono mixture.")
    parser.add_argument("--input", required=True, help="Input mixture WAV file.")
    parser.add_argument("--outdir", default="output", help="Directory for separated tracks.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                        help="Compute device. CPU is the most reliable; mps uses the Apple GPU.")
    parser.add_argument("--chunk-seconds", type=float, default=0.0,
                        help="Process in windows of this length. 0 means whole file at once.")
    parser.add_argument("--overlap-seconds", type=float, default=2.0,
                        help="Overlap between windows, used for stitching and re-ordering.")
    parser.add_argument("--report", default=None,
                        help="Optional JSON file for timing and window statistics.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading SepFormer model ({MODEL_SOURCE})... (first run downloads it)")
    model = load_model(args.device)

    # Load mixture, make mono, resample to the model's 8 kHz.
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
            track = track / peak  # normalize to avoid clipping / near-silent output
        out_path = os.path.join(args.outdir, f"speaker{i + 1}.wav")
        sf.write(out_path, track.squeeze(0).numpy(), MODEL_SR)
        paths.append(out_path)
        print(f"Saved {out_path}")

    stats = {
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
