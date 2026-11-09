"""Sweep the separation pipeline over recording length and talker distance.

The question this answers: when one person is close to the mic and the other is
across the room, how well does each voice come out, and does it get worse as the
recording gets longer?

Two axes:

  duration   how long the mixture is. SepFormer was trained on ~4 s segments, so
             it is worth checking 30 s, 60 s, 10 minutes.
  distance   metres from the mic for each talker, given as near,far pairs. A
             distant talker is quieter, delayed, and heavily reverberant (see
             room.py), and reverberation is the part that really hurts.

Each condition runs through the real command-line tools in a fresh subprocess, so
reported time and peak memory belong to that condition alone.

Usage:
    python run_experiments.py --durations 30 60 --distances 0.5,0.5 0.5,2 0.5,4
    python run_experiments.py --durations 600 --distances 0.5,4 --rt60 0 0.4 0.8
"""
import argparse
import csv
import json
import os
import subprocess
import sys

PY = sys.executable
NEAR_VOICE = "Samantha"
FAR_VOICE = "Daniel"


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(" ".join(cmd))
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"Command failed: {cmd[1]}")
    return proc.stdout


def ensure_clip(voice, seconds, seed, clips_dir, cache):
    """Render a clip once per (voice, seed), long enough for every duration."""
    key = (voice, seed)
    if key in cache:
        return cache[key]
    path = os.path.join(clips_dir, f"{voice}_seed{seed}_{int(seconds)}s.wav")
    if not os.path.exists(path):
        print(f"  rendering {int(seconds)}s clip for {voice}")
        run([PY, "make_clips.py", "--voice", voice, "--seconds", str(seconds),
             "--seed", str(seed), "--out", path])
    cache[key] = path
    return path


def one_condition(seconds, near, far, rt60, chunk_s, overlap_s, outdir, clip_a, clip_b):
    tag = f"{int(seconds)}s_near{near}m_far{far}m_rt{rt60}"
    run_dir = os.path.join(outdir, tag)
    os.makedirs(run_dir, exist_ok=True)
    mixture = os.path.join(run_dir, "mixture.wav")
    refs = os.path.join(run_dir, "refs")
    est = os.path.join(run_dir, "separated")
    room_json = os.path.join(run_dir, "room.json")
    sep_json = os.path.join(run_dir, "timing.json")
    eval_json = os.path.join(run_dir, "results.json")

    run([PY, "make_mixture.py", "--a", clip_a, "--b", clip_b,
         "--seconds", str(seconds), "--distance-a", str(near), "--distance-b", str(far),
         "--rt60", str(rt60), "--out", mixture, "--save-refs", refs,
         "--info-json", room_json])

    run([PY, "separate.py", "--input", mixture, "--outdir", est,
         "--chunk-seconds", str(chunk_s), "--overlap-seconds", str(overlap_s),
         "--report", sep_json])

    run([PY, "evaluate.py", "--mixture", mixture, "--refs", refs, "--est", est,
         "--labels", "near", "far", "--json", eval_json])

    with open(room_json) as fh:
        room_info = json.load(fh)
    with open(sep_json) as fh:
        sep = json.load(fh)
    with open(eval_json) as fh:
        ev = json.load(fh)
    ca = ev["chunk_analysis"]

    # References are written ref1 = talker A (near), ref2 = talker B (far).
    near_snr, far_snr = ev["si_snr_per_source"][0], ev["si_snr_per_source"][1]

    return {
        "duration_s": round(ev["duration_s"], 1),
        "near_m": near,
        "far_m": far,
        "rt60_s": rt60,
        "level_diff_db": room_info["level_difference_db"],
        "near_drr_db": room_info["talker_a"]["drr_db"],
        "far_drr_db": room_info["talker_b"]["drr_db"],
        "mixture_si_snr_db": round(ev["mixture_si_snr"], 2),
        "si_snr_db": round(ev["si_snr"], 2),
        "si_snri_db": round(ev["si_snri"], 2),
        "near_si_snr_db": round(near_snr, 2),
        "far_si_snr_db": round(far_snr, 2),
        "identity_consistency": round(ca["consistency"], 3),
        "identity_flips": ca["n_flips"],
        "swaps_corrected": sep["n_swaps_corrected"],
        "elapsed_s": round(sep["elapsed_s"], 1),
        "real_time_factor": round(sep["real_time_factor"], 2),
        "peak_rss_mb": round(sep["peak_rss_mb"]),
    }


COLUMNS = [
    ("duration_s", "dur(s)", "{}"),
    ("near_m", "near(m)", "{}"),
    ("far_m", "far(m)", "{}"),
    ("rt60_s", "RT60", "{}"),
    ("level_diff_db", "lvl diff", "{:+.1f}"),
    ("far_drr_db", "far DRR", "{:+.1f}"),
    ("si_snri_db", "SI-SNRi", "{:+.2f}"),
    ("near_si_snr_db", "near SNR", "{:+.2f}"),
    ("far_si_snr_db", "far SNR", "{:+.2f}"),
    ("identity_consistency", "ident.", "{:.0%}"),
    ("real_time_factor", "RTF", "{:.2f}x"),
    ("peak_rss_mb", "peak MB", "{}"),
]


def print_table(rows):
    header = [c[1] for c in COLUMNS]
    body = [[c[2].format(r[c[0]]) for c in COLUMNS] for r in rows]
    widths = [max(len(header[i]), *(len(b[i]) for b in body)) for i in range(len(header))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  ".join("-" * w for w in widths))
    for b in body:
        print("  ".join(v.ljust(widths[i]) for i, v in enumerate(b)))


def main():
    parser = argparse.ArgumentParser(description="Sweep duration and talker distance.")
    parser.add_argument("--durations", type=float, nargs="+", default=[30, 60],
                        help="Mixture lengths in seconds.")
    parser.add_argument("--distances", nargs="+", default=["0.5,0.5", "0.5,2", "0.5,4"],
                        help="near,far distance pairs in metres.")
    parser.add_argument("--rt60", type=float, nargs="+", default=[0.4],
                        help="Room reverberation times to test. 0 = anechoic.")
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--outdir", default="runs/distance_sweep")
    args = parser.parse_args()

    pairs = []
    for spec in args.distances:
        try:
            near, far = (float(v) for v in spec.split(","))
        except ValueError:
            raise SystemExit(f"Bad --distances entry {spec!r}, expected near,far e.g. 0.5,4")
        pairs.append((near, far))

    os.makedirs(args.outdir, exist_ok=True)
    clips_dir = os.path.join(args.outdir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    max_seconds = max(args.durations)

    cache, rows = {}, []
    clip_a = ensure_clip(NEAR_VOICE, max_seconds, 1, clips_dir, cache)
    clip_b = ensure_clip(FAR_VOICE, max_seconds, 2, clips_dir, cache)

    total = len(args.durations) * len(pairs) * len(args.rt60)
    done = 0
    for seconds in args.durations:
        for near, far in pairs:
            for rt60 in args.rt60:
                done += 1
                print(f"[{done}/{total}] {seconds:.0f}s  near={near}m far={far}m "
                      f"rt60={rt60}s ...", end=" ", flush=True)
                row = one_condition(seconds, near, far, rt60, args.chunk_seconds,
                                    args.overlap_seconds, args.outdir, clip_a, clip_b)
                rows.append(row)
                print(f"near {row['near_si_snr_db']:+.2f} dB, "
                      f"far {row['far_si_snr_db']:+.2f} dB, "
                      f"identity {row['identity_consistency']:.0%}")

                with open(os.path.join(args.outdir, "results.csv"), "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"\n{len(rows)} conditions, saved to {os.path.join(args.outdir, 'results.csv')}\n")
    print_table(rows)


if __name__ == "__main__":
    main()
