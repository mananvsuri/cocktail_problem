"""Create a synthetic single-channel mixture from two clean voice clips.

Handy for testing the separation pipeline without recording, and for having
ground-truth sources to compare against.

Beyond a plain sum, this can hold the mixture to an exact duration, set the
level difference between the two talkers, and delay the second talker so the
overlap is partial rather than total. Use --save-refs to write the ground-truth
sources that line up sample-for-sample with the mixture, which is what the
metrics need.

The main feature is --distance-a / --distance-b: place each talker a given number
of metres from the microphone. That runs each clip through a simulated room (see
room.py), so the distant talker arrives quieter, later, and smeared with
reverberation, the same way it would in a real room. This is the realistic
"one person close, one person across the room" case.

Usage:
    python make_mixture.py --a voice1.wav --b voice2.wav --out mixture.wav

    # one talker 0.5 m from the mic, the other 4 m away
    python make_mixture.py --a voice1.wav --b voice2.wav --seconds 600 \
        --distance-a 0.5 --distance-b 4.0 \
        --out mixture.wav --save-refs refs/
"""
import argparse
import json
import math
import os

import soundfile as sf
import torch
import torchaudio

import room as room_sim

TARGET_SR = 16000
EPS = 1e-8


def load_mono(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [time, channels]
    mono = data.mean(axis=1)
    sig = torch.from_numpy(mono).unsqueeze(0)  # [1, time]
    if sr != TARGET_SR:
        sig = torchaudio.functional.resample(sig, sr, TARGET_SR)
    return sig


def fit_length(sig, length):
    """Loop or trim a signal to exactly `length` samples."""
    if sig.shape[1] < length:
        reps = int(length / sig.shape[1]) + 1
        sig = sig.repeat(1, reps)
    return sig[:, :length]


def rms(sig):
    return float(torch.sqrt(torch.mean(sig**2) + EPS))


def build_mixture(a, b, seconds=None, sir_db=0.0, offset=0.0,
                  distance_a=None, distance_b=None, rt60=0.4,
                  dims=room_sim.DEFAULT_ROOM, azimuth_a=-25.0, azimuth_b=25.0,
                  height=1.6):
    """Return (mixture, ref_a, ref_b, info), all [1, time] and mutually aligned.

    When distance_a / distance_b are given, each talker is convolved with a room
    impulse response for that distance and the natural level difference is kept
    as-is: the whole point is that the far talker really is quieter and more
    reverberant. --sir-db is then an extra nudge on top, not an equalisation.

    The references are the reverberant images of each talker at the mic, i.e. what
    each voice actually contributes to the mixture. Scoring against the dry clips
    instead would also demand dereverberation, which this model does not claim to do.
    """
    if seconds:
        length = int(seconds * TARGET_SR)
        a, b = fit_length(a, length), fit_length(b, length)
    else:
        length = min(a.shape[1], b.shape[1])
        a, b = a[:, :length], b[:, :length]

    # Delay the second talker so the two only partially overlap.
    if offset:
        pad = int(abs(offset) * TARGET_SR)
        pad = min(pad, length - 1)
        if offset > 0:
            b = torch.cat([torch.zeros(1, pad), b[:, : length - pad]], dim=1)
        else:
            a = torch.cat([torch.zeros(1, pad), a[:, : length - pad]], dim=1)

    info = {"mode": "gain"}

    if distance_a is not None or distance_b is not None:
        # Start both talkers at the same source strength, so any level difference
        # in the mixture comes from the room and the distance, not the clips.
        b = b * (rms(a) / (rms(b) + EPS))

        da = distance_a if distance_a is not None else 1.0
        db = distance_b if distance_b is not None else 1.0
        a, info_a = room_sim.simulate_at_distance(
            a, da, TARGET_SR, dims=dims, rt60=rt60, azimuth_deg=azimuth_a, height=height)
        b, info_b = room_sim.simulate_at_distance(
            b, db, TARGET_SR, dims=dims, rt60=rt60, azimuth_deg=azimuth_b, height=height)
        info = {"mode": "room", "talker_a": info_a, "talker_b": info_b}

    if sir_db:
        b = b * (10 ** (-sir_db / 20.0))

    mix = a + b
    peak = float(mix.abs().max())
    if peak > 0:
        scale = 1.0 / peak  # normalize to prevent clipping
        mix, a, b = mix * scale, a * scale, b * scale

    info["level_difference_db"] = round(20 * math.log10((rms(a) + EPS) / (rms(b) + EPS)), 2)
    info["duration_s"] = round(length / TARGET_SR, 2)
    return mix, a, b, info


def main():
    parser = argparse.ArgumentParser(description="Mix two clips into one mono file.")
    parser.add_argument("--a", required=True, help="First voice WAV (talker A).")
    parser.add_argument("--b", required=True, help="Second voice WAV (talker B).")
    parser.add_argument("--out", default="mixture.wav", help="Output mixture WAV.")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Force this exact duration (clips are looped or trimmed).")
    parser.add_argument("--sir-db", type=float, default=0.0,
                        help="Extra level offset applied to B, in dB. Leave at 0 when "
                             "using distances so the level difference stays physical.")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Delay B by this many seconds (negative delays A instead).")
    parser.add_argument("--distance-a", type=float, default=None,
                        help="Distance of talker A from the mic, in metres.")
    parser.add_argument("--distance-b", type=float, default=None,
                        help="Distance of talker B from the mic, in metres.")
    parser.add_argument("--rt60", type=float, default=0.4,
                        help="Room reverberation time in seconds. 0 = anechoic "
                             "(level and delay only, no reverb).")
    parser.add_argument("--room", default="10,7,3",
                        help="Room size in metres as lx,ly,lz.")
    parser.add_argument("--azimuth-a", type=float, default=-25.0,
                        help="Direction of talker A from the mic, in degrees.")
    parser.add_argument("--azimuth-b", type=float, default=25.0,
                        help="Direction of talker B from the mic, in degrees.")
    parser.add_argument("--save-refs", default=None,
                        help="Directory to write the aligned ground-truth sources.")
    parser.add_argument("--info-json", default=None,
                        help="Write the room / level details to this JSON file.")
    args = parser.parse_args()

    dims = tuple(float(v) for v in args.room.split(","))
    if len(dims) != 3:
        raise SystemExit("--room needs three numbers, e.g. 10,7,3")

    a = load_mono(args.a)
    b = load_mono(args.b)
    mix, ref_a, ref_b, info = build_mixture(
        a, b, args.seconds, args.sir_db, args.offset,
        args.distance_a, args.distance_b, args.rt60, dims,
        args.azimuth_a, args.azimuth_b)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    sf.write(args.out, mix.squeeze(0).numpy(), TARGET_SR)
    print(f"Saved mixture to {args.out} ({mix.shape[1] / TARGET_SR:.1f}s)")

    if info["mode"] == "room":
        for label, key in (("A (near)", "talker_a"), ("B (far) ", "talker_b")):
            d = info[key]
            print(f"  talker {label}: {d['distance_m']:.2f} m, "
                  f"DRR {d['drr_db']:+.1f} dB, "
                  f"delay {d['propagation_delay_ms']:.1f} ms, "
                  f"RT60 {d['rt60_measured_s']:.2f}s")
        print(f"  room {info['talker_a']['room_m']} m, "
              f"level difference A-B {info['level_difference_db']:+.1f} dB")
    else:
        print(f"  gain mode, level difference A-B {info['level_difference_db']:+.1f} dB")

    if args.save_refs:
        os.makedirs(args.save_refs, exist_ok=True)
        for name, sig in (("ref1.wav", ref_a), ("ref2.wav", ref_b)):
            path = os.path.join(args.save_refs, name)
            sf.write(path, sig.squeeze(0).numpy(), TARGET_SR)
            print(f"Saved reference {path}")

    if args.info_json:
        with open(args.info_json, "w") as fh:
            json.dump(info, fh, indent=2)
        print(f"Wrote {args.info_json}")


if __name__ == "__main__":
    main()
