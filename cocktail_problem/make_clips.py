"""Generate long single-speaker test clips using the macOS `say` engine.

This gives repeatable test material with controllable "tone": a different system
voice changes timbre and accent, `--rate` changes speaking speed, `--pbas`
changes the synthesiser's pitch baseline, and `--semitones` applies a true
pitch shift after synthesis. Handy for probing where a separation model starts
to struggle (two similar voices are much harder than a male/female pair).

List the installed voices with:  say -v '?'

Usage:
    python make_clips.py --voice Samantha --seconds 60 --out clips/samantha_60.wav
    python make_clips.py --voice Daniel --rate 150 --semitones -2 --seconds 60 \
        --out clips/daniel_low_60.wav
"""
import argparse
import os
import random
import subprocess
import tempfile

import soundfile as sf
import torch
import torchaudio

TARGET_SR = 16000

# Generic filler sentences. Varied length and phonetic content so the clips are
# not just the same phrase looping, which would make separation artificially easy.
SENTENCES = [
    "The morning train was late again, so I walked the last few blocks.",
    "She measured the flour twice before adding it to the bowl.",
    "There are seventeen boxes stacked against the far wall of the garage.",
    "I keep forgetting whether the meeting moved to Tuesday or Thursday.",
    "The old radio still works if you jiggle the antenna a little.",
    "We should probably check the weather before deciding on the route.",
    "He described the painting as quiet, which I thought was a strange word.",
    "Every summer the lake drops about half a meter by the end of August.",
    "My neighbor grows tomatoes, peppers, and an alarming amount of mint.",
    "The report needs three more pages of analysis and a proper conclusion.",
    "Turn left after the bridge, then follow the gravel road to the end.",
    "I bought the cheaper headphones and regretted it within a week.",
    "The library closes early on Sundays during the winter months.",
    "Nobody could explain why the elevator only stops on even floors.",
    "A good soup takes patience and considerably more salt than you expect.",
    "The engineers argued about the tolerance for nearly an hour.",
    "Rain started just as we finished setting up the tables outside.",
    "She speaks four languages but insists she is fluent in only two.",
    "The kitten knocked the glass off the counter and looked genuinely proud.",
    "We drove past miles of wind turbines turning slowly in the haze.",
    "His handwriting is so small that I need better light to read it.",
    "The recipe calls for lemon zest, but orange works in a pinch.",
    "They repainted the hallway a color somewhere between grey and green.",
    "I found an old ticket stub tucked inside the back of the book.",
    "The bus driver waited an extra minute for the woman with the stroller.",
    "Most of the noise came from the ventilation fan above the door.",
    "He learned to solder by ruining about a dozen circuit boards.",
    "The path narrows near the top and the view is worth the climb.",
    "Our printer only jams when someone important is waiting for a document.",
    "By the time the coffee arrived, the conversation had moved on entirely.",
]


def build_text(seconds, rate_wpm, seed):
    """Assemble enough sentences to cover `seconds` of speech, with margin."""
    rng = random.Random(seed)
    words_needed = int(seconds * (rate_wpm / 60.0) * 1.6) + 40
    pool, out, count = list(SENTENCES), [], 0
    while count < words_needed:
        rng.shuffle(pool)
        for s in pool:
            out.append(s)
            count += len(s.split())
            if count >= words_needed:
                break
    return " ".join(out)


def synthesize(text, voice, rate_wpm, pbas, path_aiff):
    if pbas is not None:
        text = f"[[pbas {pbas}]] {text}"
    cmd = ["say", "-v", voice, "-r", str(rate_wpm), "-o", path_aiff, text]
    subprocess.run(cmd, check=True, capture_output=True)


def load_resampled(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    sig = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    if sr != TARGET_SR:
        sig = torchaudio.functional.resample(sig, sr, TARGET_SR)
    return sig


def make_clip(voice, seconds, out_path, rate_wpm=175, pbas=None, semitones=0.0, seed=0):
    """Render one clip of exactly `seconds` duration and return its tensor."""
    sig = None
    # `say` duration is only loosely predictable, so grow the text until long enough.
    for attempt in range(6):
        text = build_text(seconds * (1.0 + 0.35 * attempt), rate_wpm, seed + attempt)
        with tempfile.TemporaryDirectory() as tmp:
            aiff = os.path.join(tmp, "clip.aiff")
            synthesize(text, voice, rate_wpm, pbas, aiff)
            sig = load_resampled(aiff)
        if sig.shape[1] >= int(seconds * TARGET_SR):
            break

    need = int(seconds * TARGET_SR)
    if sig.shape[1] < need:  # still short: repeat the clip to fill
        reps = int(need / sig.shape[1]) + 1
        sig = sig.repeat(1, reps)
    sig = sig[:, :need]

    if semitones:
        sig = torchaudio.functional.pitch_shift(sig, TARGET_SR, semitones)

    peak = sig.abs().max()
    if peak > 0:
        sig = sig / peak * 0.9

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, sig.squeeze(0).numpy(), TARGET_SR)
    return sig


def main():
    parser = argparse.ArgumentParser(description="Synthesize a long test clip with `say`.")
    parser.add_argument("--voice", required=True, help="macOS voice name, e.g. Samantha.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Exact clip duration.")
    parser.add_argument("--out", required=True, help="Output WAV path.")
    parser.add_argument("--rate", type=int, default=175, help="Speaking rate in words/minute.")
    parser.add_argument("--pbas", type=int, default=None,
                        help="Synthesizer pitch baseline (roughly 30-70). Voice dependent.")
    parser.add_argument("--semitones", type=float, default=0.0,
                        help="Pitch shift applied after synthesis, in semitones.")
    parser.add_argument("--seed", type=int, default=0, help="Text shuffling seed.")
    args = parser.parse_args()

    sig = make_clip(args.voice, args.seconds, args.out, args.rate,
                    args.pbas, args.semitones, args.seed)
    print(f"Saved {args.out} ({sig.shape[1] / TARGET_SR:.1f}s, voice={args.voice}, "
          f"rate={args.rate}, pbas={args.pbas}, semitones={args.semitones})")


if __name__ == "__main__":
    main()
