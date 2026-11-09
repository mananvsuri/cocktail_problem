# Cocktail Party — Two-Voice Separation

Separate two overlapping voices from a single-microphone (mono) recording using a
pre-trained deep-learning speech separation model (SpeechBrain SepFormer).

No API keys, no cloud services — everything runs locally on your Mac.

## How it works
1. **Record** two people speaking at once from the laptop mic → `mixture.wav`
2. **Separate** the mixture into two tracks → `output/speaker1.wav`, `output/speaker2.wav`
3. **Listen** and compare

The separation model is `speechbrain/sepformer-wsj02mix`, a SepFormer network
trained to split a mono mixture of two speakers. It runs at 8 kHz, so the code
resamples your recording automatically.

## Setup (macOS Apple Silicon, Python 3.14)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Option 1 — record two people live
```bash
python record.py --seconds 10 --out mixture.wav
python separate.py --input mixture.wav --outdir output
```
The first `separate.py` run downloads the model (a few hundred MB), then it is cached.

### Option 2 — test with two existing clips (recommended first)
```bash
python make_mixture.py --a voice1.wav --b voice2.wav --out mixture.wav
python separate.py --input mixture.wav --outdir output
```

### Option 3 — long recording, one talker near the mic and one far away
This is the realistic meeting-room case: someone speaking right at the laptop
while someone else talks from across the room. One command does everything:

```bash
./run_long_test.sh 600 0.5 4.0
```

Arguments are `duration_seconds near_metres far_metres [rt60_seconds]`, all
optional. So `./run_long_test.sh` alone gives 10 minutes at 0.5 m vs 4 m, and
`./run_long_test.sh 300 1 6 0.8` gives 5 minutes at 1 m vs 6 m in a more
reverberant room. Results land in `test_<duration>s_<near>m_<far>m/`.

The four steps it runs, if you want them individually:

```bash
# 1. two long clips (see `say -v '?'` for other voices)
python make_clips.py --voice Samantha --seconds 600 --seed 1 --out clips/voice_a.wav
python make_clips.py --voice Daniel   --seconds 600 --seed 2 --out clips/voice_b.wav

# 2. put talker A 0.5 m from the mic and talker B 4 m away, then mix
python make_mixture.py --a clips/voice_a.wav --b clips/voice_b.wav --seconds 600 \
    --distance-a 0.5 --distance-b 4.0 --rt60 0.4 \
    --out mixture.wav --save-refs refs --info-json room.json

# 3. separate in overlapping windows (required past ~30s)
python separate.py --input mixture.wav --outdir separated \
    --chunk-seconds 10 --overlap-seconds 2

# 4. score, reporting the near and far talkers separately
python evaluate.py --mixture mixture.wav --refs refs --est separated \
    --labels near far
```

To sweep several distances and durations at once:
```bash
python run_experiments.py --durations 30 60 --distances 0.5,0.5 0.5,2 0.5,4 0.5,6
```
Results land in `runs/distance_sweep/results.csv` plus a printed table.

**How distance is simulated.** `room.py` builds a room impulse response with the
image-source method and convolves each talker with it, so moving someone back
changes all three real cues at once: level (falls as 1/distance), arrival delay
(~3 ms per metre), and direct-to-reverberant ratio. DRR is the one that matters —
at 0.5 m it is about +8 dB, at 4 m about −10 dB, meaning the distant voice arrives
mostly as reverberation. Set `--rt60 0` for an anechoic run if you want the level
cue on its own.

**`--chunk-seconds` matters.** Whole-file inference degrades badly on long audio:
at 60s it scored +0.44 dB SI-SNR improvement with the two speakers swapping tracks
mid-file, versus +5.88 dB and no swaps in 10s windows. It also used 4.7 GB of RAM
versus 1.4 GB. Whole-file memory grows steeply with length, so on an 8 GB machine
keep one-pass runs under about 60s.

Other knobs for harder tests: `--offset` (partial rather than total overlap),
`--sir-db` (extra level offset on top of distance), `--room` (room size), and
`--semitones` / `--pbas` / `--rate` on `make_clips.py` (change a voice's pitch
or tempo).

## Notes
- **Microphone permission:** the first time you run `record.py`, macOS may ask to
  allow microphone access, or you may need to enable it under
  *System Settings → Privacy & Security → Microphone* for your terminal app.
- **Device:** separation defaults to CPU (most reliable). To try the Apple GPU:
  `python separate.py --input mixture.wav --device mps`.
- **Best results:** the model was trained on clean two-speaker mixtures. A quiet
  room and a decent mic level help a lot. Heavy background noise or more than two
  speakers will degrade quality.

## Files
- `run_long_test.sh` — one command: build a near/far mixture, separate it, score it
- `record.py` — capture audio from the mic
- `make_clips.py` — synthesize long single-speaker clips with a chosen voice/pitch/rate
- `make_mixture.py` — mix two clips, optionally placing each talker at a distance
- `room.py` — image-source room simulation (distance, reverberation, delay)
- `separate.py` — separate a mixture into two speaker tracks, whole-file or windowed
- `evaluate.py` — score separated tracks (SI-SNR, SI-SNRi, speaker-identity stability)
- `metrics.py` — the metric implementations
- `run_experiments.py` — sweep durations and talker distances, write a results table
