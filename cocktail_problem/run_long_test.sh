#!/bin/bash
# One-command test: two people talking at once, one close to the mic and one
# further away, then separate them and score the result.
#
#   ./run_long_test.sh                 -> 10 minutes, 0.5 m vs 4 m
#   ./run_long_test.sh 300             -> 5 minutes,  0.5 m vs 4 m
#   ./run_long_test.sh 600 0.5 6       -> 10 minutes, 0.5 m vs 6 m
#   ./run_long_test.sh 600 1 2 0.6     -> distances 1 m / 2 m, RT60 0.6 s
#
# Arguments, all optional:
#   $1 duration in seconds  (default 600 = 10 minutes)
#   $2 near talker distance in metres (default 0.5)
#   $3 far  talker distance in metres (default 4.0)
#   $4 room reverberation time RT60 in seconds (default 0.4, use 0 for no reverb)
#
# Everything lands in a folder named test_<duration>s_<near>m_<far>m.

set -e
cd "$(dirname "$0")"

DUR=${1:-600}
NEAR=${2:-0.5}
FAR=${3:-4.0}
RT60=${4:-0.4}
PY=../.venv/bin/python
OUT="test_${DUR}s_${NEAR}m_${FAR}m"

if [ ! -x "$PY" ]; then
  echo "Cannot find the Python environment at $PY"
  echo "Run this from inside the cocktail_problem folder."
  exit 1
fi

mkdir -p "$OUT"
MINUTES=$(echo "$DUR" | awk '{printf "%.1f", $1/60}')

echo "======================================================"
echo " Near / far talker separation test"
echo "   duration : ${DUR}s (${MINUTES} min)"
echo "   talker A : ${NEAR} m from mic  (close)"
echo "   talker B : ${FAR} m from mic  (far)"
echo "   room RT60: ${RT60} s"
echo "   output   : $OUT/"
echo "======================================================"

echo
echo "[1/4] Making two ${DUR}s voice clips..."
$PY make_clips.py --voice Samantha --seconds "$DUR" --seed 1 --out "$OUT/voice_a.wav"
$PY make_clips.py --voice Daniel   --seconds "$DUR" --seed 2 --out "$OUT/voice_b.wav"

echo
echo "[2/4] Placing them in the room and mixing..."
$PY make_mixture.py --a "$OUT/voice_a.wav" --b "$OUT/voice_b.wav" \
    --seconds "$DUR" --distance-a "$NEAR" --distance-b "$FAR" --rt60 "$RT60" \
    --out "$OUT/mixture.wav" --save-refs "$OUT/refs" --info-json "$OUT/room.json"

echo
echo "[3/4] Separating (10s windows, takes roughly as long as the audio)..."
$PY separate.py --input "$OUT/mixture.wav" --outdir "$OUT/separated" \
    --chunk-seconds 10 --overlap-seconds 2 --report "$OUT/timing.json"

echo
echo "[4/4] Scoring against the clean originals..."
$PY evaluate.py --mixture "$OUT/mixture.wav" --refs "$OUT/refs" \
    --est "$OUT/separated" --labels "near_${NEAR}m" "far_${FAR}m" \
    --json "$OUT/results.json"

echo
echo "======================================================"
echo " Done."
echo "   Mixed input:  $OUT/mixture.wav"
echo "   Speaker 1:    $OUT/separated/speaker1.wav"
echo "   Speaker 2:    $OUT/separated/speaker2.wav"
echo "   Room details: $OUT/room.json"
echo "   Numbers:      $OUT/results.json"
echo "======================================================"
