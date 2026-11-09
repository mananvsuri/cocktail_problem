"""Record audio from the laptop microphone into a mono WAV file.

Usage:
    python record.py --seconds 10 --out mixture.wav
"""
import argparse
import sys

import sounddevice as sd
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Record mono audio from the mic.")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="How long to record (seconds).")
    parser.add_argument("--out", default="mixture.wav",
                        help="Output WAV file path.")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sample rate in Hz.")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices and exit.")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    n_frames = int(args.seconds * args.sr)
    print(f"Recording {args.seconds:.0f}s at {args.sr} Hz. Have both people speak now...")
    try:
        audio = sd.rec(n_frames, samplerate=args.sr, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:  # noqa: BLE001
        print(f"\nRecording failed: {exc}", file=sys.stderr)
        print("On macOS you may need to grant microphone access to your terminal "
              "in System Settings > Privacy & Security > Microphone.", file=sys.stderr)
        sys.exit(1)

    sf.write(args.out, audio, args.sr)
    print(f"Saved recording to {args.out}")


if __name__ == "__main__":
    main()
