# Cocktail Party Problem - Speech Separation

A Python implementation for separating overlapping speech from two speakers in a single-channel audio recording. Uses pre-trained SepFormer models from SpeechBrain for real-time voice separation.

## Overview

This project implements the cocktail party problem solution using deep learning. Given a mono recording of two people speaking simultaneously, it separates the mixture into individual speaker tracks.

The system includes:
- Real-time recording from microphone
- Multiple pre-trained separation models (WSJ02mix and WHAM)
- Distance and room simulation for testing
- Evaluation metrics (SI-SNR, SI-SNRi)
- Batch processing for long recordings

## Requirements

- Python 3.8+
- macOS (tested on Apple Silicon)
- ~2GB disk space for models

## Installation

```bash
cd cocktail_problem
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Record two people speaking and separate their voices:

```bash
cd cocktail_problem
python record.py --seconds 10 --out mixture.wav
python separate.py --input mixture.wav --outdir output
```

The first run downloads the model automatically.

## Usage

### Basic Separation

Separate any mono audio file:
```bash
python separate.py --input mixture.wav --outdir output
```

### Create Test Mixtures

Mix two existing audio files:
```bash
python make_mixture.py --a voice1.wav --b voice2.wav --out mixture.wav
```

### Generate Synthetic Speech

Create long test clips with different voices:
```bash
python make_clips.py --voice Samantha --seconds 60 --out voice_a.wav
```

### Simulate Distance and Room Acoustics

Place speakers at different distances with reverberation:
```bash
python make_mixture.py --a voice1.wav --b voice2.wav \
    --distance-a 0.5 --distance-b 4.0 --rt60 0.4 --out mixture.wav
```

### Long Recording Processing

For recordings over 30 seconds, use chunked processing:
```bash
python separate.py --input long_mixture.wav --outdir output \
    --chunk-seconds 10 --overlap-seconds 2
```

### Complete Test Pipeline

Run an end-to-end test with near/far speakers:
```bash
./run_long_test.sh 60 0.5 4.0
```
Arguments: duration (seconds), near distance (meters), far distance (meters)

### Batch Experiments

Run multiple tests with different conditions:
```bash
python run_experiments.py --durations 30 60 --distances 0.5,0.5 0.5,2 0.5,4
```

### Evaluation

Measure separation quality:
```bash
python evaluate.py --mixture mixture.wav --refs refs --est output
```

## Models

Two pre-trained models are supported:

1. **sepformer-wsj02mix** (default) - Trained on clean speech
2. **sepformer-wham** - Trained with noise and reverberation

Use the WHAM model for noisy environments:
```bash
python separate_wham.py --input recording.wav --outdir output
```

## Project Structure

```
cocktail_problem/
├── record.py              # Capture audio from microphone
├── separate.py            # Main separation script (WSJ02mix)
├── separate_wham.py       # Separation with WHAM model
├── make_clips.py          # Generate synthetic speech
├── make_mixture.py        # Mix two audio files
├── room.py                # Room acoustics simulation
├── evaluate.py            # Calculate separation metrics
├── metrics.py             # SI-SNR and SI-SNRi implementations
├── run_experiments.py     # Batch testing
├── run_long_test.sh       # Complete test pipeline
└── requirements.txt       # Python dependencies
```

## Technical Details

- Operates at 8kHz (resampled automatically)
- Uses transformer-based SepFormer architecture
- Supports CPU and Apple MPS acceleration
- Implements permutation-invariant training approach
- Cross-fade stitching for chunked processing

## Performance

Typical separation improvements (SI-SNRi):
- Clean speech: +10 to +15 dB
- Near/far (0.5m/4m): +5 to +8 dB
- With background noise: +3 to +6 dB

Processing speed: 0.5-1.5x real-time on M1/M2 Macs (CPU mode)

## Notes

- First run downloads models (few hundred MB)
- Microphone permission required for recording
- Best results with two speakers in quiet environment
- Quality degrades with background noise or 3+ speakers
- Long recordings require chunked processing to avoid memory issues

## License

Code is provided as-is for research and educational purposes.

Models are from SpeechBrain: https://github.com/speechbrain/speechbrain
