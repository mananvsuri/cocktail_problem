"""Place a talker at a chosen distance from the microphone.

Moving someone away from a mic is not just "make it quieter". Three things
change together, and all three matter to a separation model:

  1. Level. The direct sound falls off as 1/distance.
  2. Direct-to-reverberant ratio (DRR). Reflected energy is spread fairly evenly
     through a room, while the direct sound keeps dropping, so a distant talker
     arrives smeared in reverb. This is the cue that actually makes a voice sound
     far away, and it is the part a plain gain change cannot fake.
  3. Arrival time. Sound covers about 343 m/s, so 4 m of extra distance is ~12 ms
     of extra delay.

The room impulse response is built with the image-source method (Allen & Berkley,
1979): every wall reflection is modelled as a mirrored copy of the source, each
copy attenuated by the wall reflection coefficient and delayed by its travel
distance. Convolving a dry clip with this response puts that clip in the room at
that distance.

Wall absorption is derived from a target RT60 via Sabine's equation.
"""
import math

import numpy as np
import torch

SPEED_OF_SOUND = 343.0  # m/s at ~20 C


def sabine_absorption(dims, rt60):
    """Wall absorption coefficient giving the requested RT60 in a shoebox room."""
    lx, ly, lz = dims
    volume = lx * ly * lz
    surface = 2 * (lx * ly + lx * lz + ly * lz)
    alpha = 0.161 * volume / (surface * max(rt60, 1e-3))
    return float(min(max(alpha, 1e-4), 0.99))


def image_source_rir(dims, source, mic, sr, rt60=0.4, max_order=None,
                     c=SPEED_OF_SOUND):
    """Room impulse response from `source` to `mic` in a shoebox room.

    Args:
        dims: (lx, ly, lz) room size in metres.
        source, mic: (x, y, z) positions in metres.
        sr: sample rate.
        rt60: target reverberation time in seconds. 0 gives a pure delay+gain
            (anechoic), which is useful for isolating the level cue.
        max_order: cap on total reflection order. Auto-chosen when None.

    Returns:
        1-D float32 torch tensor.
    """
    dims = np.asarray(dims, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    mic = np.asarray(mic, dtype=np.float64)

    direct_dist = float(np.linalg.norm(source - mic))

    if rt60 <= 0:  # anechoic: single direct-path impulse
        length = int(direct_dist / c * sr) + 8
        rir = np.zeros(length, dtype=np.float64)
        delay = direct_dist / c * sr
        idx, frac = int(delay), delay - int(delay)
        amp = 1.0 / (4 * math.pi * max(direct_dist, 1e-3))
        rir[idx] += amp * (1 - frac)
        if idx + 1 < length:
            rir[idx + 1] += amp * frac
        return torch.from_numpy(rir).float()

    alpha = sabine_absorption(dims, rt60)
    beta = math.sqrt(max(1.0 - alpha, 1e-6))  # pressure reflection coefficient

    rir_len = int(sr * rt60 * 1.3) + 128
    max_dist = c * rir_len / sr

    if max_order is None:
        max_order = int(math.ceil(max_dist / (2 * float(dims.min())))) + 1
    max_order = int(min(max_order, 14))  # keeps the image count manageable

    # Per-axis mirrored source positions. For integer n and u in {0, 1} the image
    # sits at 2*n*L + (-1)^u * x, having undergone |2n - u| reflections on that axis.
    axis_pos, axis_ord = [], []
    ns = np.arange(-max_order, max_order + 1)
    for ax in range(3):
        pos, ordr = [], []
        for n in ns:
            for u in (0, 1):
                pos.append(2.0 * n * dims[ax] + ((-1.0) ** u) * source[ax])
                ordr.append(abs(2 * n - u))
        axis_pos.append(np.array(pos))
        axis_ord.append(np.array(ordr, dtype=np.int32))

    gx, gy, gz = np.meshgrid(*axis_pos, indexing="ij")
    ox, oy, oz = np.meshgrid(*axis_ord, indexing="ij")
    order = (ox + oy + oz).ravel()
    px, py, pz = gx.ravel(), gy.ravel(), gz.ravel()

    keep = order <= max_order
    px, py, pz, order = px[keep], py[keep], pz[keep], order[keep]

    dist = np.sqrt((px - mic[0]) ** 2 + (py - mic[1]) ** 2 + (pz - mic[2]) ** 2)
    keep = (dist < max_dist) & (dist > 1e-6)
    dist, order = dist[keep], order[keep]

    amp = (beta ** order) / (4 * math.pi * dist)
    delay = dist / c * sr

    rir = np.zeros(rir_len + 1, dtype=np.float64)
    idx = np.floor(delay).astype(np.int64)
    frac = delay - idx
    valid = (idx >= 0) & (idx + 1 <= rir_len)
    np.add.at(rir, idx[valid], amp[valid] * (1.0 - frac[valid]))
    np.add.at(rir, idx[valid] + 1, amp[valid] * frac[valid])

    return torch.from_numpy(rir[:rir_len]).float()


def place_source(mic, distance, dims, azimuth_deg=0.0, height=None, margin=0.3):
    """Find a point `distance` metres from `mic` that fits inside the room.

    Tries the requested azimuth first, then rotates in 5-degree steps until the
    point lands inside the room walls (minus a margin). Raises if the distance
    simply does not fit.
    """
    mic = np.asarray(mic, dtype=np.float64)
    dims = np.asarray(dims, dtype=np.float64)
    z = mic[2] if height is None else height
    z = float(np.clip(z, margin, dims[2] - margin))

    for step in range(72):
        for sign in (1, -1):
            ang = math.radians(azimuth_deg + sign * step * 5.0)
            pos = np.array([mic[0] + distance * math.cos(ang),
                            mic[1] + distance * math.sin(ang),
                            z])
            if np.all(pos > margin) and np.all(pos < dims - margin):
                return pos
    raise ValueError(
        f"A distance of {distance} m does not fit in a {dims.tolist()} m room. "
        f"Use a bigger --room or a smaller distance."
    )


def apply_rir(sig, rir, out_len=None):
    """Convolve [1, time] with an impulse response, trimmed to `out_len`.

    Trimming keeps every source on the same timeline as the mixture, so the
    ground-truth references stay sample-aligned.
    """
    n = sig.shape[-1] + rir.shape[-1] - 1
    nfft = 1 << (n - 1).bit_length()
    spec = torch.fft.rfft(sig, nfft) * torch.fft.rfft(rir.unsqueeze(0), nfft)
    wet = torch.fft.irfft(spec, nfft)[..., :n]
    return wet[..., : out_len if out_len else sig.shape[-1]]


def direct_to_reverberant_db(rir, sr, direct_window_ms=2.5):
    """DRR in dB: energy in the direct-arrival window vs everything after it.

    Lower means more reverberant, which is what "further away" sounds like.
    """
    peak = int(torch.argmax(rir.abs()))
    half = max(int(direct_window_ms * 1e-3 * sr), 1)
    lo, hi = max(peak - half, 0), min(peak + half + 1, rir.shape[-1])
    direct = float(torch.sum(rir[lo:hi] ** 2))
    total = float(torch.sum(rir**2))
    late = max(total - direct, 1e-12)
    return 10 * math.log10(max(direct, 1e-12) / late)


def estimate_rt60(rir, sr):
    """RT60 estimated from the Schroeder decay curve (-5 dB to -35 dB slope)."""
    energy = torch.flip(torch.cumsum(torch.flip(rir**2, [0]), 0), [0])
    curve = 10 * torch.log10(energy / energy.max() + 1e-12)
    try:
        i5 = int((curve <= -5).nonzero()[0])
        i35 = int((curve <= -35).nonzero()[0])
    except IndexError:
        return float("nan")
    if i35 <= i5:
        return float("nan")
    slope = (-35.0 + 5.0) / ((i35 - i5) / sr)  # dB per second
    return float(-60.0 / slope)


DEFAULT_ROOM = (10.0, 7.0, 3.0)  # a mid-size meeting room


def default_mic(dims):
    """Mic near one end of the room, so large distances still fit inside."""
    return (1.5, dims[1] / 2.0, 1.5)


def simulate_at_distance(sig, distance, sr, dims=DEFAULT_ROOM, mic=None,
                         rt60=0.4, azimuth_deg=0.0, height=1.6):
    """Render a dry [1, time] clip as heard from `distance` metres away.

    Returns (wet_signal, info_dict).
    """
    dims = tuple(float(d) for d in dims)
    if mic is None:
        mic = default_mic(dims)
    source = place_source(mic, distance, dims, azimuth_deg, height)
    rir = image_source_rir(dims, source, mic, sr, rt60)
    wet = apply_rir(sig, rir, out_len=sig.shape[-1])
    info = {
        "distance_m": round(float(np.linalg.norm(np.asarray(source) - np.asarray(mic))), 3),
        "source_pos": [round(float(v), 2) for v in source],
        "mic_pos": [round(float(v), 2) for v in mic],
        "room_m": list(dims),
        "rt60_target_s": rt60,
        "rt60_measured_s": round(estimate_rt60(rir, sr), 3) if rt60 > 0 else 0.0,
        "drr_db": round(direct_to_reverberant_db(rir, sr), 2),
        "propagation_delay_ms": round(
            float(np.linalg.norm(np.asarray(source) - np.asarray(mic))) / SPEED_OF_SOUND * 1000, 2),
    }
    return wet, info
