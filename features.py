"""Log-mel features from a phonocardiogram.

Uses scipy only. Heart sounds in this dataset are 4 kHz mono, so the Nyquist
limit is 2 kHz; murmurs live roughly between 25 and 600 Hz, which is why the
filterbank stops at 1 kHz rather than spending resolution on empty band.
"""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, stft

SAMPLE_RATE = 4000
SEG_SECONDS = 4.0
SEG_LEN = int(SAMPLE_RATE * SEG_SECONDS)
HOP_SECONDS = 2.0
N_MELS = 64
N_FFT = 256
HOP = 128          # 32 ms frames: ample for a sustained murmur, half the memory
FMIN, FMAX = 25.0, 1000.0
MAX_SEGMENTS = 8   # a 26 s recording yields ~11; 8 evenly spaced is plenty


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """Triangular mel filterbank, built once and reused."""
    edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        left = (freqs - lo) / max(mid - lo, 1e-9)
        right = (hi - freqs) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0, None)
    return fb


_FB = mel_filterbank()


def bandpass(x, sr=SAMPLE_RATE, lo=25.0, hi=800.0):
    """Drop the rumble below 25 Hz and the hiss above 800 Hz.

    Both are recording artefacts far more often than they are heart sounds.
    """
    sos = butter(4, [lo / (sr / 2), min(hi / (sr / 2), 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def read_wav(path) -> np.ndarray:
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))
    if sr != SAMPLE_RATE:
        n = int(round(x.size * SAMPLE_RATE / sr))
        x = np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)
    return x


def log_mel(seg: np.ndarray) -> np.ndarray:
    """One segment -> (N_MELS, frames) log-mel, per-segment normalised."""
    _, _, Z = stft(seg, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT - HOP,
                   padded=False, boundary=None)
    power = (np.abs(Z) ** 2).astype(np.float32)
    mel = _FB @ power
    logmel = np.log(mel + 1e-8)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)


def segments(x: np.ndarray):
    """Fixed windows with overlap, capped and evenly spaced across the recording.

    Capping keeps the feature cache in memory. Spreading the cap across the
    whole recording rather than taking the first N matters, because a murmur
    may only be audible during part of it.
    """
    if x.size < SEG_LEN:
        x = np.pad(x, (0, SEG_LEN - x.size))
    hop = int(SAMPLE_RATE * HOP_SECONDS)
    starts = list(range(0, max(1, x.size - SEG_LEN + 1), hop))
    if len(starts) > MAX_SEGMENTS:
        pick = np.linspace(0, len(starts) - 1, MAX_SEGMENTS).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(pick)]
    for start in starts:
        yield x[start:start + SEG_LEN]


#: The mel filterbank already discards everything outside 25-1000 Hz, so the
#: Butterworth stage is close to redundant. It is also the one step that is
#: genuinely hard to reproduce bit-for-bit in a browser, and a model that sees
#: different features in the product than it saw in training is worse than a
#: marginally less accurate one. Off by default; the flag keeps the comparison
#: reproducible.
USE_BANDPASS = False


def recording_features(path) -> np.ndarray:
    """(n_segments, 1, N_MELS, frames) ready for the network."""
    x = read_wav(path)
    if USE_BANDPASS:
        x = bandpass(x)
    feats = [log_mel(s) for s in segments(x)]
    return np.stack(feats)[:, None, :, :].astype(np.float32)
