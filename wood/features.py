import librosa
import numpy as np
import torch


def compute_hcqt(y, cfg):
    """Return float32 (harmonic, time, frequency); silence maps to -80 dB."""
    channels = []
    for harmonic in (0.5, 1, 2, 3, 4):
        cqt = np.abs(librosa.cqt(
            y, sr=cfg.sr, hop_length=cfg.hop,
            fmin=librosa.midi_to_hz(cfg.min_midi) * harmonic,
            bins_per_octave=12, n_bins=cfg.n_pitches))
        cqt /= max(float(cqt.max()), 1e-10)
        channels.append(np.maximum(20 * np.log10(np.maximum(cqt, 1e-4)), -80).T)
    return torch.from_numpy(np.stack(channels).astype(np.float32))


def audio_features(path, cfg, exemplar=False):
    y, _ = librosa.load(path, sr=cfg.sr)
    if exemplar:
        y = np.pad(y[:cfg.sr], (0, max(0, cfg.sr - len(y))))
    return compute_hcqt(y, cfg)


def full_window(features, t, window):
    """Previous + current frames, with silence padding at either boundary."""
    result = features.new_full((features.shape[0], 2 * window, features.shape[2]), -80)
    start, end = max(0, t - window), min(features.shape[1], t + window)
    dest = start - (t - window)
    result[:, dest:dest + end - start] = features[:, start:end]
    return result
