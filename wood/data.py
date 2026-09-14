from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .features import audio_features, full_window


def audio_hash(folder):
    import hashlib
    with (Path(folder) / 'full.wav').open('rb') as audio:
        return hashlib.file_digest(audio, 'sha256').hexdigest()


def exclude_held_out(folders, held_out):
    """Compare audio content, not folder names; never move/delete source files."""
    hashes = {audio_hash(p) for p in held_out}
    return [p for p in folders if audio_hash(p) not in hashes]


def song_folders(root):
    folders = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / 'full.wav').exists())
    if not folders:
        raise ValueError(f'No song folders in {root}')
    return folders


def empty_state(length, cfg):
    state = torch.zeros(length, cfg.n_events, 3)
    state[..., 0] = cfg.n_pitches
    return state


def load_audio_song(folder, cfg):
    folder = Path(folder)
    exemplars = sorted(folder.glob('exemplar_*.wav'))
    if len(exemplars) != 1:
        raise ValueError(f'{folder}: expected one exemplar WAV')
    _, pitch, velocity, _ = exemplars[0].stem.split('_')
    return dict(song_id=folder.name, features=audio_features(folder / 'full.wav', cfg),
                exemplar=audio_features(exemplars[0], cfg, exemplar=True),
                pitch_ex=int(librosa.note_to_midi(pitch)), vel_ex=int(velocity))


def load_song(folder, cfg):
    song = load_audio_song(folder, cfg)
    T = song['features'].shape[1]
    onset, offset, frame, velocity = [torch.zeros(T, cfg.n_pitches) for _ in range(4)]
    intervals, pitches = [], []
    for row in pd.read_csv(Path(folder) / 'midi_export.csv').itertuples():
        p = int(row.pitch) - cfg.min_midi
        if not 0 <= p < cfg.n_pitches:
            continue
        sf = max(0, int(row.start_sec * cfg.sr / cfg.hop))
        ef = min(T, int(row.end_sec * cfg.sr / cfg.hop))
        if sf >= T or ef <= sf:
            continue
        onset[sf, p] = 1
        if ef < T:
            offset[ef, p] = 1
        frame[sf:ef, p] = 1
        velocity[sf:ef, p] = row.velocity
        intervals.append([row.start_sec, row.end_sec])
        pitches.append(row.pitch)
    state = empty_state(T, cfg)
    last_onset = torch.zeros(cfg.n_pitches)
    for t in range(T):
        last_onset[onset[t].bool()] = t
        active = torch.where(frame[t].bool())[0]
        if len(active) > cfg.n_events:
            raise ValueError(f'{folder}: polyphony exceeds n_events={cfg.n_events}; increase it')
        for i, p in enumerate(active):
            state[t, i] = torch.tensor([p, velocity[t, p], t - last_onset[p]])
    song.update(onset=onset, offset=offset, frame=frame, velocity=velocity, state=state,
                intervals=np.asarray(intervals, dtype=float).reshape(-1, 2),
                pitches=np.asarray(pitches))
    return song


class WindowDataset(Dataset):
    def __init__(self, folders, cfg):
        self.cfg = cfg
        self.songs = [load_song(p, cfg) for p in folders]
        self.index = [(i, t) for i, s in enumerate(self.songs)
                      for t in range(0, len(s['frame']), cfg.window)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, t = self.index[idx]
        song, cfg = self.songs[i], self.cfg
        w = cfg.window
        previous = empty_state(w, cfg)
        count = min(t, w)
        if count:
            previous[-count:] = song['state'][t-count:t]
        valid = min(w, len(song['frame']) - t)
        targets = {}
        for key in ('onset', 'offset', 'velocity'):
            target = torch.zeros(w, cfg.n_pitches)
            target[:valid] = song[key][t:t+valid]
            targets[key] = target / 127 if key == 'velocity' else target
        return dict(full=full_window(song['features'], t, w), ex=song['exemplar'],
                    pitch=torch.tensor(song['pitch_ex']), vel=torch.tensor([song['vel_ex']/127]),
                    previous=previous, mask=(torch.arange(w) < valid).float(), **targets)
