import numpy as np
import torch
from .data import empty_state
from .features import full_window


def update_state(previous, onsets, offsets, velocities, cfg):
    """Decode rising onset edges; an offset can also retrigger an active pitch."""
    cur = previous.clone()
    result = []
    for t in range(len(onsets)):
        for p in torch.where(offsets[t])[0]:
            cur[cur[:, 0] == p] = cur.new_tensor([cfg.n_pitches, 0, 0])
        cur[cur[:, 0] != cfg.n_pitches, 2] += 1
        for p in torch.where(onsets[t])[0]:
            rows = torch.where(cur[:, 0] == p)[0]
            if not len(rows):
                rows = torch.where(cur[:, 0] == cfg.n_pitches)[0]
            if len(rows):
                cur[rows[0]] = torch.stack((p.float(), velocities[t, p], p.new_tensor(0).float()))
        cur = cur[torch.argsort(cur[:, 0])]
        result.append(cur.clone())
    return torch.stack(result)


@torch.inference_mode()
def predict_song(model, song, cfg, device, onset_threshold=0.5, offset_threshold=0.5):
    model.eval()
    w = cfg.window
    previous = empty_state(w, cfg).to(device)
    ex = song['exemplar'].unsqueeze(0).to(device)
    pitch = torch.tensor([song['pitch_ex']], device=device)
    vel = torch.tensor([[song['vel_ex']/127]], device=device)
    above_last = torch.zeros(cfg.n_pitches, dtype=torch.bool, device=device)
    states = []
    for t in range(0, song['features'].shape[1], w):
        full = full_window(song['features'], t, w).unsqueeze(0).to(device)
        on, off, velocity = model(full, ex, pitch, vel, previous.unsqueeze(0))
        above = on[0].sigmoid() >= onset_threshold
        offsets = off[0].sigmoid() >= offset_threshold
        prior = torch.cat((above_last[None], above[:-1]))
        starts = above & (~prior | offsets)
        above_last = above[-1]
        previous = update_state(previous[-1], starts, offsets, (velocity[0]*127).round(), cfg)
        states.append(previous.cpu())
    state = torch.cat(states)[:song['features'].shape[1]]
    frame = torch.zeros(len(state), cfg.n_pitches)
    onset = torch.zeros_like(frame)
    for t, rows in enumerate(state):
        for p, v, duration in rows:
            if int(p) < cfg.n_pitches:
                frame[t, int(p)] = 1
                onset[t, int(p)] = float(duration == 0)
    notes = []
    for p in range(cfg.n_pitches):
        start = None
        for t in range(len(frame) + 1):
            active = t < len(frame) and bool(frame[t, p])
            restart = t < len(frame) and bool(onset[t, p])
            if start is not None and (not active or restart):
                notes.append((p + cfg.min_midi, start*cfg.hop/cfg.sr, t*cfg.hop/cfg.sr))
                start = None
            if active and start is None:
                start = t
    return dict(frame=frame, onset=onset, notes=np.asarray(notes).reshape(-1, 3))
