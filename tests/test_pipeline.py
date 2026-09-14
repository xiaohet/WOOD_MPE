import json
from pathlib import Path
import numpy as np
import pytest
import torch
from wood.config import Config
from wood.data import empty_state, WindowDataset, exclude_held_out
from wood.features import full_window, compute_hcqt
from wood.models import UNetMPE
from wood.engine import make_model, batch_loss
from wood.decoding import update_state, predict_song
from wood.metrics import score_song


torch.set_num_threads(2)


def test_boundary_window():
    x = torch.arange(3.).reshape(1, 3, 1)
    assert full_window(x, 0, 4).flatten().tolist() == [-80]*4 + [0, 1, 2, -80]
    assert full_window(x, 2, 2).flatten().tolist() == [0, 1, 2, -80]


def test_content_duplicate_exclusion(tmp_path):
    folders = [tmp_path / name for name in ('train', 'other', 'test')]
    for folder, content in zip(folders, [b'same', b'different', b'same']):
        folder.mkdir()
        (folder / 'full.wav').write_bytes(content)
    assert exclude_held_out(folders[:2], folders[2:]) == [folders[1]]


def test_silence_features():
    s = compute_hcqt(np.zeros(44100, dtype=np.float32), Config())
    assert torch.isfinite(s).all()
    assert (s == -80).all()


@pytest.mark.parametrize('length', [16, 174, 178])
def test_salience_preserves_time(length):
    net = UNetMPE(pitches=84).eval()
    with torch.no_grad():
        assert net(torch.randn(1, 5, length, 84)).shape == (1, length, 84)


def test_history_and_padding_loss():
    cfg = Config(d_model=16)
    ds = WindowDataset.__new__(WindowDataset)
    ds.cfg = cfg
    T, w = cfg.window+2, cfg.window
    state = empty_state(T, cfg)
    state[:, 0, 2] = torch.arange(T)
    song = dict(features=torch.zeros(5, T, 84), exemplar=torch.zeros(5, w, 84),
                pitch_ex=60, vel_ex=100, state=state)
    song.update({key:torch.zeros(T, 84) for key in ('frame', 'onset', 'offset', 'velocity')})
    ds.songs, ds.index = [song], [(0, w)]
    sample = ds[0]
    assert sample['previous'].shape == (w, cfg.n_events, 3)
    assert torch.equal(sample['previous'][:, 0, 2], torch.arange(w))
    assert sample['mask'].sum() == 2
    batch = {k:v.unsqueeze(0) for k, v in sample.items()}
    model = make_model(cfg).eval()
    loss = batch_loss(model, batch, 'cpu')
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    for key in ('onset', 'offset', 'velocity'):
        batch[key][:, 2:] = 1
    assert torch.allclose(loss, batch_loss(model, batch, 'cpu'))


def test_repeated_note_and_offset():
    cfg = Config()
    on, off = torch.zeros(4, 84, dtype=torch.bool), torch.zeros(4, 84, dtype=torch.bool)
    on[0, 40] = on[2, 40] = True
    off[3, 40] = True
    states = update_state(empty_state(1, cfg)[0], on, off, torch.ones(4, 84)*90, cfg)
    assert states[0, 0, 2] == 0
    assert states[1, 0, 2] == 1
    assert states[2, 0, 2] == 0
    assert states[3, 0, 0] == cfg.n_pitches


def test_metric_perfect_notes():
    frame = torch.tensor([[0., 1.], [0., 1.]])
    ref = dict(frame=frame, intervals=np.array([[0., 1.]]), pitches=np.array([60]))
    pred = dict(frame=frame, notes=np.array([[60., 0., 1.]]))
    result = score_song(ref, pred)
    assert result['frame_f1'] == result['note_with_offset_f1'] == 1


def test_inference_carries_state_and_trims_tail():
    cfg = Config(win_sec=4*512/44100)

    class FixedModel(torch.nn.Module):
        def forward(self, full, ex, pitch, vel, previous):
            on = torch.full((1, 4, 84), -100.)
            off = torch.full_like(on, -100.)
            # A continuously high onset across the window boundary must not
            # retrigger the same sustained note in the second window.
            on[0, :, 40] = 100
            return on, off, torch.ones_like(on)*0.5

    song = dict(features=torch.zeros(5, 6, 84), exemplar=torch.zeros(5, 4, 84),
                pitch_ex=60, vel_ex=100)
    result = predict_song(FixedModel(), song, cfg, 'cpu')
    assert result['frame'].shape == (6, 84)
    assert result['frame'][:, 40].sum() == 6
    assert len(result['notes']) == 1
    assert result['notes'][0, 2] == pytest.approx(6*512/44100)


def test_notebook_is_valid_json_and_python():
    nb = json.loads(Path('WOOD_codes.ipynb').read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<notebook>', 'exec')
