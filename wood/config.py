from dataclasses import asdict, dataclass
import torch


@dataclass
class Config:
    sr: int = 44100
    hop: int = 512
    n_pitches: int = 84
    min_midi: int = 24
    n_events: int = 8
    d_model: int = 256
    win_sec: float = 1.0
    seed: int = 42

    @property
    def window(self):
        import math
        return math.ceil(self.win_sec * self.sr / self.hop)

    def to_dict(self):
        return asdict(self)


def get_device(name='auto'):
    if name == 'auto':
        if torch.cuda.is_available():
            name = 'cuda'
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            name = 'xpu'
        elif torch.backends.mps.is_available():
            name = 'mps'
        else:
            name = 'cpu'
    return torch.device(name)
