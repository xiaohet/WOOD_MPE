import random
import numpy as np
import torch
from .models import WOODModel
from .config import Config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(cfg):
    return WOODModel(n_events=cfg.n_events, n_pitches=cfg.n_pitches, d_model=cfg.d_model)


def batch_loss(model, batch, device):
    b = {k:v.to(device) for k, v in batch.items()}
    on, off, vel = model(b['full'], b['ex'], b['pitch'], b['vel'], b['previous'])
    mask = b['mask'][..., None]
    denom = (mask.sum() * on.shape[-1]).clamp_min(1)
    on_loss = (torch.nn.functional.binary_cross_entropy_with_logits(on, b['onset'], reduction='none')*mask).sum()/denom
    off_loss = (torch.nn.functional.binary_cross_entropy_with_logits(off, b['offset'], reduction='none')*mask).sum()/denom
    vel_mask = mask*b['onset']
    vel_loss = ((vel-b['velocity']).abs()*vel_mask).sum()/vel_mask.sum().clamp_min(1)
    return on_loss + off_loss + 0.5*vel_loss


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'model' not in checkpoint:
        raise ValueError('This is a legacy checkpoint. Use the archived notebook for historical reproduction; retrain the corrected model with train.py.')
    cfg = Config(**checkpoint['config'])
    model = make_model(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    return model, cfg, checkpoint
