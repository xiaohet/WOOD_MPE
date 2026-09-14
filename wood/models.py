"""Original WOOD architecture with aligned salience frames."""
import torch
from torch import nn
from torch.nn import functional as F

def crop_to_match(enc, dec):
    '''
    Center-crop enc to match dec's time/freq dims.
    '''
    _,_,Te,Fe = enc.shape
    _,_,Td,Fd = dec.shape
    te = (Te - Td) // 2
    fe = (Fe - Fd) // 2
    return enc[:, :, te:te+Td, fe:fe+Fd]

class UNetMPE(nn.Module):
  def __init__(self, freq_bins=84, pitches=128, h=5):
    super().__init__()
    def conv_block(in_c, out_c):
      return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
      )

    self.enc1 = conv_block(h, 16)
    self.enc2 = conv_block(16, 32)
    self.enc3 = conv_block(32, 64)

    self.pool = nn.MaxPool2d((2,2))
    self.bottleneck = conv_block(64, 128)

    self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
    self.dec3 = conv_block(128, 64)

    self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
    self.dec2 = conv_block(64, 32)

    self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
    self.dec1 = conv_block(32, 16)

    self.out_conv = nn.Conv2d(16, pitches, kernel_size=1)

  def forward(self, x):
    original_time = x.shape[2]
    x = F.pad(x, (0, (-x.shape[3]) % 8, 0, (-x.shape[2]) % 8))
    e1 = self.enc1(x)
    p1 = self.pool(e1)

    e2 = self.enc2(p1)
    p2 = self.pool(e2)

    e3 = self.enc3(p2)
    p3 = self.pool(e3)

    b = self.bottleneck(p3)

    u3 = self.up3(b)
    e3_cropped = crop_to_match(e3, u3)
    u3 = torch.cat([u3, e3_cropped], dim=1)
    d3 = self.dec3(u3)

    u2 = self.up2(d3)
    e2_cropped = crop_to_match(e2, u2)
    u2 = torch.cat([u2, e2_cropped], dim=1)
    d2 = self.dec2(u2)

    u1 = self.up1(d2)
    e1_cropped = crop_to_match(e1, u1)
    u1 = torch.cat([u1, e1_cropped], dim=1)
    d1 = self.dec1(u1)

    out = self.out_conv(d1)
    out = out.mean(dim=-1)
    out = out.transpose(1,2)
    return out[:, :original_time]

class WOODModel(nn.Module):
  def __init__(self, n_events=7, n_pitches=84, d_model=256, h=5):
    super().__init__()
    self.n = n_events
    self.h = h

    # windowed HCQT encoder
    self.win_conv = nn.Sequential(
      nn.Conv2d(h, 32, (3,3), padding=(1,1)),
      nn.ReLU(),
      nn.Conv2d(32, 64, (3,3), padding=(1,1)),
      nn.ReLU(),
      nn.Conv2d(64, d_model, (3,3), padding=(1,1)),
      nn.ReLU(),
    )

    # windowed HCQT salience encoder
    self.salience_net = UNetMPE(freq_bins=n_pitches, pitches=n_pitches, h=h)
    self.salience_proj = nn.Linear(n_pitches, d_model)

    # exemplar HCQT encoders
    self.ex_conv = nn.Sequential(
      nn.Conv2d(h, 32, (3,3), padding=(1,1)),
      nn.ReLU(),
      nn.Conv2d(32, 64, (3,3), padding=(1,1)),
      nn.ReLU(),
      nn.Conv2d(64, d_model, (3,3), padding=(1,1)),
      nn.ReLU(),
    )
    self.ex_pool = nn.AdaptiveAvgPool2d((1,1))
    self.ex_pitch_emb = nn.Embedding(129, d_model)
    self.ex_vel_proj = nn.Linear(1, d_model)

    # note event encoders
    self.pitch_emb = nn.Embedding(129, d_model)
    self.vel_emb = nn.Embedding(129, d_model)

    self.dur_mlp = nn.Sequential(
      nn.Linear(1, d_model),
      nn.ReLU(),
      nn.Linear(d_model, d_model)
    )

    self.row_mlp = nn.Sequential(
      nn.Linear(d_model*3, d_model),
      nn.ReLU(),
      nn.Linear(d_model, d_model)
    )
    self.prev_gate = nn.Parameter(torch.tensor(0.5))

    # fuse embeddings
    self.fusion_mlp = nn.Sequential(
      nn.Linear(d_model*4, d_model),
      nn.ReLU(),
      nn.Linear(d_model, d_model)
    )

    # temporal model
    self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=False)

    # output heads
    self.onset_head = nn.Linear(d_model, n_pitches)
    self.offset_head = nn.Linear(d_model, n_pitches)
    self.vel_head    = nn.Linear(d_model, n_pitches)

  def forward(self, full_win, ex, ex_pitch, ex_vel, prev_state):
    """
    full_win: (B, H, Tw, F), previous + current window
    ex: (B, H, Te, F)
    prev_state: (B, T_previous, n, 3)
    """

    B = full_win.shape[0]

    # window encoding
    w = full_win.transpose(2,3)
    w_h = self.win_conv(w)
    w_h = w_h.mean(2).transpose(1,2) # (B,Tw,d)
    Tw = w_h.size(1) # total window length
    hw = Tw // 2  # half window length

    # salience U-Net model
    sal = self.salience_net(full_win) # (B, T_sal, n_pitches)
    sal_h = self.salience_proj(sal)

    # exemplar embeddings
    ex_h = self.ex_conv(ex)
    ex_h = self.ex_pool(ex_h).squeeze(-1).squeeze(-1) # (B,d)

    ex_pitch_emb = self.ex_pitch_emb(ex_pitch) # (B,d)
    ex_vel_emb   = self.ex_vel_proj(ex_vel)
    ex_all = ex_h + ex_pitch_emb + ex_vel_emb

    # prev_state encoding
    prev_state = prev_state[:,:,:self.n,:]
    pitch_idx = prev_state[:,:,:,0].long() # (B,Tw,n)
    vel_idx = prev_state[:,:,:,1].long() # (B,Tw,n)
    dur = prev_state[:,:,:,2:3].float() # (B,Tw,n,1)

    pitch_vec = self.pitch_emb(pitch_idx) # (B,n,d)
    vel_vec = self.vel_emb(vel_idx) # (B,n,d)
    dur_vec = self.dur_mlp(dur) # (B,n,d)

    row_vec = self.row_mlp(torch.cat([pitch_vec, vel_vec, dur_vec], dim=-1))
    rows_agg = row_vec.mean(dim=(1,2)) # (B,d)
    rows_agg = self.prev_gate * rows_agg

    # fusion per time frame
    ex_b = ex_all.unsqueeze(1).expand(-1, Tw, -1) # (B,Tw,d)
    rows_b = rows_agg.unsqueeze(1).expand(-1, Tw, -1) # (B,Tw,d)
    fused = self.fusion_mlp(torch.cat([w_h, sal_h, ex_b, rows_b], dim=-1)) # (B,Tw,d)

    # temporal model
    out_seq, _ = self.gru(fused)
    out_seq = out_seq[:, hw:, :] # only take a size of number of current frames

    # heads
    onset  = self.onset_head(out_seq)
    offset = self.offset_head(out_seq)
    vel = torch.sigmoid(self.vel_head(out_seq))

    return onset, offset, vel
