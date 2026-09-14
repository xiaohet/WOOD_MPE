"""Transcribe a folder containing full.wav and exemplar_NOTE_VELOCITY_TEMPO.wav."""
import argparse
from pathlib import Path
import pandas as pd
from wood.config import get_device
from wood.data import load_audio_song
from wood.engine import load_checkpoint
from wood.decoding import predict_song


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder')
    p.add_argument('--checkpoint', default='runs/baseline/best.pt')
    p.add_argument('--device', default='auto')
    p.add_argument('--output', default='runs/transcription.csv')
    args = p.parse_args()
    device = get_device(args.device)
    model, cfg, ckpt = load_checkpoint(args.checkpoint, device)
    th = ckpt.get('thresholds', dict(onset=0.5, offset=0.5))
    pred = predict_song(model, load_audio_song(args.folder, cfg), cfg, device, th['onset'], th['offset'])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred['notes'], columns=['pitch', 'start_sec', 'end_sec']).to_csv(args.output, index=False)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
