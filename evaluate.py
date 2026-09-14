"""Evaluate every song in the held-out test directory."""
import argparse
import json
from pathlib import Path
from wood.config import get_device
from wood.data import load_song, song_folders, audio_hash
from wood.engine import load_checkpoint
from wood.decoding import predict_song
from wood.metrics import score_song, macro_average


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/baseline/best.pt')
    p.add_argument('--data', default='Data/test')
    p.add_argument('--device', default='auto')
    p.add_argument('--output', default='runs/baseline/test_metrics.json')
    args = p.parse_args()
    device = get_device(args.device)
    model, cfg, ckpt = load_checkpoint(args.checkpoint, device)
    thresholds = ckpt.get('thresholds', dict(onset=0.5, offset=0.5))
    rows = {}
    for folder in song_folders(args.data):
        if audio_hash(folder) in ckpt.get('development_audio_hashes', []):
            raise ValueError(f'{folder}: audio overlaps checkpoint training/validation data')
        song = load_song(folder, cfg)
        prediction = predict_song(model, song, cfg, device, thresholds['onset'], thresholds['offset'])
        rows[folder.name] = score_song(song, prediction)
        print(folder.name, rows[folder.name], flush=True)
    result = dict(per_song=rows, macro_average=macro_average(list(rows.values())), thresholds=thresholds)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(result['macro_average'])


if __name__ == '__main__':
    main()
