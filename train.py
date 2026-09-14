"""Train WOOD with song-level validation and autoregressive threshold selection."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from wood.config import Config, get_device
from wood.data import WindowDataset, song_folders, exclude_held_out, audio_hash
from wood.engine import batch_loss, make_model, seed_all
from wood.decoding import predict_song
from wood.metrics import score_song, macro_average


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='Data/train')
    p.add_argument('--validation', help='Separate validation directory; otherwise hold out training songs')
    p.add_argument('--test-data', default='Data/test', help='Held-out directory: matching audio is excluded from training/validation')
    p.add_argument('--output', default='runs/baseline')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-events', type=int, default=8)
    p.add_argument('--max-batches', type=int, help='Limit training batches per epoch for smoke checks')
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.001, 0.01, 0.1, 0.5])
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or any(not 0 < t < 1 for t in args.thresholds):
        p.error('epochs/batch-size must be positive and thresholds must be between 0 and 1')
    cfg = Config(seed=args.seed, n_events=args.n_events)
    seed_all(cfg.seed)

    folders = song_folders(args.data)
    held_out = song_folders(args.test_data)
    original = folders
    folders = exclude_held_out(folders, held_out)
    excluded = [str(p) for p in original if p not in folders]
    if excluded:
        print(f'Excluded exact test-audio duplicates: {excluded}', flush=True)
    if args.validation:
        valid = exclude_held_out(song_folders(args.validation), held_out)
        if {audio_hash(x) for x in valid} & {audio_hash(x) for x in folders}:
            p.error('Training and validation audio overlap')
    else:
        import random
        random.Random(cfg.seed).shuffle(folders)
        if len(folders) < 2:
            p.error('Need at least two songs for a train/validation split')
        import math
        n_valid = max(1, math.ceil(len(folders)*0.2))
        valid, folders = folders[:n_valid], folders[n_valid:]
    if not folders or not valid:
        p.error('No training or validation songs remain after excluding held-out audio')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'best.pt').exists():
        p.error('Output already contains best.pt; choose a new --output directory')
    metadata = dict(config=cfg.to_dict(), arguments=vars(args),
                    train=[str(x.resolve()) for x in folders], validation=[str(x.resolve()) for x in valid],
                    excluded_test_duplicates=excluded, torch_version=str(torch.__version__))
    (output/'config.json').write_text(json.dumps(metadata, indent=2))
    print('Loading training and validation features...', flush=True)
    
    training, validation = WindowDataset(folders, cfg), WindowDataset(valid, cfg)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    device = get_device(args.device)
    print(f'Device: {device}; train songs: {len(folders)}; validation songs: {len(valid)}', flush=True)
    model = make_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float('inf')
    history = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, device)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if args.max_batches and step+1 >= args.max_batches:
                break
        model.eval()
        with torch.inference_mode():
            val_loss = sum(batch_loss(model, b, device).item()*len(b['full']) for b in val_loader)/len(validation)
        row = dict(epoch=epoch+1, train_loss=sum(losses)/len(losses), validation_loss=val_loss)
        history.append(row)
        print(row, flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg.to_dict(),
                            epoch=epoch+1, validation_loss=best, format_version=1,
                            development_audio_hashes=[audio_hash(x) for x in folders+valid]), output/'best.pt')
        (output/'history.json').write_text(json.dumps(history, indent=2))
    checkpoint = torch.load(output/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    best_score, chosen, trials = -1, None, []

    for on in args.thresholds:
        for off in args.thresholds:
            metrics = macro_average([score_song(s, predict_song(model, s, cfg, device, on, off)) for s in validation.songs])
            trials.append(dict(onset=on, offset=off, **metrics))
            if metrics['note_f1'] > best_score:
                best_score, chosen = metrics['note_f1'], dict(onset=on, offset=off)
    checkpoint['thresholds'] = chosen
    torch.save(checkpoint, output/'best.pt')
    (output/'threshold_search.json').write_text(json.dumps(trials, indent=2))
    print(f'Saved {output / "best.pt"}; validation-selected thresholds: {chosen}')


if __name__ == '__main__':
    main()
