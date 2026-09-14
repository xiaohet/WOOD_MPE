# WOOD multi-pitch transcription

Exemplar-conditioned transcription of polyphonic audio. The original HCQT + U-Net + GRU architecture is retained, with corrected data handling and temporal alignment.

## Quick start on Windows (CPU)

Run from this project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe train.py --device cpu --batch-size 2
.\.venv\Scripts\python.exe evaluate.py --device cpu
.\.venv\Scripts\python.exe transcribe.py Data/test/03 --device cpu
```

Replace the example song folder with an actual folder in Data/test. Training on CPU may be slow. For a pipeline smoke check use `--epochs 1 --max-batches 1 --thresholds 0.5 --output runs/smoke`; this does not produce a useful trained model. HCQT preprocessing runs on CPU, including on GPU machines. Features are held in RAM for the small supplied dataset.

## Colab GPU with Python files

Python files use the same GPU as notebook cells. Upload/copy the entire project folder to Google Drive (exclude `.venv`), open `WOOD_codes.ipynb` in Colab, and select **Runtime > Change runtime type > GPU**. Its setup cell mounts Drive and changes into `/content/drive/MyDrive/WOOD_MPE` (edit this path if needed).

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/WOOD_MPE
%pip install -r requirements.txt
!python train.py --device cuda --output runs/colab
!python evaluate.py --device cuda --checkpoint runs/colab/best.pt --output runs/colab/test_metrics.json
```

The GPU belongs to the remote Colab runtime; your PC needs no CUDA hardware. Colab GPU availability and runtime limits vary: https://research.google.com/colaboratory/faq.html . For faster file access you can copy the data to `/content`, use `--data /content/Data/train`, and keep `--output` on Drive so checkpoints survive a runtime reset.

Other remote GPU machines can run the same commands. Locally, CUDA requires compatible NVIDIA hardware; supported AMD GPUs can use ROCm and supported Intel GPUs can use PyTorch XPU, subject to hardware/OS/build compatibility. Check the official installation instructions: https://pytorch.org/get-started/locally/ . `--device auto` tries CUDA/ROCm, XPU, MPS, then CPU. Only CPU has been exercised here; other backends must support this model's operations. Explicit `--device cuda` will fail rather than silently train on CPU when unavailable.

Hardware-specific guides: [AMD compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibility.html), [Intel XPU setup](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html). The installed local environment is a CPU build; its lack of CUDA/XPU availability does not by itself identify the physical GPU in your PC.

## Files and data

- `wood/features.py`: HCQT extraction and boundary padding.
- `wood/data.py`: shared audio/annotation loading and training windows.
- `wood/models.py`: U-Net and WOODModel.
- `wood/decoding.py`: autoregressive inference and note reconstruction.
- `wood/metrics.py`: frame precision/recall/F1 and tolerance-based note metrics.
- `wood/engine.py`: losses, checkpoints and reproducible seeds.
- `train.py`, `evaluate.py`, `transcribe.py`: command-line entry points (`--help`).

Each song folder contains `full.wav`, one `exemplar_NOTE_VELOCITY_TEMPO.wav`, and (for training/evaluation) `midi_export.csv` with pitch, start_sec, end_sec, velocity columns. Transcription requires no annotation CSV. Output CSV contains MIDI pitch, start time and end time in seconds. The velocity head is trained and used internally but exported notes currently omit velocity.

Defaults: 44.1 kHz, 512-sample hop, MIDI 24–107, five harmonic channels, eight simultaneous note slots (training song 23 reaches eight). Exceeding the slot limit raises an error; increase `--n-events` rather than silently dropping notes. Out-of-range reference pitches are excluded from metrics. Sample rate and pitch settings live in `wood/config.py` and are stored in checkpoints.

## Validation and reproducibility

The supplied training directory has eight songs, but train/12 and test/12 contain byte-identical audio and annotations. Training excludes audio matching `--test-data` (default `Data/test`) by SHA-256 without changing files or using test labels. Of the seven remaining songs, the seeded split reserves two for validation and trains on five. Evaluation checks test audio against the checkpoint's training/validation hashes and rejects overlap. Test labels are used only in evaluation. Exact hashing cannot detect different renders of the same composition; organize a separate validation directory with `--validation` if related material crosses splits. If using a different dataset, supply its held-out directory through `--test-data`.

Training saves the best validation-loss checkpoint, loss history, configuration, split paths, optimizer state, and epoch. It then searches onset/offset thresholds on complete validation songs using autoregressive note F1, saving the selected values in the checkpoint. Evaluation uses those thresholds unchanged and reports per-song and macro-averaged frame and note metrics. Note matching uses mir_eval defaults: 50 ms onset tolerance, 50 cents pitch tolerance, and (for offset-aware scores) max(50 ms, 20% of reference duration).

`requirements.txt` gives portable dependency bounds. `requirements-local-lock.txt`, when present, records the CPU environment used for verification; it is not a CUDA installation recipe. For a GPU environment use the appropriate PyTorch build and save that environment's own dependency versions.

## Corrections and remaining modeling limitations

The U-Net now pads its input on the right to multiples of eight, decodes at that size, then trims to the original timeline. The extra temporal average-pooling and zero-padding were removed. Previous-note history preserves the time dimension. Training and inference start windows at frame zero. Boundary audio padding uses the feature silence floor; padded target frames are excluded from losses. HCQT normalization handles silence and passes the configured hop explicitly. Data loading no longer depends on notebook globals.

The decoder supports rearticulated pitches and preserves onset state across window boundaries. Training still conditions on ground-truth note history while inference conditions on predictions; this exposure mismatch remains a research limitation. The original architecture also uses the full current window, so it is not sample-by-sample causal. Sparse onset/offset targets and lack of a direct frame head remain modeling choices to investigate after establishing a baseline.

`notebooks/WOOD_original.ipynb` preserves the original notebook byte-for-byte. `Model_Example/mpe_exemplar_wood.pth` remains untouched. Corrected preprocessing/alignment changes the model's behavior, so new entry points require a new-format checkpoint from retraining and reject the legacy checkpoint with an explanation. Do not interpret smoke-check scores as evidence of improved transcription performance.

## Verified in this workspace

Eleven regression tests passed, and the Python modules compile. A CPU smoke run completed one optimizer update on actual audio, a complete validation pass, threshold selection, checkpoint save/load, evaluation of all three test songs, and CSV transcription of a two-second audio-only excerpt. Results and the verification record are in `runs/smoke_verified/`. This checkpoint is only for pipeline verification. Full retraining and GPU execution have not been performed. Librosa emits a short-signal padding warning for the one-second exemplar; the silence test verifies finite features.
