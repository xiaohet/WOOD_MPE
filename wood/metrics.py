import librosa
import mir_eval
import numpy as np


def score_song(song, prediction):
    ref = song['frame'].numpy().astype(bool)
    est = prediction['frame'].numpy().astype(bool)
    tp, fp, fn = (ref & est).sum(), (~ref & est).sum(), (ref & ~est).sum()
    scores = dict(frame_precision=float(tp/max(tp+fp, 1)),
                  frame_recall=float(tp/max(tp+fn, 1)),
                  frame_f1=float(2*tp/max(2*tp+fp+fn, 1)))
    notes = prediction['notes']
    for label, ratio in [('note', None), ('note_with_offset', 0.2)]:
        precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            song['intervals'], librosa.midi_to_hz(song['pitches']),
            notes[:, 1:3], librosa.midi_to_hz(notes[:, 0]), offset_ratio=ratio)
        scores.update({f'{label}_precision':float(precision), f'{label}_recall':float(recall),
                       f'{label}_f1':float(f1)})
    return scores


def macro_average(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
