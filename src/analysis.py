from collections import Counter

import numpy as np
from scipy.signal import find_peaks

from .io import get_annotations


def compute_rr_stats(signal, start_sample, fs, window_sec=10, onset_offset=0,
                     height=None, height_offset=None, distance=100, label=''):
    """Detect peaks in a signal window and report beat count / RR-interval stats (mean, std, CV)."""
    window = signal[start_sample + onset_offset: start_sample + onset_offset + fs * window_sec]
    if height is None:
        height = window.mean() + (height_offset if height_offset is not None else 0)
    peaks, _ = find_peaks(window, height=height, distance=distance)
    if len(peaks) > 1:
        rr = np.diff(peaks) / fs
        print(f"{label} @ {start_sample/fs:.0f}s: n_beats={len(peaks)}, mean RR={rr.mean():.3f}, "
              f"std RR={rr.std():.3f}, CV={rr.std()/rr.mean():.3f}")
        return rr
    print(f"{label} @ {start_sample/fs:.0f}s: only {len(peaks)} peak(s) found — no clear beat structure")
    return None


def label_neighbor_counts(records_list, base_path, target_label):
    """Count which labels most often appear immediately before/after target_label across records."""
    before, after = Counter(), Counter()
    for rec in records_list:
        _, _, notes_raw = get_annotations(rec, base_path)
        notes = [n for n in notes_raw if n]
        for i, note in enumerate(notes):
            if note == target_label:
                if i > 0:
                    before[notes[i - 1]] += 1
                if i + 1 < len(notes):
                    after[notes[i + 1]] += 1
    return before, after
