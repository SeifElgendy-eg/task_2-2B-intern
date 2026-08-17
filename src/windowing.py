import pandas as pd

from .config import MIN_MAJORITY
from .labeling import cudb_label_intervals_v2, vfdb_label_intervals


def window_record(intervals, record_len, window_s, step_s, fs, min_majority=0.5):
    """Slide fixed-size windows across a record; label each by majority class."""
    window_samples = int(window_s * fs)
    step_samples = int(step_s * fs)
    windows = []

    pos = 0
    while pos + window_samples <= record_len:
        w_start, w_end = pos, pos + window_samples
        class_time = {}
        for s, e, cls in intervals:
            overlap = max(0, min(e, w_end) - max(s, w_start))
            if overlap > 0:
                class_time[cls] = class_time.get(cls, 0) + overlap
        if class_time:
            best_cls, best_time = max(class_time.items(), key=lambda x: x[1])
            if best_time / window_samples >= min_majority:
                windows.append((w_start, w_end, best_cls))
        pos += step_samples
    return windows


def build_windowed_dataset(record_list, is_cudb_list, window_s, step_s,
                           cudb_path, vfdb_path, fs, min_majority=MIN_MAJORITY):
    """Build a window table from the final CUDB/VFDB interval builders."""
    cudb_set = set(is_cudb_list)
    all_windows = []

    for rec in record_list:
        if rec in cudb_set:
            intervals, record_len = cudb_label_intervals_v2(rec, cudb_path, fs)
            db = 'CUDB'
        else:
            intervals, record_len = vfdb_label_intervals(rec, vfdb_path)
            db = 'VFDB'

        for start, end, label in window_record(
            intervals, record_len, window_s, step_s, fs, min_majority
        ):
            all_windows.append({
                'record': rec,
                'db': db,
                'start': start,
                'end': end,
                'label': label,
            })

    return pd.DataFrame(all_windows)
