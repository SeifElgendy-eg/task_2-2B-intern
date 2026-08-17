import numpy as np

from .io import load_signal, get_annotations
from .config import (
    EXCLUDED_RECORDS, TRANSITIONAL_PADDING_S, SPIKE_EVENT_GAP_S,
    SPIKE_EVENT_TOL,
)
from .annotations import cudb_episodes, vfdb_episodes, find_nan_blocks


def class_breakdown_pct(intervals, record_len, fs):
    """Given (start, end, class) intervals, return % of record time spent in each of the 3 classes."""
    totals = {'Normal': 0, 'Dangerous': 0, 'Transitional': 0}
    for s, e, cls in intervals:
        totals[cls] += (e - s)
    total_s = record_len / fs
    return {k: round(100 * v / fs / total_s, 1) for k, v in totals.items()}


CLASS_MAP_VFDB = {
    '(N': 'Normal', '(PM': 'Normal', '(B': 'Normal', '(BI': 'Normal', '(NOD': 'Normal',
    '(VT': 'Dangerous', '(SVTA': 'Dangerous', '(VF': 'Dangerous',
    '(VFL': 'Dangerous', '(VFIB': 'Dangerous', '(AFIB': 'Dangerous',
    '(ASYS': 'Transitional', '(VER': 'Transitional',
    '(HGEA': 'Transitional', '(SBR': 'Transitional', '(NSR': 'Transitional',
    '(NOISE': None,
}


def vfdb_label_intervals(rec, vfdb_path):
    if rec in EXCLUDED_RECORDS:
        return [], 0
    samples, _, notes = get_annotations(rec, vfdb_path)
    segments = [(s, n) for s, n in zip(samples, notes) if n]
    _, record_len = load_signal(rec, vfdb_path)

    intervals = []
    for i, (s, label) in enumerate(segments):
        end = segments[i+1][0] if i+1 < len(segments) else record_len
        cls = CLASS_MAP_VFDB.get(label)
        if cls is not None:
            intervals.append((s, end, cls))
    return intervals, record_len


def cudb_label_intervals(rec, cudb_path):
    samples, symbols, _ = get_annotations(rec, cudb_path)
    _, record_len = load_signal(rec, cudb_path)
    episodes, _, unclosed = cudb_episodes(rec, cudb_path)
    tilde_samples = [s for s, sym in zip(samples, symbols) if sym == '~']

    intervals = []
    prev_end = 0
    for (s, e) in episodes:
        if s > prev_end:
            has_tilde = any(prev_end <= t < s for t in tilde_samples)
            intervals.append((prev_end, s, 'Transitional' if has_tilde else 'Normal'))
        intervals.append((s, e, 'Dangerous'))
        prev_end = e
    if prev_end < record_len:
        has_tilde = any(prev_end <= t < record_len for t in tilde_samples)
        intervals.append((prev_end, record_len, 'Transitional' if has_tilde else 'Normal'))
    return intervals, record_len


def evidence_based_transitional_regions(rec, cudb_path, fs, pad_s=TRANSITIONAL_PADDING_S):
    """Find Transitional regions using actual NaN/clipping evidence, not a fixed window guess."""
    signal, record_len = load_signal(rec, cudb_path)
    nan_blocks = find_nan_blocks(signal)

    valid = signal[~np.isnan(signal)]
    sig_min, sig_max = valid.min(), valid.max()
    clip_mask = (signal == sig_min) | (signal == sig_max)
    clip_indices = np.where(clip_mask)[0]

    pad = int(pad_s * fs)
    regions = []
    for s, e in nan_blocks:
        regions.append((max(0, s - pad), min(record_len, e + pad)))
    if len(clip_indices) > 2:
        breaks = np.where(np.diff(clip_indices) > fs)[0]
        starts = np.insert(clip_indices[breaks + 1], 0, clip_indices[0])
        ends = np.append(clip_indices[breaks], clip_indices[-1])
        for s, e in zip(starts, ends):
            regions.append((max(0, s - pad), min(record_len, e + pad)))

    regions.sort()
    merged = []
    for s, e in regions:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def count_spike_events(signal, start_sample, end_sample, fs, gap_s=SPIKE_EVENT_GAP_S, tol=SPIKE_EVENT_TOL):
    """Count discrete times the signal touches its global min/max ceiling within a window."""
    valid = signal[~np.isnan(signal)]
    sig_max, sig_min = valid.max(), valid.min()
    window = signal[start_sample:end_sample]
    at_extreme = (window >= sig_max - tol) | (window <= sig_min + tol)
    extreme_indices = np.where(at_extreme)[0]
    if len(extreme_indices) == 0:
        return 0
    breaks = np.where(np.diff(extreme_indices) > int(gap_s*fs))[0]
    starts = np.insert(extreme_indices[breaks+1], 0, extreme_indices[0])
    return len(starts)


def cudb_label_intervals_v2(rec, cudb_path, fs):
    signal, record_len = load_signal(rec, cudb_path)
    episodes, _, unclosed = cudb_episodes(rec, cudb_path)
    evidence_regions = evidence_based_transitional_regions(rec, cudb_path, fs)

    intervals = []
    prev_end = 0
    for (ep_s, ep_e) in episodes:
        if ep_s > prev_end:
            cursor = prev_end
            for r_s, r_e in evidence_regions:
                if r_e <= prev_end or r_s >= ep_s:
                    continue
                r_s, r_e = max(r_s, prev_end), min(r_e, ep_s)
                if r_s > cursor:
                    intervals.append((cursor, r_s, 'Normal'))
                intervals.append((r_s, r_e, 'Transitional'))
                cursor = r_e
            if cursor < ep_s:
                intervals.append((cursor, ep_s, 'Normal'))

        cursor = ep_s
        for r_s, r_e in evidence_regions:
            overlap_s, overlap_e = max(r_s, ep_s), min(r_e, ep_e)
            if overlap_s >= overlap_e:
                continue
            n_events = count_spike_events(signal, overlap_s, overlap_e, fs)
            if n_events >= 2:
                if overlap_s > cursor:
                    intervals.append((cursor, overlap_s, 'Dangerous'))
                intervals.append((overlap_s, overlap_e, 'Transitional'))
                cursor = overlap_e
        if cursor < ep_e:
            intervals.append((cursor, ep_e, 'Dangerous'))

        prev_end = ep_e

    if prev_end < record_len:
        cursor = prev_end
        for r_s, r_e in evidence_regions:
            if r_e <= prev_end:
                continue
            r_s, r_e = max(r_s, prev_end), min(r_e, record_len)
            if r_s > cursor:
                intervals.append((cursor, r_s, 'Normal'))
            intervals.append((r_s, r_e, 'Transitional'))
            cursor = r_e
        if cursor < record_len:
            intervals.append((cursor, record_len, 'Normal'))

    return intervals, record_len


def flag_special_handling_cudb(row):
    """Flag CUDB records with the existing signal-quality thresholds."""
    return (row['nan_pct'] > 0.9) or (row['clip_count'] > 990)


def flag_special_handling_vfdb(row):
    """Flag VFDB records with the existing short-fragment episode rule."""
    if row['n_episodes'] == 0:
        return False
    avg_episode_len = row['total_episode_duration'] / row['n_episodes']
    return avg_episode_len < 10


def vfdb_unclosed(rec, vfdb_path):
    """Return whether the final VFDB dangerous label continues to record end."""
    _, _, unclosed = vfdb_episodes(rec, vfdb_path)
    return unclosed
