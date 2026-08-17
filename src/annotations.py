import os
import wfdb
import numpy as np

from .config import SHOCK_WINDOW_S

DANGEROUS_LABELS = {'(VF', '(VFL', '(VFIB', '(VT', '(SVTA'}


def cudb_episodes(rec, base_path):
    """Return ([(start, end)] episode boundaries, record_len, unclosed_flag) for a CUDB
    record, derived from its '[' / ']' rhythm markers."""
    rec_path = os.path.join(base_path, rec)
    ann = wfdb.rdann(rec_path, 'atr')
    record_len = wfdb.rdrecord(rec_path).p_signal.shape[0]
    markers = [(s, sym) for s, sym in zip(ann.sample, ann.symbol) if sym in ['[', ']']]

    episodes = []
    unclosed = False
    open_sample = None
    for s, sym in markers:
        if sym == '[':
            open_sample = s
        elif sym == ']' and open_sample is not None:
            episodes.append((open_sample, s))
            open_sample = None
    if open_sample is not None:
        episodes.append((open_sample, record_len - 1))
        unclosed = True
    return episodes, record_len, unclosed


def vfdb_episodes(rec, base_path):
    """Return ([(start, end)] episode boundaries, record_len, unclosed_flag) for a VFDB
    record, derived from aux_note labels found in DANGEROUS_LABELS."""
    rec_path = os.path.join(base_path, rec)
    ann = wfdb.rdann(rec_path, 'atr')
    record_len = wfdb.rdrecord(rec_path).p_signal.shape[0]
    samples = ann.sample
    notes = [n.strip('\x00') for n in ann.aux_note]
    segments = [(s, n) for s, n in zip(samples, notes) if n]

    episodes = []
    for i, (s, label) in enumerate(segments):
        if label in DANGEROUS_LABELS:
            end = segments[i + 1][0] if i + 1 < len(segments) else record_len - 1
            episodes.append((s, end))
    unclosed = len(segments) > 0 and segments[-1][1] in DANGEROUS_LABELS
    return episodes, record_len, unclosed


def record_metrics(signal, episodes, fs):
    """Compute NaN, clipping, episode-duration, and shock-boundary summary metrics for one record."""
    record_len = len(signal)
    nan_mask = np.isnan(signal)
    nan_count = nan_mask.sum()
    nan_pct = 100 * nan_count / record_len

    valid = signal[~nan_mask]
    if len(valid) == 0:
        return dict(nan_count=nan_count, nan_pct=nan_pct, clip_count=0,
                    total_episode_duration=0, shock_boundaries=0)

    sig_min, sig_max = valid.min(), valid.max()
    clip_count = int(np.sum((signal == sig_min) | (signal == sig_max)))
    if clip_count <= 2:
        clip_count = 0

    total_duration = sum((e - s) / fs for s, e in episodes)

    shock_boundaries = 0
    window = int(SHOCK_WINDOW_S * fs)
    for s, e in episodes:
        lo, hi = max(0, e - window), min(record_len, e + window)
        local = signal[lo:hi]
        has_nan = np.isnan(local).any()
        has_clip = clip_count > 0 and (np.sum((local == sig_min) | (local == sig_max)) > 0)
        if has_nan or has_clip:
            shock_boundaries += 1

    return dict(nan_count=int(nan_count), nan_pct=round(nan_pct, 2), clip_count=clip_count,
                total_episode_duration=round(total_duration, 1), shock_boundaries=shock_boundaries)


def find_nan_blocks(signal):
    """Return a list of (start, end) sample indices for contiguous NaN blocks in a signal."""
    nan_mask = np.isnan(signal)
    diffs = np.diff(nan_mask.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if nan_mask[0]:
        starts = np.insert(starts, 0, 0)
    if nan_mask[-1]:
        ends = np.append(ends, len(signal))
    return list(zip(starts, ends))
