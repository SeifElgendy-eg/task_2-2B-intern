import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .io import load_signal, get_annotations


def plot_episodes(rec, base_path, episodes, fs, title=None, mark_nans=False, figsize=(40, 10)):
    """Plot a full record's waveform with shaded VF/VT episode regions; optionally mark NaN samples."""
    signal, record_len = load_signal(rec, base_path)
    time = np.arange(record_len) / fs
    plt.figure(figsize=figsize)
    plt.plot(time, signal, linewidth=0.5)
    for i, (start, end) in enumerate(episodes):
        plt.axvspan(start / fs, end / fs, color='red', alpha=0.15 if mark_nans else 0.2,
                    label='VF/VT episode' if i == 0 else None)
    if mark_nans:
        nan_mask = np.isnan(signal)
        plt.plot(time[nan_mask], np.zeros(nan_mask.sum()), '|', color='black',
                 markersize=15, label='NaN sample')
    plt.xlabel('Time (s)')
    plt.ylabel('ECG (mV)')
    plt.title(title or f'{rec} with annotated VF/VT episode(s)')
    plt.legend()
    plt.show()


def plot_rhythm_zones(rec, base_path, fs, figsize=(20, 6)):
    """Plot a full record with colored background zones for each labeled rhythm segment (VFDB-style)."""
    signal, record_len = load_signal(rec, base_path)
    samples, _, notes = get_annotations(rec, base_path)
    segments = [(s, n) for s, n in zip(samples, notes) if n]
    time = np.arange(record_len) / fs

    unique_labels = sorted(set(n for _, n in segments))
    cmap = plt.get_cmap('tab10')
    label_to_color = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                     gridspec_kw={'height_ratios': [4, 0.5]})
    ax1.plot(time, signal, linewidth=0.4, color='black')
    for i in range(len(segments)):
        start = segments[i][0]
        end = segments[i + 1][0] if i + 1 < len(segments) else record_len
        label = segments[i][1]
        ax1.axvspan(start / fs, end / fs, color=label_to_color[label], alpha=0.25)
    ax1.set_ylabel('ECG (mV)')
    ax1.set_title(f'{rec}: raw signal with colored rhythm zones')

    legend_handles = [Patch(facecolor=label_to_color[l], alpha=0.5, label=l) for l in unique_labels]
    ax1.legend(handles=legend_handles, loc='upper right', ncol=len(unique_labels) // 2 or 1, fontsize=8)

    for i in range(len(segments)):
        start = segments[i][0]
        end = segments[i + 1][0] if i + 1 < len(segments) else record_len
        label = segments[i][1]
        ax2.axvspan(start / fs, end / fs, color=label_to_color[label])
    ax2.set_yticks([])
    ax2.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def find_label_examples(records_list, base_path, target_label, max_examples=3):
    """Find up to max_examples (record, start_sample) occurrences of a given aux_note rhythm label."""
    examples = []
    for rec in records_list:
        samples, _, notes = get_annotations(rec, base_path)
        for s, note in zip(samples, notes):
            if note == target_label and len(examples) < max_examples:
                examples.append((rec, s))
    return examples


def plot_rhythm_examples(examples, base_path, label, fs, onset_offset=250, window_sec=10, color=None):
    """Plot a row of short signal windows, one per (record, start_sample) example of a rhythm label."""
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    for col, (rec, start_sample) in enumerate(examples):
        signal, _ = load_signal(rec, base_path)
        window = signal[start_sample + onset_offset: start_sample + onset_offset + fs * window_sec]
        time = np.arange(len(window)) / fs
        axes[col].plot(time, window, linewidth=0.7, color=color)
        axes[col].set_title(f"{label} — rec {rec} @ {start_sample/fs:.0f}s")
        axes[col].set_xlabel('Time (s)')
    axes[0].set_ylabel('ECG (mV)')
    plt.tight_layout()
    plt.show()


def plot_rhythm_examples_grid(label_to_examples, base_path, fs, onset_offset=250, window_sec=10):
    """Plot a grid of short signal windows: one row per label, columns = that label's examples."""
    labels = list(label_to_examples.keys())
    n_rows = len(labels)
    n_cols = max(len(v) for v in label_to_examples.values())
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for row, label in enumerate(labels):
        examples = label_to_examples[label]
        for col, (rec, start_sample) in enumerate(examples):
            signal, _ = load_signal(rec, base_path)
            window = signal[start_sample + onset_offset: start_sample + onset_offset + fs * window_sec]
            time = np.arange(len(window)) / fs
            ax = axes[row, col]
            ax.plot(time, window, linewidth=0.7)
            ax.set_title(f"{label} — rec {rec} @ {start_sample/fs:.0f}s", fontsize=10)
            if col == 0:
                ax.set_ylabel('ECG (mV)')
            if row == n_rows - 1:
                ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
