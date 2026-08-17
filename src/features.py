import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from scipy.interpolate import Akima1DInterpolator

from src.annotations import find_nan_blocks

from .io import load_signal


def extract_amplitude_features(window):
    """7 amplitude/morphology features for a single window (no clean beat detection needed)."""
    peak_to_peak = window.max() - window.min()
    std = window.std()
    rms = np.sqrt(np.mean(window**2))

    centered = window - window.mean()
    zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)
    zcr = zero_crossings / len(window)

    if std < 1e-8:  # essentially flat/constant window — skew/kurtosis undefined, not meaningless
        sk, kurt = 0.0, 0.0
    else:
        sk = skew(window)
        kurt = kurtosis(window)

    return dict(peak_to_peak=peak_to_peak, std=std, rms=rms, zcr=zcr, skew=sk, kurtosis=kurt)


def impute_gap(signal, gap_s, gap_e, sig_min, sig_max, fs, context=50, tol_frac=0.05, max_context=1000):
    """Fill one NaN gap. If both borders sit at the recorder's clipping extreme, hold flat at
    that extreme (evidence the true signal likely stayed pinned there). Otherwise, fit an Akima
    curve using nearby real signal (expanding the search window if needed) and clip the result
    to the record's observed range so it can't fabricate a physically implausible value."""
    tol = tol_frac * (sig_max - sig_min)
    before = signal[max(0, gap_s - 5):gap_s]
    after = signal[gap_e:min(len(signal), gap_e + 5)]
    before = before[~np.isnan(before)]
    after = after[~np.isnan(after)]

    before_extreme = len(before) > 0 and (np.any(before >= sig_max - tol) or np.any(before <= sig_min + tol))
    after_extreme = len(after) > 0 and (np.any(after >= sig_max - tol) or np.any(after <= sig_min + tol))

    if before_extreme and after_extreme:
        pre_val = before[-1] if len(before) > 0 else sig_max
        post_val = after[0] if len(after) > 0 else sig_max
        midpoint = gap_s + (gap_e - gap_s) // 2
        filled = np.empty(gap_e - gap_s)
        filled[:midpoint - gap_s] = pre_val
        filled[midpoint - gap_s:] = post_val
        return filled, 'hold_at_extreme'
    else:
        current_context = context
        x_known, y_known = np.array([]), np.array([])
        while len(x_known) < 2 and current_context <= max_context:
            fit_start = max(0, gap_s - current_context)
            fit_end = min(len(signal), gap_e + current_context)
            x_known = np.concatenate([np.arange(fit_start, gap_s), np.arange(gap_e, fit_end)])
            y_known = np.concatenate([signal[fit_start:gap_s], signal[gap_e:fit_end]])
            valid_mask = ~np.isnan(y_known)
            x_known, y_known = x_known[valid_mask], y_known[valid_mask]
            current_context *= 2

        x_gap = np.arange(gap_s, gap_e)
        if len(x_known) < 2:
            # genuinely no usable neighboring signal even at max_context — can't interpolate
            return np.full(gap_e - gap_s, np.nan), 'failed'

        akima_fit = Akima1DInterpolator(x_known, y_known)(x_gap)
        return np.clip(akima_fit, sig_min, sig_max), 'clipped_akima'


def impute_record_signal(rec, base_path, fs=250):
    signal, _ = load_signal(rec, base_path)
    cleaned = signal.copy()
    imputed_mask = np.zeros(len(signal), dtype=bool)
    hold_extreme_mask = np.zeros(len(signal), dtype=bool)
    valid = signal[~np.isnan(signal)]
    sig_min, sig_max = valid.min(), valid.max()

    for gap_s, gap_e in find_nan_blocks(signal):
        filled, method = impute_gap(signal, gap_s, gap_e, sig_min, sig_max, fs=fs)
        cleaned[gap_s:gap_e] = filled
        imputed_mask[gap_s:gap_e] = True
        if method == 'hold_at_extreme':
            hold_extreme_mask[gap_s:gap_e] = True

    return cleaned, sig_min, sig_max, imputed_mask, hold_extreme_mask

def extract_features_for_dataset(windows_df, cudb_set, cudb_path, vfdb_path, clip_tol=0.05, fs=250):
    windows_df = windows_df.copy()
    windows_df['record'] = windows_df['record'].astype(str)
    records_in_set = windows_df['record'].unique()

    cleaned_signals, clip_thresholds, imputed_masks, hold_extreme_masks = {}, {}, {}, {}
    for rec in records_in_set:
        base_path = cudb_path if rec in cudb_set else vfdb_path
        cleaned, sig_min, sig_max, imp_mask, hold_mask = impute_record_signal(rec, base_path, fs=fs)
        cleaned_signals[rec] = cleaned
        clip_thresholds[rec] = (sig_min, sig_max)
        imputed_masks[rec] = imp_mask
        hold_extreme_masks[rec] = hold_mask

    rows = []
    for _, row in windows_df.iterrows():
        w_start, w_end = row['start'], row['end']
        window = cleaned_signals[row['record']][w_start:w_end]
        sig_min, sig_max = clip_thresholds[row['record']]

        if len(window) < 10 or np.isnan(window).any():
            continue

        feats = extract_amplitude_features(window)
        feats['clip_fraction'] = np.mean((window <= sig_min + clip_tol) | (window >= sig_max - clip_tol))
        feats.update(compute_imputation_features(
            imputed_masks[row['record']][w_start:w_end],
            hold_extreme_masks[row['record']][w_start:w_end],
        ))
        feats.update(compute_frequency_features(window, fs))
        feats['sample_entropy'] = compute_sample_entropy(window)
        feats.update(compute_autocorrelation_features(window, fs))
        feats.update(compute_derivative_features(window))

        feats['record'] = row['record']
        feats['db'] = row['db']
        feats['label'] = row['label']
        rows.append(feats)

    return pd.DataFrame(rows)

def compute_imputation_features(imputed_mask, hold_extreme_mask):
    return dict(
        fraction_imputed=imputed_mask.mean(),
        used_hold_at_extreme=float(hold_extreme_mask.any()),
    )


def compute_frequency_features(window, fs, vf_band=(3, 10)):
    n = len(window)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    power = np.abs(np.fft.rfft(window - window.mean())) ** 2

    if power.sum() == 0:
        return dict(dominant_freq=0.0, vf_band_power_ratio=0.0, spectral_entropy=0.0)

    dominant_freq = freqs[np.argmax(power[1:]) + 1] if len(power) > 1 else 0.0  # skip DC

    band_mask = (freqs >= vf_band[0]) & (freqs <= vf_band[1])
    vf_band_power_ratio = power[band_mask].sum() / power.sum()

    p_norm = power / power.sum()
    p_norm = p_norm[p_norm > 0]
    spectral_entropy = -np.sum(p_norm * np.log2(p_norm)) / np.log2(len(p_norm)) if len(p_norm) > 1 else 0.0

    return dict(dominant_freq=dominant_freq, vf_band_power_ratio=vf_band_power_ratio, spectral_entropy=spectral_entropy)


def compute_sample_entropy(window, m=2, r_frac=0.2):
    """Vectorized SampEn(m, r) — measures signal predictability/complexity."""
    n = len(window)
    r = r_frac * window.std()
    if r == 0 or n < m + 2:
        return 0.0

    def _phi(m):
        templates = np.array([window[i:i+m] for i in range(n - m + 1)])
        dists = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        count = np.sum(dists <= r, axis=1) - 1  # exclude self-match
        return count.sum()

    B = _phi(m)
    A = _phi(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return -np.log(A / B)


def compute_autocorrelation_features(window, fs, min_lag_s=0.2):
    centered = window - window.mean()
    full_corr = np.correlate(centered, centered, mode='full')
    autocorr = full_corr[len(full_corr)//2:]
    autocorr = autocorr / (autocorr[0] + 1e-12)

    min_lag = int(min_lag_s * fs)
    if len(autocorr) <= min_lag + 1:
        return dict(periodicity_strength=0.0, periodicity_lag_s=0.0)

    search_region = autocorr[min_lag:]
    peak_idx = np.argmax(search_region)
    periodicity_strength = search_region[peak_idx]
    periodicity_lag_s = (peak_idx + min_lag) / fs

    return dict(periodicity_strength=periodicity_strength, periodicity_lag_s=periodicity_lag_s)


def compute_derivative_features(window):
    d = np.diff(window)
    turning_points = np.sum(np.diff(np.sign(d)) != 0)
    return dict(
        mean_abs_diff=np.mean(np.abs(d)),
        std_diff=d.std(),
        max_abs_diff=np.max(np.abs(d)),
        turning_points=turning_points / len(window),  # normalized so window length doesn't bias it
    )