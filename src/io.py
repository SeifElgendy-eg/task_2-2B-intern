import os
import wfdb


def load_signal(rec, base_path):
    """Load a record's primary (channel-0) signal and its length in samples."""
    record = wfdb.rdrecord(os.path.join(base_path, rec))
    signal = record.p_signal[:, 0]
    return signal, signal.shape[0]


def get_annotations(rec, base_path):
    """Return (sample indices, symbols, cleaned aux_notes) for one record."""
    ann = wfdb.rdann(os.path.join(base_path, rec), 'atr')
    notes = [n.strip('\x00') for n in ann.aux_note]
    return ann.sample, ann.symbol, notes


def scan_record(rec_path):
    """Return every annotation's (sample, symbol, aux_note) for one record, plus record length."""
    ann = wfdb.rdann(rec_path, 'atr')
    record_len = wfdb.rdrecord(rec_path).p_signal.shape[0]
    return list(zip(ann.sample, ann.symbol, ann.aux_note)), record_len
