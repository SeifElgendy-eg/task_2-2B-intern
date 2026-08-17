from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

CUDB_PATH = DATA_DIR / 'cu-ventricular-tachyarrhythmia-database-1.0.0'
VFDB_PATH = DATA_DIR / 'mit-bih-malignant-ventricular-ectopy-database-1.0.0'

FS = 250

RANDOM_SEED = 42
TEST_FRACTION = 0.20
MIN_MAJORITY = 0.50

WINDOW_CONFIGS = {
    2: 0.5,
    5: 1.25,
}

EXCLUDED_RECORDS = {'605'}

TRANSITIONAL_PADDING_S = 1
SPIKE_EVENT_GAP_S = 0.5
SPIKE_EVENT_TOL = 0.05

CLIP_TOL = 0.05
SHOCK_WINDOW_S = 3  # +/-3s search window around an episode boundary for shock evidence
