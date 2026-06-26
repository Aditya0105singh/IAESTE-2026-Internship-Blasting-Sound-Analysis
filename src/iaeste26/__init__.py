"""iaeste26 package."""

__version__ = "0.1.0"

from .main import run
from .parser import parse_filename, load_wav
from .dataset import scan_dataset, export_csv, summary
from .preprocessing import preprocess, bandpass_filter, normalize, trim_edges
from .features import extract_features, mfcc_features
from .visualization import (
    plot_waveform, plot_spectrogram,
    plot_rms_vs_pressure, plot_rms_vs_mix,
    plot_sensor_comparison, plot_spectral_centroid_vs_pressure,
)
from .ml import cross_session_eval, compare_models, session_normalize, within_material_loso
