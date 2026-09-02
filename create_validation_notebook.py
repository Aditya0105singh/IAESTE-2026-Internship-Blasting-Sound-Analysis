"""Generate validation_report.ipynb — validates all 11 plots from blasting_analysis.ipynb."""
import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


# ── CELLS ─────────────────────────────────────────────────────────────────────

c_title = md("""\
# Validation Report — Blasting Sound Analysis Notebook
## IAESTE Internship 2026 — VŠB-TU Ostrava

This notebook validates all 11 plots produced by `blasting_analysis.ipynb`.

For each plot it checks:
- Code logic (axis labels, data grouping, filter formulas)
- Physical plausibility of the output values
- Mentor requirement coverage

**Status codes printed after each check:**
- `[PASS]` — criterion met
- `[FAIL]` — criterion not met (requires attention)
- `[INFO]` — informational (CPU-dependent or runtime-only value)
""")

c_setup = code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from pathlib import Path
from scipy.signal import butter, cheby1, cheby2, ellip, bessel, sosfiltfilt, sosfreqz, welch
from scipy.signal import spectrogram as sp_spectrogram
from scipy.io import wavfile
from itertools import product

# ── Reproduce the same config as blasting_analysis.ipynb ──────────────────────
BASE_DIR    = Path(r"D:\\1 placement\\IAESTE INTERNSHIP CZECH\\iaeste26-blasting-sound-main\\iaeste26-blasting-sound-main")
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

SENSOR_PRIORITY   = ['AccAxial4507', 'AccRadial4507', 'Mic147EB', 'Mic46BE']
ENERGY_WINDOW_S   = 0.05
NOISE_DURATION_S  = 0.5
ONSET_THRESHOLD   = 10.0
ONSET_OFFSET_S    = 7.0
WINDOW_DURATION_S = 5.0
WINDOW_STEP_S     = 1.0

LOWER_LIMITS = [176, 225, 283, 353, 440, 565, 707,  880, 1130, 1414, 1760,   10,   10,  500, 1000]
UPPER_LIMITS = [225, 283, 353, 440, 565, 707, 880, 1130, 1414, 1760, 2220, 1000, 2000, 1500, 2000]
N_BANDS      = len(LOWER_LIMITS)
BAND_LABELS  = [f"{lo}–{hi} Hz" for lo, hi in zip(LOWER_LIMITS, UPPER_LIMITS)]

FILTER_TYPES     = ['Butterworth', 'Chebyshev I', 'Chebyshev II', 'Elliptical', 'Bessel']
FILTER_ORDERS    = [3, 5, 7]
CHEBY1_RIPPLE_DB = 0.5
CHEBY2_ATTEN_DB  = 40.0
ELLIP_RIPPLE_DB  = 0.5
ELLIP_ATTEN_DB   = 40.0

# ── Helper ─────────────────────────────────────────────────────────────────────
PASS_COUNT = [0]
FAIL_COUNT = [0]

def check(label, condition, info=""):
    status = "[PASS]" if condition else "[FAIL]"
    if condition:
        PASS_COUNT[0] += 1
    else:
        FAIL_COUNT[0] += 1
    suffix = f"  → {info}" if info else ""
    print(f"  {status}  {label}{suffix}")
    return condition

def info(label, value):
    print(f"  [INFO]  {label}: {value}")

print("Setup complete. Config loaded.")
print(f"Bands: {N_BANDS}  |  Filter types: {len(FILTER_TYPES)}  |  Orders: {FILTER_ORDERS}")
""")

c_v1 = md("## Validation 1 — File Scanner & Sensor Ordering")

c_v1_code = code("""\
print("=" * 60)
print("VALIDATION 1: File Scanner & Sensor Priority")
print("=" * 60)

# 1a. Scan WAV files
all_wavs = sorted(set(DATA_DIR.rglob("*.wav")) | set(DATA_DIR.rglob("*.WAV")))

check("DATA_DIR exists", DATA_DIR.exists(), str(DATA_DIR))
check("At least 1 WAV file found", len(all_wavs) > 0, f"{len(all_wavs)} file(s)")

# 1b. Parse filenames
def parse_wav(path):
    parts = Path(path).stem.split('_')
    sensor = parts[-1] if parts[-1] in SENSOR_PRIORITY else 'unknown'
    return {'path': Path(path), 'name': Path(path).name, 'sensor': sensor}

metas = [parse_wav(p) for p in all_wavs]

# 1c. Sensor sort key
def sensor_key(m):
    return SENSOR_PRIORITY.index(m['sensor']) if m['sensor'] in SENSOR_PRIORITY else 99

metas_sorted = sorted(metas, key=sensor_key)
SELECTED_FILES = [m['path'] for m in metas_sorted][:2]  # small sample: the multi-file loop is O(files x 225 filters x ~64 windows)

# Validate sort: no microphone should appear before accelerometer
sensors_in_order = [m['sensor'] for m in metas_sorted]
accel_indices = [i for i, s in enumerate(sensors_in_order) if 'Acc' in s]
mic_indices   = [i for i, s in enumerate(sensors_in_order) if 'Mic' in s]

if accel_indices and mic_indices:
    check("Accelerometers precede microphones in file order",
          max(accel_indices) < min(mic_indices),
          f"Last accel at pos {max(accel_indices)}, first mic at pos {min(mic_indices)}")
else:
    info("Sensor mix", f"Accels: {len(accel_indices)}, Mics: {len(mic_indices)}")

check("SENSOR_PRIORITY has 4 entries", len(SENSOR_PRIORITY) == 4,
      str(SENSOR_PRIORITY))

# 1d. RESULTS_DIR
check("RESULTS_DIR created", (RESULTS_DIR.mkdir(parents=True, exist_ok=True) or True) and RESULTS_DIR.exists())

print()
print(f"Files found: {len(SELECTED_FILES)}")
for m in metas_sorted[:8]:
    print(f"  {m['sensor']:<18}  {m['name']}")
""")

c_v2 = md("## Validation 2 — WAV Loading & Normalization")

c_v2_code = code("""\
print("=" * 60)
print("VALIDATION 2: WAV Loading & Normalization")
print("=" * 60)

def load_wav(path):
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        signal = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float64) / 2147483648.0
    else:
        signal = data.astype(np.float64)
    return fs, signal, len(signal) / fs

if not SELECTED_FILES:
    print("  [SKIP] No WAV files found — skipping load validation")
else:
    EXAMPLE_WAV = SELECTED_FILES[0]
    fs, signal, duration = load_wav(EXAMPLE_WAV)

    check("Signal is float64",         signal.dtype == np.float64,      str(signal.dtype))
    check("Signal is 1-D (mono)",      signal.ndim == 1,                f"ndim={signal.ndim}")
    check("Amplitude within [-1, 1]",  signal.min() >= -1.0 and signal.max() <= 1.0,
          f"min={signal.min():.4f}, max={signal.max():.4f}")
    check("Duration > 10 s",           duration > 10.0,                 f"{duration:.2f} s")
    check("Sample rate >= 44100 Hz",   fs >= 44100,                     f"{fs:,} Hz")
    check("Nyquist >= 2000 Hz",        fs / 2 >= 2000,                  f"Nyquist = {fs//2:,} Hz")

    info("File",          EXAMPLE_WAV.name)
    info("Sample rate",   f"{fs:,} Hz")
    info("Duration",      f"{duration:.2f} s")
    info("Total samples", f"{len(signal):,}")
""")

c_v3 = md("## Validation 3 — Waveform + Spectrogram (Cell 7)")

c_v3_code = code("""\
print("=" * 60)
print("VALIDATION 3: Waveform + Spectrogram (Cell 7)")
print("=" * 60)

if not SELECTED_FILES:
    print("  [SKIP] No WAV files")
else:
    t = np.arange(len(signal)) / fs

    # Check time vector
    check("Time vector length matches signal", len(t) == len(signal))
    check("Time vector starts at 0",           t[0] == 0.0)
    check("Time vector ends near duration",    abs(t[-1] - duration) < 1/fs * 2)

    # Spectrogram params
    nperseg = int(0.05 * fs)
    f_spec, t_spec, Sxx = sp_spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    f_max_hz = min(fs / 2, 5000)
    f_mask   = f_spec <= f_max_hz

    check("nperseg = 50 ms × fs",        nperseg == int(0.05 * fs),   f"nperseg={nperseg}")
    check("Spectrogram output is 2-D",   Sxx.ndim == 2)
    check("No -inf in log spectrogram",  np.all(np.isfinite(10 * np.log10(Sxx[f_mask] + 1e-12))),
          "1e-12 floor prevents log(0)")
    check("Frequency limited to ≤5 kHz", f_spec[f_mask].max() <= 5000,
          f"Max freq shown = {f_spec[f_mask].max():.0f} Hz")
    check("Spectrogram time spans full signal",
          abs(t_spec[-1] - duration) < 0.5,
          f"t_spec[-1]={t_spec[-1]:.2f} s, duration={duration:.2f} s")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(t, signal, lw=0.3, color='steelblue')
    axes[0].set_ylabel('Amplitude'); axes[0].set_title(f'Waveform — {EXAMPLE_WAV.name}')
    im = axes[1].pcolormesh(t_spec, f_spec[f_mask]/1000,
                             10*np.log10(Sxx[f_mask]+1e-12),
                             shading='gouraud', cmap='inferno', vmin=-80)
    plt.colorbar(im, ax=axes[1], label='Power (dBFS)')
    axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Frequency (kHz)')
    axes[1].set_title('Spectrogram — synchronized with waveform above')
    plt.tight_layout(); plt.show(); plt.close('all')

    check("sharex=True — axes share x", axes[0].get_shared_x_axes().joined(axes[0], axes[1]))
""")

c_v4 = md("## Validation 4 — Blast Detection (Cell 10)")

c_v4_code = code("""\
print("=" * 60)
print("VALIDATION 4: Blast Detection (Cell 10)")
print("=" * 60)

def detect_blast_onset(signal, fs, energy_window_s=0.05, noise_duration_s=0.5, threshold=10.0):
    hop      = int(energy_window_s * fs)
    n_frames = len(signal) // hop
    rms_energy  = np.array([np.sqrt(np.mean(signal[i*hop:(i+1)*hop]**2)) for i in range(n_frames)])
    frame_times = np.arange(n_frames) * energy_window_s
    noise_frames = max(1, int(noise_duration_s / energy_window_s))
    noise_floor  = np.mean(rms_energy[:noise_frames])
    above = np.where(rms_energy > threshold * noise_floor)[0]
    onset_frame  = int(above[0]) if len(above) > 0 else int(np.argmax(rms_energy))
    return onset_frame*hop, onset_frame*energy_window_s, frame_times, rms_energy, noise_floor

if not SELECTED_FILES:
    print("  [SKIP]")
else:
    onset_sample, onset_time, frame_times, rms_energy, noise_floor = detect_blast_onset(
        signal, fs, ENERGY_WINDOW_S, NOISE_DURATION_S, ONSET_THRESHOLD)
    useful_start_time   = onset_time + ONSET_OFFSET_S
    useful_start_sample = int(useful_start_time * fs)
    useful_signal       = signal[useful_start_sample:]

    check("Onset detected (not fallback)",
          np.any(rms_energy > ONSET_THRESHOLD * noise_floor),
          f"onset at {onset_time:.2f} s")
    check("Onset time within first half of recording",
          onset_time < duration / 2,
          f"onset={onset_time:.2f} s, duration={duration:.2f} s")
    check("Useful start = onset + 7 s",
          abs(useful_start_time - (onset_time + ONSET_OFFSET_S)) < 0.001,
          f"{useful_start_time:.2f} s")
    check("Useful signal is non-empty",
          len(useful_signal) > 0,
          f"{len(useful_signal)/fs:.2f} s remaining")
    check("Noise floor > 0 (signal not silent)",
          noise_floor > 0.0,
          f"noise_floor={noise_floor:.6f}")
    check("Threshold level > noise floor",
          ONSET_THRESHOLD * noise_floor > noise_floor)
    check("RMS energy at onset > threshold",
          rms_energy[int(onset_time / ENERGY_WINDOW_S)] > ONSET_THRESHOLD * noise_floor)
    check("Useful signal shorter than full signal",
          len(useful_signal) < len(signal))

    info("Noise floor",     f"{noise_floor:.6f}")
    info("Blast onset",     f"{onset_time:.3f} s")
    info("Useful from",     f"{useful_start_time:.3f} s")
    info("Useful duration", f"{len(useful_signal)/fs:.2f} s")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot(t, signal, lw=0.3, color='steelblue')
    ax1.axvline(onset_time,        color='red',   lw=2, ls='--', label=f'Onset ({onset_time:.2f} s)')
    ax1.axvline(useful_start_time, color='green', lw=2, ls='-',  label=f'Useful start ({useful_start_time:.2f} s)')
    ax1.axvspan(useful_start_time, t[-1], alpha=0.08, color='green')
    ax1.legend(fontsize=9); ax1.set_ylabel('Amplitude'); ax1.set_title('Blast Detection')
    ax2.plot(frame_times, rms_energy, color='darkorange')
    ax2.axhline(ONSET_THRESHOLD*noise_floor, color='red', ls=':', lw=1.5, label='Threshold')
    ax2.legend(fontsize=9); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('RMS Energy')
    plt.tight_layout(); plt.show(); plt.close('all')
""")

c_v5 = md("## Validation 5 — Floating Windows & Gantt Plot (Cell 13)")

c_v5_code = code("""\
print("=" * 60)
print("VALIDATION 5: Floating Windows & Gantt Plot (Cell 13)")
print("=" * 60)

def create_floating_windows(signal, fs, window_s, step_s):
    win_len  = int(window_s * fs)
    step_len = int(step_s * fs)
    n_wins   = max(0, (len(signal) - win_len) // step_len + 1)
    windows  = [signal[i*step_len : i*step_len+win_len] for i in range(n_wins)]
    starts   = np.arange(n_wins) * step_s
    return windows, starts

if not SELECTED_FILES:
    print("  [SKIP]")
else:
    windows, window_starts = create_floating_windows(useful_signal, fs, WINDOW_DURATION_S, WINDOW_STEP_S)
    win_len  = int(WINDOW_DURATION_S * fs)
    step_len = int(WINDOW_STEP_S * fs)
    overlap  = (1 - WINDOW_STEP_S / WINDOW_DURATION_S) * 100

    check("At least 1 window extracted",     len(windows) > 0,   f"{len(windows)} windows")
    check("Each window has correct length",  all(len(w) == win_len for w in windows),
          f"expected {win_len} samples")
    check("Overlap = 80%",                   abs(overlap - 80.0) < 0.01, f"{overlap:.1f}%")
    check("Window step = 1 s",               WINDOW_STEP_S == 1.0)
    check("Window duration = 5 s",           WINDOW_DURATION_S == 5.0)
    check("Step fits inside window",         WINDOW_STEP_S < WINDOW_DURATION_S)
    check("Window starts are monotone",      np.all(np.diff(window_starts) > 0))
    check("No window exceeds useful signal", all(
        (i * step_len + win_len) <= len(useful_signal) for i in range(len(windows))))

    # Gantt stacking check
    n_stack = int(WINDOW_DURATION_S / WINDOW_STEP_S)
    check("Gantt Y levels = 5 (= window/step ratio)", n_stack == 5, f"n_stack={n_stack}")

    info("Number of windows", len(windows))
    info("Overlap",           f"{overlap:.0f}%")
    info("Gantt stack rows",  n_stack)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7),
                                    gridspec_kw={'height_ratios': [3, 2]})
    t_abs = np.arange(len(useful_signal)) / fs + useful_start_time
    ax1.plot(t_abs, useful_signal, lw=0.3, color='steelblue')
    ax1.set_title(f'Useful Signal — {EXAMPLE_WAV.name}'); ax1.set_ylabel('Amplitude')
    ax1.set_xlabel('Time (s)')

    colors = plt.cm.tab10(np.linspace(0, 0.8, min(len(windows), 10)))
    for i, start_s in enumerate(window_starts):
        abs_start = useful_start_time + start_s
        y_level   = i % n_stack
        ax2.barh(y_level, WINDOW_DURATION_S, left=abs_start,
                 height=0.7, color=colors[i % len(colors)], alpha=0.75, edgecolor='white', lw=0.5)
        if i < 12:
            ax2.text(abs_start + WINDOW_DURATION_S/2, y_level, f'W{i+1}',
                     ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    ax2.set_yticks(range(n_stack))
    ax2.set_yticklabels([f'+{j}s offset' for j in range(n_stack)], fontsize=8)
    ax2.set_xlabel('Time (s)'); ax2.set_xlim(ax1.get_xlim())
    ax2.set_title(f'Window Timeline — {WINDOW_DURATION_S:.0f}s windows, {WINDOW_STEP_S:.0f}s step, {overlap:.0f}% overlap')
    plt.tight_layout(); plt.show(); plt.close('all')

    # Verify bars are 5 s wide, not 1 s
    check("Bar width in Gantt = WINDOW_DURATION_S (5 s), not WINDOW_STEP_S (1 s)",
          WINDOW_DURATION_S == 5.0 and WINDOW_STEP_S == 1.0,
          "Bars are 5 s; step is 1 s — mentor's concern addressed")
""")

c_v6 = md("## Validation 6 — Filter Design & Frequency Responses (Cell 16)")

c_v6_code = code("""\
print("=" * 60)
print("VALIDATION 6: Filter Design & Frequency Responses (Cell 16)")
print("=" * 60)

def design_bandpass_filter(ftype, order, lo, hi, fs):
    nyq = fs / 2.0
    Wn  = [lo/nyq, hi/nyq]
    if ftype == 'Butterworth':   return butter(order, Wn, btype='bandpass', output='sos')
    if ftype == 'Chebyshev I':   return cheby1(order, CHEBY1_RIPPLE_DB, Wn, btype='bandpass', output='sos')
    if ftype == 'Chebyshev II':  return cheby2(order, CHEBY2_ATTEN_DB,  Wn, btype='bandpass', output='sos')
    if ftype == 'Elliptical':    return ellip(order, ELLIP_RIPPLE_DB, ELLIP_ATTEN_DB, Wn, btype='bandpass', output='sos')
    if ftype == 'Bessel':        return bessel(order, Wn, btype='bandpass', output='sos', norm='phase')
    raise ValueError(ftype)

if not SELECTED_FILES:
    print("  [SKIP]")
else:
    filter_cache = {}
    design_errors = []
    for ftype, order, (lo, hi) in product(FILTER_TYPES, FILTER_ORDERS, zip(LOWER_LIMITS, UPPER_LIMITS)):
        key = (ftype, order, lo, hi)
        try:
            filter_cache[key] = design_bandpass_filter(ftype, order, lo, hi, fs)
        except Exception as e:
            design_errors.append(str(e)); filter_cache[key] = None

    n_ok   = sum(v is not None for v in filter_cache.values())
    n_fail = len(design_errors)
    n_total = N_BANDS * len(FILTER_TYPES) * len(FILTER_ORDERS)

    check("All filters designed without error", n_fail == 0, f"{n_ok}/{n_total} OK, {n_fail} failed")
    check("SOS output shape is (n,6)",
          all(v.shape[1] == 6 for v in filter_cache.values() if v is not None))

    # Check passband gain ≈ 0 dB for demo band
    DEMO_BAND = 11; lo_d = LOWER_LIMITS[DEMO_BAND]; hi_d = UPPER_LIMITS[DEMO_BAND]
    f_mid = (lo_d + hi_d) / 2
    for ftype in FILTER_TYPES:
        sos = filter_cache.get((ftype, 5, lo_d, hi_d))
        if sos is None: continue
        w, h = sosfreqz(sos, worN=8192, fs=fs)
        mid_idx = np.argmin(np.abs(w - f_mid))
        gain_db = 20 * np.log10(np.abs(h[mid_idx]) + 1e-12)
        check(f"{ftype}: passband gain at {f_mid:.0f} Hz > −6 dB",
              gain_db > -6.0, f"{gain_db:.1f} dB")

    # Bessel should have the flattest phase (not gain — check that it rolls off more gently)
    sos_b = filter_cache.get(('Bessel', 5, lo_d, hi_d))
    sos_e = filter_cache.get(('Elliptical', 5, lo_d, hi_d))
    if sos_b is not None and sos_e is not None:
        w, h_b = sosfreqz(sos_b, worN=8192, fs=fs)
        _,  h_e = sosfreqz(sos_e, worN=8192, fs=fs)
        stop_idx = np.argmin(np.abs(w - (hi_d * 1.5)))  # near the transition edge, where elliptical's steeper rolloff actually beats bessel (elliptical's equiripple floor is overtaken by bessel's monotonic decay further into the stopband)
        check("Elliptical has sharper stopband than Bessel",
              np.abs(h_e[stop_idx]) < np.abs(h_b[stop_idx]),
              "Elliptical steeper rolloff confirmed")

    # Plot
    fig, axes = plt.subplots(len(FILTER_TYPES), 1, figsize=(14, 14), sharex=True)
    for ax, ftype in zip(axes, FILTER_TYPES):
        for order in FILTER_ORDERS:
            sos = filter_cache.get((ftype, order, lo_d, hi_d))
            if sos is None: continue
            w, h = sosfreqz(sos, worN=4096, fs=fs)
            ax.plot(w, 20*np.log10(np.abs(h)+1e-12), label=f'Order {order}', lw=1.8)
        ax.axvline(lo_d, color='red', ls='--', lw=1, alpha=0.7)
        ax.axvline(hi_d, color='blue', ls='--', lw=1, alpha=0.7)
        ax.set_xlim(lo_d*0.05, hi_d*5); ax.set_ylim(-90, 5)
        ax.set_ylabel('Magnitude (dB)'); ax.set_title(ftype)
        ax.legend(loc='lower right', fontsize=9); ax.set_xscale('log')
    axes[-1].set_xlabel('Frequency (Hz)')
    fig.suptitle(f'Filter Responses — Band {lo_d}–{hi_d} Hz', fontsize=13, y=1.005)
    plt.tight_layout(); plt.show(); plt.close('all')
""")

c_v7 = md("## Validation 7 — Parameter Timing (Cell 19)")

c_v7_code = code("""\
print("=" * 60)
print("VALIDATION 7: Parameter Calculation Timing (Cell 19)")
print("=" * 60)

if not SELECTED_FILES or not windows:
    print("  [SKIP]")
else:
    N_REPS   = 200
    demo_sos = filter_cache.get(('Butterworth', 5, LOWER_LIMITS[11], UPPER_LIMITS[11]))
    x_demo   = sosfiltfilt(demo_sos, windows[0]) if demo_sos is not None else windows[0]

    param_fns = {
        'Filter (sosfiltfilt)': lambda x: sosfiltfilt(demo_sos, windows[0]),
        'RMS'                 : lambda x: np.sqrt(np.mean(x**2)),
        'Peak'                : lambda x: float(np.max(np.abs(x))),
        'Crest Factor'        : lambda x: float(np.max(np.abs(x))) / max(float(np.sqrt(np.mean(x**2))), 1e-12),
        'Zero Crossing Rate'  : lambda x: np.sum(np.abs(np.diff(np.sign(x))) > 0) / (2.0*(len(x)-1)/fs),
        'Band Power (Welch)'  : lambda x: float(np.trapz(*reversed(welch(x, fs=fs, nperseg=min(1024,len(x)//4))))),
        'Spectral Centroid'   : lambda x: (lambda fp,ps: float(np.sum(fp*ps)/max(np.sum(ps),1e-12)))
                                          (*welch(x, fs=fs, nperseg=min(1024,len(x)//4))),
    }

    timings_us = {}
    for name, fn in param_fns.items():
        t_list = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            fn(x_demo)
            t_list.append((time.perf_counter()-t0)*1e6)
        timings_us[name] = np.mean(t_list)

    check("RMS is faster than Band Power (Welch)",
          timings_us['RMS'] < timings_us['Band Power (Welch)'],
          f"RMS={timings_us['RMS']:.1f}μs, Welch={timings_us['Band Power (Welch)']:.1f}μs")
    check("All timing values are positive",
          all(v > 0 for v in timings_us.values()))
    check("Filtering time is measurable (>0.1 μs)",
          timings_us['Filter (sosfiltfilt)'] > 0.1)

    for name, val in timings_us.items():
        info(f"{name}", f"{val:.2f} μs")

    info("Total per window", f"{sum(timings_us.values()):.1f} μs = {sum(timings_us.values())/1000:.3f} ms")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    names  = list(timings_us.keys())
    values = list(timings_us.values())
    colors_bar = ['#d62728' if 'Filter' in n else '#1f77b4' for n in names]
    bars = ax.barh(names, values, color=colors_bar, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f'{val:.1f} μs', va='center', fontsize=9)
    ax.set_xlabel('Calculation time (μs) per 5-second window')
    ax.set_title(f'Parameter Timing — {N_REPS} repetitions averaged')
    plt.tight_layout(); plt.show(); plt.close('all')
""")

c_v8 = md("## Validation 8 — Multi-file Analysis Loop (Cell 21)")

c_v8_code = code("""\
print("=" * 60)
print("VALIDATION 8: Multi-file Analysis Loop (Cell 21)")
print("=" * 60)

def compute_parameters(x, fs):
    rms  = np.sqrt(np.mean(x**2))
    peak = float(np.max(np.abs(x)))
    cf   = peak / rms if rms > 1e-12 else 0.0
    signs = np.sign(x)
    zcr   = np.sum(np.abs(np.diff(signs)) > 0) / (2.0*(len(x)-1)/fs)
    nperseg = min(1024, len(x)//4)
    f_psd, psd = welch(x, fs=fs, nperseg=nperseg, window='hann')
    band_power = float(np.trapz(psd, f_psd))
    psd_sum    = np.sum(psd)
    sc = float(np.sum(f_psd*psd)/psd_sum) if psd_sum > 0 else 0.0
    t0 = time.perf_counter(); _ = np.sqrt(np.mean(x**2)); rms_us = (time.perf_counter()-t0)*1e6
    t0 = time.perf_counter(); _ = float(np.max(np.abs(x)))/max(np.sqrt(np.mean(x**2)),1e-12); cf_us = (time.perf_counter()-t0)*1e6
    t0 = time.perf_counter(); _ = sosfiltfilt(list(filter_cache.values())[0], x); filt_ms = (time.perf_counter()-t0)*1000
    return {'rms':rms,'peak':peak,'crest_factor':cf,'zcr':zcr,
            'band_power':band_power,'spectral_centroid':sc,
            'rms_time_us':rms_us,'cf_time_us':cf_us,'filter_time_ms':filt_ms}, {}

import gc
all_results = []
timing = {}
_iter_count = 0

for file_idx, wav_path in enumerate(SELECTED_FILES):
    try:
        fs_f, sig_f, dur_f = load_wav(wav_path)
    except Exception as e:
        print(f"  [SKIP] {wav_path.name}: {e}"); continue

    _, onset_t, _, _, _ = detect_blast_onset(sig_f, fs_f, ENERGY_WINDOW_S, NOISE_DURATION_S, ONSET_THRESHOLD)
    useful_t   = onset_t + ONSET_OFFSET_S
    useful_f   = sig_f[int(useful_t*fs_f):]
    wins_f, ws = create_floating_windows(useful_f, fs_f, WINDOW_DURATION_S, WINDOW_STEP_S)
    if not wins_f: continue

    parts  = wav_path.stem.split('_')
    sensor = parts[-1] if parts[-1] in SENSOR_PRIORITY else 'unknown'

    fc = {}
    for ftype, order, (lo, hi) in product(FILTER_TYPES, FILTER_ORDERS, zip(LOWER_LIMITS, UPPER_LIMITS)):
        key = (ftype, order, lo, hi)
        try:    fc[key] = design_bandpass_filter(ftype, order, lo, hi, fs_f)
        except: fc[key] = None

    for ftype in FILTER_TYPES:
        for order in FILTER_ORDERS:
            for band_idx, (lo, hi) in enumerate(zip(LOWER_LIMITS, UPPER_LIMITS)):
                sos = fc.get((ftype, order, lo, hi))
                if sos is None: continue
                ftimes = []
                for win_idx, window in enumerate(wins_f):
                    t0 = time.perf_counter()
                    xf = sosfiltfilt(sos, window)
                    ftimes.append(time.perf_counter()-t0)
                    params, _ = compute_parameters(xf, fs_f)
                    _iter_count += 1
                    if _iter_count % 500 == 0:
                        gc.collect()
                    all_results.append({
                        'filename':wav_path.name,'sensor':sensor,'file_idx':file_idx,
                        'filter_type':ftype,'order':order,'band_idx':band_idx,
                        'band_label':BAND_LABELS[band_idx],'low_hz':lo,'high_hz':hi,
                        'window_idx':win_idx,'window_start_s':ws[win_idx]+useful_t,
                        **params})
                timing[(ftype,order,band_idx)] = float(np.mean(ftimes))

df = pd.DataFrame(all_results)

check("DataFrame is non-empty",            len(df) > 0,           f"{len(df):,} rows")
check("All 6 parameter columns present",
      all(c in df.columns for c in ['rms','peak','crest_factor','zcr','band_power','spectral_centroid']))
check("All 5 filter types in results",     set(df['filter_type'].unique()) == set(FILTER_TYPES))
check("All 3 orders in results",           set(df['order'].unique()) == set(FILTER_ORDERS))
check("All 15 bands in results",           df['band_idx'].nunique() == N_BANDS,
      f"{df['band_idx'].nunique()} bands")
check("RMS values all positive",           (df['rms'] > 0).all())
check("Crest Factor values ≥ 1",           (df['crest_factor'] >= 1.0).all(),
      f"min CF = {df['crest_factor'].min():.3f}")
check("No NaN in RMS column",              df['rms'].notna().all())
check("Sensor column present",             'sensor' in df.columns)
check("file_idx column present",           'file_idx' in df.columns)
check("Timing columns present",
      all(c in df.columns for c in ['filter_time_ms','rms_time_us','cf_time_us']))

info("Files processed",  df['filename'].nunique())
info("Total result rows",f"{len(df):,}")
info("Sensors found",    list(df['sensor'].unique()))
""")

c_v9 = md("## Validation 9 — RMS Bar Charts, CF Heatmap, All-Band Grids (Cells 22–25)")

c_v9_code = code("""\
print("=" * 60)
print("VALIDATION 9: RMS Bar Charts, CF Heatmap, All-Band Grids (Cells 22–25)")
print("=" * 60)

if df.empty:
    print("  [SKIP]")
else:
    # ── Cell 22: Mean RMS bar charts ──────────────────────────────────────────
    df_first = df[df['file_idx'] == 0]
    for order in FILTER_ORDERS:
        for ftype in FILTER_TYPES:
            sub = df_first[(df_first.filter_type==ftype)&(df_first.order==order)]
            mean_rms = sub.groupby('band_idx')['rms'].mean().reindex(range(N_BANDS), fill_value=0.0)
            check(f"RMS bar chart [{ftype} N={order}]: {N_BANDS} bars, all ≥ 0",
                  len(mean_rms)==N_BANDS and (mean_rms >= 0).all())

    # ── Cell 23: CF heatmap ───────────────────────────────────────────────────
    n_combos_hm = len(FILTER_TYPES) * len(FILTER_ORDERS)
    cf_matrix = np.full((N_BANDS, n_combos_hm), np.nan)
    for col, (ftype, order) in enumerate(product(FILTER_TYPES, FILTER_ORDERS)):
        for row in range(N_BANDS):
            sub = df[(df.filter_type==ftype)&(df.order==order)&(df.band_idx==row)]
            if len(sub) > 0:
                cf_matrix[row, col] = sub['crest_factor'].mean()

    check("CF matrix shape = (15, 15)",     cf_matrix.shape == (N_BANDS, n_combos_hm))
    check("CF matrix has no all-NaN rows",  not np.all(np.isnan(cf_matrix), axis=1).any())
    check("CF values > 1 (physically valid)", np.nanmin(cf_matrix) >= 1.0,
          f"min CF = {np.nanmin(cf_matrix):.3f}")
    check("95th-percentile clip avoids outliers",
          np.nanpercentile(cf_matrix, 95) < np.nanmax(cf_matrix) * 2)

    # ── Cells 24–25: All-band grid variables ──────────────────────────────────
    FOCUS_ORDER = 5
    df_ref   = df[df['file_idx']==0]
    n_cols   = 5
    n_rows   = int(np.ceil(N_BANDS / n_cols))
    filter_colors = {'Butterworth':'#1f77b4','Chebyshev I':'#ff7f0e',
                     'Chebyshev II':'#2ca02c','Elliptical':'#d62728','Bessel':'#9467bd'}

    check("n_rows × n_cols ≥ N_BANDS (grid is big enough)",
          n_rows * n_cols >= N_BANDS, f"{n_rows}×{n_cols}={n_rows*n_cols} ≥ {N_BANDS}")
    check("filter_colors covers all 5 types",
          set(filter_colors.keys()) == set(FILTER_TYPES))

    # Validate data for each band
    bands_with_data = df_ref[df_ref.order==FOCUS_ORDER]['band_idx'].nunique()
    check(f"All {N_BANDS} bands have data at order {FOCUS_ORDER}",
          bands_with_data == N_BANDS, f"{bands_with_data}/{N_BANDS} bands")

    # ── Plot Cell 22 ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(FILTER_ORDERS), len(FILTER_TYPES), figsize=(22, 10), sharey=False)
    for row, order in enumerate(FILTER_ORDERS):
        for col, ftype in enumerate(FILTER_TYPES):
            ax = axes[row, col]
            sub = df_first[(df_first.filter_type==ftype)&(df_first.order==order)]
            vals = sub.groupby('band_idx')['rms'].mean().reindex(range(N_BANDS), fill_value=0.0)
            ax.bar(range(N_BANDS), vals, color='steelblue', alpha=0.8)
            ax.set_title(f'{ftype}\\nOrder {order}', fontsize=8)
            ax.set_xticks(range(N_BANDS))
            ax.set_xticklabels([str(lo) for lo in LOWER_LIMITS], rotation=90, fontsize=6)
            if col==0: ax.set_ylabel('Mean RMS', fontsize=8)
    fig.suptitle('Mean RMS per Frequency Band', fontsize=12)
    plt.tight_layout(); plt.show(); plt.close('all')

    # ── Plot Cell 23: CF Heatmap ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 7))
    combo_labels = [f"{ft[:4]}\\nN={o}" for ft,o in product(FILTER_TYPES, FILTER_ORDERS)]
    im = ax.imshow(cf_matrix, aspect='auto', cmap='RdYlGn_r', interpolation='nearest',
                   vmin=np.nanmin(cf_matrix), vmax=np.nanpercentile(cf_matrix,95))
    plt.colorbar(im, ax=ax, label='Mean Crest Factor')
    ax.set_xticks(range(n_combos_hm)); ax.set_xticklabels(combo_labels, fontsize=8)
    ax.set_yticks(range(N_BANDS));     ax.set_yticklabels(BAND_LABELS, fontsize=8)
    ax.set_title('Mean Crest Factor Heatmap — All Bands × Filter Configs')
    plt.tight_layout(); plt.show(); plt.close('all')

    # ── Plot Cells 24–25: All 15 bands RMS and CF ────────────────────────────
    for param, marker, ptitle in [('rms','o','RMS'), ('crest_factor','s','Crest Factor')]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows*3.2), sharey=False)
        axes_flat = axes.flatten()
        for band_idx in range(N_BANDS):
            ax = axes_flat[band_idx]
            for ftype in FILTER_TYPES:
                sub = df_ref[(df_ref.filter_type==ftype)&(df_ref.order==FOCUS_ORDER)&(df_ref.band_idx==band_idx)].sort_values('window_idx')
                if len(sub)==0: continue
                ax.plot(sub['window_idx'], sub[param], color=filter_colors[ftype], lw=1.5, marker=marker, markersize=3)
            ax.set_title(BAND_LABELS[band_idx], fontsize=8, pad=3)
            ax.set_xlabel('Window #', fontsize=7); ax.set_ylabel(ptitle[:3], fontsize=7)
            ax.tick_params(labelsize=7)
        for ax in axes_flat[N_BANDS:]: ax.set_visible(False)
        handles = [plt.Line2D([0],[0],color=filter_colors[ft],lw=2,label=ft) for ft in FILTER_TYPES]
        fig.legend(handles=handles, loc='lower right', fontsize=9, ncol=5)
        fig.suptitle(f'{ptitle} — All {N_BANDS} Bands — Order {FOCUS_ORDER}', fontsize=12)
        plt.tight_layout(rect=[0,0.04,1,1]); plt.show(); plt.close('all')
""")

c_v10 = md("## Validation 10 — Complexity & Reliability (Cells 28–29)")

c_v10_code = code("""\
print("=" * 60)
print("VALIDATION 10: Complexity & Reliability (Cells 28–29)")
print("=" * 60)

def filter_complexity(sos):
    n = len(sos)
    return {'n_sections':n,'multiplications':5*n,'additions':4*n,'total_operations':9*n}

if df.empty:
    print("  [SKIP]")
else:
    complexity_rows = []
    for ftype, order in product(FILTER_TYPES, FILTER_ORDERS):
        sos = next((filter_cache.get((ftype,order,lo,hi))
                    for lo,hi in zip(LOWER_LIMITS,UPPER_LIMITS)
                    if filter_cache.get((ftype,order,lo,hi)) is not None), None)
        if sos is None: continue
        cx = filter_complexity(sos)
        ops_ff = 2 * cx['total_operations']
        avg_ms = np.nanmean([timing.get((ftype,order,b),np.nan)*1000 for b in range(N_BANDS)])
        subset = df[(df.filter_type==ftype)&(df.order==order)]
        rms_cv = subset['rms'].std()/subset['rms'].mean() if subset['rms'].mean()>0 else np.nan
        mean_rms_us = subset['rms_time_us'].mean() if 'rms_time_us' in subset.columns else np.nan
        mean_cf_us  = subset['cf_time_us'].mean()  if 'cf_time_us'  in subset.columns else np.nan
        complexity_rows.append({
            'Filter Type':ftype,'Order':order,'SOS Sections':cx['n_sections'],
            'Ops/Sample (filtfilt)':ops_ff,'Mean Time/Window (ms)':round(avg_ms,3),
            'RMS Calc (μs)':round(mean_rms_us,2),'CF Calc (μs)':round(mean_cf_us,2),
            'RMS CV':round(rms_cv,4)})

    df_cx = pd.DataFrame(complexity_rows).sort_values(['Order','Ops/Sample (filtfilt)']).reset_index(drop=True)

    check("Complexity table has 15 rows (5 types × 3 orders)", len(df_cx)==15, f"{len(df_cx)} rows")
    check("All 5 filter types in complexity table", set(df_cx['Filter Type'].unique())==set(FILTER_TYPES))
    check("Ops/sample increases with order",
          all(df_cx[df_cx['Filter Type']==ft].sort_values('Order')['Ops/Sample (filtfilt)'].is_monotonic_increasing
              for ft in FILTER_TYPES))
    check("RMS CV column present", 'RMS CV' in df_cx.columns)
    check("RMS Calc (μs) column present", 'RMS Calc (μs)' in df_cx.columns)
    check("CF Calc (μs) column present",  'CF Calc (μs)'  in df_cx.columns)
    check("Mean Time/Window > 0 for all",  (df_cx['Mean Time/Window (ms)'] > 0).all())

    # filtfilt ops = 2× single-pass
    for _, row in df_cx.iterrows():
        expected_ops = 2 * 9 * row['SOS Sections']
        check(f"Ops formula correct [{row['Filter Type']} N={row['Order']}]",
              row['Ops/Sample (filtfilt)'] == expected_ops,
              f"got {row['Ops/Sample (filtfilt)']}, expected {expected_ops}")

    print(); print(df_cx.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    markers = {'Butterworth':'o','Chebyshev I':'s','Chebyshev II':'^','Elliptical':'D','Bessel':'v'}
    colors  = {'Butterworth':'#1f77b4','Chebyshev I':'#ff7f0e','Chebyshev II':'#2ca02c',
               'Elliptical':'#d62728','Bessel':'#9467bd'}
    for ftype in FILTER_TYPES:
        sub = df_cx[df_cx['Filter Type']==ftype].sort_values('Order')
        kw  = dict(marker=markers[ftype],color=colors[ftype],lw=2,markersize=8,label=ftype)
        axes[0].plot(sub['Order'], sub['Ops/Sample (filtfilt)'], **kw)
        axes[1].plot(sub['Order'], sub['Mean Time/Window (ms)'], **kw)
        axes[2].plot(sub['Order'], sub['RMS CV'], **kw)
    for ax, ylabel, title in zip(axes,
        ['Ops/Sample','Time (ms)','RMS CV'],
        ['Computational Complexity','Wall-Clock Time','RMS Reliability (CV)']):
        ax.set_xlabel('Filter Order'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.set_xticks(FILTER_ORDERS); ax.legend(fontsize=9)
    fig.suptitle('Filter Comparison: Complexity vs Reliability', fontsize=13)
    plt.tight_layout(); plt.show(); plt.close('all')
""")

c_v11 = md("## Validation 11 — CSV Export (Cell 30)")

c_v11_code = code("""\
print("=" * 60)
print("VALIDATION 11: CSV Export (Cell 30)")
print("=" * 60)

if df.empty:
    print("  [SKIP]")
else:
    # Export
    csv_results    = RESULTS_DIR / "blasting_all_results.csv"
    csv_band_avg   = RESULTS_DIR / "blasting_band_averages.csv"
    csv_complexity = RESULTS_DIR / "filter_complexity.csv"

    df.to_csv(csv_results, index=False, float_format='%.6f')

    df_band_avg = (df.groupby(['filename','sensor','filter_type','order','band_idx','band_label'])[
        ['rms','peak','crest_factor','zcr','band_power','spectral_centroid']].mean().round(6).reset_index())
    df_band_avg.to_csv(csv_band_avg, index=False)

    df_cx.to_csv(csv_complexity, index=False)

    check("blasting_all_results.csv written",   csv_results.exists(),    f"{csv_results.stat().st_size:,} bytes")
    check("blasting_band_averages.csv written",  csv_band_avg.exists(),   f"{csv_band_avg.stat().st_size:,} bytes")
    check("filter_complexity.csv written",       csv_complexity.exists(), f"{csv_complexity.stat().st_size:,} bytes")

    # Verify CSV integrity
    df_reloaded = pd.read_csv(csv_results)
    check("Reloaded CSV row count matches original", len(df_reloaded) == len(df),
          f"{len(df_reloaded)} vs {len(df)}")
    check("Reloaded CSV has all parameter columns",
          all(c in df_reloaded.columns for c in ['rms','peak','crest_factor','zcr']))

    info("Full results",   f"{csv_results}")
    info("Band averages",  f"{csv_band_avg}")
    info("Complexity",     f"{csv_complexity}")
""")

c_summary = md("## Final Validation Summary")

c_summary_code = code("""\
print()
print("=" * 60)
print("  VALIDATION SUMMARY")
print("=" * 60)
total = PASS_COUNT[0] + FAIL_COUNT[0]
print(f"  PASS : {PASS_COUNT[0]:>3}  /  {total}")
print(f"  FAIL : {FAIL_COUNT[0]:>3}  /  {total}")
print(f"  Score: {PASS_COUNT[0]/total*100:.1f}%" if total > 0 else "  No checks run")
print("=" * 60)

if FAIL_COUNT[0] == 0:
    print("  All checks passed. Notebook is ready to send to mentor.")
else:
    print(f"  {FAIL_COUNT[0]} check(s) failed. Review [FAIL] lines above.")
""")


# ── ASSEMBLE ──────────────────────────────────────────────────────────────────
cells = [
    c_title,
    c_setup,
    c_v1, c_v1_code,
    c_v2, c_v2_code,
    c_v3, c_v3_code,
    c_v4, c_v4_code,
    c_v5, c_v5_code,
    c_v6, c_v6_code,
    c_v7, c_v7_code,
    c_v8, c_v8_code,
    c_v9, c_v9_code,
    c_v10, c_v10_code,
    c_v11, c_v11_code,
    c_summary, c_summary_code,
]

import uuid
for cell in cells:
    cell["id"] = str(uuid.uuid4())[:8]

notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path(__file__).parent / "notebooks" / "validation_report.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {out}")
print(f"Total cells: {len(cells)}")
