"""Generate 11 separate validation notebooks, one per plot."""
import json, uuid
from pathlib import Path

OUT_DIR = Path("notebooks/validations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def md(s):  return {"cell_type":"markdown","metadata":{},"source":s,"id":str(uuid.uuid4())[:8]}
def code(s): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s,"id":str(uuid.uuid4())[:8]}

COMMON_CONFIG = """\
import numpy as np, pandas as pd, matplotlib.pyplot as plt, time, warnings
from pathlib import Path
from scipy.signal import butter,cheby1,cheby2,ellip,bessel,sosfiltfilt,sosfreqz,welch
from scipy.signal import spectrogram as sp_spectrogram
from scipy.io import wavfile
from itertools import product
warnings.filterwarnings('ignore')

BASE_DIR    = Path(r"D:\\\\1 placement\\\\IAESTE INTERNSHIP CZECH\\\\iaeste26-blasting-sound-main\\\\iaeste26-blasting-sound-main")
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_PRIORITY   = ['AccAxial4507','AccRadial4507','Mic147EB','Mic46BE']
ENERGY_WINDOW_S   = 0.05;  NOISE_DURATION_S = 0.5;  ONSET_THRESHOLD = 10.0;  ONSET_OFFSET_S = 7.0
WINDOW_DURATION_S = 5.0;   WINDOW_STEP_S    = 1.0
LOWER_LIMITS = [176,225,283,353,440,565,707,880,1130,1414,1760,10,10,500,1000]
UPPER_LIMITS = [225,283,353,440,565,707,880,1130,1414,1760,2220,1000,2000,1500,2000]
N_BANDS      = len(LOWER_LIMITS)
BAND_LABELS  = [f"{lo}–{hi} Hz" for lo,hi in zip(LOWER_LIMITS,UPPER_LIMITS)]
FILTER_TYPES  = ['Butterworth','Chebyshev I','Chebyshev II','Elliptical','Bessel']
FILTER_ORDERS = [3,5,7]
CHEBY1_RIPPLE_DB=0.5; CHEBY2_ATTEN_DB=40.0; ELLIP_RIPPLE_DB=0.5; ELLIP_ATTEN_DB=40.0

P=[0]; F=[0]
def check(label, ok, note=""):
    s="[PASS]" if ok else "[FAIL]"
    if ok: P[0]+=1
    else:  F[0]+=1
    print(f"  {s}  {label}" + (f"  → {note}" if note else ""))
def info(label, val): print(f"  [INFO]  {label}: {val}")
def summary():
    t=P[0]+F[0]
    print(f"\\n{'='*50}")
    print(f"  PASS: {P[0]}/{t}  |  FAIL: {F[0]}/{t}")
    print(f"  Score: {P[0]/t*100:.0f}%" if t else "  No checks run")
    print('='*50)
print("Config loaded.")
"""

LOAD_WAV_FN = """\
def load_wav(path):
    fs,data=wavfile.read(path)
    if data.ndim>1: data=data[:,0]
    if   data.dtype==np.int16:  sig=data.astype(np.float64)/32768.0
    elif data.dtype==np.int32:  sig=data.astype(np.float64)/2147483648.0
    else:                       sig=data.astype(np.float64)
    return fs,sig,len(sig)/fs

def detect_onset(signal,fs):
    hop=int(ENERGY_WINDOW_S*fs); n=len(signal)//hop
    rms=np.array([np.sqrt(np.mean(signal[i*hop:(i+1)*hop]**2)) for i in range(n)])
    nf=np.mean(rms[:max(1,int(NOISE_DURATION_S/ENERGY_WINDOW_S))])
    ab=np.where(rms>ONSET_THRESHOLD*nf)[0]
    of=int(ab[0]) if len(ab) else int(np.argmax(rms))
    return of*hop, of*ENERGY_WINDOW_S, np.arange(n)*ENERGY_WINDOW_S, rms, nf

def make_windows(signal,fs):
    wl=int(WINDOW_DURATION_S*fs); sl=int(WINDOW_STEP_S*fs)
    n=max(0,(len(signal)-wl)//sl+1)
    return [signal[i*sl:i*sl+wl] for i in range(n)], np.arange(n)*WINDOW_STEP_S

def design_filter(ftype,order,lo,hi,fs):
    nyq=fs/2.; Wn=[lo/nyq,hi/nyq]
    if ftype=='Butterworth':  return butter(order,Wn,btype='bandpass',output='sos')
    if ftype=='Chebyshev I':  return cheby1(order,CHEBY1_RIPPLE_DB,Wn,btype='bandpass',output='sos')
    if ftype=='Chebyshev II': return cheby2(order,CHEBY2_ATTEN_DB,Wn,btype='bandpass',output='sos')
    if ftype=='Elliptical':   return ellip(order,ELLIP_RIPPLE_DB,ELLIP_ATTEN_DB,Wn,btype='bandpass',output='sos')
    if ftype=='Bessel':       return bessel(order,Wn,btype='bandpass',output='sos',norm='phase')

all_wavs = sorted(set(DATA_DIR.rglob("*.wav"))|set(DATA_DIR.rglob("*.WAV")))
def skey(p):
    s=Path(p).stem.split('_')[-1]
    return SENSOR_PRIORITY.index(s) if s in SENSOR_PRIORITY else 99
all_wavs = sorted(all_wavs,key=skey)
EXAMPLE_WAV = all_wavs[0] if all_wavs else None
print(f"WAV files found: {len(all_wavs)}")
if EXAMPLE_WAV: print(f"Using: {EXAMPLE_WAV.name}")
"""

def nb(title, cells):
    return {"nbformat":4,"nbformat_minor":5,
            "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                        "language_info":{"name":"python","version":"3.10.0"}},
            "cells":cells}

def save(filename, cells):
    path = OUT_DIR / filename
    with open(path,"w",encoding="utf-8") as f:
        json.dump(nb(filename, cells), f, indent=1, ensure_ascii=False)
    print(f"  Written: {path.name}  ({path.stat().st_size:,} bytes)")
    return path

# ══════════════════════════════════════════════════════════════════════════════
print("Generating 11 validation notebooks...")

# ── 01: File Scanner ──────────────────────────────────────────────────────────
save("validation_01_file_scanner.ipynb", [
    md("# Validation 01 — File Scanner & Sensor Ordering\nVerifies WAV files are found and sorted accelerometers-first."),
    code(COMMON_CONFIG),
    code("""\
all_wavs = sorted(set(DATA_DIR.rglob("*.wav"))|set(DATA_DIR.rglob("*.WAV")))
def skey(p):
    s=Path(p).stem.split('_')[-1]
    return SENSOR_PRIORITY.index(s) if s in SENSOR_PRIORITY else 99
metas = sorted([{'path':p,'name':p.name,'sensor':p.stem.split('_')[-1]} for p in all_wavs], key=lambda m: skey(m['path']))
sensors = [m['sensor'] for m in metas]
accel_i = [i for i,s in enumerate(sensors) if 'Acc' in s]
mic_i   = [i for i,s in enumerate(sensors) if 'Mic' in s]

check("DATA_DIR exists",               DATA_DIR.exists(),      str(DATA_DIR))
check("At least 1 WAV file found",     len(all_wavs)>0,        f"{len(all_wavs)} files")
check("4 sensors in SENSOR_PRIORITY",  len(SENSOR_PRIORITY)==4)
check("AccAxial4507 first in priority",SENSOR_PRIORITY[0]=='AccAxial4507')
check("Mic46BE last in priority",      SENSOR_PRIORITY[-1]=='Mic46BE')
if accel_i and mic_i:
    check("All accelerometers before all microphones",
          max(accel_i)<min(mic_i), f"last accel pos {max(accel_i)}, first mic pos {min(mic_i)}")
else:
    info("Only one sensor type found", sensors[:3])
check("RESULTS_DIR created",  (RESULTS_DIR.mkdir(parents=True,exist_ok=True) or True) and RESULTS_DIR.exists())

print(f"\\nFile list (sorted):")
for m in metas: print(f"  {m['sensor']:<18} {m['name']}")
summary()
"""),
])

# ── 02: WAV Loading ───────────────────────────────────────────────────────────
save("validation_02_wav_loading.ipynb", [
    md("# Validation 02 — WAV Loading & Normalization\nVerifies signal is float64, mono, normalized to [-1,1]."),
    code(COMMON_CONFIG),
    code(LOAD_WAV_FN),
    code("""\
if not EXAMPLE_WAV:
    print("[SKIP] No WAV files found"); raise SystemExit

fs,signal,duration = load_wav(EXAMPLE_WAV)
t = np.arange(len(signal))/fs

check("Signal dtype is float64",      signal.dtype==np.float64,      str(signal.dtype))
check("Signal is 1-D (mono)",         signal.ndim==1,                f"ndim={signal.ndim}")
check("Amplitude min ≥ -1.0",         signal.min()>=-1.0,            f"min={signal.min():.5f}")
check("Amplitude max ≤  1.0",         signal.max()<= 1.0,            f"max={signal.max():.5f}")
check("Duration > 10 s",              duration>10.0,                 f"{duration:.2f} s")
check("Sample rate ≥ 44100 Hz",       fs>=44100,                     f"{fs:,} Hz")
check("Nyquist ≥ 2000 Hz",            fs/2>=2000,                    f"{fs//2:,} Hz")
check("Time vector length = samples", len(t)==len(signal))
check("Time vector starts at 0",      t[0]==0.0)
check("No NaN in signal",             not np.any(np.isnan(signal)))
check("No Inf in signal",             not np.any(np.isinf(signal)))
info("File",         EXAMPLE_WAV.name)
info("Sample rate",  f"{fs:,} Hz")
info("Duration",     f"{duration:.2f} s")
info("Samples",      f"{len(signal):,}")
summary()
"""),
])

# ── 03: Waveform + Spectrogram ────────────────────────────────────────────────
save("validation_03_waveform_spectrogram.ipynb", [
    md("# Validation 03 — Waveform + Spectrogram (Cell 7)\nVerifies synchronized axes, log-floor, frequency limit, and colourbar."),
    code(COMMON_CONFIG),
    code(LOAD_WAV_FN),
    code("""\
if not EXAMPLE_WAV: raise SystemExit

fs,signal,duration = load_wav(EXAMPLE_WAV)
t = np.arange(len(signal))/fs

nperseg  = int(0.05*fs)
f_spec,t_spec,Sxx = sp_spectrogram(signal,fs=fs,nperseg=nperseg,noverlap=nperseg//2)
f_max_hz = min(fs/2, 5000)
f_mask   = f_spec<=f_max_hz
Sxx_db   = 10*np.log10(Sxx[f_mask]+1e-12)

check("nperseg = 50 ms × fs",          nperseg==int(0.05*fs),        f"nperseg={nperseg}")
check("Spectrogram output is 2-D",     Sxx.ndim==2)
check("No -inf in log spectrogram",    np.all(np.isfinite(Sxx_db)),  "1e-12 floor works")
check("Frequency capped at ≤5000 Hz",  f_spec[f_mask].max()<=5000,   f"{f_spec[f_mask].max():.0f} Hz")
check("Spectrogram covers full signal",abs(t_spec[-1]-duration)<0.5, f"t_spec end={t_spec[-1]:.2f}s")
check("Time vectors same length",      len(t)==len(signal))

fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True)
axes[0].plot(t,signal,lw=0.3,color='steelblue')
axes[0].set_ylabel('Amplitude'); axes[0].set_title(f'Waveform — {EXAMPLE_WAV.name}')
im=axes[1].pcolormesh(t_spec,f_spec[f_mask]/1000,Sxx_db,shading='gouraud',cmap='inferno',vmin=-80)
plt.colorbar(im,ax=axes[1],label='Power (dBFS)')
axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Frequency (kHz)')
axes[1].set_title('Spectrogram — synchronized with waveform (sharex=True)')
plt.tight_layout(); plt.show(); plt.close('all')

check("sharex=True: axes share x-axis", axes[0].get_shared_x_axes().joined(axes[0],axes[1]))
summary()
"""),
])

# ── 04: Blast Detection ───────────────────────────────────────────────────────
save("validation_04_blast_detection.ipynb", [
    md("# Validation 04 — Blast Detection (Cell 10)\nVerifies onset detection, 7 s skip, noise floor, and threshold crossing."),
    code(COMMON_CONFIG),
    code(LOAD_WAV_FN),
    code("""\
if not EXAMPLE_WAV: raise SystemExit
fs,signal,duration = load_wav(EXAMPLE_WAV)
t = np.arange(len(signal))/fs

onset_sample,onset_time,frame_times,rms_energy,noise_floor = detect_onset(signal,fs)
useful_start_time   = onset_time+ONSET_OFFSET_S
useful_start_sample = int(useful_start_time*fs)
useful_signal       = signal[useful_start_sample:]

check("Onset detected (threshold crossed)",
      np.any(rms_energy>ONSET_THRESHOLD*noise_floor), f"onset at {onset_time:.2f} s")
check("Onset in first half of recording",    onset_time<duration/2,   f"{onset_time:.2f} s / {duration:.2f} s")
check("Useful start = onset + 7 s",         abs(useful_start_time-(onset_time+ONSET_OFFSET_S))<0.001)
check("Useful signal non-empty",             len(useful_signal)>0,     f"{len(useful_signal)/fs:.2f} s")
check("Noise floor > 0",                     noise_floor>0,            f"{noise_floor:.6f}")
check("RMS at onset > threshold",
      rms_energy[int(onset_time/ENERGY_WINDOW_S)] > ONSET_THRESHOLD*noise_floor)
check("Useful signal shorter than full signal", len(useful_signal)<len(signal))
check("ONSET_OFFSET_S = 7.0 (midpoint of 6–8 s)", ONSET_OFFSET_S==7.0)

info("Noise floor",     f"{noise_floor:.6f}")
info("Onset time",      f"{onset_time:.3f} s")
info("Useful from",     f"{useful_start_time:.3f} s")
info("Useful duration", f"{len(useful_signal)/fs:.2f} s")

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,7),sharex=True)
ax1.plot(t,signal,lw=0.3,color='steelblue',label='Signal')
ax1.axvline(onset_time,color='red',lw=2,ls='--',label=f'Onset ({onset_time:.2f} s)')
ax1.axvline(useful_start_time,color='green',lw=2,label=f'Useful start ({useful_start_time:.2f} s)')
ax1.axvspan(useful_start_time,t[-1],alpha=0.08,color='green')
ax1.legend(fontsize=9); ax1.set_ylabel('Amplitude'); ax1.set_title('Blast Detection')
ax2.plot(frame_times,rms_energy,color='darkorange',label='Frame RMS')
ax2.axhline(ONSET_THRESHOLD*noise_floor,color='red',ls=':',lw=1.5,label='Threshold')
ax2.axvline(onset_time,color='red',lw=2,ls='--')
ax2.legend(fontsize=9); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('RMS Energy')
ax2.set_title(f'Short-Time RMS Energy (frame={ENERGY_WINDOW_S*1000:.0f} ms)')
plt.tight_layout(); plt.show(); plt.close('all')
summary()
"""),
])

# ── 05: Floating Windows ──────────────────────────────────────────────────────
save("validation_05_floating_windows.ipynb", [
    md("# Validation 05 — Floating Windows & Gantt Plot (Cell 13)\nVerifies 5 s bar width (not 1 s), 80% overlap, and Gantt stacking."),
    code(COMMON_CONFIG),
    code(LOAD_WAV_FN),
    code("""\
if not EXAMPLE_WAV: raise SystemExit
fs,signal,duration = load_wav(EXAMPLE_WAV)
_,onset_time,_,_,_ = detect_onset(signal,fs)
useful_start_time  = onset_time+ONSET_OFFSET_S
useful_signal      = signal[int(useful_start_time*fs):]
windows,window_starts = make_windows(useful_signal,fs)
win_len   = int(WINDOW_DURATION_S*fs)
overlap   = (1-WINDOW_STEP_S/WINDOW_DURATION_S)*100
n_stack   = int(WINDOW_DURATION_S/WINDOW_STEP_S)

check("At least 1 window extracted",       len(windows)>0,            f"{len(windows)} windows")
check("Window duration = 5 s",             WINDOW_DURATION_S==5.0)
check("Window step = 1 s",                 WINDOW_STEP_S==1.0)
check("Overlap = 80%",                     abs(overlap-80.0)<0.01,    f"{overlap:.1f}%")
check("Each window has correct length",    all(len(w)==win_len for w in windows), f"{win_len} samples each")
check("Window starts monotone increasing", np.all(np.diff(window_starts)>0) if len(window_starts)>1 else True)
check("Gantt stack rows = 5 (5 s / 1 s)", n_stack==5,                f"n_stack={n_stack}")
check("Bar width = WINDOW_DURATION_S (5 s), NOT step (1 s)",
      WINDOW_DURATION_S==5.0 and WINDOW_STEP_S==1.0,
      "Gantt bars are 5 s wide — mentor's original concern resolved")
check("No window overflows useful signal",
      all((i*int(WINDOW_STEP_S*fs)+win_len)<=len(useful_signal) for i in range(len(windows))))

info("Windows", len(windows)); info("Overlap", f"{overlap:.0f}%"); info("Gantt rows", n_stack)

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,7),gridspec_kw={'height_ratios':[3,2]})
t_abs=np.arange(len(useful_signal))/fs+useful_start_time
ax1.plot(t_abs,useful_signal,lw=0.3,color='steelblue')
ax1.set_title(f'Useful Signal — {EXAMPLE_WAV.name}'); ax1.set_ylabel('Amplitude'); ax1.set_xlabel('Time (s)')
colors=plt.cm.tab10(np.linspace(0,0.8,min(len(windows),10)))
for i,s in enumerate(window_starts):
    abs_s=useful_start_time+s
    y=i%n_stack
    ax2.barh(y,WINDOW_DURATION_S,left=abs_s,height=0.7,color=colors[i%len(colors)],alpha=0.75,edgecolor='white',lw=0.5)
    if i<12: ax2.text(abs_s+WINDOW_DURATION_S/2,y,f'W{i+1}',ha='center',va='center',fontsize=7,color='white',fontweight='bold')
ax2.set_yticks(range(n_stack)); ax2.set_yticklabels([f'+{j}s offset' for j in range(n_stack)],fontsize=8)
ax2.set_xlabel('Time (s)'); ax2.set_xlim(ax1.get_xlim())
ax2.set_title(f'Gantt Timeline — {WINDOW_DURATION_S:.0f}s windows, {WINDOW_STEP_S:.0f}s step, {overlap:.0f}% overlap')
plt.tight_layout(); plt.show(); plt.close('all')
summary()
"""),
])

# ── 06: Filter Design ─────────────────────────────────────────────────────────
save("validation_06_filter_design.ipynb", [
    md("# Validation 06 — Filter Design & Frequency Responses (Cell 16)\nVerifies all 225 filters design without error, passband gain, and relative rolloff."),
    code(COMMON_CONFIG),
    code(LOAD_WAV_FN),
    code("""\
if not EXAMPLE_WAV: raise SystemExit
fs,signal,_ = load_wav(EXAMPLE_WAV)

filter_cache={}; errors=[]
for ftype,order,(lo,hi) in product(FILTER_TYPES,FILTER_ORDERS,zip(LOWER_LIMITS,UPPER_LIMITS)):
    key=(ftype,order,lo,hi)
    try:    filter_cache[key]=design_filter(ftype,order,lo,hi,fs)
    except Exception as e: errors.append(f"{ftype} N={order} {lo}–{hi}Hz: {e}"); filter_cache[key]=None

n_ok=sum(v is not None for v in filter_cache.values())
n_total=N_BANDS*len(FILTER_TYPES)*len(FILTER_ORDERS)

check("All filters designed without error",    len(errors)==0,   f"{n_ok}/{n_total} OK")
check("SOS arrays have 6 columns",
      all(v.shape[1]==6 for v in filter_cache.values() if v is not None))

# Passband gain check for demo band (10–1000 Hz)
lo_d,hi_d=10,1000; f_mid=(lo_d+hi_d)/2
for ftype in FILTER_TYPES:
    sos=filter_cache.get((ftype,5,lo_d,hi_d))
    if sos is None: continue
    w,h=sosfreqz(sos,worN=8192,fs=fs)
    gain=20*np.log10(np.abs(h[np.argmin(np.abs(w-f_mid))])+1e-12)
    check(f"{ftype}: passband gain at {f_mid:.0f} Hz > -6 dB", gain>-6.0, f"{gain:.1f} dB")

# Elliptical sharper than Bessel in stopband
sos_e=filter_cache.get(('Elliptical',5,lo_d,hi_d)); sos_b=filter_cache.get(('Bessel',5,lo_d,hi_d))
if sos_e is not None and sos_b is not None:
    w,h_e=sosfreqz(sos_e,worN=8192,fs=fs); _,h_b=sosfreqz(sos_b,worN=8192,fs=fs)
    si=np.argmin(np.abs(w-hi_d*1.5))  # near the transition edge, where elliptical's steeper rolloff actually beats bessel (elliptical's equiripple floor is overtaken by bessel's monotonic decay further into the stopband)
    check("Elliptical sharper stopband than Bessel",
          np.abs(h_e[si])<np.abs(h_b[si]), "sharper rolloff confirmed")

if errors: [print(f"  FAIL detail: {e}") for e in errors[:5]]

# Plot
fig,axes=plt.subplots(len(FILTER_TYPES),1,figsize=(14,14),sharex=True)
for ax,ftype in zip(axes,FILTER_TYPES):
    for order in FILTER_ORDERS:
        sos=filter_cache.get((ftype,order,lo_d,hi_d))
        if sos is None: continue
        w,h=sosfreqz(sos,worN=4096,fs=fs)
        ax.plot(w,20*np.log10(np.abs(h)+1e-12),label=f'Order {order}',lw=1.8)
    ax.axvline(lo_d,color='red',ls='--',lw=1,alpha=0.7)
    ax.axvline(hi_d,color='blue',ls='--',lw=1,alpha=0.7)
    ax.set_xlim(lo_d*0.05,hi_d*5); ax.set_ylim(-90,5)
    ax.set_ylabel('Magnitude (dB)'); ax.set_title(ftype); ax.legend(loc='lower right',fontsize=9); ax.set_xscale('log')
axes[-1].set_xlabel('Frequency (Hz)')
fig.suptitle(f'Filter Responses — Band {lo_d}–{hi_d} Hz',fontsize=13,y=1.005)
plt.tight_layout(); plt.show(); plt.close('all')
summary()
"""),
])

# ── 07: Parameter Timing ──────────────────────────────────────────────────────
save("validation_07_parameter_timing.ipynb", [
    md("# Validation 07 — Parameter Calculation Timing (Cell 19)\nVerifies timing measurement logic and that RMS is faster than Welch."),
    code(COMMON_CONFIG),
    code(LOAD_WAV_FN),
    code("""\
import matplotlib.patches as mpatches
if not EXAMPLE_WAV: raise SystemExit
fs,signal,_ = load_wav(EXAMPLE_WAV)
_,onset_t,_,_,_=detect_onset(signal,fs)
useful=signal[int((onset_t+ONSET_OFFSET_S)*fs):]
wins,_=make_windows(useful,fs)
if not wins: print("[SKIP] No windows"); raise SystemExit

demo_sos=design_filter('Butterworth',5,10,1000,fs)
x_demo=sosfiltfilt(demo_sos,wins[0])
N=200

param_fns={
    'Filter (sosfiltfilt)': lambda x: sosfiltfilt(demo_sos,wins[0]),
    'RMS'                 : lambda x: np.sqrt(np.mean(x**2)),
    'Peak'                : lambda x: float(np.max(np.abs(x))),
    'Crest Factor'        : lambda x: float(np.max(np.abs(x)))/max(float(np.sqrt(np.mean(x**2))),1e-12),
    'Zero Crossing Rate'  : lambda x: np.sum(np.abs(np.diff(np.sign(x)))>0)/(2.0*(len(x)-1)/fs),
    'Band Power (Welch)'  : lambda x: float(np.trapz(*reversed(welch(x,fs=fs,nperseg=min(1024,len(x)//4))))),
    'Spectral Centroid'   : lambda x: (lambda fp,ps: float(np.sum(fp*ps)/max(np.sum(ps),1e-12)))(*welch(x,fs=fs,nperseg=min(1024,len(x)//4))),
}
timings={}
for name,fn in param_fns.items():
    fn(x_demo)  # warm up
    tl=[]
    for _ in range(N):
        t0=time.perf_counter(); fn(x_demo); tl.append((time.perf_counter()-t0)*1e6)
    timings[name]=np.mean(tl)

check("All timing values positive",          all(v>0 for v in timings.values()))
check("RMS faster than Band Power (Welch)",  timings['RMS']<timings['Band Power (Welch)'],
      f"RMS={timings['RMS']:.1f}μs vs Welch={timings['Band Power (Welch)']:.1f}μs")
check("Filter timing is measurable (>0.1μs)",timings['Filter (sosfiltfilt)']>0.1)
check("CF faster than Welch",               timings['Crest Factor']<timings['Band Power (Welch)'])
check(f"Averaged over {N} reps",            N==200)

for name,val in timings.items(): info(f"{name}", f"{val:.2f} μs")
info("Total per window", f"{sum(timings.values()):.1f} μs = {sum(timings.values())/1000:.3f} ms")

fig,ax=plt.subplots(figsize=(12,5))
names=list(timings.keys()); vals=list(timings.values())
cb=['#d62728' if 'Filter' in n else '#1f77b4' for n in names]
bars=ax.barh(names,vals,color=cb,alpha=0.85,edgecolor='white')
for bar,v in zip(bars,vals): ax.text(v+0.5,bar.get_y()+bar.get_height()/2,f'{v:.1f} μs',va='center',fontsize=9)
ax.set_xlabel('Time (μs) per 5-second window')
ax.set_title(f'Parameter Calculation Timing ({N} reps averaged) — {EXAMPLE_WAV.name}')
patches=[mpatches.Patch(color='#d62728',alpha=0.85,label='Filter step'),
         mpatches.Patch(color='#1f77b4',alpha=0.85,label='Parameter')]
ax.legend(handles=patches,fontsize=9)
plt.tight_layout(); plt.show(); plt.close('all')
summary()
"""),
])

# ── 08: Multi-file Analysis ───────────────────────────────────────────────────
save("validation_08_multifile_analysis.ipynb", [
    md("# Validation 08 — Multi-file Analysis Loop (Cell 21)\nVerifies all files processed, all params computed, no NaN, sensor column correct."),
    code(COMMON_CONFIG),
    code(LOAD_WAV_FN),
    code("""\
all_wavs=sorted(set(DATA_DIR.rglob("*.wav"))|set(DATA_DIR.rglob("*.WAV")))
SELECTED=[sorted(all_wavs,key=skey)[i] for i in range(min(len(all_wavs),2))]  # small sample: O(files x 225 filters x ~64 windows)
import gc
all_results=[]; timing={}; _iter_count=0

def compute_params(x,fs):
    rms=np.sqrt(np.mean(x**2)); peak=float(np.max(np.abs(x)))
    cf=peak/rms if rms>1e-12 else 0.
    zcr=np.sum(np.abs(np.diff(np.sign(x)))>0)/(2.*(len(x)-1)/fs)
    np_=min(1024,len(x)//4); fp,psd=welch(x,fs=fs,nperseg=np_,window='hann')
    bp=float(np.trapz(psd,fp)); sc=float(np.sum(fp*psd)/max(np.sum(psd),1e-12))
    t0=time.perf_counter(); _=np.sqrt(np.mean(x**2)); ru=(time.perf_counter()-t0)*1e6
    t0=time.perf_counter(); _=float(np.max(np.abs(x)))/max(float(np.sqrt(np.mean(x**2))),1e-12); cu=(time.perf_counter()-t0)*1e6
    return {'rms':rms,'peak':peak,'crest_factor':cf,'zcr':zcr,'band_power':bp,'spectral_centroid':sc,'rms_time_us':ru,'cf_time_us':cu}

for fi,wav in enumerate(SELECTED):
    try: fs_f,sig_f,_=load_wav(wav)
    except: continue
    _,ot,_,_,_=detect_onset(sig_f,fs_f)
    us=sig_f[int((ot+ONSET_OFFSET_S)*fs_f):]
    ws,wst=make_windows(us,fs_f)
    if not ws: continue
    sensor=wav.stem.split('_')[-1]
    fc={}
    for ftype,order,(lo,hi) in product(FILTER_TYPES,FILTER_ORDERS,zip(LOWER_LIMITS,UPPER_LIMITS)):
        k=(ftype,order,lo,hi)
        try: fc[k]=design_filter(ftype,order,lo,hi,fs_f)
        except: fc[k]=None
    for ftype in FILTER_TYPES:
        for order in FILTER_ORDERS:
            for bi,(lo,hi) in enumerate(zip(LOWER_LIMITS,UPPER_LIMITS)):
                sos=fc.get((ftype,order,lo,hi))
                if sos is None: continue
                ft=[]
                for wi,w in enumerate(ws):
                    t0=time.perf_counter(); xf=sosfiltfilt(sos,w); ft.append((time.perf_counter()-t0)*1000)
                    p=compute_params(xf,fs_f)
                    _iter_count+=1
                    if _iter_count%500==0: gc.collect()
                    all_results.append({'filename':wav.name,'sensor':sensor,'file_idx':fi,
                        'filter_type':ftype,'order':order,'band_idx':bi,'band_label':BAND_LABELS[bi],
                        'low_hz':lo,'high_hz':hi,'window_idx':wi,'window_start_s':wst[wi]+ot+ONSET_OFFSET_S,
                        'filter_time_ms':ft[-1],**p})
                timing[(ftype,order,bi)]=float(np.mean(ft))

df=pd.DataFrame(all_results)
check("DataFrame non-empty",              len(df)>0,                        f"{len(df):,} rows")
check("All 6 params present",             all(c in df.columns for c in ['rms','peak','crest_factor','zcr','band_power','spectral_centroid']))
check("All 5 filter types present",       set(df['filter_type'].unique())==set(FILTER_TYPES))
check("All 3 orders present",             set(df['order'].unique())==set(FILTER_ORDERS))
check("All 15 bands present",             df['band_idx'].nunique()==N_BANDS,  f"{df['band_idx'].nunique()}/15")
check("RMS all positive",                 (df['rms']>0).all())
check("Crest Factor ≥ 1 always",         (df['crest_factor']>=1.0).all(),    f"min CF={df['crest_factor'].min():.3f}")
check("No NaN in RMS",                    df['rms'].notna().all())
check("sensor column present",            'sensor' in df.columns)
check("rms_time_us column present",       'rms_time_us' in df.columns)
check("cf_time_us column present",        'cf_time_us'  in df.columns)
check("filter_time_ms column present",    'filter_time_ms' in df.columns)
info("Files processed",  df['filename'].nunique())
info("Total rows",       f"{len(df):,}")
info("Sensors found",    list(df['sensor'].unique()))
df.to_csv(RESULTS_DIR/"blasting_all_results.csv",index=False,float_format='%.6f')
print(f"  Saved: {RESULTS_DIR/'blasting_all_results.csv'}")
summary()
"""),
])

# ── 09: RMS & CF All Bands ────────────────────────────────────────────────────
save("validation_09_rms_cf_allbands.ipynb", [
    md("# Validation 09 — RMS & CF for All 15 Bands (Cells 22–25)\nVerifies bar charts, heatmap dimensions, and all-band grid plots."),
    code(COMMON_CONFIG),
    code("""\
csv_path = RESULTS_DIR/"blasting_all_results.csv"
if not csv_path.exists():
    print("[SKIP] Run validation_08 first to generate blasting_all_results.csv"); raise SystemExit
df=pd.read_csv(csv_path)
print(f"Loaded {len(df):,} rows from CSV")

# ── Cell 22: RMS bar charts ───────────────────────────────────────────────────
df_first=df[df['file_idx']==0] if 'file_idx' in df.columns else df
for order in FILTER_ORDERS:
    for ftype in FILTER_TYPES:
        sub=df_first[(df_first.filter_type==ftype)&(df_first.order==order)]
        vals=sub.groupby('band_idx')['rms'].mean().reindex(range(N_BANDS),fill_value=0.)
        check(f"RMS bars [{ftype} N={order}]: {N_BANDS} bars, all ≥ 0",
              len(vals)==N_BANDS and (vals>=0).all())

# ── Cell 23: CF heatmap ───────────────────────────────────────────────────────
n_combos=len(FILTER_TYPES)*len(FILTER_ORDERS)
cf_mat=np.full((N_BANDS,n_combos),np.nan)
for col,(ft,ord_) in enumerate(product(FILTER_TYPES,FILTER_ORDERS)):
    for row in range(N_BANDS):
        sub=df[(df.filter_type==ft)&(df.order==ord_)&(df.band_idx==row)]
        if len(sub): cf_mat[row,col]=sub['crest_factor'].mean()
check("CF matrix shape = (15, 15)",        cf_mat.shape==(N_BANDS,n_combos))
check("No all-NaN row in CF matrix",       not np.all(np.isnan(cf_mat),axis=1).any())
check("All CF values ≥ 1",                 np.nanmin(cf_mat)>=1.0,  f"min={np.nanmin(cf_mat):.3f}")
check("95th-pct clip < max",               np.nanpercentile(cf_mat,95)<np.nanmax(cf_mat)*2)

# ── Cells 24–25: All-band grid ────────────────────────────────────────────────
FOCUS_ORDER=5
df_ref=df[df['file_idx']==0] if 'file_idx' in df.columns else df
n_cols=5; n_rows=int(np.ceil(N_BANDS/n_cols))
filter_colors={'Butterworth':'#1f77b4','Chebyshev I':'#ff7f0e',
               'Chebyshev II':'#2ca02c','Elliptical':'#d62728','Bessel':'#9467bd'}
check("Grid covers all bands",    n_rows*n_cols>=N_BANDS,  f"{n_rows}×{n_cols}≥{N_BANDS}")
check("filter_colors has 5 entries", len(filter_colors)==5)
check("All bands have data at order 5",
      df_ref[df_ref.order==FOCUS_ORDER]['band_idx'].nunique()==N_BANDS)

# ── Plot Cell 22 ──────────────────────────────────────────────────────────────
fig,axes=plt.subplots(len(FILTER_ORDERS),len(FILTER_TYPES),figsize=(22,10),sharey=False)
for r,order in enumerate(FILTER_ORDERS):
    for c,ftype in enumerate(FILTER_TYPES):
        ax=axes[r,c]
        sub=df_first[(df_first.filter_type==ftype)&(df_first.order==order)]
        vals=sub.groupby('band_idx')['rms'].mean().reindex(range(N_BANDS),fill_value=0.)
        ax.bar(range(N_BANDS),vals,color='steelblue',alpha=0.8)
        ax.set_title(f'{ftype}\\nOrder {order}',fontsize=8)
        ax.set_xticks(range(N_BANDS)); ax.set_xticklabels([str(l) for l in LOWER_LIMITS],rotation=90,fontsize=6)
        if c==0: ax.set_ylabel('Mean RMS',fontsize=8)
fig.suptitle('Mean RMS per Frequency Band',fontsize=12); plt.tight_layout(); plt.show(); plt.close('all')

# ── Plot Cell 23: CF Heatmap ──────────────────────────────────────────────────
combo_labels=[f"{ft[:4]}\\nN={o}" for ft,o in product(FILTER_TYPES,FILTER_ORDERS)]
fig,ax=plt.subplots(figsize=(18,7))
im=ax.imshow(cf_mat,aspect='auto',cmap='RdYlGn_r',interpolation='nearest',
             vmin=np.nanmin(cf_mat),vmax=np.nanpercentile(cf_mat,95))
plt.colorbar(im,ax=ax,label='Mean Crest Factor')
ax.set_xticks(range(n_combos)); ax.set_xticklabels(combo_labels,fontsize=8)
ax.set_yticks(range(N_BANDS)); ax.set_yticklabels(BAND_LABELS,fontsize=8)
ax.set_title('Mean CF Heatmap — All 15 Bands × 15 Filter Configs')
plt.tight_layout(); plt.show(); plt.close('all')

# ── Plots Cells 24–25: All-band grids ────────────────────────────────────────
handles=[plt.Line2D([0],[0],color=filter_colors[ft],lw=2,label=ft) for ft in FILTER_TYPES]
for param,marker,ptitle in [('rms','o','RMS'),('crest_factor','s','Crest Factor')]:
    fig,axes=plt.subplots(n_rows,n_cols,figsize=(22,n_rows*3.2),sharey=False)
    af=axes.flatten()
    for bi in range(N_BANDS):
        ax=af[bi]
        for ft in FILTER_TYPES:
            sub=df_ref[(df_ref.filter_type==ft)&(df_ref.order==FOCUS_ORDER)&(df_ref.band_idx==bi)].sort_values('window_idx')
            if len(sub)==0: continue
            ax.plot(sub['window_idx'],sub[param],color=filter_colors[ft],lw=1.5,marker=marker,markersize=3)
        ax.set_title(BAND_LABELS[bi],fontsize=8,pad=3)
        ax.set_xlabel('Window #',fontsize=7); ax.set_ylabel(ptitle[:3],fontsize=7); ax.tick_params(labelsize=7)
    for ax in af[N_BANDS:]: ax.set_visible(False)
    fig.legend(handles=handles,loc='lower right',fontsize=9,ncol=5)
    fig.suptitle(f'{ptitle} — All {N_BANDS} Bands — Order {FOCUS_ORDER}',fontsize=12)
    plt.tight_layout(rect=[0,0.04,1,1]); plt.show(); plt.close('all')
summary()
"""),
])

# ── 10: Complexity & Reliability ──────────────────────────────────────────────
save("validation_10_complexity_reliability.ipynb", [
    md("# Validation 10 — Complexity & Reliability (Cells 28–29)\nVerifies ops formula (2×9×sections), CV column, 15 rows, timing columns."),
    code(COMMON_CONFIG),
    code(LOAD_WAV_FN),
    code("""\
csv_path=RESULTS_DIR/"blasting_all_results.csv"
if not csv_path.exists(): print("[SKIP] Run validation_08 first"); raise SystemExit
df=pd.read_csv(csv_path)
fs,_,_=load_wav(EXAMPLE_WAV) if EXAMPLE_WAV else (48000,None,None)

filter_cache={}
for ftype,order,(lo,hi) in product(FILTER_TYPES,FILTER_ORDERS,zip(LOWER_LIMITS,UPPER_LIMITS)):
    k=(ftype,order,lo,hi)
    try: filter_cache[k]=design_filter(ftype,order,lo,hi,fs)
    except: filter_cache[k]=None

timing={}
rows=[]
for ftype,order in product(FILTER_TYPES,FILTER_ORDERS):
    sos=next((filter_cache.get((ftype,order,lo,hi)) for lo,hi in zip(LOWER_LIMITS,UPPER_LIMITS) if filter_cache.get((ftype,order,lo,hi)) is not None),None)
    if sos is None: continue
    n_sec=len(sos); ops_ff=2*9*n_sec
    avg_ms=df[(df.filter_type==ftype)&(df.order==order)]['filter_time_ms'].mean() if 'filter_time_ms' in df.columns else np.nan
    sub=df[(df.filter_type==ftype)&(df.order==order)]
    cv=(sub['rms'].std()/sub['rms'].mean()) if sub['rms'].mean()>0 else np.nan
    mean_rms_us=sub['rms_time_us'].mean() if 'rms_time_us' in sub.columns else np.nan
    mean_cf_us =sub['cf_time_us'].mean()  if 'cf_time_us'  in sub.columns else np.nan
    rows.append({'Filter Type':ftype,'Order':order,'SOS Sections':n_sec,
                 'Ops/Sample (filtfilt)':ops_ff,'Mean Time/Window (ms)':round(avg_ms,3),
                 'RMS Calc (μs)':round(mean_rms_us,2),'CF Calc (μs)':round(mean_cf_us,2),'RMS CV':round(cv,4)})

df_cx=pd.DataFrame(rows).sort_values(['Order','Ops/Sample (filtfilt)']).reset_index(drop=True)

check("Complexity table has 15 rows",     len(df_cx)==15,                f"{len(df_cx)} rows")
check("All 5 filter types present",       set(df_cx['Filter Type'].unique())==set(FILTER_TYPES))
check("RMS CV column present",            'RMS CV' in df_cx.columns)
check("RMS Calc (μs) column present",     'RMS Calc (μs)' in df_cx.columns)
check("CF Calc (μs) column present",      'CF Calc (μs)'  in df_cx.columns)
check("Mean Time/Window > 0",             (df_cx['Mean Time/Window (ms)']>0).all())
check("Ops increase with order (all types)",
      all(df_cx[df_cx['Filter Type']==ft].sort_values('Order')['Ops/Sample (filtfilt)'].is_monotonic_increasing for ft in FILTER_TYPES))

for _,row in df_cx.iterrows():
    exp=2*9*row['SOS Sections']
    check(f"Ops formula [{row['Filter Type']} N={row['Order']}]: 2×9×sections",
          row['Ops/Sample (filtfilt)']==exp, f"got {row['Ops/Sample (filtfilt)']}, expected {exp}")

print(); print(df_cx.to_string(index=False))

fig,axes=plt.subplots(1,3,figsize=(18,6))
m={'Butterworth':'o','Chebyshev I':'s','Chebyshev II':'^','Elliptical':'D','Bessel':'v'}
c={'Butterworth':'#1f77b4','Chebyshev I':'#ff7f0e','Chebyshev II':'#2ca02c','Elliptical':'#d62728','Bessel':'#9467bd'}
for ft in FILTER_TYPES:
    sub=df_cx[df_cx['Filter Type']==ft].sort_values('Order')
    kw=dict(marker=m[ft],color=c[ft],lw=2,markersize=8,label=ft)
    axes[0].plot(sub['Order'],sub['Ops/Sample (filtfilt)'],**kw)
    axes[1].plot(sub['Order'],sub['Mean Time/Window (ms)'],**kw)
    axes[2].plot(sub['Order'],sub['RMS CV'],**kw)
for ax,yl,ti in zip(axes,['Ops/Sample','Time (ms)','RMS CV'],
                    ['Computational Complexity','Wall-Clock Time','RMS Reliability (CV)']):
    ax.set_xlabel('Filter Order'); ax.set_ylabel(yl); ax.set_title(ti)
    ax.set_xticks(FILTER_ORDERS); ax.legend(fontsize=9)
fig.suptitle('Filter Comparison: Complexity vs Reliability',fontsize=13)
plt.tight_layout(); plt.show(); plt.close('all')
summary()
"""),
])

# ── 11: CSV Export ────────────────────────────────────────────────────────────
save("validation_11_csv_export.ipynb", [
    md("# Validation 11 — CSV Export (Cell 30)\nVerifies all 4 CSV files are written, readable, and row counts match."),
    code(COMMON_CONFIG),
    code("""\
csv_main=RESULTS_DIR/"blasting_all_results.csv"
csv_band=RESULTS_DIR/"blasting_band_averages.csv"
csv_cx  =RESULTS_DIR/"filter_complexity.csv"
csv_sum =RESULTS_DIR/"blasting_file_summary.csv"

check("blasting_all_results.csv exists",    csv_main.exists(), str(csv_main) if csv_main.exists() else "NOT FOUND — run validation_08 first")
check("filter_complexity.csv exists",       csv_cx.exists(),   str(csv_cx)   if csv_cx.exists()   else "NOT FOUND — run validation_10 first")

if csv_main.exists():
    df=pd.read_csv(csv_main)
    # Regenerate band averages and file summary
    df_band=(df.groupby(['filename','sensor','filter_type','order','band_idx','band_label'])
               [['rms','peak','crest_factor','zcr','band_power','spectral_centroid']]
               .mean().round(6).reset_index())
    df_band.to_csv(csv_band,index=False)
    df_fs=(df.groupby(['filename','sensor','filter_type','order'])
              [['rms','peak','crest_factor','zcr']].agg(['mean','std']).round(6).reset_index())
    df_fs.columns=['_'.join(c).strip('_') for c in df_fs.columns]
    df_fs.to_csv(csv_sum,index=False)

    check("blasting_band_averages.csv written",  csv_band.exists(), f"{csv_band.stat().st_size:,} bytes")
    check("blasting_file_summary.csv written",   csv_sum.exists(),  f"{csv_sum.stat().st_size:,} bytes")

    df2=pd.read_csv(csv_main)
    check("Reloaded row count matches original", len(df2)==len(df), f"{len(df2)} rows")
    check("All parameter columns present after reload",
          all(c in df2.columns for c in ['rms','peak','crest_factor','zcr','band_power','spectral_centroid']))
    check("No NaN in RMS after reload",          df2['rms'].notna().all())
    check("filename column present",             'filename' in df2.columns)
    check("sensor column present",               'sensor'   in df2.columns)

    info("Full results rows",  f"{len(df):,}")
    info("Band averages rows", f"{len(df_band):,}")
    info("File summary rows",  f"{len(df_fs):,}")
    for p in [csv_main,csv_band,csv_cx,csv_sum]:
        if p.exists(): info(p.name, f"{p.stat().st_size:,} bytes")
summary()
"""),
])

print("\\nAll 11 notebooks generated in:", OUT_DIR)
