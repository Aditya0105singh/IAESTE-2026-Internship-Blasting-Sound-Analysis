"""Visualization functions for blasting sound analysis."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import stft as scipy_stft

# Consistent style across all plots
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------------------------
# Single-signal plots
# ---------------------------------------------------------------------------

def plot_waveform(
    audio: np.ndarray,
    sr: int,
    title: str = "Waveform",
    ax: plt.Axes | None = None,
    color: str = "#2563eb",
) -> plt.Axes:
    """Plot amplitude vs time for one audio signal."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 3))

    t = np.arange(len(audio)) / sr
    ax.plot(t, audio, lw=0.4, color=color, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, t[-1])

    if standalone:
        plt.tight_layout()
    return ax


def plot_spectrogram(
    audio: np.ndarray,
    sr: int,
    title: str = "Spectrogram",
    ax: plt.Axes | None = None,
    fmax_hz: float = 10_000.0,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> plt.Axes:
    """Plot STFT magnitude spectrogram (log power, dB scale)."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    freqs, times, Zxx = scipy_stft(
        audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    power_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

    # Limit to fmax_hz
    freq_mask = freqs <= fmax_hz
    freqs = freqs[freq_mask]
    power_db = power_db[freq_mask, :]

    vmin = np.percentile(power_db, 5)
    vmax = np.percentile(power_db, 99)

    im = ax.pcolormesh(
        times, freqs / 1000, power_db,
        shading="auto", cmap="inferno", vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)

    if standalone:
        plt.colorbar(im, ax=ax, label="Power (dB)")
        plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Multi-condition comparison plots
# ---------------------------------------------------------------------------

def plot_pressure_comparison(
    recordings: list[dict],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Waveform + spectrogram grid comparing different pressure levels.

    Args:
        recordings: list of dicts, each with keys:
            audio (np.ndarray), sr (int), label (str)
        output_path: if given, save figure here

    Returns:
        matplotlib Figure
    """
    n = len(recordings)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("Pressure Comparison — Waveform & Spectrogram", fontsize=13, y=1.01)

    colors = plt.cm.plasma(np.linspace(0.2, 0.85, n))

    for i, rec in enumerate(recordings):
        plot_waveform(rec["audio"], rec["sr"], title=rec["label"],
                      ax=axes[0, i], color=colors[i])
        plot_spectrogram(rec["audio"], rec["sr"], title="",
                         ax=axes[1, i])

    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Frequency (kHz)")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


def plot_rms_vs_pressure(
    pressures: list[float],
    rms_values: dict[str, list[float]],
    title: str = "RMS Energy vs Pressure",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Line plot of RMS energy against pressure for each sensor.

    Args:
        pressures: sorted list of pressure values (x-axis)
        rms_values: dict mapping sensor name → list of RMS values (same order as pressures)
        title: plot title
        output_path: if given, save figure here
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D"]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706"]

    for idx, (sensor, vals) in enumerate(rms_values.items()):
        ax.plot(
            pressures, vals,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            label=sensor, lw=2, ms=7,
        )

    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("RMS Amplitude (normalized)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


def plot_rms_vs_mix(
    mix_ratios: list[int],
    rms_values: dict[str, list[float]],
    title: str = "RMS Energy vs Mix Ratio",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Line plot of RMS energy against abrasive mix ratio for each sensor."""
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D"]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706"]

    for idx, (sensor, vals) in enumerate(rms_values.items()):
        ax.plot(
            mix_ratios, vals,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            label=sensor, lw=2, ms=7,
        )

    ax.set_xlabel("Mix Ratio (%)")
    ax.set_ylabel("RMS Amplitude (normalized)")
    ax.set_title(title)
    ax.legend(loc="upper left")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


def plot_sensor_comparison(
    recordings: list[dict],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """4-panel plot comparing all sensors for the same blasting test.

    Args:
        recordings: list of 4 dicts, each: {audio, sr, label}
    """
    n = len(recordings)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))

    fig.suptitle("Sensor Comparison — Same Blasting Test", fontsize=13, y=1.01)

    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706"]

    for i, rec in enumerate(recordings):
        plot_waveform(rec["audio"], rec["sr"], title=rec["label"],
                      ax=axes[i, 0], color=colors[i])
        plot_spectrogram(rec["audio"], rec["sr"], title="",
                         ax=axes[i, 1])

    for ax in axes[:, 0]:
        ax.set_ylabel("Amplitude")
    for ax in axes[:, 1]:
        ax.set_ylabel("Freq (kHz)")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


def plot_spectral_centroid_vs_pressure(
    pressures: list[float],
    centroid_values: dict[str, list[float]],
    title: str = "Spectral Centroid vs Pressure",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Show how the spectral 'brightness' changes with pressure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D"]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706"]

    for idx, (sensor, vals) in enumerate(centroid_values.items()):
        ax.plot(
            pressures, [v / 1000 for v in vals],  # convert to kHz
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            label=sensor, lw=2, ms=7,
        )

    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("Spectral Centroid (kHz)")
    ax.set_title(title)
    ax.legend(loc="upper left")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig
