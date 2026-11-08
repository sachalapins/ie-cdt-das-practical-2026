"""
das_utils.py — helper functions for the Intelligent Earth CDT DAS practical.

Everything here is deliberately simple so that the notebook cells stay short
and focus on the *ideas*. Feel free to read it, but you don't need to.

Conventions used throughout
---------------------------
DAS data are 2-D numpy arrays with shape (n_time_samples, n_channels):
    axis 0 = time, axis 1 = position along the fibre ("channel").
"""

import os
import time
import urllib.request
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

# Where the practical data live. Override with the environment variable
# DAS_PRACTICAL_DATA_URL (a URL ending in '/') or DAS_PRACTICAL_DATA_DIR
# (a local folder that already holds the files).
DATA_URL = os.environ.get(
    "DAS_PRACTICAL_DATA_URL",
    "https://github.com/sachalapins/ie-cdt-das-practical-2026/releases/download/v1.0/",
)
DATA_DIR = os.environ.get("DAS_PRACTICAL_DATA_DIR", "practical_data")


def fetch(filename):
    """Download `filename` into DATA_DIR (once) and return its local path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename} ...", end=" ", flush=True)
        urllib.request.urlretrieve(DATA_URL + filename, path)
        print(f"done ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


class DASRecord:
    """A small container for one DAS record section plus its metadata."""

    def __init__(self, data, fs, dx, x0=0.0, starttime="", name="", units="strain rate (arbitrary units)", notes=""):
        self.data = np.asarray(data, dtype=np.float32)
        self.fs = float(fs)          # sampling rate (Hz)
        self.dx = float(dx)          # channel spacing (m)
        self.x0 = float(x0)          # distance of first channel along the fibre (m)
        self.starttime = str(starttime)
        self.name = str(name)
        self.units = str(units)
        self.notes = str(notes)

    # convenient derived quantities
    @property
    def nt(self):
        return self.data.shape[0]

    @property
    def nch(self):
        return self.data.shape[1]

    @property
    def t(self):
        """Time of each sample (s, relative to starttime)."""
        return np.arange(self.nt) / self.fs

    @property
    def x(self):
        """Distance of each channel along the fibre (m)."""
        return self.x0 + np.arange(self.nch) * self.dx

    def __repr__(self):
        return (f"DASRecord '{self.name}'\n"
                f"  start time      : {self.starttime}\n"
                f"  data shape      : {self.data.shape}  (time samples, channels)\n"
                f"  sampling rate   : {self.fs:g} Hz\n"
                f"  channel spacing : {self.dx:g} m\n"
                f"  first channel at: {self.x0:g} m along the fibre\n"
                f"  units           : {self.units}\n"
                + (f"  notes           : {self.notes}\n" if self.notes else ""))


def load_das(filename):
    """Load one of the practical's .npz DAS files as a DASRecord.
    If the file also holds the second fibre of the cable, it is available as `record.fibre2`."""
    f = np.load(fetch(filename), allow_pickle=False)
    meta = {k: f[k].item() for k in f.files if f[k].ndim == 0}
    rec = DASRecord(f["data"], **{k: meta[k] for k in
                                  ("fs", "dx", "x0", "starttime", "name", "units", "notes") if k in meta})
    rec.fibre2 = f["data_fibre2"].astype(np.float32) if "data_fibre2" in f.files else None
    return rec


def load_geophones(event):
    """Load 3-component geophone data for a Rutford event as an ObsPy Stream (raw counts)
    plus the station metadata (Inventory) needed to convert counts to ground velocity."""
    import obspy
    st = obspy.read(fetch(f"rutford_geophones_{event}.mseed"))
    inv = obspy.read_inventory(fetch("rutford_geophones_stations.xml"))
    return st, inv


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _clip_value(data, perc=99, vmax=None):
    """Colour-scale limit: `vmax` if given, otherwise the `perc` percentile of |data|."""
    v = vmax if vmax is not None else np.percentile(np.abs(data), perc)
    return v if v > 0 else 1.0


def plot_das(data, fs, dx=1.0, x0=0.0, t0=0.0, perc=99, vmax=None, ax=None, title="", cmap="seismic",
             tlim=None, xlim=None, colorbar=False, units=""):
    """Plot a DAS record section as an image (time down the page, distance along the fibre across).

    perc: colour scale runs from -v to +v, where v is this percentile of |data| (default 99).
    vmax: give an absolute colour-scale limit instead (e.g. vmax=50 -> colour scale -50..50).
    NB: interpolation='none' so that what you see is the true noise level, not a smoothed version.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    v = _clip_value(data, perc, vmax)
    nt, nch = data.shape
    extent = [x0, x0 + nch * dx, t0 + nt / fs, t0]
    im = ax.imshow(data, interpolation="none", cmap=cmap, vmin=-v, vmax=v, aspect="auto", extent=extent)
    ax.set_xlabel("Distance along fibre (m)")
    ax.set_ylabel("Time (s)")
    if tlim is not None:
        ax.set_ylim(max(tlim), min(tlim))
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, label=units)
    return ax


def plot_record(rec, data=None, **kw):
    """plot_das() for a DASRecord (optionally with replacement data, e.g. a filtered version)."""
    d = rec.data if data is None else data
    kw.setdefault("title", rec.name)
    return plot_das(d, rec.fs, rec.dx, rec.x0, **kw)


def compare(panels, rec, perc=99, vmax=None, same_scale=True, tlim=None, xlim=None, figsize=None):
    """Side-by-side record sections.

    panels: dict {title: 2-D array}. If same_scale, every panel uses the colour scale of the FIRST
            panel, so amplitudes are directly comparable between panels.
    """
    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=figsize or (4.6 * n, 5.5), sharey=True)
    axs = np.atleast_1d(axs)
    first = list(panels.values())[0]
    v = _clip_value(first, perc, vmax)
    for ax, (title, d) in zip(axs, panels.items()):
        plot_das(d, rec.fs, rec.dx, rec.x0, perc=perc, vmax=v if same_scale else vmax, ax=ax, title=title,
                 tlim=tlim, xlim=xlim)
        if ax is not axs[0]:
            ax.set_ylabel("")
    plt.tight_layout()
    return axs


def plot_traces(rec, channels, datasets=None, tlim=None, normalise=False):
    """Plot single-channel time series ('seismograms') from one or more versions of the data.

    datasets: dict {label: 2-D array}; defaults to the raw data.
    """
    datasets = datasets or {"raw": rec.data}
    channels = np.atleast_1d(channels)
    fig, axs = plt.subplots(len(channels), 1, figsize=(11, 2.0 * len(channels)), sharex=True, squeeze=False)
    for ax, ch in zip(axs[:, 0], channels):
        for label, d in datasets.items():
            tr = d[:, ch]
            if normalise:
                tr = tr / (np.abs(tr).max() + 1e-12)
            ax.plot(rec.t, tr, lw=0.8, label=label)
        ax.set_ylabel(f"ch {ch}\n({rec.x0 + ch * rec.dx:.0f} m)")
        ax.axhline(0, color="0.7", lw=0.5)
    axs[0, 0].legend(loc="upper right", fontsize=8)
    axs[-1, 0].set_xlabel("Time (s)")
    if tlim:
        axs[-1, 0].set_xlim(*tlim)
    plt.tight_layout()
    return axs


# ---------------------------------------------------------------------------
# Frequency-domain views
# ---------------------------------------------------------------------------

def amplitude_spectrum(data, fs, channels=None):
    """Average amplitude spectrum over the chosen channels (Welch's method). Returns (freqs, amplitude)."""
    d = data if channels is None else data[:, channels]
    nper = min(512, d.shape[0])
    f, p = signal.welch(d, fs=fs, nperseg=nper, axis=0)
    return f, np.sqrt(p.mean(axis=1))


def plot_spectra(rec, groups, data=None, ax=None, fmax=None):
    """Plot average amplitude spectra for several groups of channels.

    groups: dict {label: slice or list of channel indices}
    """
    d = rec.data if data is None else data
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    for label, chans in groups.items():
        f, a = amplitude_spectrum(d, rec.fs, chans)
        ax.semilogy(f, a, label=label, lw=1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, fmax or rec.fs / 2)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return ax


def fk_transform(data, fs, dx):
    """2-D Fourier transform: time -> frequency f (Hz), distance -> wavenumber k (cycles/m)."""
    nt, nx = data.shape
    win = np.outer(signal.windows.tukey(nt, 0.1), signal.windows.tukey(nx, 0.1))
    F = np.fft.fftshift(np.fft.fft2(data * win))
    f = np.fft.fftshift(np.fft.fftfreq(nt, 1 / fs))
    k = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    return f, k, np.abs(F)


def plot_fk(rec, data=None, ax=None, fmax=None, velocities=(), title="f–k spectrum"):
    """Plot the f–k amplitude spectrum (positive frequencies, dB). Straight lines through the origin
    are constant apparent velocity v = f / k; draw some with velocities=[...] (m/s)."""
    d = rec.data if data is None else data
    f, k, A = fk_transform(d, rec.fs, rec.dx)
    keep = f >= 0
    f, A = f[keep], A[keep]
    A = 20 * np.log10(A / A.max() + 1e-12)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.imshow(A, aspect="auto", origin="lower", cmap="magma", vmin=-60, vmax=0,
              extent=[k[0], k[-1], f[0], f[-1]])
    for v in velocities:
        kk = np.linspace(k[0], k[-1], 401)
        ax.plot(kk, np.abs(kk) * v, "--", lw=1, label=f"{v / 1000:g} km/s")
    if velocities:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, fmax or f[-1])
    ax.set_xlim(k[0], k[-1])
    ax.set_xlabel("Wavenumber (cycles / m)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Classical filters
# ---------------------------------------------------------------------------

def bandpass(data, fs, fmin, fmax, order=4):
    """Zero-phase Butterworth band-pass filter applied to every channel (along the time axis)."""
    if not (0 < fmin < fmax < fs / 2):
        raise ValueError(f"Need 0 < fmin < fmax < Nyquist ({fs / 2:g} Hz); got fmin={fmin}, fmax={fmax}.")
    sos = signal.butter(order, [fmin, fmax], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=0).astype(np.float32)


def notch(data, fs, freqs, q=30):
    """Remove narrow frequency 'lines' (e.g. a generator's hum) at each frequency in `freqs`."""
    out = data.copy()
    for f0 in np.atleast_1d(freqs):
        b, a = signal.iirnotch(f0, q, fs=fs)
        out = signal.filtfilt(b, a, out, axis=0)
    return out.astype(np.float32)


def wiener(data, size=(7, 7)):
    """2-D adaptive Wiener filter: in each (time x channel) window it estimates the local mean and variance,
    and smooths more where the local variance looks like noise. size=(n_time_samples, n_channels)."""
    d = data - data.mean(axis=0, keepdims=True)
    return signal.wiener(d, mysize=size).astype(np.float32)


def remove_common_mode(data, channels=None):
    """Subtract, at every time sample, the median across channels (noise shared by the whole cable,
    e.g. from the interrogator/laser). `channels` selects which channels to estimate it from."""
    ref = data if channels is None else data[:, channels]
    return (data - np.median(ref, axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# DAS-N2N model
# ---------------------------------------------------------------------------

def load_dasn2n():
    """Load the pre-trained DAS-N2N model (GPU + half precision if available, else CPU)."""
    import torch
    from dasn2n import DASN2N
    model = DASN2N()
    model.load_weights()
    if torch.cuda.is_available():
        model = model.half().to("cuda")
        print("DAS-N2N loaded on GPU (half precision).")
    else:
        print("DAS-N2N loaded on CPU (this is fine for the Rutford data; a GPU runtime helps for the big datasets).")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} trainable parameters.")
    return model


def dasn2n_denoise(model, data, channelwise=True):
    """Run DAS-N2N. channelwise=True normalises each channel by its own mean/std before denoising
    (recommended for cables whose noise level changes along the fibre)."""
    if channelwise:
        return model.denoise_numpy(data, remove_mean_axis=0, std_norm_axis=0)
    return model.denoise_numpy(data)


# ---------------------------------------------------------------------------
# Measuring things
# ---------------------------------------------------------------------------

@contextmanager
def timer(label="", results=None):
    """with timer('bandpass', times): ...  -> prints and stores the elapsed time."""
    t = time.perf_counter()
    yield
    dt = time.perf_counter() - t
    print(f"{label}: {dt * 1000:.0f} ms")
    if results is not None:
        results[label] = dt


def _box(data, rec, box):
    (t1, t2), (c1, c2) = box
    i1, i2 = int(round(t1 * rec.fs)), int(round(t2 * rec.fs))
    return data[i1:i2, c1:c2]


def signal_and_noise(data, rec, signal_box, noise_box):
    """RMS noise (from the noise box) and noise-corrected RMS signal (from the signal box):
         signal = sqrt( RMS(signal box)^2 - RMS(noise box)^2 )
    Each box is ((t_start, t_end), (first_channel, last_channel)) with times in seconds."""
    rs = np.sqrt(np.mean(_box(data, rec, signal_box) ** 2))
    rn = np.sqrt(np.mean(_box(data, rec, noise_box) ** 2))
    return np.sqrt(max(rs ** 2 - rn ** 2, 1e-12 * rn ** 2)), rn


def snr_db(data, rec, signal_box, noise_box):
    """Signal-to-noise ratio in decibels: 20 log10(signal / noise), using signal_and_noise()."""
    sig, rn = signal_and_noise(data, rec, signal_box, noise_box)
    return 20 * np.log10(sig / (rn + 1e-12))


def draw_boxes(ax, rec, signal_box, noise_box):
    """Overlay the SNR boxes on a record-section plot."""
    from matplotlib.patches import Rectangle
    for (t, c), col, lab in [(signal_box, "lime", "signal"), (noise_box, "orange", "noise")]:
        x1, x2 = rec.x0 + c[0] * rec.dx, rec.x0 + c[1] * rec.dx
        ax.add_patch(Rectangle((x1, t[0]), x2 - x1, t[1] - t[0], fill=False, ec=col, lw=2, label=lab))
    ax.legend(loc="lower right", fontsize=8)


class Scoreboard:
    """Keeps a table of denoising methods: SNR, signal amplitude kept, run time."""

    def __init__(self, rec, signal_box, noise_box, reference=None):
        self.rec, self.sb, self.nb = rec, signal_box, noise_box
        self.reference = reference   # e.g. the band-passed data, for "signal amplitude kept"
        self.rows = []

    def add(self, method, data, seconds=np.nan, amplitude=True):
        snr = snr_db(data, self.rec, self.sb, self.nb)
        amp = np.nan
        if self.reference is not None and amplitude:
            sig = signal_and_noise(data, self.rec, self.sb, self.nb)[0]
            sig_ref = signal_and_noise(self.reference, self.rec, self.sb, self.nb)[0]
            amp = float(sig / sig_ref)
        self.rows = [r for r in self.rows if r[0] != method] + [(method, snr, amp, seconds)]
        return snr

    def show(self):
        import pandas as pd
        df = pd.DataFrame(self.rows, columns=["method", "SNR (dB)", "signal amplitude vs reference", "run time (s)"])
        return df.set_index("method").round(2)


# ---------------------------------------------------------------------------
# Synthetic noise for the "what does N2N assume?" experiment
# ---------------------------------------------------------------------------

def add_test_noise(data, kind, level=1.0, seed=0):
    """Add a known type of 'noise' to a copy of `data`.

    kind:
      'random'       - independent random noise at every sample and channel (with a spectrum that rises
                       with frequency, like real DAS optical noise)
      'common_mode'  - one random time series added identically to every channel (horizontal stripes)
      'fake_event'   - a coherent dipping wavelet across the channels (looks like a seismic arrival!)
    level: amplitude relative to the standard deviation of `data`.
    """
    rng = np.random.default_rng(seed)
    nt, nch = data.shape
    s = level * np.std(data)
    out = data.copy()
    if kind == "random":
        blue = np.diff(rng.standard_normal((nt + 1, nch)), axis=0)   # differentiated white noise = 'blue' noise
        out += s * blue / blue.std()
    elif kind == "common_mode":
        cm = rng.standard_normal(nt)
        cm = signal.sosfiltfilt(signal.butter(2, 0.2, output="sos"), cm)  # slightly smooth in time
        out += (s * cm / cm.std())[:, None]
    elif kind == "fake_event":
        t = np.arange(nt)[:, None]
        centre = 0.25 * nt + 0.45 * nt * np.arange(nch)[None, :] / nch     # a straight dipping line
        w = 6.0
        out += 4 * s * (1 - 2 * ((t - centre) / w) ** 2) * np.exp(-((t - centre) / w) ** 2)  # Ricker-like
    else:
        raise ValueError("kind must be 'random', 'common_mode' or 'fake_event'")
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Notebook "guard rails"
# ---------------------------------------------------------------------------

class TODO(Exception):
    pass


def require(*values, message="A ✏️ value in this cell has not been filled in yet (replace None with your value)."):
    """Stop the notebook with a friendly message if a TODO hasn't been filled in."""
    for value in values:
        if value is None or (isinstance(value, str) and value.strip() in ("", "?")):
            raise TODO("✋ " + message)


def predict(*predictions):
    """Call at the top of a 'reveal' cell: insists that a prediction was written down first."""
    require(*predictions, message="Write down your prediction in the cell above (any answer is fine!) "
                                  "before running this reveal cell.")
    print("Prediction recorded:", " | ".join(str(p) for p in predictions))


def check_range(name, value, lo, hi, unit=""):
    """Gentle sanity check for a numerical answer."""
    require(value, message=f"Set `{name}` in this cell first (replace None with your value).")
    if lo <= value <= hi:
        print(f"✅ {name} = {value:g} {unit} — looks sensible.")
    else:
        print(f"🤔 {name} = {value:g} {unit} — that's outside the range we'd expect. Worth another look?")
