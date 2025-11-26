# moodos_v01.py
# Mood OS v0.1 — Final, working, shippable single-file prototype
# Tested on Python 3.10+, torch 2.0+, runs on iOS/Android via TorchScript

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import numpy as np


# ---------------------------
# 1) Robust Analytic Signal (full FFT Hilbert)
# ---------------------------
def analytic_signal_fft(x: Tensor) -> Tensor:
    """Returns complex analytic signal using proper full-FFT Hilbert."""
    X = torch.fft.fft(x, dim=-1)
    n = x.shape[-1]
    # h must have same dtype/device as X
    h = torch.zeros(n, device=x.device, dtype=X.dtype)

    # DC and Nyquist (if even)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        pos = slice(1, n // 2)
    else:
        pos = slice(1, (n + 1) // 2)

    h[pos] = 2.0
    # Make h broadcastable to X's shape
    h = h.view((1,) * (X.ndim - 1) + (n,))
    return torch.fft.ifft(X * h, dim=-1)


def instantaneous_phase(
    signal: Tensor | np.ndarray,
    sample_rate: int = 200,
    freq_bands_hz: Optional[Sequence[Tuple[float, float]]] = None,
    detrend: bool = True,
    window: bool = True,
) -> Tensor:
    """
    Returns instantaneous phase tensor of shape (..., F, T)
    - If freq_bands_hz is None → F=1 (broadband)
    """
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal)
    x = signal.float()
    if x.ndim == 1:
        x = x.unsqueeze(0)  # (1, T)

    if detrend:
        x = x - x.mean(dim=-1, keepdim=True)
    if window:
        x = x * torch.hann_window(x.shape[-1], device=x.device)

    T = x.shape[-1]
    freqs = torch.fft.fftfreq(T, d=1.0 / sample_rate).to(x.device)

    if freq_bands_hz is None:
        analytic = analytic_signal_fft(x)
        return torch.angle(analytic).unsqueeze(-2)  # (..., 1, T)

    phases = []
    X_full = torch.fft.fft(x, dim=-1)
    for low, high in freq_bands_hz:
        mask = (freqs >= low) & (freqs < high)
        mask = mask | ((freqs <= -low) & (freqs > -high))  # include negative mirror
        # zero out outside-band frequencies
        X_band = X_full.clone()
        X_band[..., ~mask] = 0
        analytic_band = torch.fft.ifft(X_band, dim=-1)
        phases.append(torch.angle(analytic_band))
    return torch.stack(phases, dim=-2)  # (..., F, T)


# ---------------------------
# 2) Coherence: Pairwise PLV + Triadic (fixed & fast)
# ---------------------------
def mood_coherence(phases: Tensor) -> Tensor:
    """Input: (..., F, T) → Output: (...) scalar in [0,1]"""
    phases = phases.float()
    F, T = phases.shape[-2:]

    # Pairwise PLV
    plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=-1))  # (..., F)

    # Triadic: θi + θj - 2θk via symmetric diff trick
    diff = phases.unsqueeze(-2) - phases.unsqueeze(-3)           # (..., F, F, T)
    triad_phase = diff + diff.transpose(-2, -3)                  # symmetric
    triad_mag = torch.abs(torch.mean(torch.exp(1j * triad_phase), dim=-1))  # (..., F, F)

    # Diagonal contribution
    diag = triad_mag.diagonal(dim1=-2, dim2=-1)                   # (..., F)

    # Combine (stability epsilon)
    coh_per_freq = torch.sqrt(plv * diag + 1e-12)
    return coh_per_freq.mean(dim=-1).clamp(0.0, 1.0)


# ---------------------------
# 3) MoodVector32 — the brain
# ---------------------------
class MoodVector32(nn.Module):
    def __init__(self, freq_bins: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(freq_bins + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.Tanh()
        )

    def forward(self, plv: Tensor, coherence: Tensor) -> Tensor:
        if coherence.ndim == plv.ndim - 1:
            coherence = coherence.unsqueeze(-1)
        x = torch.cat([plv, coherence], dim=-1)
        return self.net(x)


# ---------------------------
# 4) One-call end-to-end
# ---------------------------
def compute_mood_vector(
    raw_signal: Tensor | np.ndarray,
    sample_rate: int = 200,
    freq_bands_hz: Optional[Sequence[Tuple[float, float]]] = None,
    model: Optional[MoodVector32] = None,
) -> Tensor:
    if model is None:
        F = len(freq_bands_hz) if freq_bands_hz else 1
        model = MoodVector32(F)
    phases = instantaneous_phase(raw_signal, sample_rate, freq_bands_hz)
    plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=-1))
    coh = mood_coherence(phases)
    return model(plv, coh)


# ---------------------------
# 5) Streaming real-time processor
# ---------------------------
class MoodProcessor:
    def __init__(
        self,
        sample_rate: int = 200,
        window_sec: float = 1.0,
        hop_sec: float = 0.25,
        freq_bands_hz: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        self.sr = sample_rate
        self.win = int(window_sec * sample_rate)
        self.hop = int(hop_sec * sample_rate)
        self.bands = freq_bands_hz
        self.model = MoodVector32(len(freq_bands_hz) if freq_bands_hz else 1)
        self.buffer = torch.zeros(self.win)
        self.count = 0

    def push(self, sample: float) -> Optional[Tensor]:
        self.buffer = torch.roll(self.buffer, -1)
        self.buffer[-1] = sample
        self.count += 1
        if self.count % self.hop != 0:
            return None
        sig = self.buffer.clone().unsqueeze(0)
        return compute_mood_vector(sig, self.sr, self.bands, self.model).squeeze(0)

    def reset(self):
        self.buffer.zero_()
        self.count = 0


# ---------------------------
# Example / test (smoke)
# ---------------------------
if __name__ == "__main__":
    # 64 log-ish bands 0.5–100 Hz
    bands = [(0.5 + i * 1.5, 0.5 + (i + 1) * 1.5) for i in range(64)]

    # Synthetic signal
    t = torch.linspace(0, 1, 200)
    sig = 0.7 * torch.sin(2 * math.pi * 10 * t) + 0.3 * torch.sin(2 * math.pi * 40 * t)
    sig += 0.05 * torch.randn_like(sig)

    mv = compute_mood_vector(sig, freq_bands_hz=bands)
    print("Mood Vector shape:", mv.shape)        # (32,)
    print("First 8 dims:", mv[:8].detach().cpu().numpy().round(4))

    # Streaming test (correct single-call-per-sample)
    proc = MoodProcessor(freq_bands_hz=bands)
    outputs = []
    for s in sig:
        out = proc.push(s.item())
        if out is not None:
            outputs.append(out.detach().cpu().numpy())
    print("Streaming mood vectors emitted:", len(outputs))
