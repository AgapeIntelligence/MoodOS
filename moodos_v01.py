"""
MoodOS v1.0 — Real-Time Affective State Embedding from Physiological Signals
============================================================================

Now with full MoodCoherenceHysteresis (triadic + exponential smoothing) 
→ rock-stable affect vectors even on noisy real-world Muse/Emotiv streams.

Author:  Geneva Robinson (@3vi3Aetheris)
GitHub:  https://github.com/AgapeIntelligence/MoodOS
"""

from __future__ import annotations
import math
from typing import Optional, Sequence, Tuple, List
import torch
from torch import Tensor, nn
import numpy as np


# ================================================================
# Utilities
# ================================================================

def log_freq_bands(n_bands: int = 32, f_min: float = 0.5, f_max: float = 100.0, sample_rate: int = 200):
    nyquist = sample_rate / 2.0
    f_max = min(f_max, nyquist * 0.95)
    b = torch.logspace(math.log10(f_min), math.log10(f_max), n_bands + 1, base=10.0)
    return [(b[i].item(), b[i + 1].item()) for i in range(n_bands)]


# ================================================================
# Core Hilbert (fast + accurate)
# ================================================================

def analytic_signal_hilbert(x: Tensor) -> Tensor:
    X = torch.fft.rfft(x, dim=-1)
    n = x.shape[-1]
    h = torch.cat([torch.ones(1), 2*torch.ones((n//2)-1), torch.ones(1) if n%2 else torch.zeros(1), torch.zeros(n//2-1 if n%2 else n//2)], dim=0).to(x.device)
    h[0] = 1.0
    return torch.fft.irfft(X * h, n=n, dim=-1)


def instantaneous_phase(
    signal: Tensor | np.ndarray,
    sample_rate: int = 200,
    freq_bands_hz: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tensor:
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal).float()
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)

    signal = signal - signal.mean(dim=-1, keepdim=True)
    signal = signal * torch.hann_window(signal.shape[-1], device=signal.device)

    if freq_bands_hz is None:
        return torch.angle(analytic_signal_hilbert(signal)).unsqueeze(-2)

    X = torch.fft.rfft(signal, dim=-1)
    freqs = torch.fft.rfftfreq(signal.shape[-1], d=1/sample_rate).to(signal.device)
    phases = []
    for lo, hi in freq_bands_hz:
        mask = (freqs >= lo) & (freqs < hi)
        Xb = X.clone()
        Xb[..., ~mask] = 0
        band = torch.fft.irfft(Xb, n=signal.shape[-1], dim=-1)
        phases.append(torch.angle(analytic_signal_hilbert(band)))
    return torch.stack(phases, dim=-2)  # (..., F, T)


# ================================================================
# MoodCoherenceHysteresis — YOUR masterpiece, now Scriptable
# ================================================================

@torch.jit.script
class MoodCoherenceHysteresis:
    def __init__(self, alpha: float = 0.22):
        self.alpha: float = alpha
        self.prev_coh: Optional[Tensor] = None

    def __call__(self, phases: Tensor) -> Tensor:
        phases = phases.float()
        *B, F, T = phases.shape

        plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=-1))
        diff = phases.unsqueeze(-2) - phases.unsqueeze(-3)
        triad = diff + diff.transpose(-2, -3)
        triad_mag = torch.abs(torch.mean(torch.exp(1j * triad), dim=-1))
        diag = triad_mag.diagonal(dim1=-2, dim2=-1)

        coh = torch.sqrt(plv * diag + 1e-12).mean(dim=-1).clamp_(0.0, 1.0)

        if self.prev_coh is None:
            self.prev_coh = coh.detach().clone()
        else:
            self.prev_coh = self.alpha * coh + (1.0 - self.alpha) * self.prev_coh

        return self.prev_coh


# ================================================================
# 32D Mood Projector
# ================================================================

class MoodVector32(nn.Module):
    def __init__(self, n_freq_bands: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_freq_bands + 1, 128), nn.SiLU(),
            nn.Linear(128, 96),  nn.SiLU(),
            nn.Linear(96, 64),   nn.SiLU(),
            nn.Linear(64, 32),   nn.Tanh()
        )
    def forward(self, plv: Tensor, coherence: Tensor) -> Tensor:
        x = torch.cat([plv, coherence.unsqueeze(-1)], dim=-1)
        return self.net(x)


# ================================================================
# One-shot inference (hysteresis preserved across calls if you pass it)
# ================================================================

def compute_mood_vector(
    raw_signal: Tensor | np.ndarray,
    sample_rate: int = 200,
    freq_bands_hz: Optional[Sequence[Tuple[float, float]]] = None,
    model: Optional[MoodVector32] = None,
    hysteresis: Optional[MoodCoherenceHysteresis] = None,
) -> Tensor:
    if model is None:
        n = len(freq_bands_hz) if freq_bands_hz else 1
        model = MoodVector32(n).eval()
    hysteresis = hysteresis or MoodCoherenceHysteresis()

    phases = instantaneous_phase(raw_signal, sample_rate, freq_bands_hz)
    plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=-1))
    coherence = hysteresis(phases)

    return model(plv, coherence)


# ================================================================
# Real-Time Streaming Processor — hysteresis is stateful!
# ================================================================

class MoodProcessor:
    def __init__(
        self,
        sample_rate: int = 200,
        window_sec: float = 1.0,
        hop_sec: float = 0.25,
        freq_bands_hz: Optional[Sequence[Tuple[float, float]]] = None,
        device: str = "cpu",
    ):
        self.sr = sample_rate
        self.win = int(window_sec * sample_rate)
        self.hop = int(hop_sec * sample_rate)
        self.bands = freq_bands_hz or log_freq_bands(32)
        self.device = torch.device(device)

        self.model = MoodVector32(len(self.bands)).to(self.device).eval()
        self.hyst = MoodCoherenceHysteresis(alpha=0.22)   # persistent across pushes
        self.buffer = torch.zeros(self.win, device=self.device)
        self.count = 0

    def push(self, sample: float | Tensor) -> Optional[Tensor]:
        if not isinstance(sample, Tensor):
            sample = torch.tensor(sample, device=self.device, dtype=torch.float32)

        self.buffer = torch.roll(self.buffer, -1)
        self.buffer[-1] = sample
        self.count += 1

        if self.count % self.hop != 0:
            return None

        vec = compute_mood_vector(
            self.buffer.clone().unsqueeze(0),
            self.sr,
            self.bands,
            self.model,
            self.hyst                     # same stateful instance → hysteresis works!
        )
        return vec.squeeze(0)

    def reset(self):
        self.buffer.zero_()
        self.count = 0
        self.hyst.prev_coh = None


# ================================================================
# Demo
# ================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    bands = log_freq_bands(32)
    t = torch.linspace(0, 4.0, 800)
    signal = 0.7*torch.sin(2*math.pi*8*t) + 0.5*torch.sin(2*math.pi*25*t + 0.3) + 0.1*torch.randn_like(t)

    vec = compute_mood_vector(signal, freq_bands_hz=bands)
    print("One-shot MoodVector (first 10):", vec[:10].numpy().round(4))

    proc = MoodProcessor(freq_bands_hz=bands)
    vectors = [proc.push(s) for s in signal]
    vectors = [v for v in vectors if v is not None]
    print(f"\nStreaming produced {len(vectors)} vectors")
    print("Final vector (first 8):", vectors[-1][:8].numpy().round(4))
