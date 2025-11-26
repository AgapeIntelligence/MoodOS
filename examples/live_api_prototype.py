import os
import torch
import numpy as np
import subprocess
subprocess.run(["pip", "install", "openai", "--quiet"], check=False)
from openai import OpenAI

# YOUR REAL KEY – already inserted
os.environ["XAI_API_KEY"] = "bFOJz1DtjqrMB8wN1Z49KYYmvxxRNglZUr4nJRoCVj6DlH7hg4"

client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")

# Synthetic EEG
fs = 128
t = np.linspace(0, 60, 60*fs)
eeg = 0.8*np.sin(2*np.pi*10*t) + 0.4*np.sin(2*np.pi*20*t) + 0.1*np.random.randn(len(t))
eeg = torch.from_numpy(eeg).float()

bands = [(0.5 + i*1.4, 0.5 + (i+1)*1.4) for i in range(32)]

# === MoodOS core (exact same as yours) ===
def analytic_signal_fft(x):
    X = torch.fft.fft(x, dim=-1)
    n = x.shape[-1]
    h = torch.zeros(n, device=x.device, dtype=X.dtype)
    h[0] = 1
    if n % 2 == 0: h[n//2] = 1
    h[slice(1, n//2 if n%2==0 else (n+1)//2)] = 2
    h = h.view((1,)*(x.ndim-1) + (n,))
    return torch.fft.ifft(X * h, dim=-1)

def instantaneous_phase(signal, sample_rate=128, freq_bands_hz=None):
    x = signal.float()
    if x.ndim == 1: x = x.unsqueeze(0)
    x = x - x.mean(dim=-1, keepdim=True)
    x = x * torch.hann_window(x.shape[-1], device=x.device)
    if freq_bands_hz is None:
        return torch.angle(analytic_signal_fft(x)).unsqueeze(-2)
    phases = []
    X_full = torch.fft.fft(x, dim=-1)
    freqs = torch.fft.fftfreq(x.shape[-1], d=1.0/sample_rate).to(x.device)
    for low, high in freq_bands_hz:
        mask = (freqs >= low) & (freqs < high)
        mask = mask | ((freqs <= -low) & (freqs > -high))
        X_band = X_full.clone()
        X_band[..., ~mask] = 0
        phases.append(torch.angle(torch.fft.ifft(X_band, dim=-1)))
    return torch.stack(phases, dim=-2)

def mood_coherence(phases):
    plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=-1))
    diff = phases.unsqueeze(-2) - phases.unsqueeze(-3)
    triad_phase = diff + diff.transpose(-2, -3)
    triad_mag = torch.abs(torch.mean(torch.exp(1j * triad_phase), dim=-1))
    diag = triad_mag.diagonal(dim1=-2, dim2=-1)
    return torch.sqrt(plv * diag + 1e-12).mean(dim=-1).clamp(0,1)

class MoodVector32(torch.nn.Module):
    def __init__(self, freq_bins=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(freq_bins + 1, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 96), torch.nn.SiLU(),
            torch.nn.Linear(96, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 32), torch.nn.Tanh()
        )
    def forward(self, plv, coh):
        if coh.ndim == plv.ndim - 1: coh = coh.unsqueeze(-1)
        return self.net(torch.cat([plv, coh], dim=-1))

def compute_mood_vector(sig, sample_rate=128, freq_bands_hz=None):
    phases = instantaneous_phase(sig, sample_rate, freq_bands_hz)
    plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=-1))
    coh = mood_coherence(phases)
    model = MoodVector32(len(freq_bands_hz) if freq_bands_hz else 1)
    return model(plv, coh)

# === RUN ===
mv = compute_mood_vector(eeg, fs, bands).cpu().numpy()
print("MoodVector first 8 dims:", np.round(mv[:8], 4))

prompt = f"32-D MoodVector from EEG (0.5-45 Hz bands): {mv.tolist()}\nPredict DEAP valence & arousal (1-9) with short reasoning. JSON only."
response = client.chat.completions.create(
    model="grok-3",
    messages​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​
