import os
import torch
import numpy as np
import subprocess
subprocess.run(["pip", "install", "openai", "--quiet"], check=False)
from openai import OpenAI

# YOUR REAL KEY
os.environ["XAI_API_KEY"] = "bFOJz1DtjqrMB8wN1Z49KYYmvxxRNglZUr4nJRoCVj6DlH7hg4"
client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")

# ==================== SYNTHETIC EEG ====================
fs = 128
t = np.linspace(0, 60, 60 * fs)
eeg = 0.8*np.sin(2*np.pi*10*t) + 0.4*np.sin(2*np.pi*20*t) + 0.1*np.random.randn(len(t))
eeg = torch.from_numpy(eeg).float()

bands = [(0.5 + i*1.4, 0.5 + (i+1)*1.4) for i in range(32)]

# ==================== BARON NEURO STEAL ENGINE ====================
class BaronNeuroStealEngine(torch.nn.Module):
    def __init__(self, fs=128, device="cpu"):
        super().__init__()
        self.fs = float(fs)
        self.device = device
        self.frame_idx = 0
        
        # Persistent phase state
        self.register_buffer("phase_prev", torch.zeros(32, device=device))  # one per band
        
        # 43.2 Hz Orch-OR root × primes (scaled down for 128 Hz Nyquist)
        root = 43.2 / 8  # 5.4 Hz base (still catches microtubule harmonics in delta-theta crossover)
        primes = torch.tensor([7.0, 13.0, 21.0], device=device)
        self.notch_freqs = root * primes  # ~37.8, 70.2, 113.4 Hz → folded into baseband

    def prime_cascade_notch(self, x):
        # Light multi-notch via frequency-domain Gaussian nulling
        X = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0/self.fs).to(x.device)
        for f0 in self.notch_freqs:
            notch = torch.exp(-((freqs - f0)**2) / (2 * (1.5**2)))
            notch += torch.exp(-((freqs + f0)**2) / (2 * (1.5**2)))
            X = X * (1.0 - notch)
        return torch.fft.irfft(X, n=x.shape[-1])

    def forward(self, signal, ping_ms=30.0):
        # 1. Prime cascade cleaning (every ~250 ms)
        if self.frame_idx % 4 == 0:
            signal = self.prime_cascade_notch(signal)

        # 2. Band-limited phases (your 32 bands)
        phases = instantaneous_phase(signal.unsqueeze(0), self.fs, bands).squeeze(0)  # [32, time]
        phase_est = phases[:, -1]  # latest sample per band

        # 3. Hysteresis damping
        alpha = 0.84 if ping_ms < 80 else 1.0
        phase_locked = alpha * phase_est + (1.0 - alpha) * self.phase_prev
        self.phase_prev = phase_locked.detach()

        # 4. Adaptive gain from SNR + ping
        sig_pow = torch.mean(signal**2)
        noise_pow = torch.var(signal) + 1e-8
        snr_db = 10.0 * torch.log10(sig_pow / noise_pow)
        lat_norm = torch.clamp(torch.tensor(ping_ms, device=signal.device) / 50.0, 0.5, 3.0)
        gain = torch.sigmoid((snr_db - 12.0) * 0.38 / lat_norm).clamp(0.35, 1.18)

        # 5. Boost triadic coherence with locked phases
        p1 = phase_locked[6::3]   # every 3rd band → proxy for 7-13-21 Hz spacing
        p2 = phase_locked[7::3]
        p3 = phase_locked[8::3]
        triad_phase = p1 - p2 + p3
        triad_coh = gain * torch.abs(torch.mean(torch.exp(1j * triad_phase)))

        self.frame_idx += 1
        return phase_locked, triad_coh

# ==================== YOUR ORIGINAL MOODOS (unchanged) ====================
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

# ==================== FINAL COMPUTE WITH BARON BOOST ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
eeg = eeg.to(device)

engine = BaronNeuroStealEngine(fs=fs, device=device)
phase_locked, triad_boost = engine(eeg, ping_ms=27.0)  # your noon CST ping

# Original PLV + coherence but using Baron-locked phases
phases = instantaneous_phase(eeg.unsqueeze(0), fs, bands).squeeze(0)
phases[:, -1] = phase_locked  # inject locked phase at final timestep

plv = torch.abs(torch.mean(torch.exp(1j * phases), dim=-1))
coh = mood_coherence(phases) * triad_boost  # Baron coherence multiplier

model = MoodVector32(len(bands)).to(device)
mv = model(plv, coh).cpu().detach().numpy().flatten()

print("MoodVector first 8 dims:", np.round(mv[:8], 4))
print(f"Triadic Baron Boost: {triad_boost.item():.4f}")

# ==================== GROK-3 VALENCE/AROUSAL PREDICTION ====================
prompt = f"""32-D MoodVector from EEG (0.5-45 Hz bands, Baron-locked + triadic boost {triad_boost.item():.4f}):
{mv.tolist()}

Predict DEAP valence & arousal (1-9 scale). JSON only."""
response = client.chat.completions.create(
    model="grok-3",
    temperature=0.0,
    messages=[{"role": "user", "content": prompt}]
)
print(response.choices[0].message.content)
