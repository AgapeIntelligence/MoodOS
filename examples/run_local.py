import torch
from moodos_v01 import compute_mood_vector, log_freq_bands  # Recommended: use the built-in log bands

# =============================================
# Signal Generation (10 Hz strong + 40 Hz weak + noise)
# =============================================
fs = 128  # Sampling rate in Hz
duration = 1.0  # seconds
t = torch.linspace(0, duration, fs, dtype=torch.float32)

# Clean physiological-like signal: dominant alpha (10 Hz) + weak gamma (40 Hz) + noise
sig = (
    0.7 * torch.sin(2 * torch.pi * 10 * t) +   # Strong calm/alpha rhythm
    0.3 * torch.sin(2 * torch.pi * 40 * t) +   # Weak high-arousal gamma
    0.05 * torch.randn_like(t)                # Light sensor noise
)

# =============================================
# Recommended: Log-spaced bands (much better for affect modeling)
# =============================================
bands = log_freq_bands(n_bands=32, f_min=1.0, f_max=60.0, sample_rate=fs)
# Alternative: keep your linear bands if you want to test them
# bands = [(0.5 + i*1.4, 0.5 + (i+1)*1.4) for i in range(32)]

# =============================================
# Compute MoodVector
# =============================================
with torch.no_grad():
    mv = compute_mood_vector(
        raw_signal=sig,
        sample_rate=fs,
        freq_bands_hz=bands
    )

# =============================================
# Results
# =============================================
print("MoodOS v1.0 — MoodVector Results")
print("="*50)
print(f"Signal duration   : {duration} sec @ {fs} Hz")
print(f"Frequency bands   : {len(bands)} logarithmic bands from {bands[0][0]:.1f}–{bands[-1][1]:.1f} Hz")
print(f"MoodVector shape  : {mv.shape}")
print(f"Norm (energy)     : {mv.norm().item():.4f}")
print()
print("First 12 dimensions:")
print(mv[:12].numpy().round(4))
print()
print("Last 12 dimensions:")
print(mv[-12:].numpy().round(4))
