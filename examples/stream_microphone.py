# live_mood_microphone.py
"""
MoodOS Live Demo — Real-time MoodVector from Microphone
======================================================

Listens to your microphone and prints a new 32D mood vector every 250 ms.
Great for testing, biofeedback apps, or artistic installations.

Requirements:
    pip install sounddevice torch numpy moodos-v01
"""

import numpy as np
import torch
import sounddevice as sd
from moodos_v01 import compute_mood_vector, MoodVector32, log_freq_bands

# ==========================
# Configuration
# ==========================
SAMPLE_RATE = 200          # Hz — perfect for voice & breath
WINDOW_SEC = 1.0           # Analysis window
HOP_SEC = 0.25             # Update every 250 ms → 4 Hz mood rate
DEVICE = None              # Auto-detect audio device (or set index)

# Use perceptually better log-spaced bands (recommended!)
FREQ_BANDS_HZ = log_freq_bands(n_bands=32, f_min=1.0, f_max=80.0, sample_rate=SAMPLE_RATE)

# Or keep your original linear bands:
# FREQ_BANDS_HZ = [(0.5 + i*1.4, 0.5 + (i+1)*1.4) for i in range(32)]

# ==========================
# Global state
# ==========================
win_samples = int(WINDOW_SEC * SAMPLE_RATE)
hop_samples = int(HOP_SEC * SAMPLE_RATE)
buffer = np.zeros(win_samples, dtype=np.float32)
sample_counter = 0

# Initialize model (no gradients needed)
model = MoodVector32(n_freq_bands=len(FREQ_BANDS_HZ)).eval()

print("MoodOS Live — Listening to microphone")
print(f"   Sample Rate : {SAMPLE_RATE} Hz")
print(f"   Window       : {WINDOW_SEC}s → {win_samples} samples")
print(f"   Hop          : {HOP_SEC}s → {hop_samples} samples")
print(f"   Bands        : {len(FREQ_BANDS_HZ)} log-spaced ({FREQ_BANDS_HZ[0][0]:.1f}–{FREQ_BANDS_HZ[-1][1]:.1f} Hz)")
print(f"   Mood Rate    : {1.0 / HOP_SEC:.1f} Hz")
print("-" * 60)


def audio_callback(indata: np.ndarray, frames: int, time, status) -> None:
    """Called for every audio block from the microphone."""
    global buffer, sample_counter

    if status:
        print("Audio warning:", status)

    # Convert stereo → mono if needed
    if indata.ndim > 1:
        signal = indata.mean(axis=1)
    else:
        signal = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()

    # Process each incoming sample
    for sample in signal:
        # Shift buffer and insert new sample
        buffer = np.roll(buffer, -1)
        buffer[-1] = sample
        sample_counter += 1

        # Trigger mood computation every hop
        if sample_counter % hop_samples == 0:
            # Convert to torch tensor
            window_tensor = torch.from_numpy(buffer.copy()).unsqueeze(0)  # (1, T)

            with torch.no_grad():
                mood_vector = compute_mood_vector(
                    raw_signal=window_tensor,
                    sample_rate=SAMPLE_RATE,
                    freq_bands_hz=FREQ_BANDS_HZ,
                    model=model
                ).squeeze(0)  # (32,)

            mv_np = mood_vector.cpu().numpy()

            # Pretty print first 8 dimensions + norm
            print(
                f"[{time.inputBufferAdcTime:7.2f}s] "
                f"MoodVector: {np.array2string(mv_np[:8], precision=3, floatmode='fixed', separator=', ')} "
                f"norm={np.linalg.norm(mv_np):.3f}"
            )


# ==========================
# Start streaming
# ==========================
try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=hop_samples,   # Reduces latency
        latency='low',
        device=DEVICE,
        callback=audio_callback
    ):
        print("Listening... Speak, breathe, hum, or stay silent. Ctrl+C to stop.\n")
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nStopped by user. Bye!")

except Exception as e:
    print(f"Error: {e}")
    print("Tip: Run `python -m sounddevice` to list devices and check permissions.")
