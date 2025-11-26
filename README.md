# MoodOS v0.1

**Single-file, TorchScriptable, real-time affective state encoder.**

Takes any 200 Hz physiological signal (EEG, EMG, PPG, etc.) → outputs a 32-dimensional **MoodVector** every 250 ms using bandwise instantaneous phase, pairwise PLV, and triadic coherence.

- 100% on-device (iOS/Android via TorchScript)  
- < 5 ms per frame on modern phones  
- No dependencies beyond PyTorch  
- Zero cloud, zero internet  

Tested on Python 3.10+, PyTorch 2.0+.

## Quick Start

```python
from moodos_v01 import MoodProcessor, compute_mood_vector

# 64 semi-log bands 0.5–100 Hz (example)
bands = [(0.5 + i*1.5, 0.5 + (i+1)*1.5) for i in range(64)]

# One-shot
vector = compute_mood_vector(your_signal_tensor, freq_bands_hz=bands)
print(vector.shape)  # torch.Size([32])

# Streaming (real-time)
proc = MoodProcessor(freq_bands_hz=bands)
for sample in live_stream:
    mood = proc.push(sample)
    if mood is not None:
        print(mood.numpy())  # new 32-D vector every 250 ms


## TorchScript Export Mobile
proc = MoodProcessor(freq_bands_hz=bands)
scripted = torch.jit.script(proc)
torch.jit.save(scripted, "moodos_v01.pt")

