# examples/example_realtime.py
import torch
import sounddevice as sd
from moodos_v01 import MoodProcessor

# 64 bands 0.5–100 Hz (same as README)
bands = [(0.5 + i*1.5, 0.5 + (i+1)*1.5) for i in range(64)]
proc = MoodProcessor(sample_rate=200, freq_bands_hz=bands)

print("MoodOS live microphone → MoodVector every 250 ms (Ctrl+C to stop)")

def callback(indata, frames, time, status):
    if status:
        print(status)
        return
    sample = float(indata[0, 0])  # left channel only
    mood = proc.push(sample)
    if mood is not None:
        print("MoodVector:", mood.detach().cpu().numpy().round(4))

with sd.InputStream(samplerate=44100, channels=1, dtype='float32', blocksize=1):
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStopped")
