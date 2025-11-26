import os
import torch
import numpy as np
import sounddevice as sd
from moodos_v01 import compute_mood_vector, MoodVector32

# ---------------------------
# Set your API key in environment
# ---------------------------
os.environ["XAI_API_KEY"] = "u4EXZzfDvX6MOjicLsLJZ12MjDjAFnMSimnzxlpNCg10tC7F4y"

try:
    from openai import OpenAI
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "openai", "--quiet"], check=False)
    from openai import OpenAI

client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")

# ---------------------------
# Audio + MoodOS parameters
# ---------------------------
SAMPLE_RATE = 200
WINDOW_SEC = 1.0
HOP_SEC = 0.25
FREQ_BANDS_HZ = [(0.5 + i*1.4, 0.5 + (i+1)*1.4) for i in range(32)]

win_size = int(WINDOW_SEC * SAMPLE_RATE)
hop_size = int(HOP_SEC * SAMPLE_RATE)
buffer = np.zeros(win_size, dtype=np.float32)
count = 0

model = MoodVector32(len(FREQ_BANDS_HZ))

# ---------------------------
# Streaming callback
# ---------------------------
def audio_callback(indata, frames, time, status):
    global buffer, count
    if status:
        print(status)

    data = indata.mean(axis=1) if indata.ndim > 1 else indata
    for sample in data:
        buffer = np.roll(buffer, -1)
        buffer[-1] = sample
        count += 1

        if count % hop_size != 0:
            continue

        sig_tensor = torch.from_numpy(buffer.copy()).unsqueeze(0)
        mv = compute_mood_vector(sig_tensor, SAMPLE_RATE, FREQ_BANDS_HZ, model)
        mv_np = mv.detach().cpu().numpy()
        print("MoodVector (first 8 dims):", np.round(mv_np[:8], 4))

        # ---------------------------
        # Grok prediction
        # ---------------------------
        prompt = (
            f"32-D MoodVector from EEG/audio: {mv_np.tolist()}\n"
            "Predict DEAP valence & arousal (1-9) with short reasoning. JSON only."
        )
        try:
            response = client.chat.completions.create(
                model="grok-3",
                messages=[{"role": "user", "content": prompt}]
            )
            output = response.choices[0].message.content
            print("Grok prediction:", output)
        except Exception as e:
            print("Grok API error:", e)

# ---------------------------
# Start streaming
# ---------------------------
with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback):
    print(f"Streaming microphone at {SAMPLE_RATE} Hz, press Ctrl+C to stop...")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStreaming stopped.")
