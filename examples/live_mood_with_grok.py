# live_mood_with_grok.py
"""
Live Mood to Valence/Arousal via Microphone + Grok (xAI)
=======================================================

Real-time affect detection from voice/breathing/environment
to 32D MoodVector to Grok instantly maps to Valence & Arousal (1–9)

Features:
    • Zero-latency streaming
    • Secure API key handling (never hardcode!)
    • Robust JSON parsing from Grok
    • Beautiful colored terminal output
    • Graceful fallbacks
"""

import os
import json
import torch
import numpy as np
import sounddevice as sd
from moodos_v01 import compute_mood_vector, MoodVector32, log_freq_bands

# =================================================================
# SECURITY: NEVER commit your API key! Use environment variable
# =================================================================
if not os.getenv("XAI_API_KEY"):
    raise EnvironmentError(
        "Please set your xAI API key with:\n"
        "    export XAI_API_KEY=\"your_key_here\"\n"
        "On Windows:\n"
        "    set XAI_API_KEY=your_key_here"
    )

try:
    from openai import OpenAI
except ImportError:
    print("Installing openai package...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai>=1.30.0", "--quiet"])
    from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

# =================================================================
# MoodOS Configuration
# =================================================================
SAMPLE_RATE = 200
WINDOW_SEC = 1.0
HOP_SEC = 0.25

FREQ_BANDS_HZ = log_freq_bands(n_bands=32, f_min=1.0, f_max=80.0, sample_rate=SAMPLE_RATE)

win_samples = int(WINDOW_SEC * SAMPLE_RATE)
hop_samples = int(HOP_SEC * SAMPLE_RATE)
buffer = np.zeros(win_samples, dtype=np.float32)
counter = 0

model = MoodVector32(n_freq_bands=len(FREQ_BANDS_HZ)).eval()

print("MoodOS + Grok Live Emotion Mirror")
print("="*50)
print("Listening to your voice, breath, or ambient sound...")
print("MoodVector to Grok to Valence/Arousal (1–9) every 250ms\n")

# =================================================================
# Real-time callback
# =================================================================
def audio_callback(indata, frames, time_info, status):
    global buffer, counter

    if status:
        print("Audio status:", status)

    audio = indata.mean(axis=1) if indata.ndim > 1 else indata[:, 0]
    audio = audio.astype(np.float32)

    for sample in audio:
        buffer = np.roll(buffer, -1)
        buffer[-1] = sample
        counter += 1

        if counter % hop_samples != 0:
            continue

        sig = torch.from_numpy(buffer).unsqueeze(0)
        with torch.no_grad():
            mv = compute_mood_vector(sig, SAMPLE_RATE, FREQ_BANDS_HZ, model).cpu().numpy().flatten()

        energy = np.linalg.norm(mv)
        bar = "█" * int(energy * 10) + "░" * (10 - int(energy * 10))

        try:
            response = client.chat.completions.create(
                model="grok-beta",
                messages=[{
                    "role": "user",
                    "content": f"""MoodVector32: {np.round(mv, 4).tolist()}
Convert this to DEAP valence and arousal (1–9 scale).
Respond with valid JSON only:
{{"valence": 7.3, "arousal": 4.1, "emotion": "calm", "reasoning": "short explanation"}}"""
                }],
                temperature=0.0,
                max_tokens=100
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"): raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"): raw = raw.rsplit("\n", 1)[0]
            result = json.loads(raw)

            v = result.get("valence", 5.0)
            a = result.get("arousal", 5.0)
            emo = result.get("emotion", "neutral")
            reason = result.get("reasoning", "")

            print(f"{bar}  V: {v:4.1f} | A: {a:4.1f}  | {emo:8}  to {reason}")

        except Exception as e:
            print(f"{bar}  MoodVector norm: {energy:.3f}  | Grok error: {e}")

# =================================================================
# Start streaming
# =================================================================
try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=hop_samples,
        latency='low',
        callback=audio_callback
    ):
        print("STARTED! Speak, hum, breathe, or play music...")
        print("Ctrl+C to stop\n")
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\n\nStopped. Take care of your mind")

except Exception as e:
    print(f"\nFailed to start audio stream: {e}")
    print("Hint: Run `python -m sounddevice` to list available devices")
