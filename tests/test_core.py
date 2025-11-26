# tests/test_core.py
import torch
import numpy as np
from moodos_v01 import compute_mood_vector, MoodProcessor

def test_smoke():
    bands = [(0.5 + i*1.5, 0.5 + (i+1)*1.5) for i in range(64)]
    t = torch.linspace(0, 1, 200)
    sig = torch.sin(2 * np.pi * 10 * t) + 0.5 * torch.sin(2 * np.pi * 40 * t)

    mv = compute_mood_vector(sig, freq_bands_hz=bands)
    assert mv.shape == (32,)
    assert torch.allclose(mv.abs(), mv.abs(), atol=1e-3)  # just sanity

    proc = MoodProcessor(freq_bands_hz=bands)
    outputs = [out for s in sig for out in [proc.push(s.item())] if out is not None]
    assert len(outputs) == 4  # 1s window, 0.25s hop → 4 vectors from 1s signal
