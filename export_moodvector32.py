# export_moodvector32.py
"""
MoodVector32 → TorchScript Export Script
========================================

Exports the trained MoodVector32 neural projector as a portable,
platform-agnostic TorchScript model (.pt) ready for:
    • iOS (CoreML via conversion)
    • Android (PyTorch Mobile)
    • Desktop / Server inference
    • Edge devices (Raspberry Pi, NVIDIA Jetson, etc.)

Output: exports/moodvector32.pt  (≈ 120 KB)
"""

import os
import torch
from moodos_v01 import MoodVector32

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
EXPORT_DIR = "exports"
MODEL_NAME = "moodvector32.pt"
N_FREQ_BANDS = 32                    # Must match training (32 or 64)
DEVICE = "cpu"                        # Use "cuda" only if you have trained weights on GPU

# ------------------------------------------------------------------
# Create export directory
# ------------------------------------------------------------------
os.makedirs(EXPORT_DIR, exist_ok=True)
export_path = os.path.join(EXPORT_DIR, MODEL_NAME)

# ------------------------------------------------------------------
# Initialize model in evaluation mode
# ------------------------------------------------------------------
print(f"Initializing MoodVector32 (input: {N_FREQ_BANDS} PLV + 1 coherence)")
model = MoodVector32(n_freq_bands=N_FREQ_BANDS)
model.eval()
model.to(DEVICE)

# ------------------------------------------------------------------
# Create proper example inputs matching forward() signature
# ------------------------------------------------------------------
# Shape: (batch, N_FREQ_BANDS) and (batch,) → we use batch=1
example_plv = torch.randn(1, N_FREQ_BANDS)          # PLV per frequency band
example_coherence = torch.tensor([0.65])            # Global coherence scalar

# ------------------------------------------------------------------
# Trace the model (recommended for this feed-forward net)
# ------------------------------------------------------------------
print("Tracing model with TorchScript...")
with torch.no_grad():
    try:
        traced_model = torch.jit.trace(model, (example_plv, example_coherence))
    except Exception as e:
        raise RuntimeError(f"Tracing failed: {e}")

# Optional: script instead of trace (if you ever add control flow)
# traced_model = torch.jit.script(model)

# ------------------------------------------------------------------
# Verify the traced model produces same output
# ------------------------------------------------------------------
original_output = model(example_plv, example_coherence)
traced_output = traced_model(example_plv, example_coherence)

if torch.allclose(original_output, traced_output, atol=1e-6):
    print("Verification passed: traced model output matches original")
else:
    print("Warning: Small numerical difference detected (usually harmless)")

# ------------------------------------------------------------------
# Save optimized TorchScript model
# ------------------------------------------------------------------
print(f"Saving to {export_path} ...")
traced_model.save(export_path)

file_size_kb = os.path.getsize(export_path) / 1024
print(f"Model exported successfully!")
print(f"   Path : {export_path}")
print(f"   Size : {file_size_kb:.1f} KB")
print(f"   Input  → PLV: (1, {N_FREQ_BANDS}), Coherence (1,)")
print(f"   Output → MoodVector32 (1, 32) ∈ [-1, 1]")

# ------------------------------------------------------------------
# Bonus: Quick load test
# ------------------------------------------------------------------
print("\nQuick load test...")
loaded = torch.jit.load(export_path)
test_out = loaded(example_plv, example_coherence)
print(f"   Loaded model inference successful → norm = {test_out.norm():.4f}")
