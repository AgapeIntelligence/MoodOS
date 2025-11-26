import torch
from typing import Optional, Sequence, Tuple, Union

from moodos_v01 import compute_mood_vector, MoodVector32


class MoodProcessor:
    """
    Real-time streaming mood vector extractor.

    Continuously processes a 200 Hz physiological signal and emits a new
    32-dimensional mood embedding every `hop_sec` seconds (default: 250 ms).

    Features:
        • Zero-copy buffer management
        • Supports CPU/GPU via device argument
        • Handles float, int, np.number, and Tensor inputs
        • Thread-safe if used carefully
        • TorchScript compatible
    """

    def __init__(
        self,
        sample_rate: int = 200,
        window_sec: float = 1.0,
        hop_sec: float = 0.25,
        freq_bands_hz: Optional[Sequence[Tuple[float, float]]] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self.sr = int(sample_rate)
        self.window_samples = int(window_sec * sample_rate)
        self.hop_samples = int(hop_sec * sample_rate)
        self.bands = freq_bands_hz
        self.device = torch.device(device)

        if self.hop_samples <= 0 or self.window_samples < self.hop_samples:
            raise ValueError("window_sec must be >= hop_sec and both > 0")

        n_bands = len(freq_bands_hz) if freq_bands_hz else 1
        self.model = MoodVector32(n_bands).to(self.device).eval()

        self.buffer = torch.zeros(self.window_samples, device=self.device, dtype=torch.float32)
        self.sample_count = 0

    def push(self, sample: Union[float, int, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Push a single new sample into the processor.

        Args:
            sample: New input value (scalar float/int/Tensor)

        Returns:
            32-dim mood vector if a new frame is ready, otherwise None
        """
        # Convert scalar inputs to tensor on correct device
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32, device=self.device)
        elif sample.device != self.device or sample.dtype != torch.float32:
            sample = sample.to(dtype=torch.float32, device=self.device)

        # Circular buffer update
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=0)
        self.buffer[-1] = sample
        self.sample_count += 1

        # Only compute when we've advanced by one hop
        if self.sample_count % self.hop_samples != 0:
            return None

        # Extract current window and compute mood vector
        window = self.buffer.clone().unsqueeze(0)  # Shape: (1, window_samples)
        with torch.no_grad():
            mood_vec = compute_mood_vector(
                raw_signal=window,
                sample_rate=self.sr,
                freq_bands_hz=self.bands,
                model=self.model,
            )
        return mood_vec.squeeze(0)  # Shape: (32,)

    def reset(self) -> None:
        """Reset internal buffer and counter (e.g., on stream restart)."""
        self.buffer.zero_()
        self.sample_count = 0

    def get_current_buffer(self) -> torch.Tensor:
        """Return current analysis window (for debugging/visualization)."""
        return self.buffer.clone()
