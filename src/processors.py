import math
from types import SimpleNamespace
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

# Prefer librosa for mel filterbank (to match HF/OpenAI exactly). Fallback to torchaudio.
# try:
#     import librosa
#     import numpy as np
#     _HAS_LIBROSA = True
# except Exception:
#     _HAS_LIBROSA = False
_HAS_LIBROSA = False

try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False


class TorchWhisperFeatureExtractor:
    """
    GPU-friendly feature extractor that reproduces WhisperFeatureExtractor behavior.

    Usage:
        fe = TorchWhisperFeatureExtractor(device="cuda")
        out = fe([waveform_tensor], sampling_rate=16000, return_tensors="pt", padding=True)
        mel = out.input_features  # (batch, n_mels, frames)
    """
    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        n_fft: int = 400,
        chunk_length: int = 30,  # seconds
        padding_value: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = n_fft
        self.chunk_length = chunk_length
        self.padding_value = padding_value

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # expected frames for a full chunk (30s -> 3000 frames with hop=160, sr=16000)
        self.expected_frames = int(math.ceil(self.chunk_length * self.sampling_rate / float(self.hop_length)))

        # Build mel filter-bank (shape: n_mels x n_freqs)
        self.n_freqs = self.n_fft // 2 + 1
        if _HAS_LIBROSA:
            mel = librosa.filters.mel(
                sr=self.sampling_rate,
                n_fft=self.n_fft,
                n_mels=self.feature_size,
                fmin=0.0,
                fmax=None,
                htk=False,
                norm="slaney",  # matches HF/OpenAI defaults
            ).astype(np.float32)  # shape (n_mels, n_freqs)
            # convert to torch and move to device
            self.mel_filters = torch.from_numpy(mel).to(self.device)
        elif _HAS_TORCHAUDIO:
            # torchaudio.create_fb_matrix returns (n_freqs, n_mels) — we transpose to (n_mels, n_freqs)
            fb = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_freqs,
                f_min=0.0,
                f_max=self.sampling_rate // 2,
                n_mels=self.feature_size,
                sample_rate=self.sampling_rate,
                norm="slaney",   # важно! whisper ожидает именно такую нормализацию
                mel_scale="htk", # whisper использует HTK-шкалу, а не default "slaney"
            )
            # ensure float and device; transpose to (n_mels, n_freqs)
            self.mel_filters = fb.T.to(self.device).to(torch.float32)
        else:
            raise RuntimeError(
                "To use TorchWhisperFeatureExtractor install librosa or torchaudio. "
                "librosa is recommended for exact parity with HF/Whisper."
            )

    def _pad_or_trim_waveform(self, waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
        # waveform: 1D tensor (n_samples,)
        if waveform.dim() > 1:
            # collapse channels to mono (mean)
            waveform = waveform.mean(dim=0)
        n = waveform.shape[-1]
        if n > target_samples:
            waveform = waveform[:target_samples]
        elif n < target_samples:
            pad = target_samples - n
            waveform = F.pad(waveform, (0, pad), "constant", 0.0)
        return waveform

    def _waveform_to_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: 1D torch tensor on self.device (float32), length = arbitrary
        returns: torch tensor shape (n_mels, frames) on self.device (float32)
        """
        # ensure float32 and on device
        waveform = waveform.to(dtype=torch.float32, device=self.device)

        # compute STFT with Hann window (matches librosa)
        window = torch.hann_window(self.win_length, device=self.device, dtype=torch.float32)
        # center=True, pad_mode='reflect' mirrors librosa.stft defaults
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )  # shape: (n_freqs, frames), complex

        # power spectrogram (magnitude^2)
        magnitude = stft.abs() ** 2  # (n_freqs, frames)

        # apply mel filters: mel_filters (n_mels, n_freqs)  @ magnitude (n_freqs, frames) -> (n_mels, frames)
        # using torch.matmul requires proper batching; here single sample
        mel_spec = torch.matmul(self.mel_filters, magnitude)  # (n_mels, frames)

        # log10 with numerical stability
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()  # (n_mels, frames)

        # clamp bottom to (max - 8.0) like in OpenAI/HF code for numerical stability
        max_val = float(log_spec.max().item())
        min_allowed = max_val - 8.0
        log_spec = torch.clamp(log_spec, min=min_allowed)

        # scale exactly like Whisper: (log_spec + 4.0) / 4.0
        log_spec = (log_spec + 4.0) / 4.0

        # Some implementations drop the last frame to match HF shape behavior; we'll fix final frame count below.
        return log_spec

    def __call__(
        self,
        raw_speech: List[Union[torch.Tensor, List[float]]],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = "pt",
        padding: Optional[bool] = True,
    ):
        """
        raw_speech: list of 1D torch tensors (or lists) — each sample may have varying length.
                    Tensors can already be on GPU (device), the extractor will keep them on device.
        sampling_rate: sampling rate of the input audio (should be 16000 for exact match)
        return_tensors: "pt" or "np" (we focus on "pt")
        padding: if True, pad/truncate each waveform to chunk_length (30s) -> fixed frames (3000)
                 if False, compute mel for original lengths and pad to max frames in batch.
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Expected sampling_rate={self.sampling_rate}, got {sampling_rate}. Resample beforehand.")

        # Convert all inputs to torch tensors and move to device
        waveforms: List[torch.Tensor] = []
        for s in raw_speech:
            if isinstance(s, torch.Tensor):
                w = s.to(self.device)
            else:
                # list or numpy array
                w = torch.tensor(s, dtype=torch.float32, device=self.device)
            waveforms.append(w)

        batch_size = len(waveforms)

        if padding:
            # pad/trim all waveforms to chunk_length in samples -> identical lengths (fast batch STFT)
            target_samples = int(self.sampling_rate * self.chunk_length)
            batched = torch.stack([self._pad_or_trim_waveform(w, target_samples) for w in waveforms], dim=0)  # (B, N)
            # batch STFT: torch.stft accepts batch input, returns (B, n_freqs, frames)
            window = torch.hann_window(self.win_length, device=self.device, dtype=torch.float32)
            stft = torch.stft(
                batched,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )  # (B, n_freqs, frames)
            magnitude = stft.abs() ** 2  # (B, n_freqs, frames)

            # mel_filters: (n_mels, n_freqs) ; compute batch mel_spec via einsum
            # result: (B, n_mels, frames)
            mel_spec = torch.einsum("mf,bft->bmt", self.mel_filters, magnitude)

            # log10
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()  # (B, n_mels, frames)

            # per-sample clamp to (max - 8.0)
            max_vals = log_spec.amax(dim=(1, 2), keepdim=True)  # (B,1,1)
            min_allowed = max_vals - 8.0
            log_spec = torch.max(log_spec, min_allowed)

            # scale
            log_spec = (log_spec + 4.0) / 4.0  # (B, n_mels, frames)

            # Fix final time-dimension to expected_frames
            frames = log_spec.shape[-1]
            if frames < self.expected_frames:
                pad = self.expected_frames - frames
                log_spec = F.pad(log_spec, (0, pad), value=self.padding_value)  # pad time dimension on right
            elif frames > self.expected_frames:
                log_spec = log_spec[:, :, : self.expected_frames]

            input_features = log_spec  # (B, n_mels, expected_frames)
        else:
            # variable-length: compute mel per sample then pad to max frames in this batch
            mel_list = []
            for w in waveforms:
                log_spec = self._waveform_to_log_mel(w)  # (n_mels, frames)
                # ensure at most expected_frames (crop)
                if log_spec.shape[-1] > self.expected_frames:
                    log_spec = log_spec[:, : self.expected_frames]
                mel_list.append(log_spec)
            # pad to max frames among mel_list
            max_frames = max(x.shape[-1] for x in mel_list)
            padded = []
            for x in mel_list:
                if x.shape[-1] < max_frames:
                    pad = max_frames - x.shape[-1]
                    x = F.pad(x, (0, pad), value=self.padding_value)
                padded.append(x)
            input_features = torch.stack(padded, dim=0)  # (B, n_mels, max_frames)

        if return_tensors == "pt":
            out = SimpleNamespace(input_features=input_features)
            return out
        else:
            # return numpy arrays on CPU
            return SimpleNamespace(input_features=input_features.detach().cpu().numpy())
