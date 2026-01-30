"""Denoising augmentation (WavLM-style)."""

import random

import torch
import torch.nn.functional as F


@torch.no_grad()
def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix clean signal with noise at specified SNR."""
    clean_energy = (clean.pow(2).sum() / clean.numel()).clamp(min=1e-8)
    noise_energy = (noise.pow(2).sum() / noise.numel()).clamp(min=1e-8)
    
    scale = torch.sqrt(clean_energy / (10 ** (snr_db / 10) * noise_energy))
    return clean + scale * noise


@torch.no_grad()
def mix_utterances(
    wav1: torch.Tensor,
    wav2: torch.Tensor,
    max_overlap_ratio: float = 0.5
) -> torch.Tensor:
    """Mix two utterances with random overlap."""
    L1, L2 = wav1.shape[-1], wav2.shape[-1]
    
    max_mix_len = int(L1 * max_overlap_ratio)
    mix_len = random.randint(1, max(1, max_mix_len))
    
    start1 = random.randint(0, max(1, L1 - mix_len))
    start2 = random.randint(0, max(1, L2 - mix_len)) if L2 > mix_len else 0
    actual_mix_len = min(mix_len, L1 - start1, L2 - start2)
    
    if actual_mix_len <= 0:
        return wav1
    
    region1 = wav1[..., start1:start1 + actual_mix_len]
    region2 = wav2[..., start2:start2 + actual_mix_len]
    
    energy1 = (region1.pow(2).sum() / region1.numel()).clamp(min=1e-8)
    energy2 = (region2.pow(2).sum() / region2.numel()).clamp(min=1e-8)
    
    ratio_db = random.uniform(-5, 5)
    scale = torch.sqrt(energy1 * (10 ** (ratio_db / 10)) / energy2)
    
    mixed = wav1.clone()
    mixed[..., start1:start1 + actual_mix_len] += scale * region2
    
    return mixed


class DenoiseAugmentor:
    """
    Batched denoising augmentation (WavLM-style).
    
    Maintains a buffer of recent utterances for mixing/noise sources.
    
    Args:
        p_noise: Probability of adding noise
        p_mix: Probability of utterance mixing
        snr_range_noise: (min, max) SNR for noise augmentation
        snr_range_speech: (min, max) SNR for speech mixing
        buffer_size: Size of utterance buffer
    """
    
    def __init__(
        self,
        p_noise: float = 0.25,
        p_mix: float = 0.25,
        snr_range_noise: tuple = (-5, 20),
        snr_range_speech: tuple = (-5, 5),
        buffer_size: int = 64
    ):
        self.p_noise = p_noise
        self.p_mix = p_mix
        self.snr_range_noise = snr_range_noise
        self.snr_range_speech = snr_range_speech
        self.buffer_size = buffer_size
        self.utterance_buffer = []
    
    def update_buffer(self, wav_batch: torch.Tensor):
        """Store recent utterances for mixing/noise."""
        for i in range(wav_batch.shape[0]):
            w = wav_batch[i].detach().cpu().clone()
            if len(self.utterance_buffer) >= self.buffer_size:
                self.utterance_buffer.pop(0)
            self.utterance_buffer.append(w)
    
    @torch.no_grad()
    def __call__(self, wav_batch: torch.Tensor) -> tuple:
        """
        Apply augmentation to batch.
        
        Args:
            wav_batch: [B, 1, T] waveform batch
        
        Returns:
            (augmented, clean) tuple
        """
        if len(self.utterance_buffer) < 4:
            self.update_buffer(wav_batch)
            return wav_batch, wav_batch.clone()
        
        B, C, T = wav_batch.shape
        device = wav_batch.device
        augmented = wav_batch.clone()
        
        for b in range(B):
            r = random.random()
            
            if r < self.p_mix:
                # Utterance mixing
                other = random.choice(self.utterance_buffer).to(device)
                if other.dim() == 2:
                    other = other.unsqueeze(0)
                
                if other.shape[-1] < T:
                    other = F.pad(other, (0, T - other.shape[-1]))
                else:
                    start = random.randint(0, other.shape[-1] - T)
                    other = other[..., start:start + T]
                
                augmented[b] = mix_utterances(wav_batch[b], other.squeeze(0))
            
            elif r < self.p_mix + self.p_noise:
                # Add noise
                noise_src = random.choice(self.utterance_buffer).to(device)
                if noise_src.shape[-1] < T:
                    noise_src = F.pad(noise_src, (0, T - noise_src.shape[-1]))
                else:
                    start = random.randint(0, noise_src.shape[-1] - T)
                    noise_src = noise_src[..., start:start + T]
                
                snr = random.uniform(*self.snr_range_noise)
                augmented[b] = mix_at_snr(wav_batch[b], noise_src, snr)
        
        self.update_buffer(wav_batch)
        return augmented, wav_batch
