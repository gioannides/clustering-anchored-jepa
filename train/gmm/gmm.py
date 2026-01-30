"""Gradient-based GMM with diagonal covariance."""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .utils import is_main_process, broadcast_tensor


class GradientGMM(nn.Module):
    """
    Gradient-based GMM with diagonal covariance.
    
    Equivalent to sklearn.GaussianMixture(covariance_type='diag').
    
    Args:
        K: Number of mixture components
        dim: Feature dimension
        device: Compute device
    """
    
    def __init__(self, K: int, dim: int, device: str = 'cuda'):
        super().__init__()
        self.K = K
        self.dim = dim
        self.device = device
        
        # Learnable parameters
        self.means = nn.Parameter(torch.randn(K, dim) * 0.1)
        self.log_vars = nn.Parameter(torch.zeros(K, dim))
        self.log_weights = nn.Parameter(torch.zeros(K))
        
        self.to(device)
    
    @property
    def weights(self) -> torch.Tensor:
        """Mixture weights (sum to 1)."""
        return F.softmax(self.log_weights, dim=0)
    
    @property
    def covariances(self) -> torch.Tensor:
        """Diagonal covariances (clamped for stability)."""
        return torch.exp(self.log_vars).clamp(min=1e-1)
    
    def _log_prob_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for a chunk of data."""
        weights = self.weights.to(x.device)
        means = self.means
        vars = self.covariances
        
        log_weights = torch.log(weights + 1e-10)
        diff = x.unsqueeze(1) - means.unsqueeze(0)
        inv_var = 1.0 / vars
        
        # Diagonal Mahalanobis distance
        mahal = ((diff ** 2) * inv_var.unsqueeze(0)).sum(dim=-1)
        log_det = torch.log(vars).sum(dim=-1)
        
        log_prob = -0.5 * (self.dim * math.log(2 * math.pi) + log_det.unsqueeze(0) + mahal)
        log_prob = log_prob + log_weights.unsqueeze(0)
        
        return log_prob
    
    def log_prob(self, x: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Compute log probabilities with chunking for memory efficiency."""
        x = x.to(self.device)
        N = x.shape[0]
        
        if N <= chunk_size:
            return self._log_prob_chunk(x)
        
        log_probs = []
        for i in range(0, N, chunk_size):
            chunk = x[i:i + chunk_size]
            log_probs.append(self._log_prob_chunk(chunk))
        
        return torch.cat(log_probs, dim=0)
    
    def forward(self, x: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Compute negative log-likelihood loss."""
        log_prob = self.log_prob(x, chunk_size)
        log_likelihood = torch.logsumexp(log_prob, dim=1)
        return -log_likelihood.mean()
    
    @torch.no_grad()
    def assign(self, x: torch.Tensor, batch_size: int = 50000) -> torch.Tensor:
        """Hard cluster assignment (argmax)."""
        self.eval()
        x = self._preprocess_input(x)
        N = x.shape[0]
        
        if N <= batch_size:
            log_prob = self.log_prob(x)
            return log_prob.argmax(dim=1)
        
        labels = []
        for i in range(0, N, batch_size):
            chunk = x[i:i + batch_size]
            log_prob = self.log_prob(chunk)
            labels.append(log_prob.argmax(dim=1))
        
        return torch.cat(labels)
    
    @torch.no_grad()
    def soft_assign(self, x: torch.Tensor, batch_size: int = 50000) -> torch.Tensor:
        """Soft cluster assignment (posteriors)."""
        self.eval()
        x = self._preprocess_input(x)
        N = x.shape[0]
        
        if N <= batch_size:
            log_prob = self.log_prob(x)
            return F.softmax(log_prob, dim=1)
        
        probs = []
        for i in range(0, N, batch_size):
            chunk = x[i:i + batch_size]
            log_prob = self.log_prob(chunk)
            probs.append(F.softmax(log_prob, dim=1))
        
        return torch.cat(probs)
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten 3D input to 2D."""
        if x.dim() == 3:
            B, C, T = x.shape
            x = x.permute(0, 2, 1).reshape(-1, C)
        return x.to(self.device).float()
    
    @torch.no_grad()
    def init_from_data(self, data: torch.Tensor, method: str = 'kmeans++'):
        """Initialize GMM parameters from data."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        data = data.to(self.device)
        N = data.shape[0]
        
        if method == 'kmeans++':
            self._init_kmeans_plusplus(data, N)
        else:
            self._init_random(data, N)
        
        # Initialize variances from data variance
        data_var = data.var(dim=0).clamp(min=1e-4)
        self.log_vars.data = torch.log(data_var).unsqueeze(0).expand(self.K, -1).clone()
        self.log_weights.data.zero_()
        
        # Broadcast to all ranks
        broadcast_tensor(self.means.data, src=0)
        broadcast_tensor(self.log_vars.data, src=0)
        broadcast_tensor(self.log_weights.data, src=0)
    
    def _init_kmeans_plusplus(self, data: torch.Tensor, N: int):
        """K-means++ initialization."""
        idx = random.randint(0, N - 1)
        centroids = [data[idx]]
        
        pbar = tqdm(range(1, self.K), desc="K-means++ init", leave=False) if is_main_process() else range(1, self.K)
        for _ in pbar:
            cents = torch.stack(centroids)
            dists = torch.cdist(data, cents)
            min_dists = dists.min(dim=1).values
            probs = min_dists ** 2
            probs = probs / probs.sum()
            
            if N > 2**24:
                probs_np = probs.cpu().numpy()
                idx = np.random.choice(N, p=probs_np)
            else:
                idx = torch.multinomial(probs, 1).item()
            
            centroids.append(data[idx])
        
        self.means.data = torch.stack(centroids)
        if is_main_process():
            print(f"[GMM] Initialized {self.K} means via k-means++")
    
    def _init_random(self, data: torch.Tensor, N: int):
        """Random initialization."""
        indices = torch.randperm(N)[:self.K]
        self.means.data = data[indices]
        if is_main_process():
            print(f"[GMM] Initialized {self.K} means randomly")
    
    def save(self, path: str):
        """Save GMM checkpoint."""
        if not is_main_process():
            return
        
        module = self.module if hasattr(self, 'module') else self
        
        torch.save({
            'means': module.means.data.cpu(),
            'covariances': module.covariances.detach().cpu(),
            'weights': module.weights.detach().cpu(),
            'log_vars': module.log_vars.data.cpu(),
            'log_weights': module.log_weights.data.cpu(),
            'K': module.K,
            'dim': module.dim,
        }, path)
        print(f"[GMM] Saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'GradientGMM':
        """Load GMM from checkpoint."""
        data = torch.load(path, map_location='cpu', weights_only=False)
        gmm = cls(data['K'], data['dim'], device=device)
        
        if 'log_vars' in data:
            gmm.means.data = data['means'].to(device)
            gmm.log_vars.data = data['log_vars'].to(device)
            gmm.log_weights.data = data['log_weights'].to(device)
        else:
            # Legacy format
            gmm.means.data = data['means'].to(device)
            gmm.log_vars.data = torch.log(data['covariances'].clamp(min=1e-6)).to(device)
            gmm.log_weights.data = torch.log(data['weights'].clamp(min=1e-10)).to(device)
        
        return gmm


@torch.no_grad()
def compute_gmm_metrics(
    gmm: GradientGMM,
    data: torch.Tensor,
    prev_means: torch.Tensor = None,
    prev_vars: torch.Tensor = None,
    sample_size: int = 50000
) -> dict:
    """Compute GMM quality metrics."""
    gmm_module = gmm.module if hasattr(gmm, 'module') else gmm
    
    data = data.to(gmm_module.device)
    if data.shape[0] > sample_size:
        indices = torch.randperm(data.shape[0])[:sample_size]
        data = data[indices]
    
    # Assignment metrics
    labels = gmm_module.assign(data)
    unique_labels = torch.unique(labels)
    alive = len(unique_labels)
    counts = torch.bincount(labels, minlength=gmm_module.K).float()
    
    # Cluster balance
    alive_counts = counts[counts > 0]
    balance_cv = (alive_counts.std() / alive_counts.mean()).item() if len(alive_counts) > 1 else 0
    
    # Weight entropy
    weights = gmm_module.weights
    weight_entropy = -(weights * torch.log(weights + 1e-10)).sum().item()
    max_entropy = math.log(gmm_module.K)
    weight_entropy_norm = weight_entropy / max_entropy
    
    # Negative log-likelihood
    nll = gmm_module(data, chunk_size=4096).item()
    
    # Posterior entropy
    posteriors = gmm_module.soft_assign(data)
    posterior_entropy = -(posteriors * torch.log(posteriors + 1e-10)).sum(dim=1).mean().item()
    
    # Parameter change
    mean_delta = 0.0
    var_delta = 0.0
    if prev_means is not None:
        mean_delta = (gmm_module.means - prev_means).abs().mean().item()
        var_delta = (gmm_module.covariances - prev_vars).abs().mean().item()
    
    # Variance stats
    vars = gmm_module.covariances
    
    # Inter-cluster distance
    centroid_dists = torch.cdist(gmm_module.means, gmm_module.means)
    mask = torch.triu(torch.ones_like(centroid_dists), diagonal=1).bool()
    inter_cluster_dist = centroid_dists[mask].mean().item()
    
    # Intra-cluster distance
    diffs = data.unsqueeze(1) - gmm_module.means.unsqueeze(0)
    sq_dists = (diffs ** 2).sum(dim=-1)
    intra_cluster_dist = (posteriors * sq_dists).sum(dim=1).mean().item()
    
    return {
        'alive': alive,
        'balance_cv': balance_cv,
        'weight_entropy': weight_entropy_norm,
        'nll': nll,
        'posterior_entropy': posterior_entropy,
        'mean_delta': mean_delta,
        'var_delta': var_delta,
        'avg_var': vars.mean().item(),
        'min_var': vars.min().item(),
        'max_var': vars.max().item(),
        'inter_cluster_dist': inter_cluster_dist,
        'intra_cluster_dist': intra_cluster_dist,
    }
