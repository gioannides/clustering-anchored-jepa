"""GPU-accelerated Mini-Batch K-Means."""

import random

import numpy as np
import torch
from tqdm import tqdm

from .utils import is_main_process, broadcast_tensor


class MiniBatchKMeans:
    """GPU-accelerated Mini-Batch K-Means."""
    
    def __init__(self, K: int, dim: int, device: str = 'cuda'):
        self.K = K
        self.dim = dim
        self.device = device
        
        self.centroids = torch.randn(K, dim, device=device) * 0.1
        self.counts = torch.zeros(K, device=device)
    
    @torch.no_grad()
    def assign(self, x: torch.Tensor, batch_size: int = 50000) -> torch.Tensor:
        """Assign points to nearest centroid."""
        if x.dim() == 3:
            B, C, T = x.shape
            x = x.permute(0, 2, 1).reshape(-1, C)
        
        x = x.to(self.device).float()
        N = x.shape[0]
        
        if N <= batch_size:
            dists = torch.cdist(x, self.centroids)
            return dists.argmin(dim=1)
        
        labels = []
        for i in range(0, N, batch_size):
            chunk = x[i:i + batch_size]
            dists = torch.cdist(chunk, self.centroids)
            labels.append(dists.argmin(dim=1))
        
        return torch.cat(labels)
    
    @torch.no_grad()
    def init_centroids(self, data: torch.Tensor, method: str = 'kmeans++'):
        """Initialize centroids using k-means++ or random."""
        data = data.to(self.device).float()
        N = data.shape[0]
        
        if method == 'kmeans++':
            idx = random.randint(0, N - 1)
            centroids = [data[idx]]
            
            pbar = tqdm(range(1, self.K), desc="K-means++ init", leave=False) if is_main_process() else range(1, self.K)
            for _ in pbar:
                cents = torch.stack(centroids)
                
                min_dists = torch.full((N,), float('inf'), device=self.device)
                chunk_size = 50000
                for i in range(0, N, chunk_size):
                    chunk = data[i:i + chunk_size]
                    dists = torch.cdist(chunk, cents).min(dim=1).values
                    min_dists[i:i + chunk_size] = dists
                
                probs = min_dists ** 2
                probs = probs / probs.sum()
                
                if N > 2**24:
                    probs_np = probs.cpu().numpy()
                    idx = np.random.choice(N, p=probs_np)
                else:
                    idx = torch.multinomial(probs, 1).item()
                
                centroids.append(data[idx])
            
            self.centroids = torch.stack(centroids)
            if is_main_process():
                print(f"[KMeans] Initialized {self.K} centroids via k-means++")
        
        elif method == 'random':
            indices = torch.randperm(N)[:self.K]
            self.centroids = data[indices].clone()
            if is_main_process():
                print(f"[KMeans] Initialized {self.K} centroids randomly")
        
        self.counts.zero_()
        broadcast_tensor(self.centroids, src=0)
    
    @torch.no_grad()
    def partial_fit(self, data: torch.Tensor, learning_rate: float = None):
        """Mini-batch k-means update step."""
        data = data.to(self.device).float()
        labels = self.assign(data)
        
        for k in range(self.K):
            mask = labels == k
            if not mask.any():
                continue
            
            points = data[mask]
            n_new = points.shape[0]
            
            old_count = self.counts[k]
            new_count = old_count + n_new
            
            if learning_rate is not None:
                lr = learning_rate
            else:
                lr = 1.0 / (1.0 + old_count / max(n_new, 1))
            
            new_mean = points.mean(dim=0)
            self.centroids[k] = (1 - lr) * self.centroids[k] + lr * new_mean
            self.counts[k] = new_count
        
        return labels
    
    @torch.no_grad()
    def fit_batch(self, data: torch.Tensor, n_iterations: int = 10):
        """Run multiple k-means iterations on a batch (Lloyd's algorithm)."""
        data = data.to(self.device).float()
        
        for _ in range(n_iterations):
            labels = self.assign(data)
            
            new_centroids = torch.zeros_like(self.centroids)
            counts = torch.zeros(self.K, device=self.device)
            
            for k in range(self.K):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = data[mask].mean(dim=0)
                    counts[k] = mask.sum()
                else:
                    new_centroids[k] = self.centroids[k]
            
            self.centroids = new_centroids
            self.counts = counts
        
        return labels
    
    def save(self, path: str):
        """Save k-means centroids."""
        if not is_main_process():
            return
        
        torch.save({
            'centroids': self.centroids.cpu(),
            'counts': self.counts.cpu(),
            'K': self.K,
            'dim': self.dim,
        }, path)
        print(f"[KMeans] Saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'MiniBatchKMeans':
        """Load k-means centroids."""
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        if 'centroids' in data:
            K, dim = data['centroids'].shape
            kmeans = cls(K, dim, device)
            kmeans.centroids = data['centroids'].to(device)
            kmeans.counts = data.get('counts', torch.zeros(K)).to(device)
        elif 'means' in data:
            # GMM format compatibility
            K, dim = data['means'].shape
            kmeans = cls(K, dim, device)
            kmeans.centroids = data['means'].to(device)
            kmeans.counts = torch.zeros(K, device=device)
        else:
            raise ValueError(f"Unknown format in {path}")
        
        return kmeans


@torch.no_grad()
def compute_kmeans_metrics(kmeans: MiniBatchKMeans, data: torch.Tensor, 
                           prev_centroids: torch.Tensor = None, 
                           sample_size: int = 100000) -> dict:
    """Compute k-means quality metrics."""
    data = data.to(kmeans.device)
    if data.shape[0] > sample_size:
        indices = torch.randperm(data.shape[0])[:sample_size]
        data = data[indices]
    
    labels = kmeans.assign(data)
    unique_labels = torch.unique(labels)
    alive = len(unique_labels)
    counts = torch.bincount(labels, minlength=kmeans.K).float()
    
    alive_counts = counts[counts > 0]
    balance_cv = (alive_counts.std() / alive_counts.mean()).item() if len(alive_counts) > 1 else 0
    
    dists = torch.cdist(data, kmeans.centroids)
    min_dists = dists.min(dim=1).values
    inertia = (min_dists ** 2).mean().item()
    
    centroid_dists = torch.cdist(kmeans.centroids, kmeans.centroids)
    mask = torch.triu(torch.ones_like(centroid_dists), diagonal=1).bool()
    inter_cluster_dist = centroid_dists[mask].mean().item() if mask.any() else 0
    
    centroid_delta = 0.0
    if prev_centroids is not None:
        centroid_delta = (kmeans.centroids - prev_centroids).abs().mean().item()
    
    intra_dists = min_dists
    dists_sorted = dists.sort(dim=1).values
    inter_dists = dists_sorted[:, 1] if dists_sorted.shape[1] > 1 else dists_sorted[:, 0]
    silhouette = ((inter_dists - intra_dists) / torch.maximum(inter_dists, intra_dists).clamp(min=1e-8)).mean().item()
    
    return {
        'alive': alive,
        'balance_cv': balance_cv,
        'inertia': inertia,
        'inter_cluster_dist': inter_cluster_dist,
        'centroid_delta': centroid_delta,
        'silhouette': silhouette,
        'min_cluster_size': int(alive_counts.min().item()) if len(alive_counts) > 0 else 0,
        'max_cluster_size': int(alive_counts.max().item()) if len(alive_counts) > 0 else 0,
    }
