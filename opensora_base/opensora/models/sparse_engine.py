import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable

def do_nothing(x: torch.Tensor, mode: str = None):
    return x

def sparse_tensor(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, 
                                     sim_threshold: float,
                                     sim_mode: str = "cosine",
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    B, N, _ = metric.shape


    gather = torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx
        # metric = metric.to(torch.float32)
        if generator is None:
            generator = torch.default_generator
        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        # L2 distance calculation: Euclidean distance
        if sim_mode == "cosine":
            metric = metric / (metric.norm(dim=-1, keepdim=True)+ 1e-8)
            a, b = split(metric)
            scores = a @ b.transpose(-1, -2)
        elif sim_mode == "l2":
            a, b = split(metric)
            scores = torch.cdist(a, b, p=2) 
        elif sim_mode == "dot":
            a, b = split(metric)
            scores = a @ b.transpose(-1, -2)

        
        # Find the most similar greedily
        if sim_mode == "cosine":
            node_max, node_idx = scores.max(dim=-1)
        elif sim_mode == "l2":
            node_max, node_idx = scores.min(dim=-1)
        elif sim_mode == "dot":
            node_max, node_idx = scores.max(dim=-1)
            
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # Select top-r tokens by similarity scores
        top_scores = gather(node_max[..., None], dim=-2, index=edge_idx).squeeze(-1)
        valid_count = 0
        # Count the number of tokens in the top-r that exceed the threshold
        for i in range(0, top_scores.shape[0], 8):
            valid_count = max(valid_count,(top_scores[0] > sim_threshold).sum(dim=-1).item())
        r = valid_count
        if r <= 0:
            return do_nothing, do_nothing  
        
        unm_idx = edge_idx[..., r:, :]  # Unreduced Tokens
        src_idx = edge_idx[..., :r, :]  # reduced Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        
    def reduce(x: torch.Tensor) -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))

        return torch.cat([unm, dst], dim=1)

    def unreduce(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return reduce, unreduce