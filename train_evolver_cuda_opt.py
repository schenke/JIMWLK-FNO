#!/usr/bin/env python3
from __future__ import annotations
import os, re, json, math, argparse, random
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import numpy as np
import collections
import torch
import time
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import (
    SequentialLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts,
    StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, LambdaLR
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler as _GradScaler
from torch.cuda.amp import GradScaler as _CudaGradScaler
from contextlib import nullcontext
from torch.nn.parameter import UninitializedParameter

# Learn (U0, Y, params) -> U_Y from JIMWLK outputs + manifest.json
# - Lattice size N inferred from data (no --size)
# - 'ds' inferred from manifest.json (no --ds)
# - DDP + AMP ready; GroupNorm (batch=1 friendly)

# --- cache for radial binning (wrapped torus distances) ---
_RADIAL_CACHE = {}  # key: (H, W, device) -> (bins[H,W] long, counts[L] long)


import torch, torch.distributed as dist

def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def dipole_from_links18(links18: torch.Tensor,
                        offsets=None) -> torch.Tensor:
    """
    Compute a compact dipole summary N(r) for a set of lattice separations.

    Args:
        links18: (B, C>=18, H, W) with first 18 channels encoding a 3x3 complex matrix:
                 channels [0..8] = real parts row-major, [9..17] = imag parts row-major.
        offsets: list of (dy, dx) lattice displacements. If None, a small default set is used.

    Returns:
        N: (B, K) tensor, where K = len(offsets), each entry is the spatial average of
           N_r = 1 - (1/Nc) Re tr[ U(x) U^\dagger(x+r) ] over the lattice.
    """
    x = links18[:, :18]  # (B, 18, H, W)
    B, _, H, W = x.shape

    # Unpack 18 channels -> complex 3x3 matrix per site
    real = x[:, 0:9].reshape(B, 9, H, W).permute(0, 2, 3, 1).reshape(B, H, W, 3, 3)
    imag = x[:, 9:18].reshape(B, 9, H, W).permute(0, 2, 3, 1).reshape(B, H, W, 3, 3)
    U = torch.complex(real, imag)  # (B, H, W, 3, 3), complex64/complex128 depending on inputs

    if offsets is None:
        # A compact, inexpensive set of radii (tweak as you like).
        # Includes cardinal and diagonal directions at a few distances.
        offsets = [(0,1),(1,0),(1,1),(0,2),(2,0),(2,1),(1,2),(2,2),
                   (0,4),(4,0),(3,4),(4,3)]

    Nc = 3.0
    out = []
    for (dy, dx) in offsets:
        # periodic BC via roll; negative shifts also work if you include them
        U_shift = torch.roll(U, shifts=(dy, dx), dims=(1, 2))  # (B,H,W,3,3)
        # U(x) U^\dagger(x+r)
        prod = U @ U_shift.conj().transpose(-1, -2)            # (B,H,W,3,3)
        # (1/Nc) Re tr[prod]
        tr_re = prod.diagonal(dim1=-2, dim2=-1).sum(-1).real / Nc   # (B,H,W)
        S = tr_re.mean(dim=(1, 2))                                  # (B,)
        N = 1.0 - S
        out.append(N)

    return torch.stack(out, dim=1)  # (B, K)

def _q4_scalar(U: torch.Tensor, dx1: int, dy1: int, dx2: int, dy2: int) -> torch.Tensor:
    """
    U: [B, H, W, 3, 3] complex
    returns: [B, H, W] real (Tr(...) / 3)
    """
    Ux = U
    Uy = torch.roll(U, (dy1,        dx1),        dims=(1, 2))
    Uu = torch.roll(U, (dy2,        dx2),        dims=(1, 2))
    Uv = torch.roll(U, (dy1 + dy2,  dx1 + dx2),  dims=(1, 2))
    # Tr(Ux Uy^† Uu Uv^†) = sum_{i j k l} Ux_{ij} * conj(Uy_{kj}) * Uu_{kl} * conj(Uv_{il})
    tr = torch.einsum('...ij,...kj,...kl,...il->...', Ux, Uy.conj(), Uu, Uv.conj())  # [B,H,W] complex
    return tr.real / 3.0


def haar_su3(device=None, dtype=torch.complex64):
    # Complex Ginibre matrix
    X = torch.randn(3, 3, device=device, dtype=dtype) + 1j*torch.randn(3, 3, device=device, dtype=dtype)
    # QR gives unitary Q up to phases
    Q, R = torch.linalg.qr(X)
    # Make R's diagonal real/positive by absorbing phases into Q's columns
    diag = torch.diagonal(R)
    phases = diag / diag.abs().clamp_min(1e-12)
    Q = Q @ torch.diag(phases.conj())
    # Project to SU(3): det(Q) = 1
    detQ = torch.det(Q)
    Q = Q * torch.exp(-1j * torch.angle(detQ) / 3.0)
    return Q  # (3,3) complex, unitary, det=1

def conjugate_global(U, Omega):
    # U: (..., 3, 3) complex; Omega: (3,3) complex
    # Returns Omega @ U @ Omega^†
    return torch.einsum('ab,...bc,cd->...ad', Omega, U, Omega.conj().transpose(-1, -2))


@torch.no_grad()
def snapshot_params(module, buf_list=None, dtype=torch.float32):
    """
    Make/refresh a device-side snapshot of model params for delta computation.
    Returns a list of tensors with same shapes/devices as module.parameters().
    """
    params = [p.detach() for p in module.parameters() if p.requires_grad]
    if buf_list is None:
        # first time: allocate clones
        return [p.to(dtype).clone() for p in params]
    # refresh in-place (no reallocation)
    for buf, p in zip(buf_list, params):
        buf.copy_(p.to(dtype))
    return buf_list

@torch.no_grad()
def delta_w_ratio(module, prev_params, reduce_across_ranks=False, eps=1e-12):
    """
    Compute ||Δw||/||w|| where Δw = (w_now - w_prev), streaming on device.
    - module: model or unwrap(model)
    - prev_params: list returned by snapshot_params at the *previous* check
    - reduce_across_ranks: set True only if you’re sharding params (FSDP/TP).
                           For standard DDP (replicated params), leave False.
    Returns: (ratio_tensor_on_device, new_snapshot_list)
    """
    params_now = [p.detach() for p in module.parameters() if p.requires_grad]
    assert len(params_now) == len(prev_params), "snapshot doesn't match parameter list"

    num = torch.zeros((), device=params_now[0].device, dtype=torch.float32)  # ||Δw||^2
    den = torch.zeros((), device=params_now[0].device, dtype=torch.float32)  # ||w||^2

    for p_now, p_prev in zip(params_now, prev_params):
        # do math in fp32 for stability even if weights are bf16/fp16
        p32 = p_now.to(torch.float32)
        d32 = (p32 - p_prev)
        num += (d32 * d32).sum()
        den += (p32 * p32).sum()

    if reduce_across_ranks and dist.is_available() and dist.is_initialized():
        dist.all_reduce(num, op=dist.ReduceOp.SUM)
        dist.all_reduce(den, op=dist.ReduceOp.SUM)

    ratio = num.sqrt() / den.sqrt().clamp_min(eps)

    # prepare the next snapshot (after using ratio)
    new_prev = snapshot_params(module, prev_params)

    return ratio, new_prev


@torch.no_grad()
def _dist_mean_scalar(t: torch.Tensor) -> torch.Tensor:
    # t: 0-dim or 1-dim tensor on device
    if dist.is_available() and dist.is_initialized():
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t

@torch.no_grad()
def aligned_overlap_r0(U_true: torch.Tensor, U_pred: torch.Tensor) -> torch.Tensor:
    """
    Cheap, gauge-independent scalar in [0,1].
    U_*: [B,H,W,3,3] complex64/128 on device.
    """
    C = (U_true.conj().transpose(-1, -2) @ U_pred).mean(dim=(1, 2))  # [B,3,3]
    # 3x3 SVD per sample; use float32/complex64 for speed
    U, S, Vh = torch.linalg.svd(C)  # S: [B,3], nonnegative real
    overlap = (S.sum(dim=-1) / 3.0).mean()  # average over batch -> scalar tensor
    return overlap

def softplus_inverse(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # invert: softplus(z) = x  =>  z = softplus^{-1}(x)
    x = x.clamp_min(eps)
    return x + torch.log(-torch.expm1(-x))  # stable for x>0


def _ensure18(arr: np.ndarray, N: int) -> np.ndarray:
    """
    Normalize a Wilson-line array to float32 [N, N, 18] (9 real + 9 imag).
    Accepts:
      - complex [N, N, 3, 3]
      - float [N, N, 18]
      - float [N, N, 3, 3, 2] with last axis (Re, Im)
      - float [N, N, 9]  (assume real-only; pad imag=0)
    """
    a = np.asarray(arr)
    # Already 18 channels?
    if a.ndim == 3 and a.shape == (N, N, 18):
        return a.astype(np.float32, copy=False)

    # Complex [N,N,3,3]
    if a.ndim == 4 and a.shape[-2:] == (3, 3) and np.iscomplexobj(a):
        real = a.real.reshape(N, N, 9)
        imag = a.imag.reshape(N, N, 9)
        return np.concatenate([real, imag], axis=-1).astype(np.float32, copy=False)

    # Packed real/imag as last dim 2: [N,N,3,3,2]
    if a.ndim == 5 and a.shape[-3:-1] == (3, 3) and a.shape[-1] == 2:
        real = a[..., 0].reshape(N, N, 9)
        imag = a[..., 1].reshape(N, N, 9)
        return np.concatenate([real, imag], axis=-1).astype(np.float32, copy=False)

    # Real-only 9 channels -> pad imag with zeros
    if a.ndim == 3 and a.shape == (N, N, 9):
        zeros = np.zeros_like(a, dtype=np.float32)
        return np.concatenate([a.astype(np.float32, copy=False), zeros], axis=-1)

    raise ValueError(f"Unexpected Wilson-line array shape {a.shape}; "
                     f"expected (N,N,3,3) complex or (N,N,18) float.")


class DevMeter:
    """Device-side running mean (no host sync)."""
    def __init__(self, device, dtype=torch.float32):
        self.sum = torch.zeros((), device=device, dtype=dtype)
        self.n   = torch.zeros((), device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, x: torch.Tensor, k: int = 1):
        # x is a scalar tensor on device
        self.sum += x.detach().to(self.sum.dtype)
        self.n   += k

    @torch.no_grad()
    def mean(self) -> torch.Tensor:
        return self.sum / self.n.clamp_min(1)

def _pack_and_tolist(*scalars: torch.Tensor):
    """Batch a few scalar tensors into one tiny transfer (single sync)."""
    if not scalars:
        return []
    stk = torch.stack([s.detach() for s in scalars])
    return stk.cpu().tolist()

def _avgpool_circular(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    pad = k // 2
    return F.avg_pool2d(F.pad(x, (pad,pad,pad,pad), mode='circular'),
                        kernel_size=k, stride=1)


def compute_curriculum_knobs(epoch: int, args, device):
    """Compute per-epoch curriculum values; always returns defined knobs."""
    # Targets
    rollout_k_tgt = int(getattr(args, "rollout_k", 0))
    cons_tgt      = float(getattr(args, "rollout_consistency", 0.0) or 0.0)
    sg_tgt        = float(getattr(args, "semigroup_weight", 0.0) or 0.0)

    # Warmups
    rk_wu = int(getattr(args, "rollout_warmup_epochs", 0) or 0)
    cw_wu = int(getattr(args, "cons_warmup_epochs", 0) or 0)
    sg_wu = int(getattr(args, "sg_warmup_epochs", 0) or 0)

    def ramp(t, T):
        if T is None or T <= 0:
            return 1.0
        return max(0.0, min(1.0, (epoch) / float(T)))

    # rollout k schedule (start from 1 if target>=1; allow 0 to disable)
    if rollout_k_tgt <= 0:
        rollout_k_curr = 0
    else:
        r = ramp(epoch, rk_wu)
        rollout_k_curr = max(1, int(round(1 + r * (rollout_k_tgt - 1))))

    # weights schedule
    cons_w = cons_tgt * ramp(epoch, cw_wu)
    sg_w   = sg_tgt   * ramp(epoch, sg_wu)

    # Optionally skip heavy penalties on CPU if such a flag exists and is True
    skip_heavy = bool(getattr(args, "skip_heavy_on_cpu", True))
    if device.type == "cpu" and skip_heavy:
        cons_w = 0.0
        sg_w   = 0.0

    return rollout_k_curr, float(cons_w), float(sg_w)

# ========= Differentiable radial binning & Qs helpers =========
def _scatter_mean_1d(src: 'torch.Tensor', index: 'torch.LongTensor', n_bins: int) -> 'torch.Tensor':
    """
    Differentiable mean over bins along a flat vector.
    src:   [N] float tensor (requires grad ok)
    index: [N] long tensor with values in [0, n_bins-1] (no grad needed)
    returns: [n_bins] mean per bin (zeros where count==0)
    """
    device = src.device
    sums = torch.zeros(n_bins, device=device, dtype=src.dtype)
    counts = torch.zeros(n_bins, device=device, dtype=src.dtype)
    sums = sums.scatter_add(0, index, src)
    ones = torch.ones_like(src, dtype=src.dtype)
    counts = counts.scatter_add(0, index, ones)
    # avoid div by zero
    out = torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))
    return out

def radial_profile_mean(values: 'torch.Tensor', bin_index: 'torch.LongTensor', n_bins: int) -> 'torch.Tensor':
    """
    values: [..., N] last-dim are samples to bin; returns [..., n_bins]
    bin_index: [N] int bin per sample (constant wrt model)
    """
    flat = values.reshape(-1, values.shape[-1])
    outs = []
    for row in flat:
        outs.append(_scatter_mean_1d(row, bin_index, n_bins))
    return torch.stack(outs, dim=0).reshape(*values.shape[:-1], n_bins)

def soft_qs_from_curve(r_centers: 'torch.Tensor', S_r: 'torch.Tensor', tau: float = float(torch.exp(torch.tensor(-0.25)))) -> 'torch.Tensor':
    """
    Smooth estimator of saturation scale Qs using soft-argmin around S(r)=tau.
    r_centers: [B, R] or [R]
    S_r:       [B, R] or [R]
    returns Qs: [B] (or scalar) where Qs ~ sqrt(2)/r*, with r* ≈ sum w_i r_i
    """
    if r_centers.ndim == 1:
        r = r_centers[None, :]
        S = S_r[None, :]
        squeeze = True
    else:
        r = r_centers
        S = S_r
        squeeze = False
    # temperature for softness
    sigma = 0.03  # tuneable
    # weights peak where S~tau
    w = torch.softmax(-torch.abs(S - tau) / sigma, dim=-1)
    r_hat = (w * r).sum(dim=-1)
    Qs = (2.0 ** 0.5) / r_hat.clamp_min(1e-6)
    return Qs.squeeze(0) if squeeze else Qs

# ---- Complex packing helpers ----
CHANNEL_LAYOUT = "split"

def pack_to_complex(v18: 'torch.Tensor') -> 'torch.Tensor':
    """v18 [...,18] -> [...,3,3] complex"""
    v18 = v18.contiguous()
    if CHANNEL_LAYOUT == "pair":
        v = v18.view(*v18.shape[:-1], 3, 3, 2)
        real, imag = v[..., 0], v[..., 1]
    else:
        real = v18[..., :9].view(*v18.shape[:-1], 3, 3)
        imag = v18[...,  9:].view(*v18.shape[:-1], 3, 3)
    return torch.complex(real.float(), imag.float())

def unpack_to_18(U: 'torch.Tensor') -> 'torch.Tensor':
    """U [...,3,3] complex -> [...,18] real"""
    real, imag = U.real, U.imag
    if CHANNEL_LAYOUT == "pair":
        out = torch.stack([real, imag], dim=-1).view(*U.shape[:-2], 18)
    else:
        out = torch.cat([real.reshape(*U.shape[:-2], 9),
                         imag.reshape(*U.shape[:-2], 9)], dim=-1)
    return out


# ------------------------- EMA helper -------------------------
class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        # only track trainable params
        self.shadow = {k: p.detach().clone()
                       for k, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[k].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self, model):
        """Swap EMA weights into live model (backup raw)."""
        self._backup = {}
        for k, p in model.named_parameters():
            if p.requires_grad:
                self._backup[k] = p.data.clone()
                p.data.copy_(self.shadow[k])

    @torch.no_grad()
    def swap_out(self, model):
        """Restore raw weights after eval."""
        for k, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self._backup[k])
        self._backup = None

def unwrap(model):
    return model.module if isinstance(model, DDP) else model
        
# === SU(3) algebra helpers (use your existing self.lambdas = λ/2 basis) ===
def _alpha_to_S(alpha: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """
    alpha: [..., C] with C=8    lambdas: [C,3,3] (Hermitian generators λ/2, complex)
    Returns Hermitian S = ∑_a alpha_a T_a, T_a = λ_a/2.
    Dtype/device are aligned to lambdas.
    """
    lams = lambdas
    a = alpha.to(dtype=lams.dtype, device=lams.device)
    return torch.einsum('...a,aji->...ij', a, lams)

def _S_to_alpha(S: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """
    Project Hermitian S back to α_a = 2 Re tr(S T_a).
    S: [...,3,3] ; lambdas: [C,3,3]
    Returns [..., C]
    """
    # match lambdas to S
    lams = lambdas.to(dtype=S.dtype, device=S.device)
    # 2 Re tr(S T_a)  (einsum handles the trace over ij)
    return 2.0 * torch.real(torch.einsum('...ij,aji->...a', S, lams))

def _bch_alpha(alpha1: torch.Tensor, alpha2: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """
    First-order BCH in S-space (Hermitian):
      L = log(U) Anti-Hermitian, S = i L Hermitian.
      BCH on L: L = L1 + L2 + 1/2 [L1,L2] + ...
      Translate to S: S = S1 + S2 + (i/2)[S1,S2] + ...
    Return α(S) up to the commutator term.
    alpha1, alpha2: [B,C]
    """
    # ensure dtype/device alignment inside helpers
    S1 = _alpha_to_S(alpha1, lambdas)        # [B,3,3], complex
    S2 = _alpha_to_S(alpha2, lambdas)        # [B,3,3], complex
    comm = S1 @ S2 - S2 @ S1                 # anti-Hermitian
    S_bch = S1 + S2 + 0.5j * comm            # Hermitian
    return _S_to_alpha(S_bch, lambdas)       # [B,C]

def _pairwise_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x:[N,d], y:[M,d] -> [N,M] Euclidean distances (supports autograd in x)."""
    # torch.cdist is fine here; if unavailable, expand and norm.
    return torch.cdist(x, y, p=2)

def _energy_distance_alpha(alpha_true: torch.Tensor,
                           alpha_pred: torch.Tensor) -> torch.Tensor:
    """
    Energy distance between α distributions (flattened samples).
    alpha_true, alpha_pred: [N,C] and [M,C].
    Returns scalar loss (larger -> worse).
    """
    # 2E||X-Y|| - E||X-X'|| - E||Y-Y'||
    d_xy = _pairwise_l2(alpha_true, alpha_pred)
    d_xx = _pairwise_l2(alpha_true, alpha_true)
    d_yy = _pairwise_l2(alpha_pred, alpha_pred)
    # exclude zero-diagonal in self distances to reduce bias
    n = max(alpha_true.shape[0], 1)
    m = max(alpha_pred.shape[0], 1)
    exy = d_xy.mean()
    exx = (d_xx.sum() - torch.diagonal(d_xx, 0).sum()) / max(n*(n-1), 1)
    eyy = (d_yy.sum() - torch.diagonal(d_yy, 0).sum()) / max(m*(m-1), 1)
    return 2.0*exy - exx - eyy


def _safe_unscale_and_grad_norm(optimizer, scaler=None, model=None):
    try:
        if scaler is not None:
            scaler.unscale_(optimizer)
    except Exception:
        pass
    if model is None:
        return 0.0
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad.detach() ** 2).sum()
    return float(total.sqrt().detach().cpu().item())  # one sync when you actually need a float

def _grad_to_weight_ratio(model):
    device = next(model.parameters()).device
    gn2 = torch.zeros((), device=device)
    wn2 = torch.zeros((), device=device)
    for p in model.parameters():
        if not p.requires_grad:
            continue
        wn2 += (p.data.float() ** 2).sum()
        if p.grad is not None:
            gn2 += (p.grad.detach().float() ** 2).sum()
    if wn2 == 0:
        return 0.0
    ratio = (gn2.sqrt() / wn2.sqrt().clamp_min(1e-30))
    return float(ratio.detach().cpu().item())  # one sync


class EpochMeters:
    def __init__(self, device=None, dtype=torch.float32):
        self._sum = {}
        self._count = {}
        self.device = torch.device("cpu") if device is None else device
        self.dtype = dtype

    def add(self, name: str, value, n: int = 1):
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=self.device, dtype=self.dtype)
        else:
            value = value.detach().to(self.device, self.dtype)
        self._sum[name]   = self._sum.get(name, torch.zeros((), device=self.device, dtype=self.dtype)) + value * int(n)
        self._count[name] = self._count.get(name, torch.zeros((), device=self.device, dtype=torch.int64)) + int(n)

    def mean(self, name: str, default=float("nan")):
        s = self._sum.get(name); c = self._count.get(name)
        if s is None or c is None or int(c.item()) == 0: 
            return default
        return (s / c)  # <- returns a tensor; convert to float only when printing

    def reset(self):
        self._sum.clear(); self._count.clear()
        
# ------------------------- IO: read Wilson lines -------------------------

def read_wilson_binary(path: Path, size: int | None = None) -> np.ndarray:
    """
    Read IP-Glasma/JIMWLK 'method 2' binary: 18 doubles/site (Re/Im for 9 entries).
    Supports a small header (0,2,4,8,16,32,64 doubles). Returns complex128 [N,N,3,3].
    If size is None, infer N (and header) from file length.
    If size is given, prefer a header that makes the payload exactly 18*N*N.
    """
    a = np.fromfile(path, dtype=np.float64)
    tried_hdrs = (0, 2, 4, 8, 16, 32, 64)

    def infer_from_any_header():
        for h in tried_hdrs:
            rem = a.size - h
            if rem > 0 and rem % 18 == 0:
                N2 = rem // 18
                N = int(round(math.sqrt(N2)))
                if N * N == N2:
                    return N, h
        return None, None

    if size is None:
        N, hdr = infer_from_any_header()
        if N is None:
            raise ValueError(f"{path}: cannot infer lattice size from {a.size} doubles (tried headers {tried_hdrs}).")
    else:
        # Choose a header that yields exactly 18*N*N payload
        hdr = None
        for h in tried_hdrs:
            rem = a.size - h
            if rem == 18 * size * size:
                hdr = h; N = size; break
        if hdr is None:
            raise ValueError(f"{path}: cannot match requested size {size}; total doubles={a.size}")
    payload = a[hdr: hdr + 18 * N * N]
    M = payload.reshape(N * N, 9, 2)
    U = (M[..., 0] + 1j * M[..., 1]).reshape(N, N, 3, 3)
    return U

def steps_to_Y(steps: int, ds: float) -> float:
    # rcJIMWLK Langevin mstochaapping
    return (math.pi**2) * ds * steps


# ------------------------- Dataset -------------------------

def _to_path(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()

class JimwlkEvolverDataset(Dataset):
    """
    Layout:
      root/run_00000/, run_00001/, ...
        manifest.json:
          {
            "ds": 0.0004,
            "params": {"runningCoupling":0,..., "m_GeV":0.2, "Lambda_QCD_GeV":0.09, "mu0_GeV":0.28, "seed":123},
            "output_dir": "evolved_wilson_lines",
            "snapshots": [
              {"steps": 0,  "path": "evolved_wilson_lines/U_steps_0"},
              {"steps": 20, "path": "evolved_wilson_lines/U_steps_20"},
              ...
            ]
          }
    """
    def __init__(self, root: Path,
                 split: str = "train", split_frac: float = 0.9, seed: int = 0,
                 cache_initial: bool = True, gauge_aug_p=0.5, split_by: str = "run",
                 ):
        super().__init__()
        self.root = Path(root)
        self.N = None
        self.gauge_aug_p = gauge_aug_p
        self.split = split
        self.cache_initial = bool(cache_initial)

        rng = random.Random(seed)

        # Gather all runs that have a manifest
        all_runs = sorted([p for p in self.root.glob("run_*") if (p / "manifest.json").exists()])

        # Infer ds across runs (must be consistent)
        ds_vals = []
        for rd in all_runs:
            try:
                man = json.loads((rd / "manifest.json").read_text())
                if "ds" in man:
                    ds_vals.append(float(man["ds"]))
            except Exception:
                pass
        if not ds_vals:
            raise ValueError("No 'ds' found in any manifest.json under the data root.")
        if abs(max(ds_vals) - min(ds_vals)) > 1e-12:
            raise ValueError(f"Inconsistent ds across runs: {sorted(set(ds_vals))}")
        self.ds = ds_vals[0]

        # Split by run (so val runs are unseen unless split_by != "run")
        if split_by == "run":
            rng.shuffle(all_runs)
            cut = int(round(len(all_runs) * split_frac))
            self.runs = all_runs[:cut] if split == "train" else all_runs[cut:]
        else:
            groups = defaultdict(list)
            for rd in all_runs:
                try:
                    man = json.loads((rd / "manifest.json").read_text())
                except Exception:
                    continue
                params = man.get("params", {})
                key = (
                    float(params.get("m_GeV") or params.get("m") or 0.0),
                    float(params.get("Lambda_QCD_GeV") or params.get("Lambda_QCD") or 0.0),
                    float(params.get("mu0_GeV") or params.get("mu0") or 0.0),
                    int(params.get("runningCoupling") or params.get("rc") or 0),
                )
                groups[key].append(rd)
            keys = list(groups.keys())
            rng.shuffle(keys)
            cut = int(round(len(keys) * split_frac))
            chosen = keys[:cut] if split == "train" else keys[cut:]
            self.runs = [rd for k in chosen for rd in groups[k]]

        # Per-run snapshots sorted by Y and per-run params
        self.snapshots_by_run: List[List[Dict[str, Any]]] = []
        self.params_by_run: List[Dict[str, float]] = []
        self.anchor_idx_by_run: List[int] = []
        self.anchor_cache_by_run: List[Optional[np.ndarray]] = []

        # Build run structures + infer N from each run's anchor
        for run_idx, rd in enumerate(self.runs):
            man = json.loads((rd / "manifest.json").read_text())
            params = man.get("params", {})
            snaps = man.get("snapshots", [])

            if not snaps:
                # Fallback: list files by pattern if manifest lacks 'snapshots'
                out_dir = _to_path(man.get("output_dir", "evolved_wilson_lines"), rd)
                for p in sorted(out_dir.glob("U_steps_*")):
                    m = re.search(r"U_steps_(\d+)$", p.name)
                    if m:
                        snaps.append({"steps": int(m.group(1)), "path": str(p.relative_to(rd))})

            # Materialize snapshot list with Y and absolute path
            run_snaps: List[Dict[str, Any]] = []
            for s in snaps:
                steps = int(s["steps"])
                Y = steps_to_Y(steps, self.ds)
                path = _to_path(s["path"], rd)
                run_snaps.append({"steps": steps, "Y": float(Y), "path": path})

            # Sort by Y and choose anchor = argmin Y (usually Y==0)
            run_snaps.sort(key=lambda s: s["Y"])
            if not run_snaps:
                continue
            a0 = 0  # after sort, min-Y is index 0
            Ya = run_snaps[a0]["Y"]
            if abs(Ya) > 1e-6:
                # Not fatal: we still anchor at min Y
                print(f"[warn] run {rd.name}: smallest Y = {Ya:.6g} (not exactly 0) — anchoring here.")

            # Read anchor once for N inference and optional cache
            U_anchor = read_wilson_binary(run_snaps[a0]["path"], size=None)
            U_anchor = np.array(U_anchor, copy=True)
            N_here = U_anchor.shape[0]
            if self.N is None:
                self.N = N_here
            elif N_here != self.N:
                raise ValueError(f"Mixed lattice sizes detected: saw N={self.N} and N={N_here}")

            base18 = np.concatenate(
                [U_anchor.real.reshape(N_here, N_here, 9), U_anchor.imag.reshape(N_here, N_here, 9)],
                axis=-1
            ).astype(np.float32, copy=False) if self.cache_initial else None
            del U_anchor
            
            self.snapshots_by_run.append(run_snaps)
            self.params_by_run.append({
                "m": float(params.get("m_GeV") or params.get("m") or 0.0),
                "Lambda_QCD": float(params.get("Lambda_QCD_GeV") or params.get("Lambda_QCD") or 0.0),
                "mu0": float(params.get("mu0_GeV") or params.get("mu0") or 0.0),
            })
            self.anchor_idx_by_run.append(a0)
            self.anchor_cache_by_run.append(base18 if self.cache_initial else None)
            self._anchor_seen = set()     
#            self.anchor_cache_by_run.append(base18)

        if self.N is None:
            raise ValueError("No valid runs/snapshots found.")

        # Build entries:
        #     * TRAIN: we keep lightweight entries (one per run) and sample b>0 on the fly (same-run, anchored).
        #     * VAL:   build a deterministic list of pairs (anchor -> each Y_b>Ya) for stable metrics.
        self.entries: List[Dict[str, Any]] = []

        
        if self.split == "val":
            # Deterministic list of pairs: anchor -> every later snapshot
             for run_idx, run_snaps in enumerate(self.snapshots_by_run):
                    a0 = self.anchor_idx_by_run[run_idx]
                    Ya = run_snaps[a0]["Y"]
                    for b in range(len(run_snaps)):
                        Yb = run_snaps[b]["Y"]
                        if Yb < Ya:
                            continue
                        if b == a0:
                            continue
                         
                        # if b == a0:  # <-- exclude Y=Ya if desired
                        #     continue
                        self.entries.append({
                            "mode": "pair",
                            "run_idx": run_idx,
                            "a_idx": a0,
                            "b_idx": b,
                            "Ya": float(Ya),
                            "Yb": float(Yb),
                            "Y":  float(Yb),  # kept for any downstream logging that expects 'Y'
                            **self.params_by_run[run_idx],
                        })
        else:
            # TRAIN: one lightweight entry per run; __getitem__ will sample b>0 on the fly
            for run_idx, run_snaps in enumerate(self.snapshots_by_run):
                a0 = self.anchor_idx_by_run[run_idx]
                Ya = run_snaps[a0]["Y"]
                for b in range(len(run_snaps)):
                    Yb = run_snaps[b]["Y"]
                    if Yb < Ya:
                        continue
                    if b == a0:
                        continue

                    self.entries.append({
                        "mode": "pair",
                        "run_idx": run_idx,
                        "a_idx": a0,
                        "b_idx": b,
                        "Ya": float(Ya),
                        "Yb": float(Yb),
                        "Y":  float(Yb),  # kept for any downstream logging that expects 'Y'
                        **self.params_by_run[run_idx],
                    })
                
        assert self.entries, "Dataset entries are empty."
        print(f"[{self.split}] using {len(self.runs)} runs, inferred lattice size N={self.N}, ds={self.ds}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        import numpy as np

        # # --- tiny, local helpers (you can hoist these to module scope for speed) ---
        # def ch18_to_su3_np(arr18):
        #     # arr18: [H, W, 18] float32  ->  U: [H, W, 3, 3] complex64
        #     R = arr18[..., :9].reshape(*arr18.shape[:2], 3, 3)
        #     I = arr18[...,  9:].reshape(*arr18.shape[:2], 3, 3)
        #     return R.astype(np.float32) + 1j * I.astype(np.float32)

        # def su3_to_18_np(U):
        #     # U: [H, W, 3, 3] complex64  ->  [H, W, 18] float32
        #     R = U.real.reshape(*U.shape[:2], 9)
        #     I = U.imag.reshape(*U.shape[:2], 9)
        #     return np.concatenate([R, I], axis=-1).astype(np.float32)

        # def haar_su3_np():
        #     # Mezzadri method: QR of complex Ginibre -> rephase -> det=1
        #     X  = (np.random.randn(3,3) + 1j*np.random.randn(3,3)).astype(np.complex64)
        #     Q, R = np.linalg.qr(X)
        #     # make diag(R) real/positive by absorbing phases into Q
        #     d = np.diag(R)
        #     phases = d / np.clip(np.abs(d), 1e-12, None)
        #     Q = Q @ np.diag(np.conj(phases))
        #     # project to SU(3)
        #     detQ = np.linalg.det(Q)
        #     Q = Q * np.exp(-1j * np.angle(detQ) / 3.0)
        #     return Q.astype(np.complex64)  # (3,3)

        # def conjugate_global_np(U, Omega):
        #     # U: [H, W, 3, 3] complex; Omega: [3,3] complex
        #     # returns Omega @ U @ Omega^†
        #     return np.einsum('ab,xybc,cd->xyad', Omega, U, np.conjugate(Omega.T), optimize=True)

        # -------------------------------------------------------------------------
        N = self.N
        e = self.entries[idx]

        # ---------- Two-time path: always SAME-RUN anchor -> target ----------
        run_idx = int(e["run_idx"])
        snaps = self.snapshots_by_run[run_idx]
        a0 = self.anchor_idx_by_run[run_idx]
        Ya = float(snaps[a0]["Y"])

        # Use prebuilt deterministic pair for BOTH train and val
        if e.get("mode") == "pair":
            a = int(e["a_idx"]); b = int(e["b_idx"])
            Yb = float(snaps[b]["Y"])
        else:
            # legacy fallback (shouldn't be hit with the new builder)
            if len(snaps) > 1:
                b = a0 + 1
            else:
                b = a0
            a = a0
            Yb = float(snaps[b]["Y"])

        # Sanity: ensure same-run and Yb >= Ya
        pa = Path(snaps[a]["path"]).parents[1].name if len(Path(snaps[a]["path"]).parents) >= 2 else None
        pb = Path(snaps[b]["path"]).parents[1].name if len(Path(snaps[b]["path"]).parents) >= 2 else None
        assert pa == pb, f"Cross-run pair detected: {pa} vs {pb}"
        if Yb < Ya:
            a, b = b, a
            Ya, Yb = Yb, Ya

        # Read anchor and target
        # Use cached anchor (18ch) if available
        Ua18 = self.anchor_cache_by_run[run_idx]
        if Ua18 is None:
            Ua18 = _ensure18(read_wilson_binary(snaps[a]["path"], size=N), N)
            # cache lazily per-worker for this run to avoid rereads next time
            if run_idx not in self._anchor_seen:
                self.anchor_cache_by_run[run_idx] = Ua18
                self._anchor_seen.add(run_idx)

        Ub18 = _ensure18(read_wilson_binary(snaps[b]["path"], size=N), N)

        Ua18 = np.array(Ua18, copy=True)  # [H,W,18], float32
        Ub18 = np.array(Ub18, copy=True)


        # Compose input channels: [U(Ya) 18ch] + [Y (1ch)] + [params 3ch]
        y_scalar = (Yb - Ya)
        base18 = torch.from_numpy(Ua18).permute(2,0,1)       # [18,H,W]
        y_s    = torch.tensor(y_scalar, dtype=torch.float32) # []
        theta  = torch.tensor([e["m"], e["Lambda_QCD"], e["mu0"]], dtype=torch.float32)  # [3]
        target = torch.from_numpy(Ub18).permute(2,0,1)       # [18,H,W]
        return base18, y_s, theta, target

#     def __getitem__(self, idx):
#         N = self.N
#         e = self.entries[idx]

#         # ---------- Two-time path: always SAME-RUN anchor -> target ----------
#         run_idx = int(e["run_idx"])
#         snaps = self.snapshots_by_run[run_idx]
#         a0 = self.anchor_idx_by_run[run_idx]
#         Ya = float(snaps[a0]["Y"])

#         # Use prebuilt deterministic pair for BOTH train and val
#         if e.get("mode") == "pair":
#             a = int(e["a_idx"]); b = int(e["b_idx"])
#             Yb = float(snaps[b]["Y"])
#         else:
#             # legacy fallback (shouldn't be hit with the new builder)
#             if len(snaps) > 1:
#                 b = a0 + 1
#             else:
#                 b = a0
#             a = a0
#             Yb = float(snaps[b]["Y"])


#         # Sanity: ensure same-run and Yb >= Ya
#         pa = Path(snaps[a]["path"]).parents[1].name if len(Path(snaps[a]["path"]).parents) >= 2 else None
#         pb = Path(snaps[b]["path"]).parents[1].name if len(Path(snaps[b]["path"]).parents) >= 2 else None
#         # (Both snapshots come from the same 'run_xxxxx' directory)
#         assert pa == pb, f"Cross-run pair detected: {pa} vs {pb}"
#         if Yb < Ya:
#             a, b = b, a
#             Ya, Yb = Yb, Ya

#         # Read anchor and target
#         # Use cached anchor (18ch) if available
#         Ua18 = self.anchor_cache_by_run[run_idx]
#         if Ua18 is None:
#             Ua18 = _ensure18(read_wilson_binary(snaps[a]["path"], size=N), N)
#             # cache lazily per-worker for this run to avoid rereads next time
#             if run_idx not in self._anchor_seen:
#                 self.anchor_cache_by_run[run_idx] = Ua18
#                 self._anchor_seen.add(run_idx)
# #            Ua18 = _ensure18(read_wilson_binary(snaps[a]["path"], size=N), N)
#         Ub18 = _ensure18(read_wilson_binary(snaps[b]["path"], size=N), N)

#         Ua18 = np.array(Ua18, copy=True)
#         Ub18 = np.array(Ub18, copy=True)

        
#         # Compose input channels: [U(Ya) 18ch] + [Y (1ch)] + [params 3ch]
#         y_scalar = (Yb - Ya) 
#         base18 = torch.from_numpy(Ua18).permute(2,0,1)       # [18,H,W]
#         y_s    = torch.tensor(y_scalar, dtype=torch.float32) # []
#         theta  = torch.tensor([e["m"], e["Lambda_QCD"], e["mu0"]], dtype=torch.float32)  # [3]
#         target = torch.from_numpy(Ub18).permute(2,0,1)       # [18,H,W]
#         return base18, y_s, theta, target


#         # Ymap = np.full((N, N, 1), np.float32(y_scalar), dtype=np.float32)
#         # params_map = np.stack([
#         #     np.full((N, N), e["m"], dtype=np.float32),
#         #     np.full((N, N), e["Lambda_QCD"], dtype=np.float32),
#         #     np.full((N, N), e["mu0"], dtype=np.float32)
#         # ], axis=-1)
#         # x = np.concatenate([Ua18, Ymap, params_map], axis=-1)

#         # return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(Ub18).permute(2, 0, 1)

# ------------------------- FNO model -------------------------

class SpectralConv2d(nn.Module):
    """2D Fourier layer: width -> width (truncated to (modes1, modes2))."""
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.m1, self.m2 = modes1, modes2
        scale = 0.02
        # Separate weights for top and bottom bands (height is two-sided in rfft2)
        self.weight1_re = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * scale)
        self.weight1_im = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * scale)
        self.weight2_re = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * scale)
        self.weight2_im = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * scale)

    @staticmethod
    def compl_mul2d(a: 'torch.Tensor', b: 'torch.Tensor') -> 'torch.Tensor':
        # a: [B,in,Hf,Wf] complex, b: [in,out,Hf,Wf] complex -> [B,out,Hf,Wf] complex
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x.float(), norm="ortho")  # [B,C,Hf,Wf] complex64
        Hf, Wf = x_ft.shape[-2], x_ft.shape[-1]
        m1, m2 = min(self.m1, Hf), min(self.m2, Wf)

        out_ft = torch.zeros(B, self.out_ch, Hf, Wf, dtype=x_ft.dtype, device=x_ft.device)
        if m1 > 0 and m2 > 0:
            re1 = self.weight1_re[..., :m1, :m2].float()
            im1 = self.weight1_im[..., :m1, :m2].float()
            re2 = self.weight2_re[..., :m1, :m2].float()
            im2 = self.weight2_im[..., :m1, :m2].float()
            w1 = torch.complex(re1, im1)  # complex64
            w2 = torch.complex(re2, im2)  # complex64

            # top band (low positive vertical freqs)
            out_ft[:, :, :m1,  :m2] = self.compl_mul2d(x_ft[:, :, :m1,  :m2], w1)
            # bottom band (low negative vertical freqs)
            out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], w2)

        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")  # [B,out,H,W] float32
        return x_out.to(x.dtype)


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w        = nn.Conv2d(width, width, kernel_size=1)
        self.act      = nn.GELU()
    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        return self.act(self.spectral(x) + self.w(x))



class RBFEmbed(nn.Module):
    def __init__(self, K: int, learnable: bool = True, init_sigma: float = 0.20):
        super().__init__()
        # centers in [0,1]; log-widths so widths stay positive
        centers = torch.linspace(0., 1., K)
        log_widths = torch.log(torch.full((K,), init_sigma))

        if learnable:
            self.rbf_centers = nn.Parameter(centers)         # [K]
            self.rbf_log_widths = nn.Parameter(log_widths)   # [K]
        else:
            self.register_buffer("rbf_centers", centers)         # [K]
            self.register_buffer("rbf_log_widths", log_widths)   # [K]

    def forward(self, y01: torch.Tensor) -> torch.Tensor:
        # y01 is in [0,1]; shape [B] or [B,1]
        widths = self.rbf_log_widths.exp()              # [K]
        z = (y01[..., None] - self.rbf_centers) / widths   # [B,K]
        time_feat = torch.exp(-0.5 * z * z)                 # [B,K]
        return time_feat

    
# class RBFEmbed(nn.Module):
#     def __init__(self, K=12, learnable_centers=False, sigma=None):
#         super().__init__()
#         self.K = K
#         centers = torch.linspace(0.0, 1.0, K)
#         self.centers = nn.Parameter(centers, requires_grad=learnable_centers)
#         self.sigma = sigma if sigma is not None else (0.5 / K)
#     def forward(self, y01: 'torch.Tensor') -> 'torch.Tensor':
#         y = y01[:, None]
#         d2 = (y - self.centers[None, :]) ** 2
#         return torch.exp(-0.5 * d2 / (self.sigma ** 2))

class TimeConditioner(nn.Module):
    def __init__(self, n_blocks: int, ch: int, emb_dim: int,
                 hidden: int = 64, film_mode: str = "scale_only",
                 gamma_scale: float = 1.5, beta_scale: float = 1.0,
                 gate_temp: float = 2.0):
        super().__init__()
        assert film_mode in ("scale_only", "scale_shift")
        self.n_blocks = n_blocks
        self.ch = ch
        self.film_mode = film_mode
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale
        self.gate_temp = gate_temp
        self.shared = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.block_heads = nn.ModuleList()
        for _ in range(n_blocks):
            out_dim = (ch + 1) if (film_mode == "scale_only") else (2*ch + 1)
            self.block_heads.append(nn.Linear(hidden, out_dim))
    def forward(self, t_emb: 'torch.Tensor'):
        h = self.shared(t_emb)
        outs = [head(h) for head in self.block_heads]
        parsed = []
        for o in outs:
            if self.film_mode == "scale_only":
                gamma = torch.tanh(o[:, :self.ch]) * self.gamma_scale
                beta  = None
                gate  = torch.sigmoid(self.gate_temp * o[:, -1:])
            else:
                gamma = torch.tanh(o[:, :self.ch]) * self.gamma_scale
                beta  = torch.tanh(o[:, self.ch:2*self.ch]) * self.beta_scale
                gate  = torch.sigmoid(self.gate_temp * o[:, -1:])
            parsed.append((gamma, beta, gate))
        return parsed

class DeltaYCalibrator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 16), torch.nn.SiLU(),
            torch.nn.Linear(16, 16), torch.nn.SiLU(),
            torch.nn.Linear(16, 1)
        )
    def forward(self, dy):  # dy: [B,1]
        # positive, bounded-ish scale; softplus + clamp for stability
        s = torch.nn.functional.softplus(self.net(dy))  # >=0
        return torch.clamp(s, 0.25, 4.0)  # you can widen later


    
class SU3HeadGellMannStochastic(nn.Module):
    """
    Predicts mean & (diag) log-std for 8 Lie-algebra coeffs.
    Trains with reparameterization: alpha = mu + sigma * eps.

    Changes vs. your version:
      - Build a bounded, per-sample normalized y_eff ∈ [0,1) from physical Y.
      - Use y_eff to gate both sigma and the exponent scale (no double scaling).
      - Keep softplus strictly on real tensors.
      - Identity snap uses the physical Y (not rescaled).
    """
    def __init__(self, width: int,
                 identity_eps: float = 0.0,
                 clamp_alphas: float | None = None,
                 alpha_scale: float | None = 1.,     # per-pixel L2 cap (recommended)
                 alpha_vec_cap: float | None = 15.,   # per-pixel L2 cap (recommended)
                 A_cap: float | None = None,         # or cap in matrix space
                 sigma0: float = 1.4,                # init noise scale
                 sigma_mode: str | None = None,
                 noise_kernel: int | None = None,
                 #sigma_mode: str = "conv",          # "diag" (current), "conv", or "spectral"
                 #noise_kernel: int = 5,             # conv kernel for spatial coupling
                 ):
        super().__init__()
        self.proj_mu   = nn.Conv2d(width, 8, 1)
        self.proj_logs = nn.Conv2d(width, 8, 1)

        # Init
        nn.init.xavier_uniform_(self.proj_mu.weight, gain=1e-2)
        nn.init.constant_(self.proj_mu.bias, 0.0)
        nn.init.constant_(self.proj_logs.weight, 0.0)
        nn.init.constant_(self.proj_logs.bias, float(math.log(sigma0)))

        self.identity_eps   = float(identity_eps)
        self.alpha_scale    = float(alpha_scale)
        self.clamp_alphas   = clamp_alphas
        self.alpha_vec_cap  = float(alpha_vec_cap)
        self.A_cap          = A_cap

        # learnable schedule s(Y)=kappa*softplus(gamma0 + gamma1*Y_eff)
        self.gamma0 = nn.Parameter(torch.tensor(-3.2, dtype=torch.float32))
        self.gamma1 = nn.Parameter(torch.tensor(0.0,  dtype=torch.float32))
        self.kappa  = nn.Parameter(torch.tensor(6.0,  dtype=torch.float32))

        # make initial noise gentle (overrides sigma0 above intentionally)
        if hasattr(self, "proj_logs") and getattr(self.proj_logs, "bias", None) is not None:
            torch.nn.init.constant_(self.proj_logs.bias, -4.6)

        self.projL = nn.Conv2d(width, 8, kernel_size=1)  # α_L^a
        self.projR = nn.Conv2d(width, 8, kernel_size=1)  # α_R^a
        self.log_cL = nn.Parameter(torch.zeros(()))      # (kept for compatibility)
        self.log_cR = nn.Parameter(torch.zeros(()))

        # gain for mapping physical Y -> bounded y_eff in the exponent

        # λ/2 basis (complex Hermitian), shared with deterministic head
        L = torch.zeros(8, 3, 3, dtype=torch.complex64)
        L[0,0,1] = L[0,1,0] = 1;        L[1,0,1] = -1j; L[1,1,0] = 1j
        L[2,0,0] = 1; L[2,1,1] = -1;    L[3,0,2] = L[3,2,0] = 1
        L[4,0,2] = -1j; L[4,2,0] = 1j;  L[5,1,2] = L[5,2,1] = 1
        L[6,1,2] = -1j; L[6,2,1] = 1j;  s3 = 1.0 / (3.0**0.5)
        L[7,0,0] = s3; L[7,1,1] = s3;   L[7,2,2] = -2*s3
        self.register_buffer("lambdas", L / 2.0, persistent=False)

        # runtime monitors
        self.last_A_fro_mean = torch.tensor(0.0)
        self.last_sigma_mean = torch.tensor(0.0)


        if sigma_mode is None:
            a = globals().get("args", None)
            sigma_mode = getattr(a, "sigma_mode", "conv")
        if noise_kernel is None:
            a = globals().get("args", None)
            noise_kernel = getattr(a, "noise_kernel", 5)

        self.sigma_mode = sigma_mode
        self.noise_kernel = noise_kernel
        self.width = width     
        
        self.alpha_channels = int(self.proj_mu.out_channels)  # 8 or 16
        C= self.alpha_channels

        self.C = C
        self.sigma_floor = 0.
        init_sigma = 0.03
        # mu_head: width -> C
        self.mu_head    = nn.Conv2d(self.width, C, kernel_size=1, bias=True)        
        # σ head: C -> C   (because you call sigma_head(mu))
        self.sigma_head = nn.Conv2d(C, C, kernel_size=1, bias=True)   # <<< change here

        nn.init.constant_(self.sigma_head.bias, -2.0)
        nn.init.zeros_(self.sigma_head.weight)

        with torch.no_grad():
            self.sigma_head.bias.fill_(math.log(math.exp(init_sigma) - 1))
        
        # σ head: C -> C   (because you call sigma_head(mu))
        self.sigma_head = nn.Conv2d(C, C, kernel_size=1, bias=True)
        nn.init.constant_(self.sigma_head.bias, -2.0)
        nn.init.zeros_(self.sigma_head.weight)
        with torch.no_grad():
            self.sigma_head.bias.fill_(math.log(math.exp(init_sigma) - 1))

        # NEW: scalar physical diffusion amplitude head (physical s ≥ 0)
        self.amp_head = nn.Conv2d(self.width, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.amp_head.weight)
        nn.init.constant_(self.amp_head.bias, -2.0)  # softplus(-2) ≈ 0.12

        self.op_gain = torch.nn.Parameter(torch.tensor(1.0))  # learnable overall gain
        
        # ---- build operator (conv) ----
        if self.sigma_mode == "conv":
            k = int(self.noise_kernel); pad = k // 2
            self.noise_dw = torch.nn.Conv2d(C, C, kernel_size=k, padding=pad, groups=C, bias=False)
            self.noise_pw = torch.nn.Conv2d(C, C, kernel_size=1, bias=False)

            # Identity init so eta ≈ eps initially (not tiny)
            with torch.no_grad():
                self.noise_dw.weight.zero_()
                self.noise_dw.weight[:, 0, pad, pad] = 1.0  # center tap per channel
                self.noise_pw.weight.zero_()
                for c in range(C):
                    self.noise_pw.weight[c, c, 0, 0] = 1.0

        elif self.sigma_mode == "spectral":
            self.spec_k0 = torch.nn.Parameter(torch.full((C,), 0.20))
            self.spec_p  = torch.nn.Parameter(torch.full((C,), 2.00))

    
    def _cap_alphas(self, a: torch.Tensor) -> torch.Tensor:
        # optional per-channel tanh clamp
        if self.clamp_alphas is not None:
            a = torch.tanh(a) * float(self.clamp_alphas)
        # per-pixel vector L2 cap (recommended)
        if (self.alpha_scale is not None) and (self.alpha_scale != 1.0):
            a = a * float(self.alpha_scale)
        if self.alpha_vec_cap is not None:
            vnorm = a.norm(dim=1, keepdim=True).clamp(min=1e-6)
            a = a * (float(self.alpha_vec_cap) / vnorm).clamp(max=1.0)
        return a

    def _assemble(self, alphas: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
        T = self.lambdas.to(device=device)                                   # [8,3,3] complex64
        S = torch.einsum("bahw,aij->bhwij", alphas.to(T.dtype), T)           # [B,H,W,3,3]
        if self.A_cap is not None:  # exact cap on ||A||_F==||S||_F
            S_f = (S.real.square().sum(dim=(-2,-1)) + 1e-12).sqrt()
            S = S * (float(self.A_cap) / S_f).unsqueeze(-1).unsqueeze(-1).clamp(max=1.0)
        A = 1j * S
        # monitor Frobenius norm of the Hermitian part (before 1j)
        self.last_A_fro_mean = (S.real.square().sum(dim=(-2,-1)).sqrt().mean()).detach()
        return A, S

    #only works for nsamples=1 when sampling
    def forward(self, h: torch.Tensor, base18: torch.Tensor, Ymap: torch.Tensor,
                nsamples: int = 1, sample: bool | None = None, dY=None) -> torch.Tensor:
        # Optional fast exit at Y≈0
        if sample is None:
            sample = self.training # sample during training, deterministic in validation
        eps = float(self.identity_eps)
        if eps > 0.0 and (Ymap.abs() <= eps).all():
            return base18

        U0 = pack_to_complex(base18.permute(0, 2, 3, 1).float())   # [B,H,W,3,3]

        # Project means & log-stdevs
        C =self.C
        assert C in (8, 16), f"expect 8 or 16 channels for Gell-Mann coeffs, got {C}"

        #mu = self.mu_head(h)                  # [B, C, H, W]
        mu     = self.proj_mu(h).float()        # [B, C, H, W], C in {8,16}
        sigma_logits = self.sigma_head(mu)    # [B, C, H, W]  <- now channels match
        sigma_amp = self.sigma_floor + F.softplus(sigma_logits)

        logsig = self.proj_logs(h).float()      # [B, C, H, W]

        # NEW: physical diffusion amplitude (scalar field per pixel)
        amp = F.softplus(self.amp_head(h)).float()      # [B,1,H,W] ≥ 0
        self.last_amp = amp

        y_eff = Ymap[:, 0, :, :].to(mu.real.dtype)                 # [B,H,W]

        # Broadcast helpers
        y       = y_eff.unsqueeze(-1).unsqueeze(-1)  # [B,H,W,1,1]  (real)
        y_sigma = y_eff.unsqueeze(1)                 # [B,1,H,W]    (real)

        # σ(Y) = y_eff * softplus(logσ̂)  (vanishes at Y=0, nonzero for Y>0 batches)
        sigma = y_sigma * F.softplus(logsig)

        self.last_mu = mu
        self.last_logsig = logsig
        self.last_y_sigma = y_sigma

        # Sampling
        if sample:
            epsn = torch.randn_like(mu)  # [B,C,H,W]

            if self.sigma_mode == "diag":
                eta_core = epsn

            elif self.sigma_mode == "conv":
                eta_core = self.noise_pw(self.noise_dw(epsn))

            elif self.sigma_mode == "spectral":
                B, C, H, W = mu.shape
                Ef = torch.fft.rfft2(epsn, norm="ortho")
                ky = torch.fft.rfftfreq(H, d=1.0).to(Ef.device)
                kx = torch.fft.rfftfreq(W, d=1.0).to(Ef.device)
                Ky = ky.view(-1, 1).expand(Ef.shape[-2], Ef.shape[-1])
                Kx = kx.view(1, -1).expand(Ef.shape[-2], Ef.shape[-1])
                Kr = torch.sqrt(Kx**2 + Ky**2)
                k0 = torch.clamp(self.spec_k0, min=1e-3).view(1, C, 1, 1)
                p  = torch.relu(self.spec_p).view(1, C, 1, 1)
                G  = (1.0 / (1.0 + (Kr.view(1,1,*Kr.shape) / k0)**p)).to(Ef.dtype)
                Ef = Ef * G
                eta_core = torch.fft.irfft2(Ef, s=(H, W), norm="ortho")

            # --- normalize eta_core to unit RMS (detach denom to avoid weird grads)
            den = eta_core.pow(2).mean(dim=(2,3), keepdim=True).add(1e-12).sqrt().detach()
            eta_core = eta_core / den


            # overall learnable gain + per-pixel amplitude
#            eta = self.op_gain * amp * eta_core   # (was sigma_amp)
            eta = sigma * eta_core
            
            B = base18.shape[0]
            if dY is None:
                # default to per-unit step if caller didn't supply it
                dY = torch.ones(B, device=base18.device, dtype=base18.dtype)
                dY = dY.view(B, 1, 1, 1)

            mu_step  = mu  * dY            # μ ΔY
            eta_step = eta * torch.sqrt(dY)  # s √ΔY
            a_all    = mu_step + eta_step     # (was mu + eta)
            
            # debug hooks
            self.last_sigma_mode = self.sigma_mode
            self.last_eta_rms = eta.detach().pow(2).mean()
            self.last_sigma_mean = amp.detach().mean()  # alias to amp for compat
            self.last_sigma_min  = amp.detach().min()
            self.last_sigma_max  = amp.detach().max()
        else:
            a_all = mu
            # # σ(Y) = y_eff * softplus(logσ̂)  => vanishes at Y=0 and grows smoothly
            # sigma = y_sigma * F.softplus(logsig)
            self.last_sigma_mean = sigma.mean().detach()
            
        # Split into Left/Right sets
        if C == 16:
            aL, aR = torch.split(a_all, 8, dim=1)    # each [B,8,H,W]
        else:
            aL = a_all
            aR = a_all

        # Cap alphas
        aL = self._cap_alphas(aL)
        aR = self._cap_alphas(aR)

        # Assemble anti-Hermitian matrices A_L, A_R: [B,H,W,3,3] (complex)
        AL, _ = self._assemble(aL, device=h.device)
        AR, _ = self._assemble(aR, device=h.device)

        # Gain schedule s(Y) = kappa * softplus(gamma0 + gamma1 * y_eff)
        # (real, >=0, broadcast-friendly)
        gain  = self.kappa * F.softplus(self.gamma0 + self.gamma1 * y)  # [B,H,W,1,1]
        scale = y * gain                                                # [B,H,W,1,1], real

        # Effective generators; real 'scale' broadcasts over complex matrices
        AeffL = scale * AL * self.dy_calib(scale)  
        AeffR = scale * AR * self.dy_calib(scale)  

        # Strang composition
        GLh = torch.linalg.matrix_exp(-0.5 * AeffL)
        GR  = torch.linalg.matrix_exp(-1.0 * AeffR)
        U   = GLh @ U0 @ GR @ GLh

        # Monitor the effective update size (complex Frobenius norm)
        self.last_A_fro_mean = (AeffL.abs().square().sum(dim=(-2, -1)).sqrt().mean()).detach()

        # Pack back to 18 real-imag channels
        out18 = unpack_to_18(U).permute(0, 3, 1, 2).to(h.dtype)

        # Optional identity snap at Y≈0 using *physical* Y (not rescaled)
        if eps > 0.0:
            y_abs0 = y_eff[:, 0, 0].abs()
            if (y_abs0 <= eps).any():
                mask = (y_abs0 <= eps)
                out18 = out18.clone()
                out18[mask] = base18[mask]

                
        return out18

class SU3HeadGellMann(nn.Module):
    def __init__(self, width: int, identity_eps: float = 0.0,
                 alpha_scale: float = 1.0, clamp_alphas: float | None = None, alpha_vec_cap: float|None = 15.):
        super().__init__()
        self.proj8 = nn.Conv2d(width, 8, 1)
        self.identity_eps = float(identity_eps)
        self.alpha_scale  = float(alpha_scale)
        self.clamp_alphas = clamp_alphas
        self.alpha_vec_cap = alpha_vec_cap
        self.last_A_fro_mean = torch.tensor(0.0)
        
        # Complex Hermitian Gell-Mann matrices λ_a  (shape [8,3,3], complex)
        L = torch.zeros(8, 3, 3, dtype=torch.complex64)
        # λ1
        L[0,0,1] = L[0,1,0] = 1
        # λ2
        L[1,0,1] = -1j; L[1,1,0] = 1j
        # λ3
        L[2,0,0] = 1; L[2,1,1] = -1
        # λ4
        L[3,0,2] = L[3,2,0] = 1
        # λ5
        L[4,0,2] = -1j; L[4,2,0] = 1j
        # λ6
        L[5,1,2] = L[5,2,1] = 1
        # λ7
        L[6,1,2] = -1j; L[6,2,1] = 1j
        # λ8
        s3 = 1.0 / (3.0**0.5)
        L[7,0,0] = s3; L[7,1,1] = s3; L[7,2,2] = -2*s3
        L = L / 2.
        self.register_buffer("lambdas", L, persistent=False)

    def forward(self, h: 'torch.Tensor', base18: 'torch.Tensor', Ymap: 'torch.Tensor') -> 'torch.Tensor':
        with torch.amp.autocast("cpu", enabled=False):   # keep disabled; MPS has no autocast mode
            # Early exit for identity region (batchwise)
            eps = self.identity_eps
            if eps > 0.0:
                y_abs = Ymap[:, 0, 0, 0].abs()
                if (y_abs <= eps).all():
                    return base18

            # U0 once we know we won’t early-return for the whole batch
            U0 = pack_to_complex(base18.permute(0, 2, 3, 1).float())  # [B,H,W,3,3]

            # Coefficients (real), clamp to avoid huge angles
            raw_alphas = self.proj8(h).float()            # [B,8,H,W]
            cap = self.clamp_alphas if (self.clamp_alphas is not None) else 2.0
            alphas = torch.tanh(raw_alphas) * float(cap)
            if self.alpha_scale != 1.0:
                alphas = alphas * float(self.alpha_scale)

            # NEW: L2 cap on the 8-d vector per (B,H,W)
            if self.alpha_vec_cap is not None:
                vnorm = alphas.norm(dim=1, keepdim=True).clamp(min=1e-6)
                scale = (float(self.alpha_vec_cap) / vnorm).clamp(max=1.0)
                alphas = alphas * scale


            # Build S and A (keep basis complex!)
            T = self.lambdas.to(device=h.device)          # complex64, [8,3,3]
            alphas_c = alphas.to(dtype=T.dtype)           # complex64
            S = torch.einsum("bahw,aij->bhwij", alphas_c, T)   # complex Hermitian
            A = 1j * S                                    # anti-Hermitian

            # Broadcast Y: [B,1,H,W] -> [B,H,W,1,1]
#            y = Ymap.squeeze(1).unsqueeze(-1).unsqueeze(-1).to(A.dtype)
            y = Ymap[:, 0, :, :].unsqueeze(-1).unsqueeze(-1).to(A.real.dtype)
            # Exponentiate and apply
            G = torch.linalg.matrix_exp(-y * A)
            U = G @ U0

            # expose an aux term we can regularize if needed
            A_fro = (A.real.square() + A.imag.square()).sum(dim=(-2,-1)).sqrt()
            self.last_A_fro_mean = A_fro.mean()

            out18 = unpack_to_18(U).permute(0, 3, 1, 2).to(h.dtype)

            # Partial identity masking (per-sample)
            if eps > 0.0:
                y_abs = Ymap[:, 0, 0, 0].abs()
                if (y_abs <= eps).any():
                    mask = (y_abs <= eps)
                    out18 = out18.clone()
                    out18[mask] = base18[mask]

            return out18


import torch, torch.nn as nn

class EvolverFNO(nn.Module):
    """Lift -> [FNOBlock]*n with Y/θ-conditioned gating/FiLM -> SU3HeadGellMann."""
    def __init__(self, in_ch=22, width=64, modes1=12, modes2=12, n_blocks=4,
                 identity_eps: float = 0.0, alpha_scale: float = 4.0,
                 clamp_alphas=None, alpha_vec_cap=15.0,
                 # conditioning
                 y_index: int = 18,
                 film_mode: str = "scale_only",
                 rbf_K: int = 12,
                 film_hidden: int = 64,
                 gamma_scale: float = 1.5,
                 beta_scale: float = 1.0,
                 gate_temp: float = 2.0,
                 y_min: float | None = None,
                 y_max: float | None = None,
                 rbf_gamma: float = 0.,
                 rbf_min_width: float =0.,
                 y_map: str = "linear"):
        super().__init__()
        self.y_index = int(y_index)
        self.width = int(width)

        self.rbf_gamma = rbf_gamma
        self.rbf_min_width = rbf_min_width
        
        self.film_hidden = film_hidden

        self.rbf_K = rbf_K
        self.block_norm = torch.nn.GroupNorm(num_groups=1, num_channels=self.width)
        
        # Trunk sees only the spatial channels (base18)
        self.lift = nn.Conv2d(18, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_blocks)])

        # SU(3) head: will receive base18 and a broadcasted Y map
        self.head = SU3HeadGellMannStochastic(
            width,
            identity_eps=identity_eps,
            alpha_scale=alpha_scale,
            clamp_alphas=clamp_alphas,
            alpha_vec_cap=alpha_vec_cap,
        )
        self.head.dy_calib = DeltaYCalibrator()

        # Y & θ embeddings → per-block FiLM + gates

        # learnable per-center widths for both time and theta
        self.time_embed  = RBFEmbed(K=rbf_K, learnable=True, init_sigma=0.20)
        self.theta_embed = RBFEmbed(K=rbf_K, learnable=True, init_sigma=0.20)


        # (optional) small “pre-FiLM” head to clean up the RBFs
        # If you want to be able to toggle this, put it behind a flag – but simplest is to keep it on.
        pre_film_in = 2 * rbf_K
        pre_film_h  = self.film_hidden  # you already parse --film_hidden
        self.pre_film = torch.nn.Sequential(
            torch.nn.LazyLinear(self.film_hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.film_hidden),
            torch.nn.Linear(self.film_hidden, self.film_hidden),
            torch.nn.SiLU(),
        )

        # self.pre_film = nn.Sequential(
        #     nn.LayerNorm(pre_film_in),
        #     nn.Linear(pre_film_in, pre_film_h),
        #     nn.SiLU(),
        #     nn.Linear(pre_film_h, pre_film_h),
        # )

        self.beta_scale = beta_scale
        self.gamma_scale = gamma_scale
        self.film_mode = film_mode
        self.gate_temp = gate_temp
        # FiLM now takes the pre_film hidden size instead of raw 2*K

        self.time_cond = TimeConditioner(
            ch=self.width,
            n_blocks=len(self.blocks),
            emb_dim=pre_film_h,
            hidden=self.film_hidden,
            film_mode=self.film_mode,
            gamma_scale=self.gamma_scale,
            beta_scale=self.beta_scale,
            gate_temp=self.gate_temp,
        )



        # --- figure out the conditioning vector dim coming out of pre_film ---
        def _last_linear_out_dim(mod: nn.Module):
            # walk down to the last Linear anywhere inside mod
            if isinstance(mod, nn.Linear):
                return mod.out_features
            # recurse into containers
            children = list(mod.children())
            for child in reversed(children):
                d = _last_linear_out_dim(child)
                if d is not None:
                    return d
            return None

        # if you built pre_film to end in a Linear -> (activation), grab that Linear's out_features
        pre_film_out_dim = _last_linear_out_dim(self.pre_film)
        
        # fallback if pre_film is Identity or weird: use your intended hidden size, else the raw cond input size
        if pre_film_out_dim is None:
            # if you created pre_film with film_hidden > 0, use that; otherwise use cond_in_dim = out_dim(time_embed)+out_dim(theta_embed)
            KY = getattr(self.time_embed, "out_features")
            KT = getattr(self.theta_embed, "out_features")
            cond_in_dim = KY + KT
            pre_film_out_dim = getattr(self, "film_hidden", None) or cond_in_dim

        self.pre_film_out_dim = pre_film_out_dim  # stash for clarity

        # --- tail calibrator (fixed) ---
        self.tail_gate = nn.Sequential(
            nn.LayerNorm(self.pre_film_out_dim),
            nn.Linear(self.pre_film_out_dim, self.width),
            nn.Sigmoid(),
        )
        self.tail_conv = nn.Conv2d(self.width, self.width, kernel_size=1, bias=True)



        # self.tail_gate = nn.Sequential(
        #     nn.LayerNorm(self.pre_film[-1].out_features if isinstance(self.pre_film[-1], nn.Linear) else self.pre_film[-1].num_features),
        #     nn.Linear(self.pre_film[-1].out_features if isinstance(self.pre_film[-1], nn.Linear) else self.pre_film[-1].num_features, self.width),
        #     nn.Sigmoid(),
        # )
        # self.tail_conv = nn.Conv2d(self.width, self.width, kernel_size=1, bias=True)


        
        # self.time_embed  = RBFEmbed(K=rbf_K)  # eY ∈ ℝ^K
        # self.theta_embed = nn.Sequential(
        #     nn.Linear(3, film_hidden), nn.SiLU(), nn.Linear(film_hidden, rbf_K)  # eθ ∈ ℝ^K
        # )
        # self.time_cond   = TimeConditioner(
        #     n_blocks=len(self.blocks), ch=width, emb_dim=rbf_K*2,
        #     hidden=film_hidden, film_mode=film_mode,
        #     gamma_scale=gamma_scale, beta_scale=beta_scale, gate_temp=gate_temp
        # )



        self.y_map = y_map
        self.y_min = y_min
        self.y_max = y_max
        # Buffers for normalizing Y → [0,1]
        self.register_buffer("y_min_buf", torch.tensor(0.0), persistent=False)
        self.register_buffer("y_max_buf", torch.tensor(1.0), persistent=False)
        if (y_min is not None) and (y_max is not None):
            self.y_min_buf.fill_(float(y_min)); self.y_max_buf.fill_(float(y_max))

        # build centers/widths once
        t = torch.linspace(0, 1, self.rbf_K)
        centers = t**self.rbf_gamma                      # <-- bias toward high y when gamma<1
        # widths ~ half the local spacing, with a floor
        edges = torch.cat([centers[:1], centers, centers[-1:]])
        spacings = torch.diff(edges)
        widths = 0.5 * spacings.clamp_min(self.rbf_min_width)
        self.register_buffer("rbf_centers", centers)
        self.register_buffer("rbf_widths",  widths)
            
    # --- helpers ---
    def _y01_from_scalar(self, y_scalar: torch.Tensor) -> torch.Tensor:
        # Ensure y_min/y_max are tensors on the same device
        y_min = self.y_min
        y_max = self.y_max
        if not torch.is_tensor(y_min): y_min = torch.tensor(y_min, device=y_scalar.device)
        if not torch.is_tensor(y_max): y_max = torch.tensor(y_max, device=y_scalar.device)
        
        if self.y_map == "linear":
            denom = (y_max - y_min).clamp_min(1e-9)
            y01 = (y_scalar - y_min) / denom
            return y01.clamp(0.0, 1.0)

        elif self.y_map == "tanh":
            # Smoothly compress large ΔY; map to [0,1]
            g = 1. 
            num = torch.tanh(g * (y_scalar - y_min))
            den = torch.tanh(g * (y_max - y_min)).clamp_min(1e-9)
            y01 = (num / den).clamp(0.0, 1.0)
            return y01

        else:
            raise ValueError(f"Unknown y_map: {self.y_map}")

    # def _y01_from_scalar(self, Y_scalar: torch.Tensor) -> torch.Tensor:
    #     # self.y_min_buf / self.y_max_buf are tensors on the right device already
    #     denom = (self.y_max_buf - self.y_min_buf)
    #     # torch.where keeps it all on device, no Python scalars
    #     y01 = torch.where(
    #         denom > 0,
    #         (Y_scalar - self.y_min_buf) / denom.clamp_min(1e-12),
    #         torch.sigmoid(Y_scalar),
    #     )
    #     return y01.clamp_(0.0, 1.0)


    # --- trunk encoding (base18 + scalars Y,θ) ---
    def encode_trunk_from_components(
        self,
        base18: torch.Tensor,
        Y_scalar: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        # Trunk features
        h = self.lift(base18)                      # [B, C, H, W]
        B, C = h.shape[:2]

        # ----- build 2-D conditioning vector [B, *] -----
        # y in [0,1], then time embed -> [B, KY]
        y01 = self._y01_from_scalar(Y_scalar).to(h.device, h.dtype)
        # some embedders want [B] and some [B,1]; this handles both
        eY = self.time_embed(y01 if y01.dim() == 1 else y01.squeeze(-1))
        if eY.dim() > 2:
            # collapse any stray dims (shouldn't happen, but safe)
            eY = eY.flatten(1)
        # theta embed -> [B, KT] (make sure it's 2-D)
        eT = self.theta_embed(theta.to(h.device, h.dtype))
        if eT.dim() > 2:
            # prefer pooling over flatten if yours is spatial; otherwise flatten(1)
            # eT = eT.mean(dim=tuple(range(2, eT.dim())))
            eT = eT.flatten(1)
        # if embed returns [B], make it [B,1]
        if eT.dim() == 1:
            eT = eT.unsqueeze(-1)

        # concatenate -> [B, KY + KT]
        h_cond_in = torch.cat([eY, eT], dim=1)

        # map to conditioner hidden, then per-block FiLM params
        h_cond = self.pre_film(h_cond_in)          # [B, Hf]
        cond   = self.time_cond(h_cond)            # len == len(self.blocks)

        # ----- FiLM residual blocks -----
        for i, b in enumerate(self.blocks):
            h_in = h
            h_b  = b(h)
            h_b  = self.block_norm(h_b)                 # <— normalize block output first
            

            gamma, beta, gate = cond[i]  # shapes: [B,W], [B,W] or None, gate: [B,1] or [B,W]

            # upgrade scalar gate -> per-channel
            if gate.dim() == 2 and gate.shape[1] == 1:
                gate = gate.expand(-1, self.width)
            elif gate.dim() == 1:
                gate = gate[:, None].expand(-1, self.width)

            h = h_in + gate.view(-1, self.width, 1, 1) * (h_b - h_in)
            h = h * (1.0 + gamma.view(-1, self.width, 1, 1))
            if beta is not None:
                h = h + beta.view(-1, self.width, 1, 1)


        tail_gate = self.tail_gate(h_cond).view(h.shape[0], self.width, 1, 1)
        h = h + tail_gate * self.tail_conv(h)
        return h


    # --- clean API: components in, SU(3) out ---
    def forward(self, base18, Y_scalar, theta, 
                sample: bool | None = None,
                nsamples: int = 1,
                dY=None):
        B, _, H, W = base18.shape
        h = self.encode_trunk_from_components(base18, Y_scalar, theta)
        Ymap = Y_scalar.view(B, 1, 1, 1).expand(B, 1, H, W)
        return self.head(h, base18, Ymap, nsamples=nsamples, sample=sample, dY=dY)
    # def forward(self, base18: torch.Tensor,
    #                        Y_scalar: torch.Tensor,
    #                        theta: torch.Tensor) -> torch.Tensor:
    #     B, _, H, W = base18.shape
    #     h = self.encode_trunk_from_components(base18, Y_scalar, theta)
    #     # Build a Y "map" on-the-fly for the head (broadcast; no dataset storage)
    #     Ymap = Y_scalar.view(B,1,1,1).expand(B,1,H,W)
    #     return self.head(h, base18, Ymap)

def _reduce(x: torch.Tensor, reduction: str):
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError(f"Unknown reduction: {reduction}")

class GroupLoss(nn.Module):
    """
    Frobenius loss on 3x3 with optional:
      - trace, unitarity, det regularizers (as before)
      - geodesic loss on SU(3) (mean squared eigen-angles)
      - direction alignment (cosine between deltas wrt U0)
      - dipole anchor (gauge-invariant, few axial r's)

    New weights (geo_weight, dir_weight, dipole_weight) default to 0 so
    existing behavior is preserved unless you set them.
    """
    def __init__(
        self,
        w_frob: float = 0,
        w_trace: float = 0.0,
        w_unit: float = 0.1,
        w_det: float  = 0.0,
        *,
        project_before_frob: bool = True,
        geo_weight: float = 0.0,
        dir_weight: float = 0.0,
        dipole_weight: float = 0.0,
        dipole_offsets=((1,0),(0,1),(2,0),(0,2)),
    ):
        super().__init__()
        self.w_frob  = float(w_frob)
        self.w_trace = float(w_trace)
        self.w_unit  = float(w_unit)
        self.w_det   = float(w_det)

        self.project_before_frob = bool(project_before_frob)
        self.geo_weight   = float(geo_weight)
        self.dir_weight   = float(dir_weight)
        self.dipole_weight   = float(dipole_weight)
        self.dip_offsets  = tuple(dipole_offsets)
        self.register_buffer("I3", torch.eye(3, dtype=torch.cfloat), persistent=False)

    # -------- helpers (kept inside class for self-containment) --------
    @staticmethod
    def _pack18_to_U(x18: 'torch.Tensor') -> 'torch.Tensor':
        # x18: [B, 18, H, W]  ->  U: [B, H, W, 3, 3] complex
        return pack_to_complex(x18.permute(0,2,3,1).contiguous()).to(torch.complex64)

    @staticmethod
    def _su3_project(A, iters: int = 3, eps: float = 1e-6):
        """
        Fast SU(3) projection via Newton–Schulz polar factorization (GPU-friendly).
        A: (..., 3, 3) complex tensor. Returns approx. unitary with det = 1.
        """
        X = A.to(torch.complex64)
        I = torch.eye(3, dtype=X.dtype, device=X.device)

        # Newton–Schulz: X_{k+1} = 1/2 * X_k * (3I - X_k^H X_k)
        for _ in range(iters):
            XHX = X.conj().transpose(-1, -2) @ X
            X   = 0.5 * (X @ (3*I - XHX))

        # Determinant correction to SU(3): multiply by det(X)^{-1/3}
        det   = torch.linalg.det(X)                    # complex
        mag   = det.abs().clamp_min(eps)               # real, safe from zero
        ang   = torch.angle(det)                       # real angle in (-pi, pi]

        # det^{-1/3} = mag^{-1/3} * exp(-i * ang / 3)  (principal branch)
        mag_corr   = mag.pow(-1.0/3.0)                 # real
        phase_corr = torch.polar(torch.ones_like(ang), -ang/3.0)  # complex64, unit magnitude

        corr = (mag_corr * phase_corr).to(X.dtype)     # complex64
        X = X * corr[..., None, None]
        return X

    @staticmethod
    def _geo_theta2(Ut, Up):
        # True geodesic on SU(3): sum of squared principal angles
        # assumes Ut, Up already near-unitary (projected as you do above)
        Delta = Ut.conj().transpose(-1, -2) @ Up                    # (...,3,3)
        w = torch.linalg.eigvals(Delta.to(torch.complex128))        # (...,3)
        theta = torch.atan2(w.imag, w.real)                         # principal phases in (-pi,pi]
        theta = theta - theta.mean(dim=-1, keepdim=True)            # det≈1 ⇒ sum theta ≈ 0
        d2 = (theta**2).sum(dim=-1)                                 # (...,)
        return d2.mean()

    def geo_lowpass_theta2_from18(
        self,
        U_true18: torch.Tensor,    # truth FIRST (like _geo_theta2)
        U_pred18: torch.Tensor,    # prediction second
        k: int = 3,
        detach_true: bool = True,
        project_pred: bool = True
    ) -> torch.Tensor:
        Up = self._pack18_to_complex33(U_pred18)   # pred
        Ut = self._pack18_to_complex33(U_true18)   # true

        if project_pred:
            Up = self._su3_project(Up)             # keep grad
        if detach_true:
            with torch.no_grad():
                Ut = self._su3_project(Ut)
        else:
            Ut = self._su3_project(Ut)

        Delta = Ut.conj().transpose(-2, -1) @ Up   # U_true† U_pred
        w = torch.linalg.eigvals(Delta.to(torch.complex128))
        theta = torch.atan2(w.imag, w.real).to(torch.float32)
        theta = theta - theta.mean(dim=-1, keepdim=True)

        B,H,W,_ = theta.shape
        th = theta.permute(0,3,1,2).reshape(B*3,1,H,W)
        th_lp = _avgpool_circular(th, k=k).view(B,3,H,W).permute(0,2,3,1)

        return (th_lp.pow(2).sum(dim=-1)).mean()

    @staticmethod
    def _dir_cosine(U0: 'torch.Tensor', Ut: 'torch.Tensor', Up: 'torch.Tensor') -> 'torch.Tensor':
        # cosine( (Ut-U0), (Up-U0) ) averaged over batch
        d_true = (Ut - U0).reshape(U0.shape[0], -1)
        d_pred = (Up - U0).reshape(U0.shape[0], -1)
        num = (d_true.conj()*d_pred).real.sum(dim=1)
        den = d_true.norm(dim=1) * d_pred.norm(dim=1) + 1e-8
        return (num / den).mean()

    # ------------------------------------------------------------------

    def _components(self, yhat: torch.Tensor, y: torch.Tensor, reduction: str = "mean"):
        # yhat, y: [B,18,H,W]
        Uh_raw = self._pack18_to_U(yhat)
        U_raw  = self._pack18_to_U(y)

        # Only project if a *projected* base term actually needs it
        need_proj = self.project_before_frob and (self.w_frob != 0.0 or self.w_trace != 0.0)
        Uh = self._su3_project(Uh_raw) if need_proj else Uh_raw
        U  = self._su3_project(U_raw)  if need_proj else U_raw

        frob_map = None
        if self.w_frob != 0.0:
            d = Uh - U
            frob_map = d.real.pow(2) + d.imag.pow(2)   # [B,H,W,3,3]

        trace_map = None
        if self.w_trace != 0.0:
            trh = Uh.diagonal(dim1=-2, dim2=-1).sum(-1)  # [B,H,W]
            tr  =  U.diagonal(dim1=-2, dim2=-1).sum(-1)
            trace_map = (trh.real - tr.real).pow(2) + (trh.imag - tr.imag).pow(2)

        unit_map = None
        if self.w_unit != 0.0:
            I = self.I3.to(dtype=Uh_raw.dtype, device=Uh_raw.device)  # register I3 once in __init__
            UU = Uh_raw.conj().transpose(-2, -1) @ Uh_raw
            unit_map = (UU - I).abs().pow(2)       # [B,H,W,3,3]

        det_map = None
        if self.w_det != 0.0:
            det_map = (torch.linalg.det(Uh_raw).abs() - 1.0).pow(2)  # [B,H,W]

        def _reduce(t):
            if t is None:
                # scalar zero per-sample
                return torch.zeros((yhat.shape[0],), device=Uh_raw.device, dtype=torch.float32)
            if t.dim() == 5:  # [B,H,W,3,3] -> sum over matrix
                t = t.sum(dim=(-1, -2))
            return t.mean(dim=(1, 2))  # [B]

        if reduction == "none":
            return frob_map, trace_map, unit_map, det_map, Uh, U, Uh_raw
        else:
            frob     = _reduce(frob_map)
            trace_ms = _reduce(trace_map)
            unit     = _reduce(unit_map)
            det      = _reduce(det_map)
            return frob, trace_ms, unit, det, Uh, U, Uh_raw

    # def _components(self, yhat: 'torch.Tensor', y: 'torch.Tensor', reduction: str = "mean"):
    #     # yhat, y: [B,18,H,W]

    #     Uh_raw = self._pack18_to_U(yhat)
    #     U_raw  = self._pack18_to_U(y)

    #     # For the main data geometry, optionally project both to SU(3)
    #     Uh = self._su3_project(Uh_raw) if self.project_before_frob else Uh_raw
    #     U  = self._su3_project(U_raw)  if self.project_before_frob else U_raw

    #     # Frobenius map (on chosen geometry)
    #     d = Uh - U
    #     frob_map = d.real.pow(2) + d.imag.pow(2)     # [B,H,W,3,3]

    #     # Optional trace term (on chosen geometry)
    #     if self.w_trace != 0.0:
    #         trh = Uh.diagonal(dim1=-2, dim2=-1).sum(-1)  # [B,H,W]
    #         tr  =  U.diagonal(dim1=-2, dim2=-1).sum(-1)
    #         trace_map = (trh.real - tr.real).pow(2) + (trh.imag - tr.imag).pow(2)
    #     else:
    #         trace_map = None

    #     # Unitarity & det penalties are computed on the *raw* prediction (pre-projection),
    #     # so the network learns to be unitary before the external projection.
    #     I = torch.eye(3, device=Uh_raw.device, dtype=Uh_raw.dtype)
    #     UU = Uh_raw.conj().transpose(-2, -1) @ Uh_raw
    #     unit_map = (UU - I).abs().pow(2)             # [B,H,W,3,3]

    #     det_map = None
    #     if self.w_det != 0.0:
    #         det_map = (torch.linalg.det(Uh_raw).abs() - 1.0).pow(2)  # [B,H,W]

    #     # reducers
    #     def _reduce(t):
    #         if t is None: return torch.zeros((), device=Uh.device)
    #         if t.dim() == 5:    # [B,H,W,3,3] -> sum over matrix, mean over H,W
    #             t = t.sum(dim=(-1,-2))
    #         return t.mean(dim=(1,2))  # [B]

    #     if reduction == "none":
    #         return frob_map, trace_map, unit_map, det_map, Uh, U, Uh_raw
    #     else:
    #         frob     = _reduce(frob_map)        # [B]
    #         trace_ms = _reduce(trace_map)       # [B]
    #         unit     = _reduce(unit_map)        # [B]
    #         det      = _reduce(det_map)         # [B]
    #         return frob, trace_ms, unit, det, Uh, U, Uh_raw


    @staticmethod
    def _log_unitary(U: torch.Tensor, *, proj_iters: int = 1, log_dtype=torch.complex64) -> torch.Tensor:
        # project + eig in complex64 unless you *know* you need c128
        Uc = GroupLoss._su3_project(U.to(log_dtype), iters=proj_iters)
        w, V = torch.linalg.eig(Uc)             # complex64 eigendecomp
        theta = torch.atan2(w.imag, w.real)
        Ldiag = 1j * theta
        L = V @ torch.diag_embed(Ldiag) @ torch.linalg.inv(V)
        return 0.5 * (L - L.conj().transpose(-1, -2))  # anti-Hermitian

    # @staticmethod
    # def _log_unitary(U: 'torch.Tensor') -> 'torch.Tensor':
    #     """
    #     Matrix logarithm for (near-)unitary 3x3 via eigendecomposition.
    #     Returns an anti-Hermitian generator in the principal branch.
    #     """
    #     # promote & lightly project to unitary for stability
    #     Uc = GroupLoss._su3_project(U.to(torch.complex128))
    #     # eigendecomposition: U = V diag(e^{iθ}) V^{-1}

    #     w, V = torch.linalg.eig(Uc)                         # (...,3), (...,3,3)
        
    #     theta = torch.atan2(w.imag, w.real)                 # principal angles in (-π, π]
    #     Ldiag = 1j * theta                                  # diag entries of log(U)
    #     L = V @ torch.diag_embed(Ldiag) @ torch.linalg.inv(V)
    #     # enforce anti-Hermitian numerically
    #     L = 0.5 * (L - L.conj().transpose(-1, -2))
    #     return L.to(U.dtype)


    @staticmethod
    def _dir_cosine_lie(U0: 'torch.Tensor', Ut: 'torch.Tensor', Up: 'torch.Tensor') -> 'torch.Tensor':
        """
        Cosine similarity between Lie generators per lattice site:
          Lt = log(Ut U0†), Lp = log(Up U0†).
        Mean over sites with ||Lt|| > tiny.
        Shapes: [B,H,W,3,3] complex (near SU(3))
        """
        U0h = U0.conj().transpose(-1, -2)
        Gt  = Ut @ U0h
        Gp  = Up @ U0h
        Lt  = GroupLoss._log_unitary(Gt)   # [B,H,W,3,3] anti-Hermitian
        Lp  = GroupLoss._log_unitary(Gp)

        ip = (Lt.conj() * Lp).real.sum(dim=(-1,-2))                 # [B,H,W]
        nt = (Lt.conj() * Lt).real.sum(dim=(-1,-2)).sqrt()
        np = (Lp.conj() * Lp).real.sum(dim=(-1,-2)).sqrt()

        cos_site = ip / (nt * np + 1e-8)
        mask = nt > 1e-7
        return cos_site[mask].mean() if mask.any() else cos_site.new_tensor(0.0)

    @staticmethod
    def _dir_cosine_global(U0: torch.Tensor, Ut: torch.Tensor, Up: torch.Tensor) -> torch.Tensor:
        """
        Cosine between global update vectors: vec(Ut-U0) and vec(Up-U0).
        Works on raw complex entries (Frobenius inner product), flattened over H,W,3,3.
        """
        d_true = (Ut - U0).reshape(U0.shape[0], -1)   # [B, HW*9*2] as complex flattened
        d_pred = (Up - U0).reshape(U0.shape[0], -1)

        # Frobenius inner product -> Re(⟨d_true, d_pred⟩) / (||d_true||·||d_pred||)
        num = (d_true.conj() * d_pred).real.sum(dim=1)
        den = d_true.norm(dim=1) * d_pred.norm(dim=1) + 1e-8
        cos = num / den
        return cos.mean()

    def _direction_loss(self, U0, U_true, U_pred):
        # Drive cosine → 1 (not just above a tiny margin)
        cos = self._dir_cosine_global(U0, U_true, U_pred)      # scalar
        return (1.0 - cos)                                     # scalar (mean over batch already)
   
    def forward(self, yhat: 'torch.Tensor', y: 'torch.Tensor', base18: 'torch.Tensor' | None = None) -> 'torch.Tensor':
        """
        If base18 is provided ([B,18,H,W]), the direction-alignment term is enabled.
        """
        frob, trace_mse, unit, det, Uh, U, Uh_raw = self._components(yhat, y, reduction="mean")  # [B], [B], ...

        # main components
        total = self.w_frob * frob + self.w_trace * trace_mse + self.w_unit * unit + self.w_det * det

        # geodesic (SU(3)) term
        if self.geo_weight != 0.0:
#            geo = self._geo_theta2(U, Uh) * self.geo_weight
            geo = self.geo_lowpass_theta2_from18(U, Uh) * self.geo_weight
            total = total + geo

        if self.dir_weight != 0.0 and base18 is not None:
            U0 = self._pack18_to_U(base18)
            dir_loss = self._direction_loss(U0, U, Uh)
            total = total + self.dir_weight * dir_loss

        return total.mean().float()

    @torch.no_grad()
    def components_mean(self, yhat: 'torch.Tensor', y: 'torch.Tensor', base18: 'torch.Tensor' | None = None):
        """
        Backward-compatible: returns the classic 4-tuple (frob, trace_mse, unit, det).
        Use components_dict(...) below for extended metrics.
        """
        f, tr, un, de, Uh, U, Uh_raw = self._components(yhat, y, reduction="mean")
        return float(f.mean()), float(tr.mean()), float(un.mean()), float(de.mean())

    @torch.no_grad()
    def components_dict(self, yhat: 'torch.Tensor', y: 'torch.Tensor', base18: 'torch.Tensor' | None = None):
        """
        New: returns a dict with the original 4 plus any enabled extended metrics.
        """
        f, tr, un, de, Uh, U, Uh_raw = self._components(yhat, y, reduction="mean")
        out = {
            "frob": float(f.mean()),
            "trace_mse": float(tr.mean()),
            "unit": float(un.mean()),
            "det": float(de.mean()),
        }
        if self.geo_weight != 0.0:
            out["geo_theta2"] = float(self._geo_theta2(U, Uh))
        if self.dir_weight != 0.0 and base18 is not None:
            U0 = self._pack18_to_U(base18)
            out["dir_cosine"] =  float(self._dir_cosine_lie(U0, U, Uh))
        return out


# ==================== Qs-augmented loss (dipole + Q_s) ====================
# This subclass extends GroupLoss with a gauge-invariant dipole- and Q_s-based loss.
# Usage (inside train() where criterion is created):
#     criterion = GroupLossWithQs(Sread_b
#         w_frob=0.0, w_unit=0.0,       # (optional) turn off matrix-space losses
#         dipole_weight=1.0,            # match the dipole correlator S(r)
#         dipole_offsets=((1,0),(2,0),(4,0),(0,1),(0,2),(0,4)),
#         qs_weight=1.0,                # match the saturation scale Q_s extracted from S(r)
#         qs_threshold=0.5,             # solve for r* where N(r*)=threshold (N=1-S)
#         qs_on='N',                    # compute threshold on N (1-S) or directly on S
#         qs_local=True                 # if True, compute site-wise Q_s(x,y); else global per image
#     )
#
# Notes:
# - Q_s is computed by finding the (interpolated) radius r* where N(r) or S(r) crosses `qs_threshold`.
#   Then Q_s = 1 / r* (optionally scaled by `qs_scale`, default 1.0).
# - Requires at least two distinct |r| in `dipole_offsets`.
# - The dipole pieces are computed on the SU(3)-projected links for robustness.
#
class GroupLossWithQs(GroupLoss):
    def __init__(
        self,
        *,
        w_frob: float = 1.0,
        w_trace: float = 0.0,
        w_unit: float = 0.1,
        w_det: float = 0.0,
        project_before_frob: bool = True,
        # existing extended terms
        geo_weight: float = 0.0,
        dir_weight: float = 0.0,
        dipole_weight: float = 0.0,
        dipole_offsets=((1,0),(0,1),(2,0),(0,2)),
        # new Q_s controls
        qs_weight: float = 0.0,
        qs_threshold: float = 0.5,
        qs_on: str = "N",
        qs_local: bool = False,
        qs_scale: float = 1.0,
        # higher-order
        quad_weight: float = 0.0,
        quad_pairs=(((1,0),(0,1)), ((2,0),(0,2))),
        rough_weight: float = 0.0,
        # Y-evolution observables
        mono_weight: float = 0.0,          # enforce N(r,Y_b) >= N(r,Y_a)
        qs_slope_weight: float = 0.0,      # enforce d/dY ln Qs^2 slope wrt pairs
        qs_soft_beta: float = 0.,
        qs_soft_slope: float = 1.,
        nll_weight: float = 0.0,
        kmin: float = 4,
        kmax: float = None,
        current_epoch: int = None,
        energy_weight: float = 0.,
        moment_weight: float = 0.0
    ):
        super().__init__(
            w_frob=w_frob, w_trace=w_trace, w_unit=w_unit, w_det=w_det,
            project_before_frob=project_before_frob,
            geo_weight=geo_weight, dir_weight=dir_weight,
            dipole_weight=dipole_weight, dipole_offsets=dipole_offsets,
        )
        self.dipole_weight = float(dipole_weight)
        self.qs_weight = float(qs_weight)
        self.qs_threshold = float(qs_threshold)
        self.qs_on = str(qs_on).upper()  # 'N' or 'S'
        assert self.qs_on in ("N", "S"), "qs_on must be 'N' or 'S'"
        self.qs_local = bool(qs_local)
        self.qs_scale = float(qs_scale)
        # higher-order
        self.quad_weight  = float(quad_weight)
        self.quad_pairs   = tuple((tuple(map(int, a)), tuple(map(int, b))) for (a,b) in quad_pairs)
        self.rough_weight = float(rough_weight)   # new arg with default 0.0
        self.mono_weight = float(mono_weight)
        self.qs_slope_weight = float(qs_slope_weight)

        self.qs_soft_beta = float(locals().get("qs_soft_beta", 0.0))
        self.qs_soft_slope = float(locals().get("qs_soft_slope", 1.0))
        self.nll_weight = float(nll_weight)
        self.nll_target_mode = str(locals().get("nll_target_mode", "none"))
        self.energy_weight = float(energy_weight)

        self.current_epoch = int(current_epoch)
        self.kmin = int(kmin)                     # e.g., 4
        self.kmax = int(kmax) if kmax is not None else None

        # === feature flags (set from train() via setattr) ===
        self.use_fullcov   = getattr(self, 'use_fullcov', False)
        self.nll_compose   = getattr(self, 'nll_compose', 'brownian')
        self.bch_weight    = getattr(self, 'bch_weight', 0.0)
        self.energy_stride = getattr(self, 'energy_stride', 4)
        # Optional: uncertainty-based balancing
        self.auto_balance = getattr(self, 'auto_balance', False)
        if self.auto_balance:
            # Register fixed keys so optimizer sees them from the start
            self.log_scales = nn.ParameterDict({
                'nll': nn.Parameter(torch.zeros(())),
                'bch': nn.Parameter(torch.zeros(())),
                'energy': nn.Parameter(torch.zeros(())),
                # You can add others similarly
            })
        self._lams_cached = None
        self.nll_stride = 2
        self.I3 = torch.eye(3, dtype=torch.cfloat)
        self.moment_weight = float(moment_weight)        
        print("[loss cfg] dipole_w=", self.dipole_weight, "qs_w=", self.qs_weight, "tr_w=", self.w_trace,
              "geo_w=", self.geo_weight, "quad_weight=", self.quad_weight, "moment_weight=", self.moment_weight, "qs_slope_w=", self.qs_slope_weight, "nll_weight=", self.nll_weight, "energy_weight=", self.energy_weight)


        
     
        # --- Gell-Mann (λ/2) basis for projection (complex Hermitian) ---
        L = torch.zeros(8, 3, 3, dtype=torch.cfloat)
        L[0,0,1] = L[0,1,0] = 1.0
        L[1,0,1] = 1j; L[1,1,0] = -1j
        L[2,0,0] = 1.0; L[2,1,1] = -1.0
        L[3,0,2] = L[3,2,0] = 1.0
        L[4,0,2] = 1j; L[4,2,0]  = -1j
        L[5,1,2] = L[5,2,1] = 1.0
        L[6,1,2] = 1j; L[6,2,1]  = -1j
        L[7,0,0] = 1.0/np.sqrt(3); L[7,1,1] = 1.0/np.sqrt(3); L[7,2,2] = -2.0/np.sqrt(3)
        self.register_buffer("lambdas", L / 2.0, persistent=False)
    #---- helpers --------------------------------------------------------

    def _wb(self, name: str, loss_value: torch.Tensor, base_weight: float) -> torch.Tensor:
        """(internal) apply optional uncertainty-based balancing to a term."""
        if not getattr(self, 'auto_balance', False):
            return base_weight * loss_value
        s = self.log_scales[name]
        return torch.exp(-2*s) * (base_weight * loss_value) + s

    def _radial_power(self, U: torch.Tensor) -> torch.Tensor:
        # [B,H,W,3,3] -> [B,L] radial mean of |FFT|^2 over 9 channels
        B,H,W = U.shape[:3]
        F = U.reshape(B,H,W,9).permute(0,3,1,2).contiguous().to(torch.complex64)
        Fk = torch.fft.fft2(F)
        P  = (Fk.conj()*Fk).real.sum(dim=1) / (H*W*3.0)     # [B,H,W]
        bins, counts = self._radial_bins(H, W, U.device)    # existing cache util
        P_flat = P.view(B, -1)
        sums = torch.zeros(B, counts.numel(), device=U.device, dtype=P.dtype)
        sums.index_add_(1, bins.view(-1), P_flat)
        Pr = sums / counts.clamp_min(1).to(sums.dtype)      # [B,L]
        return Pr[:, 1:]  # drop r=0

    @staticmethod
    def _radial_bins(H: int, W: int, device):
        key = (H, W, device)
        if key in _RADIAL_CACHE:
            return _RADIAL_CACHE[key]
        # minimal wrapped distances on a torus
        dx = torch.arange(W, device=device)
        dy = torch.arange(H, device=device)
        dx = torch.minimum(dx, W - dx)[None, :].expand(H, W)        # [H,W]
        dy = torch.minimum(dy, H - dy)[:, None].expand(H, W)        # [H,W]
        r  = torch.sqrt(dx.to(torch.float32)**2 + dy.to(torch.float32)**2)  # [H,W]
        bins = torch.round(r).to(torch.int64)                        # integer radius bins
        L = int(round(math.hypot(W // 2, H // 2))) + 1
        counts = torch.bincount(bins.view(-1), minlength=L).to(torch.int64)
        _RADIAL_CACHE[key] = (bins, counts)
        return bins, counts

    def _dipole_curve(self, U: 'torch.Tensor', local: bool, assume_su3: bool=False):
        """Compute S(r) for each axial offset in self.dip_offsets.
        Returns (r: [K], S: [B,K] if global else [B,H,W,K]).
        """
        if not assume_su3:
            U = self._su3_project(U)  # robust, gauge-invariant inputs
        B, H, W = U.shape[:3]
        S_list = []
        rs = []
        for dx, dy in self.dip_offsets:
            Us = torch.roll(U, shifts=(dy, dx), dims=(1, 2))
            prod = U @ Us.conj().transpose(-1, -2)
            Spr = torch.diagonal(prod, dim1=-2, dim2=-1).sum(-1).real / 3.0  # [B,H,W]
            S_list.append(Spr if local else Spr.mean(dim=(1,2)))             # [B,H,W] or [B]
            rs.append((dx*dx + dy*dy) ** 0.5)
        S = torch.stack(S_list, dim=-1)  # [..., K]
        r = torch.tensor(rs, device=U.device, dtype=S.dtype)  # [K]
        # sort by increasing r
        idx = torch.argsort(r)
        r = r.index_select(0, idx)
        S = S.index_select(-1, idx)
        return r, S

    def quad_mse_streaming(self, Uh_q, U_q, quad_pairs, *, stride=2, sample_sites=4096):
        Uh = Uh_q[:, ::stride, ::stride] if stride > 1 else Uh_q
        Ut = U_q [:, ::stride, ::stride] if stride > 1 else U_q

        sel = None
        if sample_sites is not None:
            H, W = Uh.shape[1:3]
            ys = torch.randint(0, H, (sample_sites,), device=Uh.device)
            xs = torch.randint(0, W, (sample_sites,), device=Uh.device)
            sel = (slice(None), ys, xs)

        B = Uh.shape[0]
        # IMPORTANT: real accumulator (don’t use Uh.new_zeros(...))
        sumsq = torch.zeros(B, device=Uh.device, dtype=Uh.real.dtype)
        n = 0
        for (dx1,dy1),(dx2,dy2) in quad_pairs:
            qh = _q4_scalar(Uh, dx1, dy1, dx2, dy2)      # [B,H,W] real
            if sel is not None: qh = qh[sel]
            qh = qh.mean(dim=(-1)) if sel else qh.mean(dim=(1,2))  # [B] real

            with torch.no_grad():
                qt = _q4_scalar(Ut, dx1, dy1, dx2, dy2)      # [B,H,W] real
                if sel is not None: qt = qt[sel]
                qt = qt.mean(dim=(-1)) if sel else qt.mean(dim=(1,2))  # [B] real

            d = (qh - qt).to(dtype=sumsq.dtype)  # keep it real
            sumsq += d * d
            n += 1

        loss_quad = (sumsq / max(n, 1)).mean()  # real scalar
        return loss_quad


   
    def _dipole_loss(self,
                     Uh: torch.Tensor,    # [B,H,W,3,3] complex
                     U:  torch.Tensor,    # [B,H,W,3,3] complex
                     *,
                     local: bool = False,         # ignored (radial avg is global)
                     use_logN: bool = True,
                     per_radius_norm: bool = True,
                     detach_target: bool = True) -> torch.Tensor:
        """
        Radial-averaged dipole loss on the lattice torus using FFT autocorrelation.
        Compares N(r)=1-S(r) between prediction and target over *radial* bins.
        - use_logN=True  -> MSE in log-space (helps dynamic range).
        - per_radius_norm=True -> equalize contribution of each radius.
        - detach_target=True -> no grads through target branch.
        Returns scalar loss tensor.
        """
        del local  # radial average is global by construction

        assert Uh.dtype.is_complex and U.dtype.is_complex, "Uh/U must be complex link fields"
        B, H, W = Uh.shape[0], Uh.shape[1], Uh.shape[2]
        device = Uh.device

        # --- radial bins & counts (cached) ---
        bins, counts = self._radial_bins(H, W, device)  # [H,W], [L]
        L = counts.numel()

        def _radial_curve(Uc: torch.Tensor, grad: bool) -> torch.Tensor:
            """
            Uc: [B,H,W,3,3] complex -> S_rad: [B, L-1] (r=0 dropped)
            Uses FFT autocorr over the 9 complex channels (fundamental rep).
            """
            # reshape to [B,9,H,W]
            F = Uc.reshape(B, H, W, 9).permute(0, 3, 1, 2).contiguous()
            ctx = (torch.enable_grad() if grad else torch.no_grad())
            with ctx:
                Fk   = torch.fft.fft2(F)                      # [B,9,H,W] complex
                corr = torch.fft.ifft2(Fk.conj() * Fk).real   # [B,9,H,W] real
                corr = corr.sum(dim=1)                        # sum over 9 -> [B,H,W]
                corr = corr / (3.0 * H * W)                   # normalize so S(0)~1

                # bin-mean over wrapped radii
                corr_flat = corr.view(B, -1)                  # [B,H*W]
                bin_ids   = bins.view(-1)                     # [H*W]
                sums = torch.zeros(B, L, device=device, dtype=corr.dtype)
                sums.index_add_(1, bin_ids, corr_flat)        # [B,L]
                S_rad = sums / counts.clamp_min(1).to(sums.dtype)  # [B,L]

                # drop r=0 bin (always ~1)
                if L > 1:
                    S_rad = S_rad[:, 1:]
                return S_rad

        # Prediction path must keep grads
        S_pred = _radial_curve(Uh, grad=True)     # [B, L']
        # Target path can be detached for memory/speed
        S_true = _radial_curve(U,  grad=not detach_target)

        # N = 1 - S
        N_pred = 1.0 - S_pred
        N_true = 1.0 - S_true

        # Per-radius normalization (equalize scale across r)
        if per_radius_norm:
            # reduce over batch: weights w[r] ~ 1 / mean|N_true[r]|
            w = 1.0 / N_true.abs().mean(dim=0).clamp_min(1e-3)   # [L']
            N_pred = N_pred * w[None, :]
            N_true = N_true * w[None, :]

        # Log-space (stabilize tails / small-N region)
        if use_logN:
            N_pred = torch.log(N_pred.clamp_min(1e-6))
            N_true = torch.log(N_true.clamp_min(1e-6))

        return F.mse_loss(N_pred, N_true)



    @torch.no_grad()
    def _make_base_grid(H, W, device, dtype):
        # pixel coordinates (0..W-1, 0..H-1)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        # cached for speed; returned so caller can reuse
        return yy, xx

    def _sample_shift_periodic(field, dx, dy, base=None):
        """
        Periodic bilinear sampler: returns field(x+dx, y+dy) with subpixel dx,dy (floats, in pixels).
        field: (B, C, H, W)
        """
        B, C, H, W = field.shape
        dtype, device = field.dtype, field.device
        yy, xx = base if base is not None else _make_base_grid(H, W, device, dtype)

        # shift in pixel coords, wrap to [0, W/H)
        x = (xx + dx) % W
        y = (yy + dy) % H

        # map to normalized grid [-1, 1] with align_corners=True
        gx = 2.0 * x / (W - 1) - 1.0
        gy = 2.0 * y / (H - 1) - 1.0
        grid = torch.stack((gx, gy), dim=-1).expand(B, H, W, 2)

        return F.grid_sample(
            field, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )


    def _isotropic_dipole_loss(self,
                     Uh: torch.Tensor,  # [B,H,W,3,3] complex
                     U:  torch.Tensor,  # [B,H,W,3,3] complex
                     *,
                     local: bool = False,
                     use_logN: bool = True,
                     per_radius_norm: bool = True,
                     detach_target: bool = True) -> torch.Tensor:
        """
        Radial-averaged dipole loss on the lattice torus using FFT autocorrelation.
        Compares N(r)=1-S(r) between prediction and target over radial bins.
        """
        del local
        assert Uh.dtype.is_complex and U.dtype.is_complex
        B, H, W = Uh.shape[0], Uh.shape[1], Uh.shape[2]
        device = Uh.device

        bins, counts = self._radial_bins(H, W, device)  # [H,W], [L]
        L = counts.numel()

        def _radial_curve(Uc: torch.Tensor, needs_grad: bool) -> torch.Tensor:
            # Uc: [B,H,W,3,3] -> S_rad: [B, L-1] (drop r=0)
            F = Uc.reshape(B, H, W, 9).permute(0, 3, 1, 2).contiguous()  # [B,9,H,W]
            ctx = (torch.enable_grad() if needs_grad else torch.no_grad())
            with ctx:
                Fk   = torch.fft.fft2(F)                    # [B,9,H,W] complex
                corr = torch.fft.ifft2(Fk.conj() * Fk).real # [B,9,H,W] real
                corr = corr.sum(dim=1)                      # sum 9 ch -> [B,H,W]
                corr = corr / (3.0 * H * W)                 # S(0)~1

                corr_flat = corr.view(B, -1)                # [B,H*W]
                bin_ids   = bins.view(-1)                   # [H*W]
                sums = torch.zeros(B, L, device=device, dtype=corr.dtype)
                sums.index_add_(1, bin_ids, corr_flat)      # [B,L]
                S_rad = sums / counts.clamp_min(1).to(sums.dtype)
                if L > 1: S_rad = S_rad[:, 1:]              # drop r=0
                return S_rad

        S_pred = _radial_curve(Uh, needs_grad=True)
        S_true = _radial_curve(U,  needs_grad=not detach_target)

        N_pred = 1.0 - S_pred
        N_true = 1.0 - S_true

        # --- per-radius equalization (RMS version, more stable at small-signal r) ---
        if per_radius_norm:
            # w[r] ~ 1 / sqrt(E[N_true^2])[r], detach so weights don't chase the target noise
            w = 1.0 / torch.sqrt((N_true**2).mean(dim=0).clamp_min(1e-8))
            w = (w / w.mean()).detach()          # normalize so mean weight = 1
            N_pred = N_pred * w[None, :]
            N_true = N_true * w[None, :]

        if use_logN:
            N_pred = torch.log(N_pred.clamp_min(1e-6))
            N_true = torch.log(N_true.clamp_min(1e-6))

        # Robust to large-ΔY tails
        return F.smooth_l1_loss(N_pred, N_true, beta=0.02)


    
    # def _isotropic_dipole_loss(self,
    #                  Uh: torch.Tensor,    # [B,H,W,3,3] complex
    #                  U:  torch.Tensor,    # [B,H,W,3,3] complex
    #                  *,
    #                  local: bool = False,         # ignored (radial avg is global)
    #                  use_logN: bool = True,
    #                  per_radius_norm: bool = True,
    #                  detach_target: bool = True) -> torch.Tensor:
    #     """
    #     Isotropic (radially-averaged) dipole loss on the lattice torus via FFT autocorrelation.
    #     Compares N(r) = 1 - S(r) between prediction Uh and target U over radial bins.

    #     - use_logN=True            -> MSE in log-space (stabilizes dynamic range)
    #     - per_radius_norm=True     -> equalize contribution of each radius
    #     - detach_target=True       -> no grads through target branch
    #     Returns a scalar loss tensor.
    #     """
    #     del local  # radial average is global by construction

    #     assert Uh.dtype.is_complex and U.dtype.is_complex, "Uh/U must be complex link fields"
    #     B, H, W = Uh.shape[0], Uh.shape[1], Uh.shape[2]
    #     device = Uh.device

    #     # --- radial bins & counts (cached elsewhere in your class) ---
    #     bins, counts = self._radial_bins(H, W, device)  # bins: [H,W] (long), counts: [L] (long)
    #     L = counts.numel()

    #     def _radial_curve(Uc: torch.Tensor, need_grad: bool) -> torch.Tensor:
    #         """
    #         Uc: [B,H,W,3,3] complex -> S_rad: [B, L-1] (r=0 dropped)
    #         Uses FFT autocorr over the 9 complex channels (fundamental rep),
    #         averaging over all translations and directions on the torus.
    #         """
    #         # reshape to [B,9,H,W]
    #         F9 = Uc.reshape(B, H, W, 9).permute(0, 3, 1, 2).contiguous()

    #         ctx = (torch.enable_grad() if need_grad else torch.no_grad())
    #         with ctx:
    #             Fk   = torch.fft.fft2(F9)                    # [B,9,H,W] complex
    #             corr = torch.fft.ifft2(Fk.conj() * Fk).real  # [B,9,H,W] real (autocorr per channel)
    #             corr = corr.sum(dim=1)                       # sum over 9 -> [B,H,W]
    #             corr = corr / (3.0 * H * W)                  # normalize so S(0) ~ 1

    #             # radial bin-mean (periodic distances)
    #             corr_flat = corr.view(B, -1)                 # [B,H*W]
    #             bin_ids   = bins.view(-1)                    # [H*W]
    #             sums = torch.zeros(B, L, device=device, dtype=corr.dtype)
    #             sums.index_add_(1, bin_ids, corr_flat)       # [B,L]
    #             S_rad = sums / counts.clamp_min(1).to(sums.dtype)  # [B,L]

    #             # drop r=0 bin (always ~1)
    #             return S_rad[:, 1:] if L > 1 else S_rad[:, :0]

    #     # Prediction path keeps grads; target can be detached
    #     S_pred = _radial_curve(Uh, need_grad=True)                 # [B, L']
    #     S_true = _radial_curve(U,  need_grad=not detach_target)    # [B, L']

    #     # N = 1 - S
    #     N_pred = 1.0 - S_pred
    #     N_true = 1.0 - S_true

    #     # Per-radius normalization (equalize scale across r)
    #     if per_radius_norm and N_true.numel() > 0:
    #         w = 1.0 / N_true.abs().mean(dim=0).clamp_min(1e-3)     # [L']
    #         N_pred = N_pred * w[None, :]
    #         N_true = N_true * w[None, :]

    #     # Log-space (stabilize tails / small-N region)
    #     if use_logN and N_true.numel() > 0:
    #         eps = 1e-6
    #         N_pred = torch.log(N_pred.clamp_min(eps))
    #         N_true = torch.log(N_true.clamp_min(eps))

    #     return F.mse_loss(N_pred, N_true)



    
    def _compute_Qs_from_U(self, U: 'torch.Tensor', *, local: bool):
        r, S = self._dipole_curve(U, local=local, assume_su3=True)  # r:[K], S:[B,K] or [B,H,W,K]
        # Decide curve to threshold: N(r)=1-S(r) or S(r)
        X = 1.0 - S if self.qs_on == "N" else S
        thr = self.qs_threshold
        if getattr(self, "qs_soft_beta", 0.0) > 0.0:
            # soft, differentiable estimate of r* where X(r)=thr
            dX = X.diff(dim=-1, prepend=X[..., :1]).abs()
            scores = -self.qs_soft_beta * (X - thr).abs()
            if getattr(self, "qs_soft_slope", 1.0) != 0.0:
                eps = 1e-12
                scores = scores + self.qs_soft_slope * torch.log(dX.clamp_min(eps))
                w = torch.softmax(scores, dim=-1)        # [..., K]
                r_star = (w * r).sum(dim=-1)         # [...]
                return 1.0 / r_star.clamp_min(eps)

        # Make X monotone non-decreasing along r to stabilize crossing
        X_mono, _ = torch.cummax(X, dim=-1)  # [...,K]
        target = torch.tensor(self.qs_threshold, device=X.device, dtype=X.dtype)

        # Count how many points are strictly below target -> index of first >= target
        below = (X_mono < target).to(torch.int64)     # [...,K]
        idx_hi = below.sum(dim=-1)                    # [...]
        K = X_mono.shape[-1]
        idx_hi = idx_hi.clamp(min=0, max=K-1)
        idx_lo = (idx_hi - 1).clamp(min=0, max=K-1)

        # Broadcast r to shape of X_mono
        r_b = r.view(*((1,)*(X_mono.ndim-1)), -1).expand_as(X_mono)

        def gather_last(t, idx):
            return torch.take_along_dim(t, idx.unsqueeze(-1), dim=-1).squeeze(-1)

        X_lo = gather_last(X_mono, idx_lo)
        X_hi = gather_last(X_mono, idx_hi)
        r_lo = gather_last(r_b, idx_lo)
        r_hi = gather_last(r_b, idx_hi)

        denom = (X_hi - X_lo).abs()
        alpha = torch.where(denom > 1e-8, (target - X_lo) / (denom + 1e-12), torch.zeros_like(denom))
        r_star = r_lo + alpha * (r_hi - r_lo)

        Qs = self.qs_scale / (r_star + 1e-8)
        return Qs  # shape: [B] or [B,H,W]



    def _quadrupole(self, U: torch.Tensor) -> torch.Tensor:
        U_q = U if self.project_before_frob else self._su3_project(U)  # [B,H,W,3,3] complex
        outs = []
        for (dx1, dy1), (dx2, dy2) in self.quad_pairs:
            q4 = _q4_scalar(U_q, dx1, dy1, dx2, dy2)      # [B,H,W]
            outs.append(q4.mean(dim=(1, 2)))              # [B]
        return torch.stack(outs, dim=-1) if outs else U_q.new_zeros((U_q.shape[0], 0))

    
    # def _quadrupole(self, U: 'torch.Tensor') -> 'torch.Tensor':
    #     if self.project_before_frob:
    #         U_q = U
    #     else:
    #         U_q  = self._su3_project(U)       # U here is raw truth when pre-proj is False
    #     outs = []
    #     for (dx1, dy1), (dx2, dy2) in self.quad_pairs:
    #         Ux = U_q
    #         Uy = torch.roll(U_q, shifts=(dy1, dx1), dims=(1, 2))
    #         Uu = torch.roll(U_q, shifts=(dy2, dx2), dims=(1, 2))
    #         Uv = torch.roll(U_q, shifts=(dy1 + dy2, dx1 + dx2), dims=(1, 2))
    #         prod = Ux @ Uy.conj().transpose(-1, -2) @ Uu @ Uv.conj().transpose(-1, -2)
    #         s4 = torch.diagonal(prod, dim1=-2, dim2=-1).sum(-1).real / 3.0   # [B,H,W]
    #         outs.append(s4.mean(dim=(1, 2)))
    #     return torch.stack(outs, dim=-1) if outs else U_q.new_zeros((U_q.shape[0], 0))

    def _get_lams(self, ref: torch.Tensor):
        if (self._lams_cached is None or
            self._lams_cached.device != ref.device or
            self._lams_cached.dtype  != ref.dtype):
            self._lams_cached = self.lambdas.to(ref.device, ref.dtype)
        return self._lams_cached
        
    def _alpha_map_from_pair(self, U0, U1, lams, *, stride=1, fast_thresh=0.15, proj_iters=1):
        """Return a_true map: [B,C,H,W]. Fast path avoids per-pixel eig when Δ≈I."""
        # optional spatial subsample before heavy math
        if stride > 1:
            U0 = U0[:, ::stride, ::stride]
            U1 = U1[:, ::stride, ::stride]

        Delta = U1.mH.contiguous() @ U0                      # [B,H',W',3,3]
        I = self.I3.to(Delta.device, Delta.dtype)

        # Fast Hermitian generator (no eig)
        S = 0.5j * (Delta.mH.contiguous() - Delta)           # [B,H',W',3,3]

        # Mask sites where Δ is not close to I (Frobenius norm per site)
        # normalize by sqrt(3) to keep threshold in ~[0,1]
        frob = torch.linalg.matrix_norm(Delta - I, ord='fro', dim=(-2,-1)) / (3.0**0.5)
        mask = (frob > fast_thresh)

        if mask.any():
            Dv = Delta.reshape(-1, 3, 3)
            mv = mask.reshape(-1)
            De = Dv[mv]
            # light projection for stability, complex64 eig is faster
            De = self._su3_project(De, iters=proj_iters)
            w, V = torch.linalg.eig(De.to(torch.complex64))
            theta = torch.atan2(w.imag, w.real)
            L = V @ torch.diag_embed(1j * theta) @ torch.linalg.inv(V)
            L = L.contiguous(); Lh = L.mH.contiguous()
            S_exact = 0.5 * (L - Lh)                          # Hermitian
            S = S.reshape(-1, 3, 3); S[mv] = S_exact; S = S.reshape(Delta.shape)

        # α projection (keep it real32)
        a = 2.0 * torch.real(torch.einsum('bhwij,aji->bahw', S, lams))
        return a

    # ---- overrides ------------------------------------------------------
    def forward(self, yhat: 'torch.Tensor', y: 'torch.Tensor',
                base18: 'torch.Tensor' | None = None,
                dy_scalar: 'torch.Tensor' | None = None,
                theta=None, Y_final=None,
                mu_pred: 'torch.Tensor' | None = None,
                logsig_pred: 'torch.Tensor' | None = None,
                drift_pred: 'torch.Tensor' | None = None,
                amp_pred: 'torch.Tensor' | None = None,
                y_gate: 'torch.Tensor' | None = None,
                return_stats: bool = False):

        frob, trace_ms, unit, det, Uh, U, Uh_raw = self._components(yhat, y, reduction="mean")

        reduction = "mean"
        
        frob     = _reduce(frob,     reduction)
        trace_ms = _reduce(trace_ms, reduction)
        unit     = _reduce(unit,     reduction)
        det      = _reduce(det,      reduction)


        # 2) Base loss (same weights you use in the base class)
        total = (self.w_frob * frob +
                 self.w_trace * trace_ms +
                 self.w_unit * unit +
                 self.w_det * det)

        stats = {"frob":frob, "trace":trace_ms, "unit":unit, "det":det}

        if self.qs_weight != 0.0 or self.dipole_weight != 0.0:
            if self.project_before_frob:
                Uh_q, U_q = Uh, U
            else:
                # Uh is raw in this branch; you also want U to be SU(3)
                Uh_q = self._su3_project(Uh_raw)  # raw prediction from _components
                U_q  = self._su3_project(U)       # U here is raw truth when pre-proj is False

        if self.qs_weight != 0.0 and len(self.dip_offsets) >= 2:
            assert self.dip_offsets and len(self.dip_offsets) > 0, "dipole_offsets is empty!"
            Qh = self._compute_Qs_from_U(Uh_q, local=self.qs_local)
            Qt = self._compute_Qs_from_U(U_q,  local=self.qs_local)
            qs_mse = torch.nn.functional.mse_loss(Qh, Qt)
            total = total + self.qs_weight * qs_mse
            # #----time----
            # if torch.cuda.is_available(): torch.cuda.synchronize()
            # print(f"[timing] Qs block: {(time.perf_counter()-t0)*1e3:.2f} ms")
            # #----end-time----
            stats["qs_mse"] = qs_mse.detach()

        if self.dipole_weight != 0.0 and len(self.dip_offsets) > 0:
            #----time----
            #if torch.cuda.is_available(): torch.cuda.synchronize();
            #t0=time.perf_counter()
            #----end-time----
            # dip = self._dipole_loss(
            #     Uh_q, U_q,
            #     local=False,              # start global (more stable); try True later if you need local
            #     use_logN=True,
            #     per_radius_norm=True,
            #     detach_target=True
            # )

            #total = total + self.dipole_weight * dip
            
            dip = self._isotropic_dipole_loss(
                 Uh_q, U_q,
                 local=False,              # start global (more stable); try True later if you need local
                 use_logN=True,
                 per_radius_norm=True,
                 detach_target=True
             )
            total = total + self.dipole_weight * dip

            #----time----
            #if torch.cuda.is_available(): torch.cuda.synchronize()
            #print(f"[timing] Dipole block: {(time.perf_counter()-t0)*1e3:.2f} ms")
            #----end-time----
            
        if self.quad_weight and self.quad_pairs:
            # optionally limit to K random pairs per step to cap work
            pairs = self.quad_pairs
            K = getattr(self, "quad_pairs_per_step", 8)
            if len(pairs) > K:
                idx = torch.randperm(len(pairs), device=Uh_q.device)[:K].tolist()
                pairs = [pairs[i] for i in idx]

            loss_quad = self.quad_mse_streaming(Uh_q, U_q, pairs, stride=2, sample_sites=4096)
            total += self.quad_weight * loss_quad
            stats["quad_mse"] = loss_quad.detach()

            
        # if self.quad_weight != 0.0 and len(self.quad_pairs) > 0:
        #     Q4h = self._quadrupole(Uh_q)
        #     with torch.no_grad():
        #         Q4t = self._quadrupole(U_q)
        #     loss_quad = torch.nn.functional.mse_loss(Q4h, Q4t)
        #     total = total + self.quad_weight * loss_quad
        #     stats["quad_mse"] = loss_quad.detach()
            
        if self.rough_weight != 0.0:
            Pp = self._radial_power(Uh_q)
            Pt = self._radial_power(U_q)
            k0, k1 = self.kmin, (self.kmax if self.kmax is not None else Pp.shape[1])
            loss_rough = F.mse_loss(torch.log(Pp[:, k0:k1].clamp_min(1e-12)),
                                    torch.log(Pt[:, k0:k1].clamp_min(1e-12)))
            total = total + self.rough_weight * loss_rough

        # --- MOMENT matching on α: drift & physical diffusion amplitude ---
        if getattr(self, 'moment_weight', 0.0) != 0.0:
            if base18 is None:
                raise RuntimeError("GroupLossWithQs: moment_weight>0 but base18 is None. Pass U0 (18-ch) to criterion(...).")

            # 1) Build α_true map from (U0,U1)
            with torch.no_grad():
                U0 = self._su3_project(self._pack18_to_U(base18), iters=1)
                U1 = self._su3_project(self._pack18_to_U(y),      iters=1)
                lams   = self.lambdas.to(device=U0.device, dtype=U0.dtype)
                stride = getattr(self, "nll_map_stride", 2)
                a_true_mm = self._alpha_map_from_pair(
                    U0, U1, lams, stride=stride,
                    fast_thresh=getattr(self, "nll_fast_thresh", 0.15),
                    proj_iters=1
                )  # [B,C',H',W']
                if stride > 1:
                    a_true_mm = F.interpolate(
                        a_true_mm, size=(U0.shape[1], U0.shape[2]), mode='bilinear', align_corners=False
                    )

            # 2) Align channel count to drift if needed
            if (mu_pred is not None) and (a_true_mm.shape[1] != mu_pred.shape[1]):
                a_true_mm = a_true_mm[:, :mu_pred.shape[1]]

            # 3) Choose drift/amp sources
            drift = drift_pred if (drift_pred is not None) else mu_pred
            amp   = amp_pred   if (amp_pred   is not None) else (F.softplus(logsig_pred) if (logsig_pred is not None) else None)
            if (drift is None) or (amp is None):
                # nothing to do this batch
                pass
            else:
                # === Anchor ALL shapes to the supervision batch ===
                B = a_true_mm.shape[0]                # <- anchor
                # Crop (never expand) drift/amp to B if they came from a larger forward
                if drift.shape[0] != B:
                    drift = drift[:B]
                if amp.shape[0] != B:
                    amp = amp[:B]

                # Make sure amp is [B,1,H,W]; if channelled, average to scalar amplitude
                if amp.dim() == 4 and amp.shape[1] != 1:
                    amp = amp.mean(dim=1, keepdim=True)

                # 4) Build ΔY with the SAME B
                if dy_scalar is None:
                    dY_mm = torch.ones(B, 1, 1, 1, device=a_true_mm.device, dtype=a_true_mm.dtype)
                else:
                    s = dy_scalar.to(device=a_true_mm.device, dtype=a_true_mm.dtype).squeeze()
                    if s.ndim == 0:
                        dY_mm = s.view(1,1,1,1).expand(B,1,1,1)
                    elif s.ndim == 1 and s.shape[0] == B:
                        dY_mm = s.view(B,1,1,1)
                    elif (s.numel() % B) == 0:
                        s = s.reshape(B, -1).mean(dim=1)
                        dY_mm = s.view(B,1,1,1)
                    else:
                        dY_mm = s.mean().view(1,1,1,1).expand(B,1,1,1)

                # Final sanity
                assert a_true_mm.shape[0] == drift.shape[0] == amp.shape[0] == dY_mm.shape[0], \
                    f"moment batch mismatch after alignment: α={tuple(a_true_mm.shape)}, drift={tuple(drift.shape)}, amp={tuple(amp.shape)}, dY={tuple(dY_mm.shape)}"

                # 5) Moments
                # shapes: drift:[B,C,H,W], amp:[B,1,H,W], dY_mm:[B,1,1,1], a_true_mm:[B,C,H,W]
                resid       = a_true_mm - drift * dY_mm                # [B,C,H,W]
                v_true_map  = resid.pow(2).mean(dim=1, keepdim=True)   # [B,1,H,W]
                v_pred_map  = (amp**2) * dY_mm                         # [B,1,H,W]
                eps = 1e-12

                # drift moment: keep as-is
                m1 = resid.pow(2).mean()

                # diffusion moment: scale-invariant (choose ONE of the two)
                # (recommended) LOG variance penalty:
                m2 = (torch.log(v_pred_map + eps) - torch.log(v_true_map + eps)).pow(2).mean()
                # (alt) ratio penalty:
                # m2 = (v_pred_map / (v_true_map + eps) - 1.0).pow(2).mean()

                # weights (set constants here or read from args)
                m1_w = getattr(self, "moment_m1_weight", 1.0)
                m2_w = getattr(self, "moment_m2_weight", 4.0)

      
                # stats / calibration
                mm_v_true = v_true_map.mean()
                mm_v_pred = v_pred_map.mean()
                stats['moment_m1'] = m1.detach()
                stats['moment_m2'] = m2.detach()

                # your two pieces m1, m2 already exist; v_true, v_pred are the per-batch variances you print
                # replace raw MSE in variance with a log-ratio penalty (scale-invariant):
                log_ratio = torch.log((mm_v_pred + eps) / (mm_v_true + eps))
                moment_var_term = (log_ratio ** 2).mean()   # replaces any v_pred-v_true term
                
                # (keep your first-moment term if you like, but weight it lightly)
                moment_loss = 0.25 * m1 + moment_var_term
                
                
                #loss_moment = m1_w * m1 + m2_w * m2
                total = total + float(self.moment_weight) * moment_loss

                stats['moment']    = moment_loss.detach()
                stats['mm_v_true'] = mm_v_true.detach()
                stats['mm_v_pred'] = mm_v_pred.detach()
                stats['mm_calib']  = (mm_v_pred / (mm_v_true + eps)).detach()

                # m1 = (a_true_mm - drift * dY_mm).pow(2).mean()

                # resid = (a_true_mm - drift * dY_mm)
                # v_true = resid.pow(2).mean(dim=1, keepdim=True)      # [B,1,H,W]
                # v_pred = (amp**2) * dY_mm                            # [B,1,H,W]
                # m2 = (v_true - v_pred).pow(2).mean()

                # # --- variance calibration (how well amp^2 * ΔY matches residual variance) ---
                # mm_v_true = resid.pow(2).mean()                    # scalar over B,C,H,W
                # mm_v_pred = v_pred.mean()                          # scalar over B,1,H,W
                # mm_ratio  = mm_v_pred / (mm_v_true + 1e-12)        # ~1.0 is perfect

                # # log into stats (detached so it's cheap to print)
                # stats['mm_v_true'] = mm_v_true.detach()
                # stats['mm_v_pred'] = mm_v_pred.detach()
                # stats['mm_calib']  = mm_ratio.detach()



                # loss_moment = m1 + m2
                # total = total + float(self.moment_weight) * loss_moment
                # stats['moment_m1'] = m1.detach()
                # stats['moment_m2'] = m2.detach()
                # stats['moment']    = loss_moment.detach()

            
        ##----time----
        #if torch.cuda.is_available(): torch.cuda.synchronize();
        #t0=time.perf_counter()
        ##----end-time----
        a_true = None
        # --- NLL on α (projected generator) -----------------------------------
        if (self.nll_weight != 0.0) and (base18 is not None) \
           and (mu_pred is not None) and (logsig_pred is not None):

            with torch.no_grad():
                # (D) if you already have U0/U1 earlier, reuse them; else:
                U0 = self._su3_project(self._pack18_to_U(base18), iters=1)
                U1 = self._su3_project(self._pack18_to_U(y),      iters=1)

                lams = self.lambdas.to(device=U0.device, dtype=U0.dtype)  # or cached getter

                # (A)+(B): fast log with stride + eig only on outliers
                stride = getattr(self, "nll_map_stride", 2)
                a_true = self._alpha_map_from_pair(
                    U0, U1, lams, stride=stride, fast_thresh=getattr(self, "nll_fast_thresh", 0.15), proj_iters=1
                )  # [B,C',H',W']

                if stride > 1:
                    a_true = torch.nn.functional.interpolate(
                        a_true, size=(U0.shape[1], U0.shape[2]), mode='bilinear', align_corners=False
                    )

                # (C) only as many channels as you’ll use
                C = mu_pred.shape[1]
                if a_true.shape[1] != C:
                    a_true = a_true[:, :C]  # or mirror only if you must

            sigma_floor, sigma_cap = 1e-2, 3.0
           
            sigma = (F.softplus(logsig_pred) + sigma_floor).clamp_max(sigma_cap)

            if getattr(self, "current_epoch", 0) < 3:
                sigma_eff = sigma.detach()
            else:
                sigma_eff = sigma

            resid = mu_pred - a_true
            baseline = math.log(sigma_floor) + 0.5 * math.log(2.0*math.pi)
            nll = 0.5 * (resid / sigma_eff)**2 + torch.log(sigma) + 0.5 * math.log(2.0*math.pi) - baseline
            loss_nll = nll.mean()
            total = total + self.nll_weight * loss_nll  # DO NOT negate

            

        #         if (self.nll_weight != 0.0) and (base18 is not None) \
#            and (mu_pred is not None) and (logsig_pred is not None):

#             # 0) Compute *target only* under no_grad to prevent any gradient leak
#             with torch.no_grad():
#                 # Pack & project to SU(3): U0=U(Y_a), U1=U(Y_b)
#                 U0 = self._su3_project(self._pack18_to_U(base18))
#                 U1 = self._su3_project(self._pack18_to_U(y))

#                 # 1) Orientation: anchor† @ target (anchor->target)
#                 #Delta = U0.conj().transpose(-1, -2) @ U1          # [B,H,W,3,3]
#                 Delta = U1.conj().transpose(-1,-2) @ U0

#                 # 2) Log on the group; S_H = i log(Δ) (Hermitian)
#                 L = self._log_unitary(Delta, proj_iters=1, log_dtype=torch.complex64)
#                 #L = self._log_unitary(Delta)                      # anti-Hermitian
#                 S = (1j) * L                                      # Hermitian

#                 # 3) Scale by *raw* ΔY if requested (no extra factors)
#                 mode = getattr(self, "nll_target_mode", "none")
#                 if mode == "per_dy":
#                     if dy_scalar is None:
#                         raise RuntimeError("per_dy target but dy_scalar is None.")
#                     dY = dy_scalar.view(-1,1,1,1,1).to(S.dtype)
#                     dY = torch.where(dY.abs() > 1e-12, dY, torch.ones_like(dY))
#                     S = S / dY
#                 elif mode == "per_ysigma":
#                     if y_gate is None:
#                         raise RuntimeError("per_ysigma target but y_gate is None.")
#                     g = y_gate.view(-1,1,1,1,1).to(S.dtype)
#                     g = torch.where(g.abs() > 1e-12, g, torch.ones_like(g))
#                     S = S / g
#                 # else: "none" -> no scaling

#                 # 4) α_true projection. IMPORTANT: what is self.lambdas?
#                 # If self.lambdas are T_a = λ_a/2 with tr(T_a T_b)=½ δ_ab, keep the 2.0 factor.
#                 # If they are λ_a with tr(λ_a λ_b)=2 δ_ab, DROP the 2.0 factor.
#                 lams = self._get_lams(S)   # / S_pred

#                 #lams = self.lambdas.to(dtype=S.dtype, device=S.device)   # S is on GPU
#                 a_true = 2.0 * torch.real(torch.einsum('bhwij,aji->bahw', S, lams))

# #                a_true = 2.0 * torch.real(torch.einsum('bhwij,aji->bahw', S, self.lambdas))  # [B,8,H,W]
                
#                 # Match channel count (8 or 16)
#                 C = mu_pred.shape[1]
#                 if C == 16:
#                     a_true = torch.cat([a_true, a_true], dim=1)
#                 elif C != 8:
#                     a_true = a_true[:, :C]

#             sigma_floor, sigma_cap = 1e-2, 3.0
#             sigma = (F.softplus(logsig_pred) + sigma_floor).clamp_max(sigma_cap)
                    
#             # Short warmup: stabilize μ learning before σ learns (still pure NLL)
#             if getattr(self, "current_epoch", 0) < 3:
#                 sigma_eff = sigma.detach()
#             else:
#                 sigma_eff = sigma

#             resid = mu_pred - a_true              # standard (pred - target)
#             # log N = const + 0.5 z^2 + log σ   (const dropped for grads)
#             nll = 0.5 * (resid / sigma_eff)**2 + torch.log(sigma)
#             loss_nll = nll.mean()
#             total = total + self.nll_weight * loss_nll
        ##----time----
        #if torch.cuda.is_available(): torch.cuda.synchronize()
        #print(f"[timing] NLL loss: {(time.perf_counter()-t0)*1e3:.2f} ms")
        ##----end-time----
        ##----time----
        #if torch.cuda.is_available(): torch.cuda.synchronize();
        #t0=time.perf_counter()
        ##----end-time----
            
        # ---- Monotonic N(r,Y): N_pred(Y_b) >= N_true(Y_a) over the axial radii you supervise ----
        if (self.mono_weight != 0.0) and (base18 is not None):
            U_base = self._pack18_to_U(base18)                  # [B,H,W,3,3] complex
            if self.project_before_frob:
                Uh_q, U_q = Uh, U
            else:
                Uh_q = self._su3_project(Uh_raw)
                U_q  = self._su3_project(U)
            # radial dipole values at supervised offsets (global, per-batch)
            r_offs, S_pred = self._dipole_curve(Uh_q, local=False, assume_su3=True)   # [K], [B,K]
            _,      S_base = self._dipole_curve(U_base, local=False, assume_su3=True) # [K], [B,K]
            N_pred, N_base = (1.0 - S_pred), (1.0 - S_base)
            mono_viol = torch.relu(N_base - N_pred)                   # [B,K]
            loss_mono = (mono_viol**2).mean()
            total = total + self.mono_weight * loss_mono

        # ---- Slope on ln Qs^2 per pair: (ln Qs^2(Y_b)-ln Qs^2(Y_a)) / ΔY  ----+        # uses true base Y_a from base18, predicted Y_b from yhat, true Y_b from y
        if (self.qs_slope_weight != 0.0) and (base18 is not None) and (dy_scalar is not None):
            U_base = self._pack18_to_U(base18)
            if self.project_before_frob:
                Uh_q, U_q = Uh, U
            else:
                Uh_q = self._su3_project(Uh_raw)
                U_q  = self._su3_project(U)
            Qp = self._compute_Qs_from_U(Uh_q,  local=False).clamp_min(1e-8)   # [B]
            Qt = self._compute_Qs_from_U(U_q,   local=False).clamp_min(1e-8)   # [B]
            Qb = self._compute_Qs_from_U(U_base,local=False).clamp_min(1e-8)   # [B]
            lnQ2_p = 2.0 * torch.log(Qp)
            lnQ2_t = 2.0 * torch.log(Qt)
            lnQ2_b = 2.0 * torch.log(Qb)
            dY = dy_scalar.view(-1).to(lnQ2_p.dtype)
            dY = torch.where(dY.abs() > 1e-12, dY, torch.full_like(dY, 1.0))   # avoid div0 if any
            slope_pred = (lnQ2_p - lnQ2_b) / dY
            slope_true = (lnQ2_t - lnQ2_b) / dY
            loss_qsl = F.mse_loss(slope_pred, slope_true)
            total = total + self.qs_slope_weight * loss_qsl


#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize();
#        t0=time.perf_counter()
#        #----end-time----

        # ------------------------------------------------------------------
        # Full-covariance α-NLL, BCH mean consistency, and energy distance
        # ------------------------------------------------------------------
        # 1) Build α_true and α_pred (maps) and α_true_bar (global per-sample)
        a_true_bar = None
        a_pred     = None

        need_a_true_bar = (
            (self.nll_weight != 0.0)
            and getattr(self, 'use_fullcov', False)
            and hasattr(self, 'param_nll')
            and (theta is not None)
        )

        need_a_pred = (getattr(self, 'energy_weight', 0.0) > 0.0)

        nll_stride = getattr(self, "nll_stride", 1)
        def _ds18(t):  # t: [B,18,H,W]
            return t if nll_stride == 1 else t[:, :, ::nll_stride, ::nll_stride]

        if base18 is not None and (need_a_true_bar or need_a_pred):
            base18_s = _ds18(base18)
            y_s      = _ds18(y)
            yhat_s   = _ds18(yhat)

            U0 = self._su3_project(self._pack18_to_U(base18_s), iters=1)

            if need_a_true_bar:
                with torch.no_grad():
                    Utru = self._su3_project(self._pack18_to_U(y_s), iters=1)
                    Delta_true = Utru.mH @ U0
                    L_true = self._log_unitary(Delta_true, proj_iters=2, log_dtype=torch.complex64)
                    S_true = (1j) * L_true
                    lams = self._get_lams(S_true)  
#lams = self.lambdas.to(device=S_true.device, dtype=S_true.dtype)  # or cached helper
                    B, H, W = S_true.shape[:3]
                    a_true_bar = 2.0 * torch.real(torch.einsum('bhwij,aji->ba', S_true, lams)) / (H * W)

            if need_a_pred:
                # Only needed for energy distance; compute directly at stride
                Upred = self._su3_project(self._pack18_to_U(yhat_s), iters=1)
                Delta_pred = Upred.mH @ U0
                L_pred = self._log_unitary(Delta_pred, proj_iters=2, log_dtype=torch.complex64)
                S_pred = (1j) * L_pred
                lams = self._get_lams(S_pred)  
                #lams = self.lambdas.to(device=S_pred.device, dtype=S_pred.dtype)
                a_pred = 2.0 * torch.real(torch.einsum('bhwij,aji->bahw', S_pred, lams))  # [B,C,H',W']


#         if base18 is not None and (need_a_true_bar or need_a_pred):
#             # Always compute U0 once
#             U0 = self._su3_project(self._pack18_to_U(base18), iters=1)

#             if need_a_true_bar:
#                 with torch.no_grad():  # α_true comes from data; no grads needed
#                     Utru = self._su3_project(self._pack18_to_U(y), iters=1)
#                     Delta_true = Utru.conj().transpose(-1, -2) @ U0
#                     L_true = self._log_unitary(Delta_true, proj_iters=1, log_dtype=torch.complex64)
#                     S_true = (1j) * L_true
#                     lams = self.lambdas.to(device=S_true.device, dtype=S_true.dtype)
#                     B, H, W = S_true.shape[:3]
#                     a_true_bar = 2.0 * torch.real(torch.einsum('bhwij,aji->ba', S_true, lams)) / (H * W)
# #                    a_true = 2.0 * torch.real(torch.einsum('bhwij,aji->bahw', S_true, lams))
# #                    a_true_bar = a_true.mean(dim=(2, 3))  # [B,C]

#             if need_a_pred:
#                 # Needed only for energy-distance; gradients flow through Upred path
#                 Upred = self._su3_project(self._pack18_to_U(yhat), iters=1)
#                 Delta_pred = Upred.conj().transpose(-1, -2) @ U0
#                 L_pred = self._log_unitary(Delta_pred, proj_iters=1, log_dtype=torch.complex64)
#                 S_pred = (1j) * L_pred
#                 lams = self.lambdas.to(device=S_true.device, dtype=S_true.dtype)
#                 a_pred = 2.0 * torch.real(torch.einsum('bhwij,aji->bahw', S_pred, lams))


#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize()
#        print(f"[timing] NLL 1: {(time.perf_counter()-t0)*1e3:.2f} ms")
#        #----end-time----
#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize();
#        t0=time.perf_counter()
#        #----end-time----

        # 2) Full-cov α-NLL on spatial average (requires composer & θ)                                                                                                                                  
        if (self.nll_weight != 0.0) and getattr(self, 'use_fullcov', False) \
           and hasattr(self, 'param_nll') and (theta is not None) and (a_true_bar is not None):
            B, C = a_true_bar.shape
            # Match channels if model uses 16-ch α basis: duplicate 8->16                                                                                                                               
            if self.param_nll.C == 16 and C == 8:
                a_true_bar = torch.cat([a_true_bar, a_true_bar], dim=1)
                C = 16
            Yv = (Y_final if Y_final is not None else dy_scalar).to(a_true_bar.dtype).view(-1)
            mu_unit, L, kappa = self.param_nll(theta)
            muY   = self.param_nll.compose_mu(mu_unit, kappa, Yv)           # [B,C]                                                                                                                     
            CovY  = self.param_nll.cov_to_Y(L, kappa, Yv)                   # [B,C,C]                                                                                                                   
            # Cholesky of CovY (jitter for numerical safety)                                                                                                                                            
            eyeC = torch.eye(C, device=CovY.device, dtype=CovY.dtype).unsqueeze(0).expand(B,-1,-1)
            sigma_floor = 1e-2  # match your diagonal case
            L_Y = torch.linalg.cholesky(CovY + (sigma_floor**2) * eyeC)
            resid = (a_true_bar - muY).unsqueeze(-1)                        # [B,C,1]                                                                                                                   
            # Solve Σ^{-1} resid via cholesky_solve                                                                                                                                                     
            sol = torch.cholesky_solve(resid, L_Y)                          # [B,C,1]                                                                                                                   
            maha = (resid.transpose(1,2) @ sol).squeeze(-1).squeeze(-1)     # [B]                                                                                                                       
            logdet = 2.0 * torch.log(torch.diagonal(L_Y, dim1=-2, dim2=-1)).sum(-1)
            logdet_shift = logdet - C * math.log(sigma_floor**2)  # = log det(Σ / σ_floor^2 I) ≥ 0                                                                                                      
            loss_nll_full = 0.5 * (5. * maha + logdet_shift)   # (scale_maha = 5.0 in your code)                                                                                                        
            loss_nll_full = loss_nll_full.mean()
            total = total + self._wb('nll', loss_nll_full, self.nll_weight)

            
#            #----time----
#            if torch.cuda.is_available(): torch.cuda.synchronize()
#            print(f"[timing] NLL 2: {(time.perf_counter()-t0)*1e3:.2f} ms")
#            #----end-time----
#            #----time----
#            if torch.cuda.is_available(): torch.cuda.synchronize();
#            t0=time.perf_counter()
#            #----end-time----

            # Mahalanobis coverage (χ² check)
            p = CovY.shape[-1]
            res = (a_true_bar - muY)                         # [B,C]
            # Solve with Cholesky for stability if you have it; else inverse:
            solve = torch.linalg.solve(CovY, res.unsqueeze(-1)).squeeze(-1)  # [B,C]
            mah2  = (res * solve).sum(-1)                   # [B]

            # # gradient ratio: is the composer dominating updates?

            # def _gnorm(params, device):
            #     tot = torch.zeros((), device=device)
            #     for p in params:
            #         if p.grad is not None:
            #             tot += (p.grad.detach()**2).sum()
            #     return torch.sqrt(tot)

            # # def _gnorm(params):
            # #     s = 0.0
            # #     for p_ in params:
            # #         if p_.grad is not None:
            # #             s += (p_.grad.detach()**2).sum()
            # #     return float(torch.sqrt(s + 1e-12))

            # # Optional: gradient norm of the NLL head, if the head was attached to the criterion
            # g_nll = float("nan")
            # if hasattr(self, "param_nll") and self.param_nll is not None:
            #     try:
            #         g_nll = float(_gnorm(self.param_nll.parameters()))
            #     except Exception:
            #         g_nll = float("nan")

            # g_core = float("nan")
            # if hasattr(self, "core_params") and self.core_params:
            #     try:
            #         g_core = float(_gnorm(self.core_params))
            #     except Exception:
            #         g_core = float("nan")
                
            # safety counters (how often we hit floors/caps)
            eps = 1e-6
            lam_min = torch.linalg.eigvalsh(CovY).min(dim=-1).values   # [B]
                      
            #push penalties for odd behavior:

            # after you form CovY = L @ L.T + eps*I
            lambda_off = 1e-3
            diag = torch.diagonal(CovY, dim1=-2, dim2=-1)                # [B,C]
            off  = CovY - torch.diag_embed(diag)                         # [B,C,C]
            total += lambda_off * off.pow(2).mean()        # e.g., lambda_off = 1e-3 to 1e-2

            # hinge penalty if smallest eigenvalue < tau
            lambda_eig = 1e-2
            tau = 1e-3
            lam_min = torch.linalg.eigvalsh(CovY).min(dim=-1).values      # [B]
            total += lambda_eig * F.relu(tau - lam_min).mean()  # tau ~ 1e-3 · ⟨diag⟩, lambda_eig ~ 1e-2

            # encourage |μ| to match |ᾱ| scale (weak)
            lambda_cal = 1e-3
            scale_ratio = (muY.abs().mean() / (a_true_bar.abs().mean() + 1e-12))
            total += lambda_cal * (scale_ratio - 1.0).abs()
            # lambda_cal ~ 1e-3 (very gentle)

            
            cov_reg_weight = 0.002
            trY = torch.diagonal(CovY, dim1=-2, dim2=-1).sum(-1)           # [B]
            cov_barrier = (trY - logdet - C).mean()                        # scalar
            total = total + self._wb('cov_barrier', cov_reg_weight * cov_barrier, 1.0)     

            offdiag_weight = 0.01
            diag = torch.diagonal(CovY, dim1=-2, dim2=-1)
            off  = CovY - torch.diag_embed(diag)
            off_pen = (off**2).mean()
            total = total + self._wb('cov_offdiag', offdiag_weight * off_pen, 1.0)

            offcorr_weight = 0.001
            std  = torch.sqrt(torch.clamp_min(torch.diagonal(CovY, dim1=-2, dim2=-1), 1e-8))  # [B,C]
            den  = std.unsqueeze(-1) * std.unsqueeze(-2) + 1e-8
            Corr = CovY / den
            Corr = Corr - torch.diag_embed(torch.ones_like(std))   # zero out diagonal
            corr_off = (Corr**2).mean()
            total = total + self._wb('cov_corr_off', offcorr_weight * corr_off, 1.0)

            mu_anchor_weight = 0.0001
            mu_anchor = F.mse_loss(muY, a_true_bar)   # both [B,C]
            total = total + self._wb('mu_anchor', mu_anchor_weight * mu_anchor, 1.0)

            # residuals and safe covariance
            res   = (a_true_bar - muY)                           # [B,C]
            B, C  = res.shape
            eps   = 1e-12
            jit   = 1e-5
            I     = torch.eye(C, device=CovY.device, dtype=CovY.dtype).unsqueeze(0)
            CovY_safe = CovY + jit * I
            
            # squared Mahalanobis per sample: rᵀ Σ⁻¹ r
            L    = torch.linalg.cholesky(CovY_safe)              # [B,C,C]
            x    = torch.cholesky_solve(res.unsqueeze(-1), L)    # [B,C,1] = Σ⁻¹ r
            mah2 = (res.unsqueeze(-2) @ x).squeeze(-1).squeeze(-1)   # [B]
            
            # penalty: mean(mah2) → p (=C)
            lambda_cal = 1e-4   # start small (1e-4…5e-4)
            total = total + lambda_cal * (mah2.mean() - float(C))**2
            
            diag = torch.diagonal(CovY_safe, dim1=-2, dim2=-1).clamp_min(eps)    # [B,C]
            sig  = diag.sqrt()                                                    # [B,C]
            den  = sig.unsqueeze(-1) * sig.unsqueeze(-2) + eps                    # [B,C,C]
            Corr = CovY_safe / den                                                # [B,C,C]
            off_mask = (1.0 - I)                                                  # [1,C,C]
            offcorr_pen = ((Corr * off_mask)**2).sum(dim=(-2,-1)).mean()          # scalar
            
            lambda_offcorr = 1e-4  # gentle; raise only if off/diag keeps growing
            total = total + lambda_offcorr * offcorr_pen

#            #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize()
#        print(f"[timing] NLL 2.5: {(time.perf_counter()-t0)*1e3:.2f} ms")
#        #----end-time----
#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize();
#        t0=time.perf_counter()
#        #----end-time----
            
        # 3) BCH mean consistency on μ (no rollouts)
        if (getattr(self, 'bch_weight', 0.0) > 0.0) and hasattr(self, 'param_nll') \
           and (theta is not None) and (a_true_bar is not None):
            Yv = (Y_final if Y_final is not None else dy_scalar).to(a_true_bar.dtype).view(-1)
            # Random split of Y into Y1 + Y2 = Y
            r = torch.rand_like(Yv)
            Y1, Y2 = r*Yv, (1.0 - r)*Yv
            mu_unit, L, kappa = self.param_nll(theta)
            mu1 = self.param_nll.compose_mu(mu_unit, kappa, Y1)             # [B,C]
            mu2 = self.param_nll.compose_mu(mu_unit, kappa, Y2)
            muY = self.param_nll.compose_mu(mu_unit, kappa, Yv)
            # BCH in S-space, then back to α
            mu_bch = _bch_alpha(mu1, mu2, lams)
            loss_bch = F.mse_loss(mu_bch, muY)
            total = total + self._wb('bch', loss_bch, getattr(self, 'bch_weight', 0.0))

#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize()
#        print(f"[timing] NLL 3: {(time.perf_counter()-t0)*1e3:.2f} ms")
#        #----end-time----
#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize();
#        t0=time.perf_counter()
#        #----end-time----

        # 4) Energy distance between α_pred and α_true distributions (subsampled grid)
        if (self.energy_weight!=0) and (a_true is not None) and (a_pred is not None):
            stride = max(int(getattr(self, 'energy_stride', 4)), 1)
            # Subsample H×W by stride to keep compute reasonable; flatten samples
            a_t = a_true[:, :, ::stride, ::stride].flatten(start_dim=2).transpose(1,2).reshape(-1, a_true.shape[1])
            a_p = a_pred[:, :, ::stride, ::stride].flatten(start_dim=2).transpose(1,2).reshape(-1, a_pred.shape[1])
            # Match channels if needed
            if a_p.shape[1] != a_t.shape[1]:
                minC = min(a_p.shape[1], a_t.shape[1])
                a_p, a_t = a_p[:, :minC], a_t[:, :minC]
            loss_energy = _energy_distance_alpha(a_t.detach(), a_p)         # no grad into data
            total = total + self._wb('energy', loss_energy, getattr(self, 'energy_weight', 0.0))

#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize()
#        print(f"[timing] NLL 4: {(time.perf_counter()-t0)*1e3:.2f} ms")
#        #----end-time----

            
        if not return_stats:
            return total
        
        stats["dip_mse"] = dip.detach()

                # ---- Monitoring stats (detach -> python floats) ----

        if (self.nll_weight != 0.0) and getattr(self, 'use_fullcov', False) \
           and hasattr(self, 'param_nll') and (theta is not None) and (a_true_bar is not None):
            with torch.no_grad():
                # per-channel std at Y
                varY = torch.diagonal(CovY, dim1=-2, dim2=-1).clamp_min(0)
                sig = torch.sqrt(varY + 1e-12)                               # [B,C]
                sigma_bar = sig.mean()
                sigma_small_frac = (sig < 1e-3).float().mean()

                # off-diagonal / diagonal energy ratio (batch-safe)
                diag = torch.diagonal(CovY, dim1=-2, dim2=-1)                # [B,C]
                off  = CovY - torch.diag_embed(diag)                         # [B,C,C]
                # Frobenius norms per batch, then mean
                off_F  = torch.linalg.matrix_norm(off,  ord='fro')           # [B]
                diag_F = torch.linalg.vector_norm(diag, ord=2, dim=-1)       # [B]
                offdiag_ratio = (off_F.mean() / (diag_F.mean() + 1e-12))
                # spectrum + condition number
                ev = torch.linalg.eigvalsh(CovY)                              # [B,C], real
                lam_min = ev.min(dim=-1).values.mean()
                lam_max = ev.max(dim=-1).values.mean()
                cond = (lam_max / (lam_min + 1e-12))
                # correlations/scale match for μ vs ᾱ
                x = a_true_bar
                y = muY
                x_c = x - x.mean(dim=1, keepdim=True)
                y_c = y - y.mean(dim=1, keepdim=True)
                corr = ( (x_c*y_c).sum(dim=1) /
                         (x_c.norm(dim=1)*y_c.norm(dim=1) + 1e-12) ).mean()
                abs_ratio = (y.abs().mean() / (x.abs().mean() + 1e-12))

                stats['alpha_nll_full']        = float(loss_nll_full.detach())
                stats['alpha_maha']            = float(maha.mean().detach())
                stats['alpha_logdet']          = float(logdet.mean().detach())
                stats['alpha_sigma_bar']       = float(sigma_bar)
                stats['alpha_sigma_small_frac']= float(sigma_small_frac)
                stats['alpha_offdiag_ratio']   = float(offdiag_ratio)
                stats['alpha_lambda_min']      = float(lam_min)
                stats['alpha_lambda_max']      = float(lam_max)
                stats['alpha_cond']            = float(cond)
                stats['alpha_corr']            = float(corr)
                stats['alpha_abs_ratio']       = float(abs_ratio)

                stats['alpha_mah2_mean'] = float(mah2.mean().detach())
                stats['alpha_mah2_med']  = float(mah2.median().detach())
                stats['alpha_mah2_p90']  = torch.quantile(mah2.float(), 0.90).item()

                #stats['alpha_mah2_p90']  = float(torch.quantile(mah2, 0.90).detach())
                stats['alpha_mah2_p']    = p                    # target mean ≈ p if calibrated

                stats['alpha_lammin_med'] = float(lam_min.median().detach())

        return total, stats

    @torch.no_grad()
    def components_dict(self, yhat: 'torch.Tensor', y: 'torch.Tensor', base18: 'torch.Tensor' | None = None):
        out = super().components_dict(yhat, y, base18=base18)
        if self.qs_weight != 0.0 and len(self.dip_offsets) >= 2:
            Uh = self._pack18_to_U(yhat)
            U  = self._pack18_to_U(y)
            Qh = self._compute_Qs_from_U(Uh, local=self.qs_local)
            Qt = self._compute_Qs_from_U(U,  local=self.qs_local)

            out["qs_mse"] = torch.nn.functional.mse_loss(Qh, Qt).detach()
            out["qs_pred_mean"] = Qh.mean().detach()
            out["qs_true_mean"] = Qt.mean().detach()
            # out["qs_mse"] = float(torch.nn.functional.mse_loss(Qh, Qt).item())
            # # optionally export means for quick sanity checks
            # out["qs_pred_mean"] = float(Qh.mean().item())
            # out["qs_true_mean"] = float(Qt.mean().item())
 
            # # Use the same dipole loss as a metric (detach to keep it no-grad)
            # try:
            #     dip = self._dipole_loss(
            #         Uh.detach(), U.detach(),
            #         local=False,            # same as your training call
            #         use_logN=True,
            #         per_radius_norm=True,
            #         detach_target=True
            #     )
            #     out["dip_mse"] = dip.detach()
            # except Exception as e:
            #     out["dip_mse_error"] = str(e)
            
            if self.quad_weight != 0.0 and len(self.quad_pairs) > 0:
                Q4h = self._quadrupole(Uh); Q4t = self._quadrupole(U)
                out["quad_mse"] = float(torch.nn.functional.mse_loss(Q4h, Q4t).detach())
        return out



class ParamNLLComposerFullCov(nn.Module):
    """
    θ=(m, Λ_QCD, μ0) -> μ_unit(θ) in R^C and diffusion D(θ)=L L^T (PSD, full-cov).
    Compose to Y via Brownian (default) or OU (diagonal κ).
    """
    def __init__(self, C: int = 8, compose: str = 'brownian'):
        super().__init__()
        self.C = C
        self.compose = compose
        H = 128
        self.backbone = nn.Sequential(
            nn.Linear(3, H), nn.SiLU(),
            nn.Linear(H, H), nn.SiLU()
        )
        self.mu_head   = nn.Linear(H, C)                  # μ per unit Y
        self.diag_head = nn.Linear(H, C)                  # softplus diag of L
        self.tril_head = nn.Linear(H, (C*(C-1))//2)       # strictly lower triangle
        self.kappa_head = nn.Linear(H, C) if compose=='ou' else None

    def compose_to_Y(self,
                     mu_unit: torch.Tensor,
                     sigma_or_L: torch.Tensor,
                     kappa: torch.Tensor | None,
                     Y: torch.Tensor):
        """
        Back-compat helper so old call sites keep working.
        Accepts either per-channel sigma per unit-Y (diag) or a full Cholesky L.
        Returns:
            muY: [B,C]      — composed mean at Y
            sigY: [B,C]     — per-channel std at Y (sqrt diag of Cov(Y))
        """
        # 1) mean composition
        muY = self.compose_mu(mu_unit, kappa, Y)  # [B,C]

        # 2) covariance composition → per-channel std
        # detect if sigma_or_L is a full Cholesky [B,C,C] or diagonal sigma [B,C]
        if sigma_or_L.dim() == 3 and sigma_or_L.shape[-1] == self.C:
            # treat as Cholesky L
            L = sigma_or_L
            CovY = self.cov_to_Y(L, kappa, Y)  # [B,C,C]
            varY = torch.diagonal(CovY, dim1=-2, dim2=-1)  # [B,C]
        else:
            # treat as diagonal sigma per unit-Y
            sigma_unit = sigma_or_L
            Yv = Y.view(-1, 1).to(sigma_unit.dtype)
            if self.compose == 'brownian' or (kappa is None):
                # Var[Y] = Y * sigma_unit^2
                varY = Yv * (sigma_unit ** 2)
            else:
                # OU: Var[Y] = sigma^2 * (1 - exp(-2 κ Y)) / (2 κ)
                varY = (sigma_unit ** 2) * (1.0 - torch.exp(-2.0 * kappa * Yv)) / (2.0 * kappa + 1e-9)

        sigY = torch.sqrt(varY.clamp_min(1e-12))  # [B,C]
        return muY, sigY


        
    def forward(self, theta: torch.Tensor):
        """
        theta: [B,3] -> μ_unit [B,C], L [B,C,C] (lower), κ [B,C] or None
        """
        h = self.backbone(theta.view(theta.shape[0], 3))
        mu_unit = self.mu_head(h)                         # [B,C]
        diag = F.softplus(self.diag_head(h)) + 1e-6       # positive
        off  = self.tril_head(h)                          # [B,C(C-1)/2]
        B, C = theta.shape[0], self.C

        # use the activation/head dtype (matches autocast) instead of theta's
        L = torch.zeros(B, C, C, device=mu_unit.device, dtype=mu_unit.dtype)
        idx = torch.tril_indices(C, C, offset=-1)
        
        # (optional but robust if anything upstream changes dtype)
        off  = off.to(L.dtype)
        diag = diag.to(L.dtype)
        
        L[:, idx[0], idx[1]] = off
        L[:, torch.arange(C), torch.arange(C)] = diag
        
        # L = theta.new_zeros((B, C, C))
        # idx = torch.tril_indices(C, C, offset=-1)
        # L[:, idx[0], idx[1]] = off
        # L[:, torch.arange(C), torch.arange(C)] = diag
        kappa = None
        if self.kappa_head is not None:
            kappa = F.softplus(self.kappa_head(h)) + 1e-6
        return mu_unit, L, kappa

    def compose_mu(self, mu_unit: torch.Tensor, kappa: torch.Tensor | None, Y: torch.Tensor):
        """Return μ(Y) from μ_unit; Brownian: Y μ_unit ; OU: (μ_unit/κ)(1-e^{-κY})."""
        Yv = Y.view(-1,1).to(mu_unit.dtype)
        if self.compose == 'brownian':
            return Yv * mu_unit
        # OU mean with diagonal κ per channel
        e = torch.exp(-kappa * Yv)
        return (mu_unit / kappa) * (1.0 - e)

    def cov_to_Y(self, L: torch.Tensor, kappa: torch.Tensor | None, Y: torch.Tensor):
        """
        Return Cov(Y). Brownian: Y * (L L^T).
        OU with diagonal κ: Cov_ij(Y) = D_ij * (1 - exp(-(κ_i+κ_j)Y)) / (κ_i+κ_j)
        """
        D = L @ L.transpose(-1, -2)                       # [B,C,C]
        if self.compose == 'brownian':
            return Y.view(-1,1,1) * D
        # OU: diagonal κ
        B, C = D.shape[0], D.shape[1]
        ki = kappa.view(B, C, 1)
        kj = kappa.view(B, 1, C)
        K = ki + kj
        e = torch.exp(-K * Y.view(-1,1,1))
        return (1.0 - e) * (D / (K + 1e-9))


    
# ------------------------- Data loaders -------------------------



def make_loaders(root, batch_size=1, workers=2, seed=0, ddp: bool=False, split_by="run",  **kwargs):
    # ↓ Avoid huge per-worker dataset copies of complex128 when using workers
    train_ds = JimwlkEvolverDataset(root, split="train", split_frac=0.9, seed=seed,
                                    split_by=split_by, cache_initial=(workers == 0))
    val_ds   = JimwlkEvolverDataset(root, split="val",   split_frac=0.9, seed=seed,
                                    split_by=split_by,   cache_initial=(workers == 0))

    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False)
    else:
        train_sampler = None; val_sampler = None

    use_pin = torch.cuda.is_available()                    # no args needed

    train_dl = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=workers, pin_memory=use_pin,
        persistent_workers=(workers > 0),          # was False
        prefetch_factor=(4 if workers > 0 else None),  # was 1
        drop_last=True)#, collate_fn=collate_base18_Y_theta)
    
    val_dl   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler,
        num_workers=workers, pin_memory=use_pin,
        persistent_workers=(workers > 0),          # was False
        prefetch_factor=(4 if workers > 0 else None),  # was 1
        drop_last=False)#, collate_fn=collate_base18_Y_theta)
    
    
    # train_dl = DataLoader(train_ds, batch_size=batch_size,
    #                       shuffle=(train_sampler is None), sampler=train_sampler,
    #                       num_workers=workers, pin_memory=use_pin,
    #                       persistent_workers=False, prefetch_factor=pf, drop_last=True,
    #                       collate_fn=collate_base18_Y_theta)

    # val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler,
    #                       num_workers=workers, pin_memory=use_pin,
    #                       persistent_workers=False, prefetch_factor=pf, drop_last=False,
    #                       collate_fn=collate_base18_Y_theta)

    # after creating val_dl
    val_Y = [float(e["Y"]) for e in getattr(val_dl.dataset, "entries", [])]
    if val_Y:
        print(f"[val] Y range from data: y_min={min(val_Y):.6g}, y_max={max(val_Y):.6g}, "
              f"count={len(val_Y)}, frac(Y==0)={sum(abs(y)<=1e-12 for y in val_Y)/len(val_Y):.2%}")

    return train_dl, val_dl, train_sampler, val_sampler


# ------------------------- Train -------------------------


# ------------------------- Rollout & consistency helpers -------------------------

def collate_base18_Y_theta(batch):
    """
    Convert legacy (x[C,H,W], y) where C>=22 into:
      base18: [B,18,H,W]
      Y_scalar: [B]        (mean over H,W of channel 18)
      theta: [B,3]         (from channels 19..21 at any pixel)
      y: target (stacked)
    """
    base_list, Y_list, th_list, tgt_list = [], [], [], []
    for x, y in batch:
        C, H, W = x.shape
        assert C >= 22, f"Expected ≥22 channels, got {C}"

        base18 = x[0:18, ...]               # [18,H,W]
        Y_map  = x[18,   ...]               # [H,W], spatially constant
        theta  = x[19:22, 0, 0]             # [3], constant map → scalars

        base_list.append(base18)
        Y_list.append(Y_map.mean())         # scalar
        th_list.append(theta)
        tgt_list.append(y)

    base18   = torch.stack(base_list, dim=0).contiguous()         # [B,18,H,W]
    Y_scalar = torch.stack(Y_list,   dim=0).to(base18.dtype)      # [B]
    theta    = torch.stack(th_list,  dim=0).to(base18.dtype)      # [B,3]
    y        = torch.stack(tgt_list, dim=0)
    return base18, Y_scalar, theta, y


def _build_x_from_parts(base18: 'torch.Tensor', y_scalar: 'torch.Tensor', params_map: 'torch.Tensor', y_index: int = 18) -> 'torch.Tensor':
    """
    Construct an input tensor [B,22,H,W] from:
      - base18: [B,18,H,W] (current base field)
      - y_scalar: [B] or [] scalar rapidity for the *target* step
      - params_map: [B,3,H,W] parameter maps (m, Lambda_QCD, mu0)
    """
    B, _, H, W = base18.shape
    if y_scalar.ndim == 0:
        y_scalar = y_scalar.unsqueeze(0).expand(B)
    Ymap = base18.new_empty(B, 1, H, W).copy_(y_scalar.view(B,1,1,1).expand(B,1,H,W))
    xnew = torch.cat([base18, Ymap, params_map], dim=1)
    return xnew



def rollout_predict(model, base18, Y_scalar, k, theta,
                    device_type="cuda", amp_dtype=torch.bfloat16, use_amp=True):
    # 1 big step: keep grads (we often compute loss on this)
    with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
        y_single = model(base18, Y_scalar * k, theta)

    # k small steps: no grads, no version counters (saves a lot of memory)
    with torch.inference_mode(), torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
        base_curr = base18
        for _ in range(k):
            base_curr = model(base_curr, Y_scalar, theta)
    y_roll = base_curr  # already detached
    return y_roll, y_single


# def rollout_predict(model, base18, Y_scalar, k, theta,
#                     device_type="cuda", autocast_dtype=torch.bfloat16):
#     # Big step: keep grads
#     with torch.autocast(device_type, dtype=autocast_dtype):
#         y_single = model(base18, Y_scalar * k, theta)

#     # k small steps: NO grads (use inference_mode to save both autograd & versioning memory)
#     with torch.inference_mode(), torch.autocast(device_type, dtype=autocast_dtype):
#         base_curr = base18
#         for _ in range(k):
#             base_curr = model(base_curr, Y_scalar, theta)
#     y_roll = base_curr  # already detached
#     return y_roll, y_single


# def rollout_predict(model, base18, Y_scalar, k, theta):
#     """
#     Compare k small steps vs 1 big step.
#     NOTE: you likely already have logic to form the next base from the previous prediction;
#     keep that logic and just pass scalars here.
#     """
#     # single big step to k * ΔY
#     Y_big = Y_scalar * k
#     y_single   = model(base18, Y_big, theta)
#     # k sequential steps of ΔY
#     # (Pseudo: you must reuse your original way of feeding the previous prediction back as 'base18' or equivalent.)
#     base_curr = base18
#     for _ in range(k):
#         y_step     = model(base_curr, Y_scalar, theta)
#         # ... update base_curr from y_step as your original rollout did ...
#     y_roll = y_step
#     return y_roll, y_single

def semigroup_compose(model, base18, Y_a, Y_b, theta):
    """
    Compare f(Y_a + Y_b, base18)  vs  f(Y_b, f(Y_a, base18)).
    base18: [B,18,H,W]
    Y_a, Y_b: [B] with Y_a + Y_b = total Y
    theta: [B,3]
    Returns (u_b_then_a, u_ab) with identical shapes.
    """
    # First step of size Y_a from the original base
#    u_a = model.forward(base18, Y_a, theta)      # predicted links after Y_a
    u_a = model(base18, Y_a, theta)
    # IMPORTANT: turn the first prediction into the next "base"
    # If your head outputs exactly the 18-channel link representation (usual case),
    # this is just the prediction itself. If your head outputs more than 18 channels,
    # slice to the first 18.
    if u_a.shape[1] == 18:
        base_after_a = u_a
    else:
        base_after_a = u_a[:, :18, ...]  # adjust if your layout differs

    # Second step of size Y_b from the intermediate state
    u_b_then_a = model(base_after_a, Y_b, theta)
    #u_b_then_a = model.forward(base_after_a, Y_b, theta)

    # One single step of size (Y_a + Y_b) from the original base
    u_ab = model(base18, Y_a + Y_b, theta)
    #u_ab = model.forward(base18, Y_a + Y_b, theta)

    return u_b_then_a, u_ab


def ramp(epoch, start, duration):
    # linear 0→1 starting at `start` over `duration` epochs
    if epoch < start: return 0.0
    if epoch >= start+duration: return 1.0
    return (epoch - start) / max(1, duration)



# ==================== TRAIN ====================

# Fixed training (with validation inside) for correct DDP+AMP+accumulation behavior
# Key changes vs your version:
#  - Exactly one backward() per microstep; uses model.no_sync() on non-final microsteps
#  - Optimizer step happens on EVERY rank only on the last microstep
#  - GradScaler used only for AMP; assertion avoided
#  - Optional Δw/‖w‖ metric computed rarely, rank0-only, with contiguous().reshape(-1)
#  - Removed duplicate/early step blocks that caused slowdowns or double-backward
#  - Clean DDP shutdown


# ---- helpers for parsing offset/pair strings ----
def _parse_offsets_list(s: str):
    """Parse strings like "(1,0),(0,2),(4,0)" into [(1,0),(0,2),(4,0)].
    Accepts spaces. Returns a list of (dx,dy) ints."""
    if isinstance(s, (list, tuple)):
        return [tuple(map(int, x)) for x in s]
    s = (s or '').strip()
    if not s:
        return [(1,0),(0,1)]
    pairs = []
    for a,b in re.findall(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', s):
        pairs.append((int(a), int(b)))
    # collapse duplicates, keep order
    seen = set(); out = []
    for p in pairs:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def _parse_rects_list(s: str):
    """Parse strings like "(1,1),(1,2),(2,2)" into [(1,1),(1,2),(2,2)].""" 
    return _parse_offsets_list(s)


def _parse_quad_pairs_list(s: str, dipole_offsets=None):
    """Parse quadrupole pair spec.
    Supported forms:
      - explicit: "((1,0),(0,1));((2,0),(0,2))"
      - radial set: "r=1,2,4,8,12,16"  -> [((r,0),(0,r)) for r in set]
      - auto from dipole offsets: "auto" or "auto_from_dipole" (axial only)
      - *NEW* broader auto: "auto_broad" / "auto_full" / "auto_all"
        which generates axial and diagonal base vectors and all non-colinear pairs.
    Returns list[ ((dx1,dy1),(dx2,dy2)), ... ].
    """
    if isinstance(s, (list, tuple)) and s and isinstance(s[0], (list, tuple)):
        return [ (tuple(map(int,a)), tuple(map(int,b))) for (a,b) in s ]
    s = (s or '').strip()
    s_lower = s.lower()
    # --- AUTO AXIAL (legacy default) ---
    if s_lower in ('', 'auto', 'auto_from_dipole', 'auto-dip', 'dip'):
        offs = dipole_offsets or []
        # Use only axial offsets present in dipole list; derive radii
        radii = sorted({ abs(dx)+abs(dy) for (dx,dy) in offs if ((dx==0) ^ (dy==0)) and (dx!=0 or dy!=0) })
        if not radii:
            radii = [1,2,4,8,12,16]
        return [ ((int(r),0),(0,int(r))) for r in radii ]
    # --- AUTO BROAD (new): axial + diagonals, all non-colinear pairs ---
    if s_lower in ('auto_broad', 'auto-full', 'auto_full', 'auto-all', 'auto_all', 'broad', 'full'):
        offs = dipole_offsets or []
        # derive radii from dipole offsets (L1) if available, else a sensible default
        radii = sorted({ abs(dx)+abs(dy) for (dx,dy) in offs if (dx!=0 or dy!=0) })
        if not radii:
            radii = [1,2,4,8,12,16]
        base = []
        for r in radii:
            # axial
            base.extend([(r,0), (0,r)])
            # diagonals
            base.extend([(r,r), (r,-r)])
        # dedupe base
        seen_b=set(); base2=[]
        for v in base:
            if v not in seen_b:
                seen_b.add(v); base2.append(v)
        base = base2
        # form all non-colinear pairs
        pairs=set()
        for i in range(len(base)):
            ax, ay = base[i]
            for j in range(i, len(base)):
                bx, by = base[j]
                # skip degenerate (zero area) parallelograms, i.e., colinear vectors
                if ax*by - ay*bx == 0:
                    continue
                a = (int(ax), int(ay)); b = (int(bx), int(by))
                # canonicalize order to dedupe: sort by (|a|_1, a) then (|b|_1, b)
                def key(v): return (abs(v[0])+abs(v[1]), v[0], v[1])
                a_, b_ = (a,b) if key(a) <= key(b) else (b,a)
                pairs.add((a_, b_))
        # return as list in a stable order
        #print("QUADPAIRS=",pairs)
        return [ (a,b) for (a,b) in sorted(pairs, key=lambda p: ((abs(p[0][0])+abs(p[0][1])), (abs(p[1][0])+abs(p[1][1])), p)) ]
    # --- Radius list (axial) ---
    if s_lower.startswith('r='):
        nums = re.findall(r'-?\d+', s[2:])
        radii = [int(x) for x in nums]
        radii = sorted({r for r in radii if r!=0})
        return [ ((int(r),0),(0,int(r))) for r in radii ]
    # explicit pair-of-pairs parser
    quad_pairs = []
    for a1,b1,a2,b2 in re.findall(r'\(\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*,\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*\)', s):
        quad_pairs.append(((int(a1),int(b1)), (int(a2),int(b2))))
    # Fallback: if string was a plain list of offsets, make symmetric pairs of identical radii (axial)
    if (not quad_pairs):
        offs = _parse_offsets_list(s)
        radii = sorted({ abs(dx)+abs(dy) for (dx,dy) in offs if ((dx==0) ^ (dy==0)) and (dx!=0 or dy!=0) })
        quad_pairs = [ ((int(r),0),(0,int(r))) for r in radii ]
    # dedupe
    seen=set(); out=[]
    for p in quad_pairs:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def safe_ratio_err(ratio, device=None):
    # Return a scalar tensor err = |log(ratio)|, robust to None/NaN/Inf/negatives/zeros.
    if ratio is None:
        return torch.tensor(float('inf'), device=device)

    if torch.is_tensor(ratio):
        r = ratio.detach()
        # reduce to scalar if needed (mean; change to median/max as you prefer)
        if r.numel() > 1:
            r = r.mean()
        r = r.float()
    else:
        r = torch.tensor(float(ratio), dtype=torch.float32, device=device)

    # sanitize: replace NaN/Inf, clamp to positive
    r = torch.nan_to_num(r, nan=1.0, posinf=1e8, neginf=1e-8)
    r = torch.clamp(r, min=1e-8, max=1e8)
    return torch.abs(torch.log(r))

def train(args):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from contextlib import nullcontext
    
    # ---- Device & DDP setup ----
    want_cpu = bool(getattr(args, "cpu", False))
    has_cuda = torch.cuda.is_available() and not want_cpu
    has_mps = (not has_cuda) and (getattr(torch.backends, "mps", None) is not None) and torch.backends.mps.is_available() and (not want_cpu)

    device_type = "cuda" if has_cuda else ("mps" if has_mps else "cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if device_type == "cuda" else 0
    is_ddp = device_type == "cuda" and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(device_type if device_type != "cuda" else f"cuda:{local_rank}")

    LOG_EVERY = int(getattr(args, "log_every", 1000))


    if (not is_ddp) or dist.get_rank()==0:
        print({
            "WORLD_SIZE": os.getenv("WORLD_SIZE"),
            "RANK": os.getenv("RANK"),
            "LOCAL_RANK": os.getenv("LOCAL_RANK"),
            "device": str(device),
            "amp": bool(getattr(args, "amp", False)),
        }, flush=True)

    
    if (not is_ddp) or dist.get_rank() == 0:
        print("DDP ", is_ddp)

    # ---- Data ----
    train_dl, val_dl, train_sampler, val_sampler = make_loaders(
        args.data_root, batch_size=args.batch, workers=args.workers,
        seed=args.seed, ddp=is_ddp, split_by=args.split_by
    )
    inferred_N = train_dl.dataset.N
    ds_value = train_dl.dataset.ds
    if (not is_ddp) or dist.get_rank() == 0:
        print(f"Inferred lattice size: N={inferred_N}, ds={ds_value}")

    y_values = [float(e["Y"]) for e in train_dl.dataset.entries]
    y_min_data = float(min(y_values)) if y_values else 0.0
    y_max_data = float(max(y_values)) if y_values else 1.0
    if (not is_ddp) or dist.get_rank() == 0:
        print(f"[train] Y range from data: y_min={y_min_data:.6g}, y_max={y_max_data:.6g}")

    def _init_ybins():
        edges = np.linspace(y_min_data, y_max_data, 7, dtype=np.float64)
        return torch.tensor(edges, device=device, dtype=torch.float64)

    # ---- Perf knobs ----
    if device_type == "cuda":
        try: torch.backends.cuda.matmul.allow_tf32 = True
        except Exception: pass
        try: torch.backends.cudnn.benchmark = True
        except Exception: pass
        torch.set_float32_matmul_precision("high")

    use_amp = bool(getattr(args, "amp", False) and device_type == "cuda")
    amp_dtype = (torch.bfloat16 if (device_type=="cuda" and torch.cuda.is_bf16_supported()) else torch.float16)
    use_scaler = (use_amp and amp_dtype is torch.float16)
    
    class _NullScaler:
        def scale(self, x): return x
        def step(self, opt): opt.step()
        def update(self): pass
        def unscale_(self, opt): pass
        def is_enabled(self): return False

    scaler = _NullScaler()
    if use_scaler:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    
    # ---- Model ----
    model = EvolverFNO(
        in_ch=22, width=args.width, modes1=args.modes, modes2=args.modes,
        n_blocks=args.blocks, identity_eps=args.identity_eps,
        alpha_scale=3.0, clamp_alphas=getattr(args, "clamp_alphas", 2.),
        y_index=args.y_channel, film_mode=args.film_mode, rbf_K=args.rbf_K,
        film_hidden=args.film_hidden, gamma_scale=args.gamma_scale,
        beta_scale=args.beta_scale, gate_temp=args.gate_temp,
        y_min=(args.y_min if args.y_min is not None else y_min_data),
        y_max=(args.y_max if args.y_max is not None else y_max_data),
        y_map=args.y_map,
        rbf_gamma = getattr(args, "rbf_gamma", 1.0),
        rbf_min_width = getattr(args, "rbf_min_width", 1e-3)
    ).to(device)



    def _has_uninitialized(m):
        for p in m.parameters(recurse=True):
            if isinstance(p, UninitializedParameter):
                return True
        return False

    # move channels_last to after lazy init
    use_channels_last = getattr(args, "channels_last", True)  # or whatever you use to gate it
    if use_channels_last:
        try:
            # Fast path if nothing is lazy
            if not _has_uninitialized(model):
                model = model.to(memory_format=torch.channels_last)
            else:
                raise ValueError("lazy")  # force the lazy-safe path
        except ValueError:
            # Lazy-safe path: run a tiny dummy forward to materialize shapes, then retry
            # new:
            batch = next(iter(train_dl))
            # Support loaders that yield 3-tuple (x, y_scalar, theta) or 4-tuple (x, y_scalar, theta, target)
            if isinstance(batch, (tuple, list)):
                if len(batch) >= 3:
                    xb, yb, tb = batch[0], batch[1], batch[2]
                else:
                    raise ValueError(f"train_dl yielded {len(batch)} items; expected at least 3 (x, y_scalar, theta).")
            elif isinstance(batch, dict):
                # Best-effort mapping of common keys
                xb = batch.get('base18') or batch.get('x') or batch.get('inputs') or batch.get('images')
                yb = batch.get('y_s')   or batch.get('y') or batch.get('Y') or batch.get('labels')
                tb = batch.get('theta') or batch.get('params') or batch.get('theta_vec')
                if xb is None or yb is None or tb is None:
                    raise ValueError(f"Unable to infer (x, y_scalar, theta) from dict keys: {list(batch.keys())}")
            else:
                raise TypeError(f"Unsupported batch type from train_dl: {type(batch).__name__}")


            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            tb = tb.to(device, non_blocking=True)
            model.eval()
            with torch.no_grad():
                _ = model(xb[:1], yb[:1], tb[:1])    # materialize all Lazy params
                model.train()
                model = model.to(memory_format=torch.channels_last)
    
# #    if torch.cuda.is_available():
# #        model = torch.compile(model, mode="reduce-overhead")   # great for backward

#     if getattr(args, "channels_last", False):
#         model = model.to(memory_format=torch.channels_last)
#     if getattr(args, "torch_compile", False):
#         try:
#             model = torch.compile(model)
#             if (not is_ddp) or dist.get_rank() == 0:
#                 print("[perf] torch.compile enabled.")
#         except Exception as e:
#             if (not is_ddp) or dist.get_rank() == 0:
#                 print(f"[perf] torch.compile failed: {e}")

    def unwrap(m):
        return m.module if isinstance(m, DDP) else m

    
    mcore = unwrap(model)
    ema = EMA(mcore, decay=args.ema_decay) if args.ema_decay and args.ema_decay>0 else None
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            bucket_cap_mb=100,
        )


    head, trunk = [], []
    for n,p in model.named_parameters():
        (head if "head" in n else trunk).append(p)

        
    # ---- Optimizer / Sched / AMP ----
    sched_names = {"head.gamma0", "head.gamma1", "head.kappa"}
    base_params = [p for n,p in model.named_parameters() if p.requires_grad and n not in sched_names]
    fast_params = [p for n,p in model.named_parameters() if p.requires_grad and n in sched_names]
    opt = torch.optim.AdamW(
        [{"params": trunk, "lr": args.lr, "weight_decay": args.weight_decay},
         {"params": head,  "lr": args.lr * 4.0, "weight_decay": 0.0}],   # ×4 head LR
        betas=(0.9, 0.98), eps=1e-8, fused=(device_type == "cuda"))

    # ---- LR scheduler (configurable) ----
    w_epochs = max(int(args.warmup_epochs), 0)
    main_epochs = max(args.epochs - w_epochs, 1)

    def _build_main_scheduler():
        if args.scheduler == "cosine":
            return CosineAnnealingLR(opt, T_max=main_epochs, eta_min=args.min_lr)
        elif args.scheduler == "cosine_wr":
            return CosineAnnealingWarmRestarts(opt, T_0=args.t0, T_mult=args.tmult, eta_min=args.min_lr)
        elif args.scheduler == "step":
            return StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.scheduler == "multistep":
            milestones = [int(m) for m in args.lr_milestones.split(",") if m.strip()]
            return MultiStepLR(opt, milestones=milestones, gamma=args.lr_gamma)
        elif args.scheduler == "exponential":
            return ExponentialLR(opt, gamma=args.lr_gamma)
        elif args.scheduler == "plateau":
            # step(val_loss) in the loop
            return ReduceLROnPlateau(opt, mode="min", factor=args.lr_gamma,
                                     patience=args.lr_patience, min_lr=args.min_lr,
                                     threshold=1e-4, verbose=False)
        elif args.scheduler == "poly":
            # polynomial decay down to min_lr over main_epochs
            base_lr = args.lr
            lr_span = max(base_lr - args.min_lr, 0.0)
            def _lambda(e):
                t = min(max(e, 0), main_epochs) / float(max(main_epochs, 1))
                return (args.min_lr + lr_span * (1.0 - t) ** args.poly_power) / base_lr
            return LambdaLR(opt, lr_lambda=_lambda)
        elif args.scheduler == "constant":
            return LambdaLR(opt, lr_lambda=lambda e: 1.0)
        else:
            raise ValueError(f"Unknown scheduler {args.scheduler!r}")

    if w_epochs > 0:
        warmup_start = args.warmup_start_factor if args.warmup_start_factor is not None else min(0.1, args.min_lr / args.lr)
        warmup = LinearLR(opt, start_factor=warmup_start, end_factor=1.0, total_iters=w_epochs)
        main_sched = _build_main_scheduler()
        sched = SequentialLR(opt, [warmup, main_sched], milestones=[w_epochs])
    else:
        sched = _build_main_scheduler()



    
    # w_epochs   = max(0, min(args.warmup_epochs, max(0, args.epochs - 1)))
    # main_epochs = max(1, args.epochs - w_epochs)
    # if w_epochs > 0:
    #     warm = LinearLR(opt, start_factor=args.warmup_start_factor, total_iters=w_epochs)
    #     cosine = CosineAnnealingLR(opt, T_max=main_epochs, eta_min=args.min_lr)
    #     sched = SequentialLR(opt, schedulers=[warm, cosine], milestones=[w_epochs])
    # else:
    #     sched = CosineAnnealingLR(opt, T_max=main_epochs, eta_min=args.min_lr)

    best = float("inf")
    os.makedirs(args.out, exist_ok=True)

    def log(msg: str):
        if (not is_ddp) or dist.get_rank() == 0:
            print(msg, flush=True)

    if (not is_ddp) or dist.get_rank() == 0:
        print("Device=", device_type)


    E1, E2 = args.E1, args.E2
    meters = EpochMeters(device=device_type)
    last_dw_ratio = None

    def flat_params(mod):
        with torch.no_grad():
            return torch.cat([
                p.detach().contiguous().reshape(-1)
                for p in mod.parameters() if p.requires_grad
            ])

    PROFILE_DW_EVERY = int(getattr(args, "profile_dw_every", 0))  # 0 = off

    # --- Simple monitors for whether α-NLL improves over time (rank 0 only) ---
    best_alpha_R2 = float("-inf")
    best_alpha_corr = float("-inf")
    best_alpha_nll = float("inf")


    core = unwrap(model)
    alpha_channels = getattr(core.head, 'alpha_channels', 8)
    #alpha_channels = getattr(model.head, 'alpha_channels', 8)  # 8 or 16 in your setup
    if args.nll_param_compose:
        model.param_nll = ParamNLLComposerFullCov(C=alpha_channels, compose=args.nll_compose).to(device)
    #     # When composing to Y, compare to α at final Y -> do NOT renormalize by ΔY inside the NLL block
    #     # If optimizer already exists, add this:
    #     opt.add_param_group({"params": model.param_nll.parameters(), "lr": args.lr})


    if args.nll_param_compose:
        opt.add_param_group({'params': model.param_nll.parameters(), 'lr': args.lr})
    if args.auto_balance:
        # Give the optimizer access to any learnable loss scales we add in the criterion
        opt.add_param_group({'params': criterion.parameters(), 'lr': args.lr})
    
    for epoch in range(1, args.epochs+1):
        # ---- staged weights ----
        e = epoch
        w_qs    = args.qs_weight #* (1.0 - 0.5 * ramp(e, start=E1, duration=E2-E1))
        w_dip   = args.dipole_weight #* (1.0 - 0.5 * ramp(e, start=E1, duration=E2-E1))
        w_geo   = args.geo_weight * ramp(e, start=E1, duration=E2-E1)
        w_dir   = args.dir_weight * ramp(e, start=E1, duration=E2-E1)
        w_tr    = args.trace_weight * ramp(e, start=E1, duration=E2-E1)
        w_quad  = args.quad_weight  * ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_nll   = args.nll_weight  * ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_energy= args.energy_weight * ramp(e, start=E1, duration=max(1,(E2-E1)))
        
        # derive correlator geometry from args
        dipole_offsets = _parse_offsets_list(getattr(args, "dipole_offsets", "(1,0),(0,1),(2,0),(0,2)"))
        quad_pairs     = _parse_quad_pairs_list(getattr(args, "quad_pairs", "auto_from_dipole"), dipole_offsets=dipole_offsets)


        # Optional cap for the number of quadrupole configs (broader sets can be large)
        max_qp = int(getattr(args, "quad_max_pairs", 0) or 0)
        if max_qp > 0 and len(quad_pairs) > max_qp:
            rnd = random.Random(1337)
            tmp = list(quad_pairs)
            rnd.shuffle(tmp)
            quad_pairs = tuple(tmp[:max_qp])

        criterion = GroupLossWithQs(
            w_frob=0., w_unit=0.01,
            geo_weight=w_geo, dir_weight=w_dir, w_trace=w_tr,
            dipole_weight=w_dip, dipole_offsets=dipole_offsets,
            qs_weight=w_qs, qs_threshold=0.5, qs_on='N', qs_local=True,
            qs_soft_beta=args.qs_soft_beta,
            qs_soft_slope=args.qs_soft_slope,
            quad_weight=w_quad, quad_pairs=quad_pairs,
            mono_weight=float(getattr(args,"mono_weight",0.0)),
            qs_slope_weight=float(getattr(args,"qs_slope_weight",0.0)),
            moment_weight=float(getattr(args,"moment_weight",0.0)),
            nll_weight = w_nll,
            current_epoch = epoch, energy_weight = w_energy
        ).to(device) 

#        criterion= torch.compile(criterion, mode="reduce-overhead")

        # === Full-cov composer & loss feature flags ===
        # Infer alpha channel count (8 or 16) from your head; default to 8.
        alpha_channels = int(getattr(getattr(model, 'head', None), 'alpha_channels', 8))
        if args.nll_param_compose:
            # Hand the composer to the criterion so it can compute μ/Σ(Y)
            criterion.param_nll = model.param_nll
            # When composing to Y, compare to final α directly (no ΔY normalization inside loss)
            setattr(criterion, 'nll_target_mode', 'none')
            # Let the loss know which composition it should use
            setattr(criterion, 'nll_compose', args.nll_compose)
            # Tell the loss we want full-cov NLL
            setattr(criterion, 'use_fullcov', bool(args.nll_fullcov))
            # Optional extras
            setattr(criterion, 'bch_weight', float(args.bch_weight))
            setattr(criterion, 'energy_weight', float(args.energy_weight))
            setattr(criterion, 'energy_stride', int(args.energy_stride))

        model.train()
        rollout_k_curr, cons_w_d, sg_w_d = compute_curriculum_knobs(epoch, args, device)

        cons_w  = cons_w_d  * ramp(e, start=E1, duration=max(1,(E2-E1)))
        sg_w  = sg_w_d  * ramp(e, start=E1, duration=max(1,(E2-E1)))
        

        if (not is_ddp) or dist.get_rank() == 0:
            print(f"[curriculum] epoch={epoch:03d} rollout_k={rollout_k_curr} cons_w={cons_w:.3g} sg_w={sg_w:.3g}")
            
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

            
        # ---- device-side running sums ----
        tr_sum = torch.zeros((), device=device, dtype=torch.float64)
        tr_cnt = torch.zeros((), device=device, dtype=torch.float64)
        val_sum = torch.zeros((), device=device, dtype=torch.float64)
        val_cnt = torch.zeros((), device=device, dtype=torch.float64)

        opt.zero_grad(set_to_none=True)
        win_meter = DevMeter(device)


        # use_cuda_timing = torch.cuda.is_available()  # timing only when CUDA exists

        # if use_cuda_timing:
        #     e = {k: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        #          for k in ["h2d", "forward", "loss", "backward", "step"]}


        for it, (base18, Y_scalar, theta, y) in enumerate(train_dl, 1):
#            if torch.cuda.is_available():
#                e["h2d"][0].record()
            base18   = base18.to(device, non_blocking=True)      # [B,18,H,W]
            Y_scalar = Y_scalar.to(device, non_blocking=True)    # [B]
            theta    = theta.to(device, non_blocking=True)       # [B,3]
            y        = y.to(device, non_blocking=True)
 #           if torch.cuda.is_available():
 #               e["h2d"][1].record()

            if getattr(args, "channels_last", False) and device_type == "cuda":
                base18 = base18.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
  #              if torch.cuda.is_available():
  #                  e["forward"][0].record()
                # forward with components (no x tensor needed)
                ##----time----
                #if torch.cuda.is_available(): torch.cuda.synchronize();
                #t0=time.perf_counter()
                ##----end-time----
                yhat_single= model(base18, Y_scalar, theta)
#                yhat_single = model.forward(base18, Y_scalar, theta)
                yhat = yhat_single
                dy_scalar = Y_scalar  # replaces _y_from_batch(x, ...)
   #             if torch.cuda.is_available():
   #                 e["forward"][1].record()
                ##----time----
                #if torch.cuda.is_available(): torch.cuda.synchronize()
                #print(f"[timing] forward: {(time.perf_counter()-t0)*1e3:.2f} ms")
                ##----end-time----
                # #----time----
                # if torch.cuda.is_available(): torch.cuda.synchronize();
                # t0=time.perf_counter()
                # #----end-time----

                core = unwrap(model)
                mu_pred  = getattr(core.head, "last_mu", None)
                amp_pred = getattr(core.head, "last_amp", None)

                # (optional) if your code still keeps a legacy per-channel sigma head:
                logsig_pred = getattr(core.head, "last_y_sigma", None)
                #mu_pred, logsig_pred, y_gate = None, None, getattr(core.head, "last_y_sigma", None)
                y_gate = None
                
                if hasattr(model, "param_nll"):
                    # θ from channels 19..21
                    #p0 = int(getattr(args, "y_channel", 18)) + 1
                    Y_final = dy_scalar.to(device).float().view(-1)       # [B]

                    # θ -> (μ_unit, σ_unit [, κ]) -> compose to Y
                    mu_unit, sigma_unit, kappa = model.param_nll(theta)                    # [B,C], [B,C], [B,C?]
                    muY, sigY = model.param_nll.compose_to_Y(mu_unit, sigma_unit, kappa, Y_final)  # [B,C]

                    # Broadcast to spatial grid expected by the loss
                    B, C, H, W = y.shape[0], muY.shape[1], y.shape[-2], y.shape[-1]
                    mu_pred      = muY.view(B, C, 1, 1).expand(B, C, H, W)
                    logsig_pred  = softplus_inverse(sigY.view(B, C, 1, 1).expand(B, C, H, W))
                else:
                    # fall back to whatever the model head produced
                    mu_pred     = getattr(core.head, "last_mu", None)
                    logsig_pred = getattr(core.head, "last_logsig", None)                    # 1) Get θ=(m, Λ_QCD, μ0) and Y_final for this batch

                
                if mu_pred is None or logsig_pred is None:
                    print("[warn] last_mu/logsig missing; NLL will be inactive.")


#                #----time----
#                if torch.cuda.is_available(): torch.cuda.synchronize();
#                t0=time.perf_counter()
#                #----end-time----
        
                loss, stats = criterion(
                    yhat, y, base18,
                    dy_scalar=Y_scalar,          # [B]
                    theta=theta,                 # [B,3]
                    Y_final=Y_scalar,            # alias as in your code
                    mu_pred=mu_pred,
                    logsig_pred=logsig_pred,
                    drift_pred=mu_pred,         # drift per channel (what head calls μ)
                    amp_pred=amp_pred,          # physical amplitude s(x,y) ≥ 0
                    y_gate=y_gate,
                    return_stats=True
                )

#                #----time----
#                if torch.cuda.is_available(): torch.cuda.synchronize()
#                print(f"[timing] loss: {(time.perf_counter()-t0)*1e3:.2f} ms")
#                #----end-time----


                # ----- Rollout consistency (state) + rollout SINGLE consistency (transition) -----
                rollout_k = int(rollout_k_curr)
                w_state   = float(getattr(args, "rollout_consistency", 0.0))            # state match
                w_single  = float(getattr(args, "rollout_single_consistency", 0.0))     # transition match

                # Disable heavy stuff on CPU if requested
                if (base18.device.type == "cpu") and bool(getattr(args, "skip_heavy_on_cpu", True)):
                    w_state = 0.0
                    w_single = 0.0

                if rollout_k > 1 and (w_state > 0.0 or w_single > 0.0):
                    # k small steps vs one big step to reach the same rapidity k*ΔY
                    yhat_roll, yhat_single_big = rollout_predict(model, base18, Y_scalar, rollout_k, theta)

                    sg_offsets = getattr(args, "sg_dipole_offsets", None)

                    # ---- STATE consistency: match states at k*ΔY ----
                    if w_state > 0.0:
                        Nd = dipole_from_links18(yhat_single_big, offsets=sg_offsets)
                        Nc = dipole_from_links18(yhat_roll,       offsets=sg_offsets)
                        rollout_state_loss = F.mse_loss(Nd, Nc)
                        loss = loss + w_state * rollout_state_loss
                        meters.add("train/rollout_state_cons", rollout_state_loss.detach(), base18.size(0))

                    # ---- SINGLE consistency: match one-step transitions from that state ----
                    if w_single > 0.0:
                        next_from_roll = model(yhat_roll.detach().clone(),       Y_scalar, theta)
                        next_from_big  = model(yhat_single_big.detach().clone(), Y_scalar, theta)
                        Nr = dipole_from_links18(next_from_roll, offsets=sg_offsets)
                        Nb = dipole_from_links18(next_from_big,  offsets=sg_offsets)
                        rollout_single_loss = F.mse_loss(Nr, Nb)
                        loss = loss + w_single * rollout_single_loss
                        meters.add("train/rollout_single_cons", rollout_single_loss.detach(), base18.size(0))

                    del yhat_roll, yhat_single_big
                else:
                    meters.add("train/rollout_state_cons",  torch.zeros((), device=base18.device), base18.size(0))
                    meters.add("train/rollout_single_cons", torch.zeros((), device=base18.device), base18.size(0))



#                 rollout_k = int(rollout_k_curr)
#                 rollout_cons_w = float(cons_w)
#                 if (base18.device.type == 'cpu') and bool(getattr(args,'skip_heavy_on_cpu', True)):
#                     rollout_cons_w = 0.0

#                 # rollout consistency regularizer (only if both enabled)
#                 if (rollout_k > 1) and (rollout_cons_w > 0.0):
#                     yhat_roll, yhat_single_big = rollout_predict(model, base18, Y_scalar, rollout_k, theta)
#                     sg_offsets = getattr(args, "sg_dipole_offsets", None)
#                     Nd = dipole_from_links18(yhat_single_big, offsets=sg_offsets)  # (B,K)
#                     Nc = dipole_from_links18(yhat_roll,       offsets=sg_offsets)  # (B,K)
#                     rollout_loss = F.mse_loss(Nd, Nc)
# #                    cons = ((yhat_roll - yhat_single) ** 2).mean()
#                     loss = loss + rollout_cons_w * rollout_loss
#                     meters.add("train/rollout_cons", rollout_loss.detach(), base18.size(0))
#                     del yhat_roll, yhat_single_big, rollout_loss
#                 else:
#                     # keep the metric stream consistent; log zero when disabled
#                     meters.add("train/rollout_cons",
#                                torch.zeros((), device=base18.device), base18.size(0))

#                 # rollout-single consistency
#                 if int(getattr(args, "rollout_k", 1)) > 1 and float(getattr(args, "rollout_consistency", 0.0)) > 0.0 and (yhat_single is not None):
#                     cons = torch.mean((yhat_single - yhat) ** 2)
#                     loss = loss + float(args.rollout_consistency) * cons
#                     meters.add("train/rollout_cons", cons.detach(), base18.size(0))

                # --- semigroup ---
                # --- semigroup on dipoles instead of Wilson lines ---
                use_sg = (args.semigroup_weight > 0) and (torch.rand(()) < getattr(args, "semigroup_prob", 1.0))
                use_sg = use_sg and not (getattr(args, "skip_heavy_on_cpu", False) and not torch.cuda.is_available())

                if use_sg:
                    # Split Y into two parts: Y_a + Y_b = Y
                    # Option 1: random split
                    split = torch.rand((), device=base18.device)
                    Y_a = Y_scalar * split         # [B]
                    Y_b = Y_scalar - Y_a           # [B]
                    # (Option 2: deterministic half-split)
                    # Y_a = 0.5 * Y_scalar
                    # Y_b = Y_scalar - Y_a
                    # (Keep your existing code that creates Y_a, Y_b and runs the model twice.)
                    #
                    # Example skeleton (match to your variable names):
                    u_a       = model(base18, Y_a, theta)
                    u_comp    = model(u_a,   Y_b, theta)          # two-step (compose)
                    u_direct  = model(base18, Y_a + Y_b, theta)   # one-step (direct)

                    # Compute dipole summaries (B, K) for both branches
                    # Optionally, pass a fixed 'offsets' from args for reproducibility.
                    sg_offsets = getattr(args, "sg_dipole_offsets", None)
                    # If args.sg_dipole_offsets is a comma-separated string like "1,2,4",
                    # you can parse it into actual (dy,dx) pairs above where you parse args.

                    Nd = dipole_from_links18(u_direct, offsets=sg_offsets)  # (B,K)
                    Nc = dipole_from_links18(u_comp,   offsets=sg_offsets)  # (B,K)

                    sg_loss = F.mse_loss(Nd, Nc)

                    loss = loss + sg_w * sg_loss

                    meters.add("train/semigroup", sg_loss.detach(), base18.size(0))

                    
                # _sg_w = float(sg_w)
                # if (base18.device.type == 'cpu') and bool(getattr(args,'skip_heavy_on_cpu', True)):
                #     _sg_w = 0.0
                # sg_p = float(getattr(args, "semigroup_prob", 0.0))


                #     # Compose and compare (pass scalars, not channels)
                #     u_comp, u_direct = semigroup_compose(model, base18, Y_a, Y_b, theta)
                #     sgl = torch.mean((u_comp - u_direct) ** 2)
                #     loss = loss + _sg_w * sgl

                
            meters.add("train/frob", stats["frob"], base18.size(0))
            meters.add("train/unit", stats["unit"], base18.size(0))
                    
            # ---- accumulation-aware backward/step ----
            accum = int(args.accum)
            loss_unscaled = loss                      # for meters/logging/comparisons
            if accum > 1:
                loss = loss / accum                   # only for backward

            micro      = (it - 1) % max(1, accum)
            last_micro = (micro == max(1, accum) - 1)

            ddp_ctx = (model.no_sync() if (is_ddp and accum > 1 and not last_micro) else nullcontext())
            with ddp_ctx:
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # One-off grads diagnostic after a backward happened
            if it == 1:
                gnorm = _safe_unscale_and_grad_norm(opt, scaler, model)
                grad_ratio = _grad_to_weight_ratio(model)
                if (not is_ddp) or dist.get_rank() == 0:
                    print(f"[grad] unscaled ||g||={gnorm:.3e}  ||g||/||w||={grad_ratio:.3e}")
                meters.add("grad/norm", gnorm)
                meters.add("grad/ratio", grad_ratio)

            if last_micro:  # do the step on every rank
                core = unwrap(model)
                do_dw = (it <= 1) or (PROFILE_DW_EVERY and it % PROFILE_DW_EVERY == 0)
                if do_dw:
                    before_vec = flat_params(core)

                if use_amp and getattr(scaler, 'is_enabled', lambda: False)():
                    # scaler.unscale_(opt)  # enable if you clip every step
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.update(core)

                if do_dw:
                    with torch.no_grad():
                        before_vec = before_vec.detach()
                        after_vec  = flat_params(core).detach()
                        dw        = (after_vec - before_vec)
                        denom     = after_vec.norm().clamp_min(1e-12)
                        dw_ratio  = (dw.norm() / denom)                    # 0-dim tensor on GPU
                        dw_ratio_mean = _dist_mean_scalar(dw_ratio)        # all ranks call this

                    if (not is_ddp) or dist.get_rank() == 0:
                        last_dw_ratio = float(dw_ratio_mean.detach().cpu().item())  # one sync, rank-0
                        print(f"[step] ||Δw||/||w|| = {last_dw_ratio:.3e}")
                    else:
                        last_dw_ratio = None  # or keep the tensor if you use it later

                    # if do_dw:
                    # with torch.no_grad():
                    #     before_vec = before_vec.detach()          # assume you saved this last step
                    #     after_vec  = flat_params(core).detach()   # keep this on GPU
                    #     dw = (after_vec - before_vec)
                    #     # avoid division by ~0
                    #     denom = after_vec.norm().clamp_min(1e-12)
                    #     dw_ratio = (dw.norm() / denom)            # 0-dim tensor on GPU
                    #     # (optional) average across ranks so the printed value is global, not local
                    #     dw_ratio_mean = _dist_mean_scalar(dw_ratio)

                    # if is_main():
                    #     last_dw_ratio = float(dw_ratio_mean.detach().cpu().item())  # one sync, rank-0 only
                    #     print(f"[step] ||Δw||/||w|| = {last_dw_ratio:.3e}")
                    # else:
                    #     # keep a tensor on device if you need it later; or skip storing on non-main
                    #     last_dw_ratio = dw_ratio_mean

        # if do_dw:
        #             after_vec = flat_params(core)
        #             dw_ratio = (after_vec - before_vec).norm() / (after_vec.norm().clamp_min(1e-12))
        #             last_dw_ratio = float(dw_ratio.item())
        #             if ((not is_ddp) or dist.get_rank()==0):
        #                 print(f"[step] ||Δw||/||w|| = {last_dw_ratio:.3e}")

            # ---- running stats ----
            tr_sum += loss_unscaled.detach().to(tr_sum.dtype)
            tr_cnt += 1
            win_meter.update(loss_unscaled.detach())
            meters.add("train/total", loss_unscaled.detach(), base18.size(0))

            # Throttled logging
            if (it % LOG_EVERY == 0) and (((not is_ddp) or dist.get_rank()==0)):
                mean_loss_t = win_meter.mean()
                (mean_loss,) = _pack_and_tolist(mean_loss_t)
                print(f"[train] ep={epoch:03d} it={it:05d}  loss(win)={mean_loss:.5g}")
                win_meter = DevMeter(device)

        # leftover grads (if epoch ended mid-accum window)
        if (tr_cnt % max(1, accum)) != 0:
            if use_amp and getattr(scaler, 'is_enabled', lambda: False)():
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(unwrap(model))

        ybins = _init_ybins()
        nbins = len(ybins)-1
        bin_loss_sum = torch.zeros(nbins, device=device, dtype=torch.float32)
        bin_count    = torch.zeros(nbins, device=device, dtype=torch.float32)


        head = getattr(model, 'head', None)
        if head is not None and hasattr(head, 'last_eta_rms'):
            print(f"[debug] mode={head.last_sigma_mode}  eta_rms={float(head.last_eta_rms):.3e}  "
                  f"sigma_mean={float(head.last_sigma_mean):.3e}  "
                  f"sigma_min={float(head.last_sigma_min):.3e}  "
                  f"sigma_max={float(head.last_sigma_max):.3e}  "
                  f"op_gain={float(head.op_gain.detach()):.3e}")

        vt = stats.get('mm_v_true'); vp = stats.get('mm_v_pred'); rc = stats.get('mm_calib')
        if (vt is not None) and (vp is not None) and (rc is not None):
            print(f"[moment calib] v_true={float(vt):.3e}  v_pred={float(vp):.3e}  ratio={float(rc):.3f}")
        # Show the physical diffusion amplitude (per pixel scalar), and what actually multiplies the noise.

        # in the training loop, after you compute moment calib
        ratio = rc # use batch means from your prints
        err_t = safe_ratio_err(ratio, device=base18.device if torch.is_tensor(base18) else None)
        err = err_t.item()
        #err = abs(math.log(max(ratio, 1e-8)))              # 0 when perfect; grows with mismatch
        mw_base = args.moment_weight
        mw = mw_base * min(1.0, err / 0.5)                 # full weight if >~65% off; fades near match
        mw = max(mw, 0.0005)                                # keep a tiny floor
        criterion.moment_weight = mw


        #core   = unwrap(model)
        #amp    = getattr(core.head, "last_amp", None)
        # if amp is not None:
        #     s_phys = amp.detach()
        #     s_used = (core.head.op_gain.detach() * s_phys)
        #     print(f"[s_phys] mean={s_phys.mean():.3e}  min={s_phys.min():.3e}  max={s_phys.max():.3e}   [s_used] mean={s_used.mean():.3e}")

        
        # try:
        #     with torch.no_grad():
        #         stats_print = stats_val if isinstance(stats_val, dict) else criterion.components_dict(yh, y, x[:, :18])
        #         if (not is_ddp) or dist.get_rank()==0:
        #             print(f"[val diag] frob={stats_print['frob']:.3e} geoθ2={stats_print.get('geo_theta2',0):.3e} "
        #                   f"dip_mse={stats_print.get('dip_mse',0):.3e} qs_mse={stats_print.get('qs_mse',0):.3e} "
        #                   f"Qs̄pred={stats_print.get('qs_pred_mean',0):.3e} Qs̄true={stats_print.get('qs_true_mean',0):.3e}")
                    
        #             # ---- Full-cov α-NLL monitor (if present) ----
        #             if 'alpha_nll_full' in stats:
        #                 print(  "[αNLL(fullcov)] "
        #                     f"NLL={stats['alpha_nll_full']:.3e}  "
        #                     f"maha={stats['alpha_maha']:.3e}  "
        #                     f"log|Σ|={stats['alpha_logdet']:.3e}  "
        #                     f"σ̄={stats['alpha_sigma_bar']:.3e}  "
        #                     f"σ<1e-3={stats['alpha_sigma_small_frac']:.3f}  "
        #                     f"off/diag={stats['alpha_offdiag_ratio']:.3f}  "
        #                     f"cond={stats['alpha_cond']:.2e}  "
        #                     f"λmin={stats['alpha_lambda_min']:.3e}  "
        #                     f"ρ(ᾱ,μ)={stats['alpha_corr']:.3f}  "
        #                     f"|μ|/|ᾱ|={stats['alpha_abs_ratio']:.3f}  "
        #                     f"alpha_mah2_mean={stats['alpha_mah2_mean']:.3f}  "
        #                     f"alpha_mah2_p90={stats['alpha_mah2_p90']:.3f}  "
        #                     f"alpha_mah2_p={stats['alpha_mah2_p']:.3f}  "
        #                     f"alpha_lammin_med={stats['alpha_lammin_med']:.3f}  "
        #                 )
                    
        #             r2_now   = stats_print.get('alpha_R2',  float('nan'))
        #             corr_now = stats_print.get('alpha_corr',float('nan'))
        #             nll_now  = stats_print.get('alpha_nll', float('nan'))
        #             msgs = []
        #             if isinstance(r2_now, float)  and math.isfinite(r2_now)  and (r2_now  > best_alpha_R2):
        #                 msgs.append(f"αR2 ↑ {best_alpha_R2:.3f} → {r2_now:.3f}")
        #                 best_alpha_R2 = r2_now
        #             if isinstance(corr_now, float) and math.isfinite(corr_now) and (corr_now > best_alpha_corr):
        #                 msgs.append(f"αcorr ↑ {best_alpha_corr:.3f} → {corr_now:.3f}")
        #                 best_alpha_corr = corr_now
        #             if isinstance(nll_now, float)  and math.isfinite(nll_now)  and (nll_now  < best_alpha_nll):
        #                 msgs.append(f"αNLL ↓ {best_alpha_nll:.3e} → {nll_now:.3e}")
        #                 best_alpha_nll = nll_now
        #             if msgs:
        #                 print("[monitor] " + " | ".join(msgs))


        #             if args.nll_param_compose and is_main():
        #                 if args.nll_compose == 'brownian':
        #                     # all GPU ops
        #                     yv = (Y_final + 1e-9).sqrt()  # [B]
        #                     mu_norm_t = (muY.abs().mean(dim=1) / (Y_final + 1e-9)).mean()  # 0-dim tensor
        #                     sig_norm_t = (sigY.abs().mean(dim=1) / yv).mean()              # 0-dim tensor
                            
        #                     # (optional) average across ranks before printing
        #                     mu_norm_t  = _dist_mean_scalar(mu_norm_t)
        #                     sig_norm_t = _dist_mean_scalar(sig_norm_t)
                            
        #                     # single host transfer each
        #                     mu_norm  = float(mu_norm_t.detach().cpu().item())
        #                     sig_norm = float(sig_norm_t.detach().cpu().item())
        #                     print(f"[scaling] ⟨|μ|/Y⟩≈{mu_norm:.3e},  ⟨|σ|/√Y⟩≈{sig_norm:.3e}")
        #                 else:  # OU
        #                     kappa_bar_t = kappa.mean()
        #                     kappa_bar_t = _dist_mean_scalar(kappa_bar_t)  # optional
        #                     kappa_bar = float(kappa_bar_t.detach().cpu().item())
        #                     print(f"[OU] κ̄≈{kappa_bar:.3e}")                                      
        # except Exception:
        #     pass

#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize();
#        t0=time.perf_counter()
#        #----end-time----
        # ----------------- VALIDATION -----------------
        # Accumulators for Y-evolution diagnostics
        val_qslope_list = []
        val_mono_list   = []
        val_dy_all, val_qp_all, val_qt_all = [], [], []
        val_q4p_all, val_q4t_all = [], []  # accumulate quadrupole preds/targets per batch
        
        if val_sampler is not None:
            try: val_sampler.set_epoch(epoch)
            except Exception: pass

        ema_eval = (ema is not None) and getattr(args, "ema_eval", True)
        if ema_eval: ema.swap_in(unwrap(model))

        model.eval()
        first_nonzeroY_probed = False


        if (not is_ddp) or dist.get_rank() == 0:
            dev = device  # same device as tensors below
            qs_stats_t = {
                "Ymax": torch.tensor(-1e30, device=dev),
                "idx_Ymax": torch.tensor([-1, -1], device=dev, dtype=torch.long),  # [batch_idx, sample_idx]
                "Qs_pred_at_Ymax": torch.tensor(0.0, device=dev),
                "Qs_true_at_Ymax": torch.tensor(0.0, device=dev),
                
                "Qsmax_true": torch.tensor(-1e30, device=dev),
                "Qs_pred_at_Qsmax_true": torch.tensor(0.0, device=dev),
                "Y_at_Qsmax_true": torch.tensor(0.0, device=dev),
                
                # only if qs_local=True
                "Qsmax_true_px": torch.tensor(-1e30, device=dev),
                "Y_at_Qsmax_true_px": torch.tensor(0.0, device=dev),
                "Qs_pred_at_Qsmax_true_px": torch.tensor(0.0, device=dev),
            }


        # # ---------- before the val loop (once per epoch, rank 0) ----------
        # if (not is_ddp) or dist.get_rank() == 0:
        #     qs_stats = {
        #         "Ymax": -1e30,
        #         "idx_Ymax": None,
        #         "Qs_pred_at_Ymax": None,
        #         "Qs_true_at_Ymax": None,
        #         "Y_at_Qsmax_true": None,
        #         "Qsmax_true": -1e30,
        #         "Qs_pred_at_Qsmax_true": None,
        #         # if you want per-pixel maxima when qs_local=True
        #         "Qsmax_true_px": -1e30,
        #         "Y_at_Qsmax_true_px": None,
        #         "Qs_pred_at_Qsmax_true_px": None,
        #     }


        with torch.no_grad():
            for i, (base18, Y_scalar, theta, y) in enumerate(val_dl):
                base18   = base18.to(device, non_blocking=True)
                Y_scalar = Y_scalar.to(device, non_blocking=True)
                theta    = theta.to(device, non_blocking=True)
                y        = y.to(device, non_blocking=True)
                if getattr(args, "channels_last", False) and device_type == "cuda":
                    base18 = base18.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
#                    yh = model.forward(base18, Y_scalar, theta)
                    yh = model(base18, Y_scalar, theta)
                    dy_scalar = Y_scalar  # [B]

                    mu_pred, logsig_pred = None, None
                    if hasattr(model, "param_nll"):
                        #p0 = int(getattr(args, "y_channel", 18)) + 1
                        Y_final = dy_scalar.to(device).float().view(-1)
                        mu_unit, sigma_unit, kappa = model.param_nll(theta)
                        muY, sigY = model.param_nll.compose_to_Y(mu_unit, sigma_unit, kappa, Y_final)
                        B, C, H, W = y.shape[0], muY.shape[1], y.shape[-2], y.shape[-1]
                        mu_pred     = muY.view(B, C, 1, 1).expand(B, C, H, W)
                        logsig_pred = softplus_inverse(sigY.view(B, C, 1, 1).expand(B, C, H, W))
                    else:
                        core = unwrap(model)
                        mu_pred     = getattr(core.head, "last_mu", None)
                        logsig_pred = getattr(core.head, "last_logsig", None)


                    # y_gate is optional; define safely for this scope
                    try:
                        core = unwrap(model)
                        y_gate = getattr(core.head, "last_y_sigma", None)
                    except Exception:
                        y_gate = None
                    

                    l, stats_val = criterion(
                        yh, y, base18,
                        dy_scalar = Y_scalar,
                        theta=theta,                     # [B,3] for the composer
                        Y_final=Y_scalar,               # alias; clearer name inside loss
                        mu_pred     = mu_pred,
                        logsig_pred = logsig_pred,
                        drift_pred=mu_pred,         # drift per channel (what head calls μ)
                        amp_pred=amp_pred,          # physical amplitude s(x,y) ≥ 0
                        y_gate = y_gate,
                        return_stats = True
                    )

                # Collect batch-level slope/mono stats (if present)
                if isinstance(stats_val, dict):
                    if "qs_slope_mse" in stats_val:
                        val_qslope_list.append(float(stats_val["qs_slope_mse"]))
                    if "mono_violation" in stats_val:
                        val_mono_list.append(float(stats_val["mono_violation"]))

                # --- Per-sample Qs for ΔY-binned diagnostics ---
                try:
                    # Use the same helpers as the loss for consistency
                    Uh = criterion._pack18_to_U(yh)  # complex [B,H,W,3,3]
                    Ut = criterion._pack18_to_U(y)
                    if not getattr(criterion, "project_before_frob", False):
                        Uh = criterion._su3_project(Uh)
                        Ut = criterion._su3_project(Ut)
                    qp = criterion._compute_Qs_from_U(Uh, local=False)  # [B]
                    qt = criterion._compute_Qs_from_U(Ut, local=False)  # [B]
                    val_qp_all.append(qp.detach().cpu().numpy())
                    val_qt_all.append(qt.detach().cpu().numpy())
                    val_dy_all.append(dy_scalar.view(-1).detach().cpu().numpy())

                except Exception:
                    pass

                # Quadrupole accumulators per batch, if available
                try:
                    if getattr(criterion, "quad_weight", 0.0) != 0.0 and len(getattr(criterion, "quad_pairs", [])) > 0:
                        Q4h = criterion._quadrupole(Uh)  # [B, n_pairs]
                        Q4t = criterion._quadrupole(Ut)  # [B, n_pairs]
                        val_q4p_all.append(Q4h.detach().cpu().numpy())
                        val_q4t_all.append(Q4t.detach().cpu().numpy())
                except Exception:
                    pass
                

                val_sum += l.detach().to(val_sum.dtype)
                val_cnt += 1
                meters.add("val/total", l.detach(), base18.size(0))

                f,tr,un,de = criterion.components_mean(yh, y)
                meters.add("val/frob", f, base18.size(0))
                meters.add("val/unit", un, base18.size(0))

                # per-sample proxy loss and Y-binning
                per = (yh - y)
                per = (per.real.pow(2) + per.imag.pow(2)) if per.is_complex() else per.pow(2)
                per = per.mean(dim=(1,2,3))  # [B]
                y_clamped = Y_scalar.clamp(min=y_min_data, max=y_max_data).to(torch.float64)
                idx = torch.bucketize(y_clamped, ybins) - 1           # [B], int64
                idx = idx.clamp_(0, nbins-1)
                bin_loss_sum.scatter_add_(0, idx, per.to(torch.float32))
                bin_count.scatter_add_(0, idx, torch.ones_like(per, dtype=torch.float32))


                # ---------- inside the val loop, after you have y_scalar, y, yh ----------
                # y_scalar must be [B] with the Y for each sample
                mask_pos = (Y_scalar > 0)

                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    Uh = criterion._pack18_to_U(yh)
                    U  = criterion._pack18_to_U(y)
                    Qh = criterion._compute_Qs_from_U(Uh, local=getattr(criterion, "qs_local", False))
                    Qt = criterion._compute_Qs_from_U(U,  local=getattr(criterion, "qs_local", False))

                # Reduce to per-sample scalars (mean over HxW) for selection/print
                if Qh.ndim > 1:
                    Qh_b = Qh.mean(dim=(-1, -2))
                    Qt_b = Qt.mean(dim=(-1, -2))
                else:
                    Qh_b, Qt_b = Qh, Qt

                if (not is_ddp) or dist.get_rank() == 0:
                    # 1) Largest Y (per epoch)
                    if mask_pos.any():
                        pos_idx = torch.nonzero(mask_pos, as_tuple=False).squeeze(1)  # [M]
                        y_pos = Y_scalar[mask_pos].view(-1)                           # [M]
                        y_max_batch, j = y_pos.max(dim=0)                             # 0-dim, idx
                        if y_max_batch > qs_stats_t["Ymax"]:
                            gidx = pos_idx[j]
                            qs_stats_t["Ymax"] = y_max_batch.detach()
                            qs_stats_t["idx_Ymax"] = torch.stack([torch.tensor(i, device=dev), gidx])
                            qs_stats_t["Qs_pred_at_Ymax"] = Qh_b[gidx].detach()
                            qs_stats_t["Qs_true_at_Ymax"] = Qt_b[gidx].detach()

                    # 2) Largest *true* Qs (per-sample scalar)
                    Qt_max_batch, k = Qt_b.max(dim=0)
                    if Qt_max_batch > qs_stats_t["Qsmax_true"]:
                        qs_stats_t["Qsmax_true"] = Qt_max_batch.detach()
                        qs_stats_t["Qs_pred_at_Qsmax_true"] = Qh_b[k].detach()
                        qs_stats_t["Y_at_Qsmax_true"] = Y_scalar[k].detach().view(())

                    # 3) Optional: largest *pixel* Qs
                    if Qt.ndim > 1:
                        Qt_px_max = Qt.amax(dim=(-1, -2))  # [B]
                        v, kk = Qt_px_max.max(dim=0)       # 0-dim, idx
                        if v > qs_stats_t["Qsmax_true_px"]:
                            qs_stats_t["Qsmax_true_px"] = v.detach()
                            qs_stats_t["Y_at_Qsmax_true_px"] = Y_scalar[kk].detach().view(())
                            qs_stats_t["Qs_pred_at_Qsmax_true_px"] = Qh_b[kk].detach()


                # mask_pos = (Y_scalar > 0)
                # with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                #     Uh = criterion._pack18_to_U(yh)  # predicted links
                #     U  = criterion._pack18_to_U(y)   # true links
                #     Qh = criterion._compute_Qs_from_U(Uh, local=getattr(criterion, "qs_local", False))
                #     Qt = criterion._compute_Qs_from_U(U,  local=getattr(criterion, "qs_local", False))

                # # Reduce to per-sample scalars (mean over HxW) for printing/selection
                # if Qh.ndim > 1:   # e.g., [B,H,W]
                #     Qh_b = Qh.mean(dim=(-1, -2))
                #     Qt_b = Qt.mean(dim=(-1, -2))
                # else:
                #     Qh_b, Qt_b = Qh, Qt

                # if (not is_ddp) or dist.get_rank() == 0:    
                # # 1) Track the sample with the largest Y in the whole val set
                #     if mask_pos.any():
                #         pos_idx = torch.nonzero(mask_pos, as_tuple=False).squeeze(1)
                #         y_pos = Y_scalar[mask_pos]
                #         y_max_batch, j = y_pos.max(dim=0)
                #         if y_max_batch.item() > qs_stats["Ymax"]:
                #             gidx = pos_idx[j]
                #             qs_stats["Ymax"] = y_max_batch.item()
                #             qs_stats["idx_Ymax"] = (i, int(gidx))
                #             qs_stats["Qs_pred_at_Ymax"] = float(Qh_b[gidx].item())
                #             qs_stats["Qs_true_at_Ymax"] = float(Qt_b[gidx].item())

                #     # 2) Track the sample with the largest *true* Qs (per-sample scalar)
                #     Qt_max_batch, k = Qt_b.max(dim=0)
                #     if Qt_max_batch.item() > qs_stats["Qsmax_true"]:
                #         qs_stats["Qsmax_true"] = float(Qt_max_batch.item())
                #         qs_stats["Qs_pred_at_Qsmax_true"] = float(Qh_b[k].item())
                #         qs_stats["Y_at_Qsmax_true"] = float(Y_scalar[k].item())


                #     # 3) (Optional) If qs_local=True, also track the largest *pixel* Qs across all samples/pixels
                #     if Qt.ndim > 1:
                #         Qt_px_max = Qt.amax(dim=(-1, -2))    # per-sample pixel max
                #         Qh_px_at_max = Qh.reshape(Qh.shape[0], -1)
                #         Qt_px_at_max = Qt.reshape(Qt.shape[0], -1)
                #         # find per-batch absolute pixel max and its sample index
                #         v, kk = Qt_px_max.max(dim=0)
                #         if v.item() > qs_stats["Qsmax_true_px"]:
                #             qs_stats["Qsmax_true_px"] = float(v.item())
                #             sidx = int(kk.item())
                #             qs_stats["Y_at_Qsmax_true_px"] = float(Y_scalar[sidx].item())
                #             # For a fair pair, report the *sample-level* predicted mean at that sample
                #             qs_stats["Qs_pred_at_Qsmax_true_px"] = float(Qh_b[sidx].item())


                
                # --- Print Qs at Y_max and at a middle Y (median over positive Y) ---
                if ((not is_ddp) or dist.get_rank() == 0) and (i == 0):  # print once per epoch on rank 0
                    mask_pos = (Y_scalar > 0)
                    if mask_pos.any():
                        # pick Y_max
                        y_pos = Y_scalar[mask_pos]
                        pos_idx = torch.nonzero(mask_pos, as_tuple=False).squeeze(1)
                        i_max = pos_idx[torch.argmax(y_pos)]

                        # pick a "middle" Y: nearest to the median of positive Y in this batch
                        y_med = y_pos.median()
                        i_mid = pos_idx[torch.argmin((y_pos - y_med).abs())]

                        # compute Qs for the whole batch once, then index
                        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                            Uh = criterion._pack18_to_U(yh)  # predicted links
                            U  = criterion._pack18_to_U(y)   # true links
                            Qh = criterion._compute_Qs_from_U(Uh, local=getattr(criterion, "qs_local", False))
                            Qt = criterion._compute_Qs_from_U(U,  local=getattr(criterion, "qs_local", False))

                        # if local=True, Qs is [B,H,W]; reduce to per-sample scalars to print
                        if Qh.ndim > 1:
                            Qh_b = Qh.mean(dim=(-1, -2))
                            Qt_b = Qt.mean(dim=(-1, -2))
                        else:
                            Qh_b, Qt_b = Qh, Qt

                        print(
                            f"[val Qs(Ymax)] ep={epoch:03d} step={i+1:05d} "
                            f"Y={Y_scalar[i_max].item():.3e}  "
                            f"Qs_pred={Qh_b[i_max].item():.3e}  Qs_true={Qt_b[i_max].item():.3e}"
                        )
                        print(
                            f"[val Qs(Ymid)] ep={epoch:03d} step={i+1:05d} "
                            f"Y≈{Y_scalar[i_mid].item():.3e}  "
                            f"Qs_pred={Qh_b[i_mid].item():.3e}  Qs_true={Qt_b[i_mid].item():.3e}"
                        )
        
                # mask_pos = (Y_scalar.abs() > 1e-12)
                # if (not is_ddp) or dist.get_rank() == 0:
                #     pos_idx = torch.nonzero(mask_pos, as_tuple=False).squeeze(1)
                #     i_max = pos_idx[torch.argmax(Y_scalar[mask_pos])]
                #     # Rebuild a minimal x=[base18|Y|theta] for the probe
                #     B, _, H, W = base18.shape
                #     Y_map     = Y_scalar.view(B, 1, 1, 1).expand(B, 1, H, W)
                #     theta_map = theta.view(B, 3, 1, 1).expand(B, 3, H, W)
                #     x_probe   = torch.cat([base18, Y_map, theta_map], dim=1)  # [B, 22, H, W]
                #     log_evolution_probe(model, x_probe[i_max:i_max+1], prefix="val(Ymax>0)", epoch=epoch, step=i+1)

        # ---------- after the val loop (once per epoch, rank 0) ----------

        # one diagnostic print per epoch (ok if it syncs internally)
        try:
            with torch.no_grad():
                # Prefer stats returned by the loss, otherwise recompute; avoid touching an undefined name
                stats_src = locals().get("stats_val", None)
                if not isinstance(stats_src, dict):
                    # Recompute components from current predictions/targets
                    # Use base18 (not x[:, :18]) since that's what you pass into the model
                    stats_src = criterion.components_dict(yhat, y, base18)

            # Rank-0 printing only
            if (not is_ddp) or dist.get_rank() == 0:
                # Safe getters → Python floats (single host transfer)
                frob         = float(stats_src.get("frob", float("nan")))
                geo_theta2   = float(stats_src.get("geo_theta2", 0.0))
                dip_mse      = float(stats_src.get("dip_mse", 0.0))
                qs_mse       = float(stats_src.get("qs_mse", 0.0))
                quad_mse     = float(stats_src.get("quad_mse", 0.0))

                print(
                    "[val diag] "
                    f"frob={frob:.3e} "
                    f"geoθ2={geo_theta2:.3e} "
                    f"dip_mse={dip_mse:.3e} "
                    f"quad_mse={quad_mse:.3e} "
                    f"qs_mse={qs_mse:.3e} "
                )

                # ---- Full-cov α-NLL monitor (only if present) ----
                if "alpha_nll_full" in stats_src:
                    line = (
                        "[αNLL(fullcov)] "
                        f"NLL={stats_src['alpha_nll_full']:.3e}  "
                        f"maha={stats_src['alpha_maha']:.3e}  "
                        f"log|Σ|={stats_src['alpha_logdet']:.3e}  "
                        f"σ̄={stats_src['alpha_sigma_bar']:.3e}  "
                        f"σ<1e-3={stats_src['alpha_sigma_small_frac']:.3f}  "
                        f"off/diag={stats_src['alpha_offdiag_ratio']:.3f}  "
                        f"cond={stats_src['alpha_cond']:.2e}  "
                        f"λmin={stats_src['alpha_lambda_min']:.3e}  "
                        f"ρ(ᾱ,μ)={stats_src['alpha_corr']:.3f}  "
                        f"|μ|/|ᾱ|={stats_src['alpha_abs_ratio']:.3f}  "
                        f"alpha_mah2_mean={stats_src['alpha_mah2_mean']:.3f}  "
                        f"alpha_mah2_p90={stats_src['alpha_mah2_p90']:.3f}  "
                        f"alpha_mah2_p={stats_src['alpha_mah2_p']:.3f}  "
                        f"alpha_lammin_med={stats_src['alpha_lammin_med']:.3f}"
                    )
                    print(line)

                # Progress monitors (use local vars that definitely exist)
                r2_now   = stats_src.get("alpha_R2",  float("nan"))
                corr_now = stats_src.get("alpha_corr",float("nan"))
                nll_now  = stats_src.get("alpha_nll", float("nan"))
                msgs = []
                if isinstance(r2_now, (float, int)) and math.isfinite(r2_now) and (r2_now > best_alpha_R2):
                    msgs.append(f"αR2 ↑ {best_alpha_R2:.3f} → {r2_now:.3f}")
                    best_alpha_R2 = r2_now
                if isinstance(corr_now, (float, int)) and math.isfinite(corr_now) and (corr_now > best_alpha_corr):
                    msgs.append(f"αcorr ↑ {best_alpha_corr:.3f} → {corr_now:.3f}")
                    best_alpha_corr = corr_now
                if isinstance(nll_now, (float, int)) and math.isfinite(nll_now) and (nll_now < best_alpha_nll):
                    msgs.append(f"αNLL ↓ {best_alpha_nll:.3e} → {nll_now:.3e}")
                    best_alpha_nll = nll_now
                if msgs:
                    print("[monitor] " + " | ".join(msgs))

                # ---- Parameter-composed NLL scaling (GPU/DDP friendly) ----
                if args.nll_param_compose:
                    if args.nll_compose == "brownian" and all(k in locals() for k in ("muY", "sigY", "Y_final")):
                        # all GPU ops
                        yv = (Y_final + 1e-9).sqrt()
                        mu_norm_t  = (muY.abs().mean(dim=1) / (Y_final + 1e-9)).mean()  # 0-dim tensor on device
                        sig_norm_t = (sigY.abs().mean(dim=1) / yv).mean()                # 0-dim tensor on device

                        # (optional) average across ranks; _dist_mean_scalar should keep the tensor on device
                        mu_norm_t  = _dist_mean_scalar(mu_norm_t)
                        sig_norm_t = _dist_mean_scalar(sig_norm_t)

                        if (not is_ddp) or dist.get_rank() == 0:
                            # single host transfers, rank-0 only
                            mu_norm  = float(mu_norm_t.detach().cpu().item())
                            sig_norm = float(sig_norm_t.detach().cpu().item())
                            print(f"[scaling] ⟨|μ|/Y⟩≈{mu_norm:.3e},  ⟨|σ|/√Y⟩≈{sig_norm:.3e}")

                    elif args.nll_compose != "brownian" and ("kappa" in locals()):
                        kappa_bar_t = kappa.mean()                 # stays on device
                        kappa_bar_t = _dist_mean_scalar(kappa_bar_t)  # optional DDP average
                        if (not is_ddp) or dist.get_rank() == 0:
                            kappa_bar = float(kappa_bar_t.detach().cpu().item())
                            print(f"[OU] κ̄≈{kappa_bar:.3e}")


                # # print occasionally
                # if use_cuda_timing and (it % 50 == 0) and ((not is_ddp) or dist.get_rank() == 0):
                #     torch.cuda.synchronize()
                #     t = {k: e[k][0].elapsed_time(e[k][1]) for k in e}
                #     print(f"ms/step: h2d={t['h2d']:.1f} fwd={t['forward']:.1f} "
                #           f"loss={t['loss']:.1f} bwd={t['backward']:.1f} step={t['step']:.1f}")
                        
                # if torch.cuda.is_available() and (it % 50 == 0) and ((not is_ddp) or dist.get_rank()==0):
                #     torch.cuda.synchronize()
                #     t = {k: e[k][0].elapsed_time(e[k][1]) for k in e}
                #     print(f"ms/step: h2d={t['h2d']:.1f} fwd={t['forward']:.1f} "
                #           f"loss={t['loss']:.1f} bwd={t['backward']:.1f} step={t['step']:.1f}")

                            
            # # ---- Parameter-composed NLL scaling (only if those tensors exist) ----
            #     if args.nll_param_compose:
            #         if args.nll_compose == "brownian" and all(k in locals() for k in ("muY", "sigY", "Y_final")):
            #             yv = (Y_final + 1e-9).sqrt()
            #             mu_norm_t  = (muY.abs().mean(dim=1) / (Y_final + 1e-9)).mean()
            #             sig_norm_t = (sigY.abs().mean(dim=1) / yv).mean()
            #             mu_norm, sig_norm = [float(t.detach().cpu().item()) for t in (mu_norm_t, sig_norm_t)]
            #             print(f"[scaling] ⟨|μ|/Y⟩≈{mu_norm:.3e},  ⟨|σ|/√Y⟩≈{sig_norm:.3e}")
            #         elif args.nll_compose != "brownian" and "kappa" in locals():
            #             kappa_bar = float(kappa.mean().detach().cpu().item())
            #             print(f"[OU] κ̄≈{kappa_bar:.3e}")

        except Exception as e:
            # Keep the job running even if diagnostics fail
            if (not is_ddp) or dist.get_rank() == 0:
                print(f"[val diag] skipped due to: {e}")


        if (not is_ddp) or dist.get_rank() == 0:
            qs_stats = {
                "Ymax": float(qs_stats_t["Ymax"].detach().cpu().item()),
                "idx_Ymax": tuple(int(x) for x in qs_stats_t["idx_Ymax"].detach().cpu().tolist()),
                "Qs_pred_at_Ymax": float(qs_stats_t["Qs_pred_at_Ymax"].detach().cpu().item()),
                "Qs_true_at_Ymax": float(qs_stats_t["Qs_true_at_Ymax"].detach().cpu().item()),
                "Qsmax_true": float(qs_stats_t["Qsmax_true"].detach().cpu().item()),
                "Qs_pred_at_Qsmax_true": float(qs_stats_t["Qs_pred_at_Qsmax_true"].detach().cpu().item()),
                "Y_at_Qsmax_true": float(qs_stats_t["Y_at_Qsmax_true"].detach().cpu().item()),
                "Qsmax_true_px": float(qs_stats_t["Qsmax_true_px"].detach().cpu().item()),
                "Y_at_Qsmax_true_px": float(qs_stats_t["Y_at_Qsmax_true_px"].detach().cpu().item()),
                "Qs_pred_at_Qsmax_true_px": float(qs_stats_t["Qs_pred_at_Qsmax_true_px"].detach().cpu().item()),
            }

            # print(f"[val Qs(global Ymax)]  Y={qs_stats['Ymax']:.6e}  "
            #       f"Qs_pred={qs_stats['Qs_pred_at_Ymax']:.6e}  Qs_true={qs_stats['Qs_true_at_Ymax']:.6e}  "
            #       f"@batch_idx={qs_stats['idx_Ymax']}")
            print(f"[val Qs(global Qsmax_true, sample-mean)] Qs_pred={qs_stats['Qs_pred_at_Qsmax_true']:.6e}  "
                  f"Qs_true={qs_stats['Qsmax_true']:.6e}  at Y={qs_stats['Y_at_Qsmax_true']:.6e}")
            # if qs_stats["Qsmax_true_px"] > -1e20:
            #     print(f"[val Qs(global Qsmax_true, pixel)]      Qs_true_pxmax={qs_stats['Qsmax_true_px']:.6e}  "
            #           f"(sample’s mean Qs_pred={qs_stats['Qs_pred_at_Qsmax_true_px']:.6e})  "
            #           f"at Y={qs_stats['Y_at_Qsmax_true_px']:.6e}")

            
            # print(f"[val Qs(global Ymax)]  Y={qs_stats['Ymax']:.6e}  "
            #       f"Qs_pred={qs_stats['Qs_pred_at_Ymax']:.6e}  Qs_true={qs_stats['Qs_true_at_Ymax']:.6e}  "
            #       f"@batch_idx={qs_stats['idx_Ymax']}")
            # print(f"[val Qs(global Qsmax_true, sample-mean)] Qs_pred={qs_stats['Qs_pred_at_Qsmax_true']:.6e}  "
            #       f"Qs_true={qs_stats['Qsmax_true']:.6e}  "
            #       f"at Y={qs_stats['Y_at_Qsmax_true']:.6e}")
            # if qs_stats["Qsmax_true_px"] > -1e20:
            #     print(f"[val Qs(global Qsmax_true, pixel)]      Qs_true_pxmax={qs_stats['Qsmax_true_px']:.6e}  "
            #           f"(sample’s mean Qs_pred={qs_stats['Qs_pred_at_Qsmax_true_px']:.6e})  "
            #           f"at Y={qs_stats['Y_at_Qsmax_true_px']:.6e}")

        if len(val_dy_all):
            dy = np.concatenate(val_dy_all)  # [N]
            qp = np.concatenate(val_qp_all)  # [N]
            qt = np.concatenate(val_qt_all)  # [N]
            # ΔY quantile bins (4 bins: quartiles)
            edges = np.quantile(dy, [0.0, 0.25, 0.5, 0.75, 1.0])
            rmse_bins, mape_bins = [], []
            eps = 1e-8
            for i in range(len(edges) - 1):
                m = (dy >= edges[i]) & (dy < edges[i+1] if i < len(edges)-2 else dy <= edges[i+1])
                if m.any():
                    rmse = float(np.sqrt(np.mean((qp[m] - qt[m])**2)))
                    mape = float(np.mean(np.abs((qp[m] - qt[m]) / (np.abs(qt[m]) + eps))))
                else:
                    rmse, mape = np.nan, np.nan
                rmse_bins.append(rmse); mape_bins.append(mape)
            # Top-quartile (largest ΔY) focus
            top_mask = dy >= edges[3]
            if top_mask.any():
                top_rmse = float(np.sqrt(np.mean((qp[top_mask] - qt[top_mask])**2)))
                top_mape = float(np.mean(np.abs((qp[top_mask] - qt[top_mask]) / (np.abs(qt[top_mask]) + eps))))
                top_pred = float(np.mean(qp[top_mask]))
                top_true = float(np.mean(qt[top_mask]))
            else:
                top_rmse = top_mape = top_pred = top_true = float("nan")

            print(f"[val Qs(ΔY bins)] edges={edges}  rmse={np.array(rmse_bins)}  mape={np.array(mape_bins)}")
            print(f"[val Qs(ΔY top quartile)] rmse={top_rmse:.3e} mape={top_mape:.3e} "
                  f"Qs_pred_mean={top_pred:.3e} Qs_true_mean={top_true:.3e}")

            # ---- Quadrupole RMSE in ΔY quantile bins (matching Qs bins) ----
            try:
                import numpy as _np
                q4_rmse_bins = None
                q4_rmse_bins_mean = None
                q4_combined = None
                qs_combined = None
                combined_total = None

                if len(val_q4p_all) > 0 and len(val_q4t_all) > 0:
                    q4p = _np.concatenate(val_q4p_all, axis=0)  # [N, n_pairs]
                    q4t = _np.concatenate(val_q4t_all, axis=0)  # [N, n_pairs]
                    n_pairs = q4p.shape[1] if q4p.ndim == 2 else 0

                    q4_rmse_bins = []
                    q4_counts_bins = []
                    for i in range(len(edges)-1):
                        m = (dy >= edges[i]) & (dy < edges[i+1] if i < len(edges)-2 else dy <= edges[i+1])
                        if m.any():
                            # RMSE per pair in this bin
                            err = _np.sqrt(_np.mean((q4p[m, :] - q4t[m, :])**2, axis=0))  # [n_pairs]
                            q4_rmse_bins.append(err.tolist())
                            q4_counts_bins.append(int(m.sum()))
                        else:
                            q4_rmse_bins.append([_np.nan] * (n_pairs))
                            q4_counts_bins.append(0)

                    # Mean across pairs per bin (ignoring NaNs)
                    q4_rmse_bins_mean = [
                        float(_np.nanmean(_np.array(bin_err))) if _np.isfinite(_np.array(bin_err)).any() else float("nan")
                        for bin_err in q4_rmse_bins
                    ]

                    # Weighted combined RMSE across all bins & pairs
                    num = 0.0; den = 0
                    for cnt, bin_err in zip(q4_counts_bins, q4_rmse_bins):
                        if cnt > 0:
                            arr = _np.array(bin_err, dtype=float)
                            mask = _np.isfinite(arr)
                            if mask.any():
                                num += cnt * float(_np.nanmean(arr[mask]**2))
                                den += cnt
                    q4_combined = float(_np.sqrt(num / max(den, 1))) if den > 0 else float("nan")
                else:
                    q4_rmse_bins = []
                    q4_rmse_bins_mean = []
                    q4_combined = float("nan")

                # Also compute a weighted-combined RMSE for Qs across bins
                qs_counts_bins = []
                for i in range(len(edges)-1):
                    m = (dy >= edges[i]) & (dy < edges[i+1] if i < len(edges)-2 else dy <= edges[i+1])
                    qs_counts_bins.append(int(m.sum()))
                num = 0.0; den = 0
                for cnt, rm in zip(qs_counts_bins, rmse_bins):
                    if _np.isfinite(rm):
                        num += cnt * (rm**2)
                        den += cnt
                qs_combined = float(_np.sqrt(num / max(den, 1))) if den > 0 else float("nan")

                # Final combined scalar (equal weight between Qs and quadrupoles)
                if _np.isfinite(qs_combined) and _np.isfinite(q4_combined):
                    combined_total = float(0.5 * (qs_combined + q4_combined))
                elif _np.isfinite(qs_combined):
                    combined_total = float(qs_combined)
                elif _np.isfinite(q4_combined):
                    combined_total = float(q4_combined)
                else:
                    combined_total = float("nan")

                # Emit a single-line, tuner-readable JSON metric
                tuner_payload = {
                    "epoch": int(epoch),
                    "qs_rmse_bins": [float(x) if _np.isfinite(x) else None for x in rmse_bins],
                    "qs_rmse_combined": qs_combined,
                    #"q4_rmse_bins_per_pair": q4_rmse_bins,            # list[len=4] of lists[len=n_pairs]
                    "q4_rmse_bins_mean": q4_rmse_bins_mean,           # per-bin mean across pairs
                    "q4_rmse_combined": q4_combined,                  # weighted across bins & pairs
                    "combined_rmse_total": combined_total             # single scalar for tuner
                }
                print("TUNER_METRIC " + json.dumps(tuner_payload), flush=True)
                print("TUNER_SCALAR ", combined_total)
            except Exception as _e:
                # Keep training robust if diagnostics fail
                if ((not is_ddp) or dist.get_rank()==0):
                    print(f"[tuner metric] skipped: {_e}")
                    
        if ema_eval: ema.swap_out(unwrap(model))

#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize()
#        print(f"[timing] validation: {(time.perf_counter()-t0)*1e3:.2f} ms")
#        #----end-time----
        # ---- Reduce across ranks ----
        if is_ddp:
            t = torch.stack([tr_sum, tr_cnt, val_sum, val_cnt], dim=0)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            tr_sum, tr_cnt, val_sum, val_cnt = t[0], t[1], t[2], t[3]

        # Convert ONCE to Python floats for printing
        train_loss, val_loss = _pack_and_tolist(tr_sum / tr_cnt.clamp_min(1), val_sum / val_cnt.clamp_min(1))

        # Step the LR scheduler (special handling for ReduceLROnPlateau)
        if isinstance(sched, ReduceLROnPlateau):
            sched.step(val_loss)   # uses validation loss to decide when to decay
        else:
            sched.step()

        # Read current LR robustly (even for Plateau which may not expose get_last_lr)
        try:
            lr = sched.get_last_lr()[0]
        except Exception:
            lr = opt.param_groups[0]["lr"]


        # sched.step()
        # lr = sched.get_last_lr()[0]

        if ((not is_ddp) or dist.get_rank()==0):
            f_tr = meters.mean("train/frob"); u_tr = meters.mean("train/unit")
            f_va = meters.mean("val/frob");   u_va = meters.mean("val/unit")
            print(f"[{epoch:03d}/{args.epochs}] train {train_loss:.5f} val {val_loss:.5f} "
                  f"lr {lr:.2e}{('  |Δw|/|w| %.3e'%last_dw_ratio) if last_dw_ratio is not None else ''}  "
                  f"frob tr/va {f_tr:.3e}/{f_va:.3e}  unit tr/va {u_tr:.3e}/{u_va:.3e}")

        # Save best
        outdir = Path(args.out)
        if ((not is_ddp) or dist.get_rank() == 0) and val_loss < best:
            best = val_loss


            ckpt = {
                "model": (model.module.state_dict() if is_ddp else model.state_dict()),
                "args": {"in_ch": 22, "width": args.width, "modes": args.modes, "blocks": args.blocks, "proj_iter": 8, "gate_temp": args.gate_temp, "alpha_vec_cap": 15},
                "meta": {"N": inferred_N, "ds": ds_value, "epoch": epoch}
            }
            torch.save(ckpt, os.path.join(args.out, "evolver_best.pt"))
            try:
                with open(outdir / "hparams.json", "w", encoding="utf-8") as f:
                    json.dump(vars(args), f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())  # helps o
            except Exception as e:
                print(f"[WARN] failed to write hparams.json: {e}")
                
    # Clean DDP shutdown
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Train FNO evolver: (U0, Y, params, formatter_class=argparse.ArgumentDefaultsHelpFormatter, formatter_class=argparse.ArgumentDefaultsHelpFormatter) -> U_Y")

    # Core IO / training schedule
    ap.add_argument("--data_root", type=Path, required=True,
                    help="Training root dir with run_*/ (each has manifest.json + evolved_wilson_lines/)")
    ap.add_argument("--out", type=str, default="outputs_evolver", help="Output dir for checkpoints")
    ap.add_argument("--epochs", type=int, default=50, help='Number of training epochs to run.')
    ap.add_argument("--batch", type=int, default=1, help="Per-device batch size")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    ap.add_argument("--accum", type=int, default=1, help="Grad accumulation steps")
    ap.add_argument("--seed", type=int, default=0, help='Random seed for reproducibility (Python/NumPy/PyTorch).')

    # Optimizer / LR
    ap.add_argument("--lr", type=float, default=2e-4, help='Initial learning rate for the optimizer.')
    ap.add_argument("--weight_decay", type=float, default=1e-7, help='Weight decay (L2 regularization) applied by the optimizer.')
    ap.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    ap.add_argument("--warmup_start_factor", type=float, default=0.1, help="Warmup factor (relative to lr)")
    ap.add_argument("--min_lr", type=float, default=0.0, help="Eta_min for cosine phase")

    # Model size
    ap.add_argument("--width", type=int, default=64, help='Base channel width of the FNO trunk.')
    ap.add_argument("--modes", type=int, default=16, help='Number of retained Fourier modes per spatial dimension in spectral convs.')
    ap.add_argument("--blocks", type=int, default=6, help='Number of FNO residual blocks (depth of the trunk).')

    # Device / precision
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA)")
    ap.add_argument("--torch_compile", action="store_true", help="Use torch.compile (CUDA recommended)")
    ap.add_argument("--channels_last", dest="channels_last", action="store_true", help="Use channels_last memory format")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    # Data split
    ap.add_argument("--split_by", choices=["run","params"], default="run",
                    help="Train/val split granularity: 'run' (default) or 'params' (hold out entire parameter sets)")

    # EMA
    ap.add_argument("--ema_decay", type=float, default=0.0, help="EMA decay, e.g. 0.999; 0 disables EMA")
    ap.add_argument("--ema_eval", type=int, default=1, help="Use EMA weights for eval when EMA is enabled")

    # Debug probes
    ap.add_argument("--debug_evo", type=int, default=1,
                    help="Print evolution probes (Y range, |alpha|, |U(Y)-U0|) during train/val")
    ap.add_argument("--debug_evo_every", type=int, default=1,
                    help="How often (in epochs) to print the probes (default: every epoch)")
    ap.add_argument("--profile_dw_every", type=int, default=0, help="Profile every X steps. SHould be large or zero for better performance on CUDA.")    

    # Y handling / conditioning
    ap.add_argument("--y_channel", type=int, default=18, help="Index of Y channel in input tensor")
    ap.add_argument("--identity_eps", type=float, default=0.0, help="If |Y|<=eps, return U0 exactly (skip update)")
    ap.add_argument("--film_mode", type=str, default="scale_only", choices=["scale_only","scale_shift"], help="FiLM conditioning mode for time/Y embedding: 'scale_only' or 'scale_shift'.")
    ap.add_argument("--rbf_K", type=int, default=12, help='Number of RBF centers used to embed Y (time) before FiLM.')
    ap.add_argument("--film_hidden", type=int, default=64, help='Hidden size of the FiLM MLP used in the time conditioner.')
    ap.add_argument("--gamma_scale", type=float, default=1.5, help='Initial scale for FiLM gamma (multiplicative) parameters.')
    ap.add_argument("--beta_scale", type=float, default=1.0, help='Initial scale for FiLM beta (additive) parameters (if used).')
    ap.add_argument("--gate_temp", type=float, default=1.0, help='Temperature for gating nonlinearity in the time conditioner; higher = sharper gates.')
    ap.add_argument("--y_min", type=float, default=None, help="Override data-inferred Y min (optional)")
    ap.add_argument("--y_max", type=float, default=None, help="Override data-inferred Y max (optional)")
    ap.add_argument("--skip_heavy_on_cpu", type=int, default=1, help="No rollouton CPU")
    

    ap.add_argument("--trace_weight",    type=float, default=0, help='Weight of trace of Wilson lines.')
    ap.add_argument("--geo_weight",    type=float, default=0, help='Weight of geodesic distance term in GroupLoss (keeps outputs on-manifold).')
    ap.add_argument("--dir_weight",    type=float, default=0, help='Weight of direction-hinge term encouraging correct ΔU direction with Y.')
    ap.add_argument("--dir_margin",    type=float, default=0.25, help='Cosine-similarity margin used by the direction-hinge term.')
    ap.add_argument("--project_before_frob", type=int, default=1, help='If 1, project prediction to SU(3) before computing Frobenius loss term.')

    # --- Dipole & Qs & higher order correlator loss flags ---
    ap.add_argument("--dipole_weight", type=float, default=0.5,
                    help="Weight of MSE loss on dipole correlator S(r) evaluated at --dipole_offsets.")
    ap.add_argument("--dipole_offsets", type=str, default="(1,0),(0,1),(2,0),(0,2),(4,0),(0,4),(8,0),(0,8),(12,0),(0,12)(16,0),(0,16)",
                    help="Comma-separated list of (dx,dy) axial separations for S(r); at least two distinct |r| needed for Qs.")
    ap.add_argument("--qs_weight", type=float, default=0.,
                    help="Weight of MSE loss on saturation scale Qs derived from the dipole correlator.")
    ap.add_argument("--qs_threshold", type=float, default=0.5,
                    help="Threshold c used to define r*: N(r*)=c if --qs_on=N, or S(r*)=c if --qs_on=S; Qs = qs_scale / r*.")
    ap.add_argument("--qs_on", type=str, choices=["N", "S", "n", "s"], default="N",
                    help="Curve to threshold for Qs: 'N' uses N(r)=1-S(r); 'S' uses S(r). Case-insensitive.")
    ap.add_argument("--qs_local", action="store_true",
                    help="Compute local Qs(x,y) per lattice site instead of a single global Qs per configuration.")
    ap.add_argument("--qs_scale", type=float, default=1.0,
                    help="Scale factor applied to 1/r* when computing Qs (i.e., Qs = qs_scale / r*).")
    #ap.add_argument("--quad_pairs", type=str, default="auto_from_dipole", help="Quadrupole pair spec: explicit like ((1,0),(0,1));((2,0),(0,2)) or r=1,2,4,8 for symmetric ((r,0),(0,r)), or auto_from_dipole to derive radii from --dipole_offsets.")
    ap.add_argument("--quad_pairs", type=str, default="auto_broad", help="Quadrupole pair spec. Options: explicit pairs like \"((1,0),(0,1));((2,0),(0,2))\", \"r=1,2,4\" (axial), \"auto_from_dipole\" (legacy axial), or \"auto_broad\" (axial+diagonals, all non-colinear pairs).")
    ap.add_argument("--quad_max_pairs", type=int, default=48, help="If >0, cap the number of quadrupole pairs after deduplication (for speed/memory).")
    # parser / ap.add_argument(...) block

    ap.add_argument("--moment_weight", type=float, default=0.0,
                help="Weight for moment-matching loss on drift and diffusion amplitude (0 disables).")
    ap.add_argument("--nll_weight", type=float, default=0.0, help="Weight for alpha-true diagonal Gaussian NLL (0 disables).")
    ap.add_argument("--nll_target_mode", type=str, default="none", choices=["per_dy","per_ysigma","none"], help="How to scale S to match head’s μ-space.")
    ap.add_argument("--quad_weight", type=float, default=0.0, help="Weight for quadrupole loss.")
    ap.add_argument("--rough_weight", type=float, default=0., help="Weight for fluctuation scale loss.")
    ap.add_argument("--kmin", type=float, default=4., help="Minimum k for which to include fluctuations.")
    ap.add_argument("--kmax", type=float, default=None, help="Max k for fluctuation scale loss.")
    ap.add_argument("--mono_weight", type=float, default=0.0, help="Weight for monotonic N(r,Y) penalty using pairs.")
    ap.add_argument("--qs_slope_weight", type=float, default=0.0, help="Weight for d/dY ln Qs^2 slope penalty using pairs.")
    ap.add_argument("--E1", type=int, default=5, help="Epoch when to start ramping up loss functions other than the dipole.")
    ap.add_argument("--E2", type=int, default=10, help="Epoch when to finish ramping up loss functions other than the dipole.")

    # Rollout & Consistency
    ap.add_argument("--rollout_k", type=int, default=1, help='Number of uniform substeps in Y for autoregressive rollout. If >1, the model is applied k times starting from U0 with intermediate Y_i = Y_final*(i/k), encouraging learning of the Y trajectory (not just the endpoint). Set to 1 to disable rollout and use single-shot predictions. Typical values: 3–8.')
    ap.add_argument("--rollout_consistency", type=float, default=0.0, help='Weight for an auxiliary MSE between single-shot output F(Y, U0) and rollout output after k substeps. Helps the network keep the single-shot and multi-step operators consistent. Use together with --rollout_k>1. Recommended range: 0.05–0.2.')
    ap.add_argument("--rollout_single_consistency", type=float, default=0.0, help='Weight for SINGLE consistency: match one-step transitions from the two states that rollout produces.')
    ap.add_argument("--semigroup_weight", type=float, default=0.0, help='Weight for the semigroup consistency loss enforcing F(Y2, F(Y1, U0)) ≈ F(Y1+Y2, U0) on random splits Y1/Y2. Adds two extra forward passes on selected batches. Small but strong physics prior. Try 0.01–0.05.')
    ap.add_argument("--semigroup_prob", type=float, default=0.0, help='Probability (0–1) of applying the semigroup consistency term on a given batch. Use <1.0 to control extra compute overhead. Example: 0.25 (apply on 25% of batches).')
    ap.add_argument("--qs_soft_beta", type=float, default=0.0,
                    help=">0: use soft, differentiable Qs estimator (temperature).")
    ap.add_argument("--qs_soft_slope", type=float, default=1.0,
                    help="Weight by local |dX/dr|^s in soft estimator (0 disables).")

    ap.add_argument('--nll_param_compose', action='store_true', help='Use parameter-only (m, Lambda_QCD, mu0) head for μ/σ and compose over Y.')
    # === New: generator & distributional training knobs ===
    ap.add_argument('--nll_fullcov', action='store_true',
                        help='Use full-covariance α-NLL with Cholesky diffusion.')
    ap.add_argument('--nll_compose', type=str, default='brownian',
                        choices=['brownian','ou'],
                        help='Composition law for μ/Σ over Y (Brownian or OU).')
    ap.add_argument('--bch_weight', type=float, default=0.0,
                        help='Weight for BCH semigroup mean-consistency loss.')
    ap.add_argument('--energy_weight', type=float, default=0.0,
                        help='Weight for distributional match (energy distance) in algebra.')
    ap.add_argument('--energy_stride', type=int, default=4,
                        help='Subsample stride over H×W for the energy distance term.')
    ap.add_argument('--auto_balance', action='store_true',
                        help='Enable uncertainty-based loss balancing of loss terms.')

    
    ap.add_argument('--sigma_mode', type=str, default='conv', choices=['diag','conv','spectral'], help='Noise-path covariance: diagonal, local conv operator, or spectral.')
    ap.add_argument('--noise_kernel', type=int, default=5, help='Kernel size for conv operator σ (odd number).')
    
    ap.add_argument("--scheduler", type=str, default="cosine",
                    choices=["cosine","cosine_wr","step","multistep","exponential","plateau","poly","constant"],
                    help="LR schedule used after the warmup phase.")
    ap.add_argument("--lr_gamma", type=float, default=0.5,
                    help="Decay factor for step/exponential/plateau schedulers.")
    ap.add_argument("--lr_step_size", type=int, default=10,
                    help="Epoch step size for StepLR.")
    ap.add_argument("--lr_milestones", type=str, default="",
                    help="Comma-separated epoch milestones for MultiStepLR, e.g. '10,15,20'.")
    ap.add_argument("--lr_patience", type=int, default=5,
                    help="Patience (epochs) for ReduceLROnPlateau.")
    ap.add_argument("--t0", type=int, default=10,
                    help="T_0 for CosineAnnealingWarmRestarts.")
    ap.add_argument("--tmult", type=float, default=2.0,
                    help="T_mult for CosineAnnealingWarmRestarts.")
    ap.add_argument("--poly_power", type=float, default=1.0,
                    help="Power for polynomial LR decay (poly).")

    ap.add_argument('--y_map', type=str, default='linear',
                    choices=['tanh','linear'],
                    help='Map from physical Y to internal Y_eff')

    ap.add_argument("--rbf_gamma", type=float, default=1.0, help="Bias time-embedding centers: <1 packs toward high Y, >1 packs toward low Y")
    ap.add_argument("--rbf_min_width", type=float, default=1e-3, help="Floor for RBF widths in y01 space")

    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()


