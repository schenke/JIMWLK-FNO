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
import contextlib

# Learn (U0, Y, params) -> U_Y from JIMWLK outputs + manifest.json
# - Lattice size N inferred from data (no --size)
# - 'ds' inferred from manifest.json (no --ds)
# - DDP + AMP ready; GroupNorm (batch=1 friendly)

# --- cache for radial binning (wrapped torus distances) ---
_RADIAL_CACHE = {}  # key: (H, W, device) -> (bins[H,W] long, counts[L] long)


import torch, torch.distributed as dist

# def get_noise_core_fn(model):
#     """Return a callable like lambda x: model.head._noise_core(x), or None."""
#     m = model.module if hasattr(model, "module") else model
#     head = getattr(m, "head", None)
#     core = getattr(head, "_noise_core", None) if head is not None else None
#     return (lambda x: core(x)) if core is not None else None

# --- Canonical SU(3) Gell-Mann matrices (optionally divided by 2) ---
def su3_gellmann_matrices(
    *, dtype: torch.dtype = torch.complex64,
    device: torch.device | None = None,
    half: bool = True,
) -> torch.Tensor:
    """Return λ_a (a=1..8) as a tensor of shape [8,3,3].
    If `half` is True (default), returns λ_a/2.
    """
    L = torch.zeros(8, 3, 3, dtype=torch.complex64, device=device)
    # λ1..λ8 (physics convention)
    L[0,0,1] = L[0,1,0] = 1
    L[1,0,1] = -1j; L[1,1,0] =  1j
    L[2,0,0] = 1;   L[2,1,1] = -1
    L[3,0,2] = L[3,2,0] = 1
    L[4,0,2] = -1j; L[4,2,0] =  1j
    L[5,1,2] = L[5,2,1] = 1
    L[6,1,2] = -1j; L[6,2,1] =  1j
    s3 = 1.0 / (3.0 ** 0.5)
    L[7,0,0] = s3; L[7,1,1] = s3; L[7,2,2] = -2 * s3
    if half:
        L = L / 2.0
    # cast/move if needed; buffers will follow the module when .to(device) is called
    return L.to(dtype=dtype, device=device) if (dtype != torch.complex64 or device is not None) else L


def _ddp_world_pg_or_none():
    return dist.group.WORLD if (dist.is_available() and dist.is_initialized()) else None


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

def _iso_radial_power(x: torch.Tensor, nbins: int | None = None,
                      exclude_dc: bool = True, window: bool = False) -> torch.Tensor:
    """
    x: [B, C, H, W] real tensor
    returns: [B, C, K] isotropic (radial) power spectrum with K bins
    """
    B, C, H, W = x.shape

    # Optional Hann window to reduce leakage if boundaries aren't periodic
    if window:
        wy = torch.hann_window(H, dtype=x.dtype, device=x.device).view(1, 1, H, 1)
        wx = torch.hann_window(W, dtype=x.dtype, device=x.device).view(1, 1, 1, W)
        x = x * wy * wx

    X = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')         # complex
    P = (X.real**2 + X.imag**2)                               # [B,C,H,W]

    # Radial frequency grid
    fy = torch.fft.fftfreq(H, device=x.device).view(H, 1).expand(H, W)
    fx = torch.fft.fftfreq(W, device=x.device).view(1, W).expand(H, W)
    r  = torch.sqrt(fy*fy + fx*fx)                            # [H,W]

    if nbins is None:
        nbins = int(min(H, W) // 2)

    edges = torch.linspace(0, r.max() + 1e-7, nbins + 1, device=x.device)
    ridx  = torch.bucketize(r.reshape(-1), edges) - 1         # [H*W], in [0..nbins-1]
    ridx  = ridx.clamp_(0, nbins - 1)

    # Valid mask: exclude the whole DC bin if requested
    valid = torch.ones_like(ridx, dtype=torch.bool)
    if exclude_dc:
        valid = ridx >= 1

    # Prepare 2D scatter inputs
    Pflat        = P.reshape(B * C, -1)                       # [B*C, H*W]
    ridx_expand  = ridx.unsqueeze(0).expand(B * C, -1)        # [B*C, H*W]
    valid_expand = valid.unsqueeze(0).expand(B * C, -1)       # [B*C, H*W]

    # Zero-out invalid contributions (keep shapes intact)
    src_vals = Pflat * valid_expand.to(Pflat.dtype)           # [B*C, H*W]

    # Accumulate sums and counts per radial bin
    sums = torch.zeros(B * C, nbins, device=x.device, dtype=P.dtype)
    cnts = torch.zeros(B * C, nbins, device=x.device, dtype=P.dtype)

    sums.scatter_add_(1, ridx_expand, src_vals)
    cnts.scatter_add_(1, ridx_expand, valid_expand.to(P.dtype))

    spec = sums / (cnts + 1e-12)                              # [B*C, nbins]
    return spec.view(B, C, nbins)


@torch.no_grad()
def _dist_mean_scalar(t: torch.Tensor) -> torch.Tensor:
    # t: 0-dim or 1-dim tensor on device
    if dist.is_available() and dist.is_initialized():
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t


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



def _pack_and_tolist(*scalars: torch.Tensor):
    """Batch a few scalar tensors into one tiny transfer (single sync)."""
    if not scalars:
        return []
    stk = torch.stack([s.detach() for s in scalars])
    return stk.cpu().tolist()

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
    """Exponential Moving Average of model parameters (robust to late-created params)."""
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        # Track only trainable params
        self.shadow = {k: p.detach().clone()
                       for k, p in model.named_parameters() if p.requires_grad}
        self._backup = None  # set by swap_in

    @torch.no_grad()
    def update(self, model):
        """EMA update; auto-registers any new trainable params that appeared after init."""
        d = self.decay
        if d == 0:
            return

        # Add or update all current trainable params
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            v = p.detach()
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v, alpha=1.0 - d)
            else:
                # New param created after EMA init (e.g., param_nll.*)
                self.shadow[k] = v.clone()

        # (Optional) prune stale entries that no longer exist in the model
        # to avoid memory growth if params are deleted/replaced.
        keys_now = {k for k, p in model.named_parameters() if p.requires_grad}
        stale = [k for k in self.shadow.keys() if k not in keys_now]
        for k in stale:
            del self.shadow[k]

    @torch.no_grad()
    def swap_in(self, model):
        """Swap EMA weights into live model (and back up raw weights). Safe if some keys are missing."""
        if self._backup is not None:
            # Already swapped in; avoid double-swap bugs.
            return
        self._backup = {}
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Back up current
            self._backup[k] = p.data.clone()
            # Use EMA if we have it; otherwise keep the raw weight
            if k in self.shadow:
                p.data.copy_(self.shadow[k])

    @torch.no_grad()
    def swap_out(self, model):
        """Restore raw weights after eval."""
        if self._backup is None:
            return
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if k in self._backup:
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
    return torch.einsum('...a,aij->...ij', a, lams)
#return torch.einsum('...a,aij->...ij', a, lams)
    
def _S_to_alpha(S: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """
    Project Hermitian S back to α_a = 2 Re tr(S T_a).
    S: [...,3,3] ; lambdas: [C,3,3]
    Returns [..., C]
    """
    # match lambdas to S
    lams = lambdas.to(dtype=S.dtype, device=S.device)
    # 2 Re tr(S T_a)  (einsum handles the trace over ij)
    return 2.0 * torch.real(torch.einsum('...ij,aij->...a', S, lams))
#return 2.0 * torch.real(torch.einsum('...ij,aji->...a', S, lams))

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

class SU3HeadGellMannStochastic(nn.Module):
    """
    Predicts per-unit-Y drift μ and (diag) log-std for 8 (or 16) Lie-algebra coeffs.
    Sampling: α_step = μ * ΔY + σ * √ΔY ∘ η, with optional spatially-correlated η.

    Default here: if dY is None, we set dY = Y (from Ymap) so a single 0→Y step
    works out of the box. For compositions, pass absolute Y in Ymap and the
    increment ΔY explicitly via dY.
    """
    def __init__(self,
                 width: int,
                 *,
                 alpha_channels: int = 8,          # 8 (shared L/R) or 16 (8L+8R)
                 identity_eps: float = 0.0,
                 clamp_alphas: float | None = None,
                 alpha_scale: float | None = 1.0,  # optional global scale
                 alpha_vec_cap: float | None = 15.0,
                 A_cap: float | None = None,
                 sigma0: float = 0.03,             # σ init per unit Y
                 sigma_mode: str = "conv",         # "diag" | "conv" | "spectral"
                 noise_kernel: int = 9):
        super().__init__()

        M = getattr(self, "spec_mixtures", 3)
        self.spec_a_raw  = nn.Parameter(torch.full((M,), -1.0))   # a_m ~ softplus → ~0.3
        self.spec_k0_raw = nn.Parameter(torch.tensor([0.3, 0.6, 0.1]))  # cycles/pixel guesses
        self.spec_p_raw  = nn.Parameter(torch.full((M,), 2.5))
        
        assert alpha_channels in (8, 16), "alpha_channels mustbe 8 or 16"
        self.C = alpha_channels
        self.identity_eps = float(identity_eps)
        self.clamp_alphas = clamp_alphas
        self.alpha_scale = float(alpha_scale) if alpha_scale is not None else 1.0
        self.alpha_vec_cap = alpha_vec_cap
        self.A_cap = A_cap
        self.sigma_mode = sigma_mode
        
        self.noise_kernel = int(noise_kernel)
        self.width = width

        # Heads
        self.proj_mu   = nn.Conv2d(width, self.C, kernel_size=1, bias=True)
        self.proj_logs = nn.Conv2d(width, self.C, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.proj_mu.weight, gain=1e-2)
        nn.init.constant_(self.proj_mu.bias, 0.0)
        nn.init.constant_(self.proj_logs.weight, 0.0)
        nn.init.constant_(self.proj_logs.bias, math.log(sigma0))  # σ≈sigma0 at start

        # Gell-Mann / 2 (Hermitian, traceless)
        self.register_buffer("lambdas", su3_gellmann_matrices(), persistent=False)

        # Noise coloring operators
        if self.sigma_mode == "conv":
            k, pad = self.noise_kernel, self.noise_kernel // 2
            self.noise_dw = nn.Conv2d(self.C, self.C, kernel_size=k, padding=pad, groups=self.C, bias=False)
            self.noise_pw = nn.Conv2d(self.C, self.C, kernel_size=1, bias=False)
            with torch.no_grad():
                self.noise_dw.weight.zero_()
                self.noise_dw.weight[:, 0, pad, pad] = 1.0  # identity depthwise
                self.noise_pw.weight.zero_()
                for c in range(self.C):
                    self.noise_pw.weight[c, c, 0, 0] = 1.0   # identity pointwise
        elif self.sigma_mode == "spectral":
            self.spec_k0 = nn.Parameter(torch.full((self.C,), 1.50))
            self.spec_p  = nn.Parameter(torch.full((self.C,), 3.00))
        elif self.sigma_mode == "diag":
            pass
        else:
            raise ValueError(f"Unknown sigma_mode: {self.sigma_mode}")

        # monitors
        self.last_A_fro_mean = torch.tensor(0.0)
        self.last_sigma_mean = torch.tensor(0.0)

    # ---- helpers ----
    def _cap_alphas(self, a: torch.Tensor) -> torch.Tensor:
        if self.clamp_alphas is not None:
            a = torch.tanh(a) * float(self.clamp_alphas)
        if (self.alpha_scale is not None) and (self.alpha_scale != 1.0):
            a = a * float(self.alpha_scale)
        if self.alpha_vec_cap is not None:
            vnorm = a.norm(dim=1, keepdim=True).clamp(min=1e-6)
            a = a * (float(self.alpha_vec_cap) / vnorm).clamp(max=1.0)
        return a

    def _assemble(self, alphas: torch.Tensor, device) -> torch.Tensor:
        # expects 8 channels
        T = self.lambdas.to(device=device)                                 # [8,3,3] complex64
        if alphas.shape[1] != 8:
            raise ValueError(f"assemble expects 8 channels, got {alphas.shape[1]}")
        S = torch.einsum("bchw,aij->bhwij", alphas.to(T.dtype), T)         # Hermitian
        if self.A_cap is not None:
            S_f = (S.real.square().sum(dim=(-2,-1)) + 1e-12).sqrt()
            S = S * (float(self.A_cap) / S_f).unsqueeze(-1).unsqueeze(-1).clamp(max=1.0)
        A = 1j * S                                                         # anti-Hermitian, traceless
        return A


    def _noise_core(self, epsn: torch.Tensor) -> torch.Tensor:
        if self.sigma_mode == "diag":
            eta_core = epsn
        elif self.sigma_mode == "conv":
            eta_core = self.noise_pw(self.noise_dw(epsn))
        else:  # spectral
            B, C, H, W = epsn.shape

            # FFT (real-to-complex) → shape [B,C,H,W//2+1]
            Ef = torch.fft.rfft2(epsn, norm="ortho")

            # Correct frequency grids: ky uses fftfreq(H), kx uses rfftfreq(W)
            device = epsn.device
            dtype_r = Ef.real.dtype

            ky = torch.fft.fftfreq(H, d=1.0, device=device)         # [H]
            kx = torch.fft.rfftfreq(W, d=1.0, device=device)        # [W//2+1]
            KY, KX = torch.meshgrid(ky, kx, indexing="ij")          # [H, W//2+1]
            Kr = torch.sqrt(KX**2 + KY**2).to(dtype_r)              # radial |k|

            # Per-channel filter: G(k) = 1 / (1 + (|k|/k0)^p)
            k0 = torch.clamp(self.spec_k0, min=1e-6).view(1, C, 1, 1).to(dtype_r)
            p  = torch.relu(self.spec_p).view(1, C, 1, 1).to(dtype_r)

            M = self.spec_a_raw.numel()
            a  = F.softplus(self.spec_a_raw).view(1,1,M,1,1)
            k0 = F.softplus(self.spec_k0_raw).add(1e-6).view(1,1,M,1,1)
            p  = F.softplus(self.spec_p_raw ).add(1e-3).view(1,1,M,1,1)
            Kr = Kr.view(1,1,1,H,W//2+1)
            term = (Kr / k0) ** p                               # [1,1,M,H,Wr]
            G = 1.0 / (1.0 + (a * term).sum(dim=2))             # [1,1,H,Wr]
            G = G.expand(1, C, H, W//2+1)                       # share across C (or learn per-C if needed)
            Ef = Ef * G

            eta_core = torch.fft.irfft2(Ef, s=(H, W), norm="ortho")      # [B,C,H,W]

        # Normalize per-channel RMS (detach denom so only shape—not amplitude—is learned)
        den = eta_core.pow(2).mean(dim=(2, 3), keepdim=True).add(1e-12).sqrt().detach()
        return eta_core / den

    def forward(self,
                h: torch.Tensor,
                base18: torch.Tensor,
                Ymap: torch.Tensor,
                *,
                nsamples: int = 1,
                sample: bool | None = None,
                dY: torch.Tensor | None = None,
                # NEW:
                return_eta: bool = True,
                eta_stride: int = 1):
        # identity snap at |Y|≈0
        if sample is None:
            sample = self.training

        B, _, H, W = base18.shape
        device = h.device

        # --- build μ, σ, dY as you already do ---
        mu     = self.proj_mu(h).float()
        logsig = self.proj_logs(h).float()

        sigma_min, sigma_max = 1e-4, 0.20
        sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(logsig)

        dtype  = mu.dtype
        if dY is None:
            dYt = Ymap.to(device=device, dtype=dtype)
        else:
            dYt = dY.to(device=device, dtype=dtype)
            if dYt.dim() == 1 or dYt.dim() == 2:
                dYt = dYt.view(B, 1, 1, 1).expand(B, 1, H, W)
        dYt = dYt.clamp_min(0)

        C = mu.shape[1]
        if dYt.shape[1] == 1 and C != 1:
            dYc = dYt.expand(B, C, H, W)
        else:
            dYc = dYt

        mu_step  = mu * dYc
        eta_step = sigma * torch.sqrt(dYc)

        # ---- (A) sampling path for α-step (full resolution) ----
        if sample:
            epsn_full = torch.randn_like(mu)                 # [B,C,H,W]
            eta_core_full = self._noise_core(epsn_full)      # uses conv/spectral/diag core
            a_all = mu_step + eta_step * eta_core_full
        else:
            a_all = mu_step
            eta_core_full = None

        # ---- (B) OPTIONAL: produce eta_core for the loss and put in extras ----
        # Do this regardless of `sample` so DDP always sees these params in forward.
        eta_core_for_loss = None
        if return_eta:
            if eta_stride > 1:
                Hs, Ws = H // eta_stride, W // eta_stride
            else:
                Hs, Ws = H, W
            # Draw noise directly at the loss resolution so the core’s own RMS renorm is correct
#            epsn_loss = torch.randn(B, C, Hs, Ws, device=device, dtype=dtype)
            epsn_loss = torch.randn_like(mu)   
            eta_core_for_loss = self._noise_core(epsn_loss)  # [B,C,Hs,Ws]

        # ---- assemble SU(3) update exactly as you have it ----
        if self.C == 16:
            aL, aR = torch.split(a_all, 8, dim=1)
        else:
            aL = a_all
            aR = a_all

        aL = self._cap_alphas(aL)
        aR = self._cap_alphas(aR)

        U0 = pack_to_complex(base18.permute(0, 2, 3, 1).to(torch.float32))
        AL = self._assemble(aL, device=device)
        AR = self._assemble(aR, device=device)

        GLh = torch.linalg.matrix_exp(+0.5 * AL)
        GR  = torch.linalg.matrix_exp(+1.0 * AR)
        U   = GLh @ U0
        U   = U @ GR
        U   = GLh @ U

        self.last_A_fro_mean = (AL.abs().square().sum(dim=(-2,-1)).sqrt().mean()).detach()
        self.last_sigma_mean = sigma.detach().mean()

        out18 = unpack_to_18(U).permute(0, 3, 1, 2).to(h.dtype)

        # small-Y snap (unchanged)
        eps = self.identity_eps
        if eps > 0.0:
            y_abs0 = Ymap[:, 0, 0, 0].abs()
            if (y_abs0 <= eps).any():
                mask = (y_abs0 <= eps)
                out18 = out18.clone()
                out18[mask] = base18[mask]

        extras = {
            "mu": mu, "logsig": logsig, "sigma": sigma,
            "dY": dYt, "alpha_step": a_all.detach(),
        }
        if return_eta and eta_core_for_loss is not None:
            extras["eta_core"] = eta_core_for_loss  # <— tensor for your spectral loss

        return out18, extras

# Simple channel-wise LayerNorm for NCHW (no spatial coupling)
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> LN over C only
        b, c, h, w = x.shape
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class EvolverFNO(nn.Module):
    """Lift -> [FNOBlock]*n with Y/θ-conditioned FiLM + residual gate -> SU3HeadGellMann(Stochastic)."""
    def __init__(self,
                 in_ch=22, width=64, modes1=12, modes2=12, n_blocks=4,
                 identity_eps: float = 0.0, alpha_scale: float = 1.0,
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
                 rbf_min_width: float = 0.,
                 y_map: str = "linear",
                 sigma_mode: str = "conv",
                 channels_last: bool = False):

        super().__init__()
        self.y_index = int(y_index)
        self.width = int(width)
        self.film_hidden = int(film_hidden)
        self.rbf_K = int(rbf_K)
        self.beta_scale = float(beta_scale)
        self.gamma_scale = float(gamma_scale)
        self.film_mode = str(film_mode)
        self.gate_temp = float(gate_temp)
        self.y_map = y_map
        self.y_min = y_min
        self.y_max = y_max
        self.channels_last = bool(channels_last)
        self.sigma_mode = sigma_mode
        # --- trunk ---
        self.lift = nn.Conv2d(18, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_blocks)])
        self.block_norm = LayerNorm2d(width)  # pre-norm (channel-only), more stable than GN(1)

        # --- SU(3) head ---
        self.head = SU3HeadGellMannStochastic(
            width,
            identity_eps=identity_eps,
            alpha_scale=alpha_scale,
            clamp_alphas=clamp_alphas,
            alpha_vec_cap=alpha_vec_cap,
            sigma_mode=self.sigma_mode
        )

        # --- Y & θ embeddings (RBF) ---
        self.time_embed  = RBFEmbed(K=rbf_K, learnable=True, init_sigma=0.20)
        self.theta_embed = RBFEmbed(K=rbf_K, learnable=True, init_sigma=0.20)

        # small “pre-FiLM” MLP to clean embeddings
        self.pre_film = nn.Sequential(
            nn.LazyLinear(self.film_hidden),
            nn.SiLU(),
            nn.LayerNorm(self.film_hidden),
            nn.Linear(self.film_hidden, self.film_hidden),
            nn.SiLU(),
        )

        self.time_cond = TimeConditioner(
            ch=self.width,
            n_blocks=len(self.blocks),
            emb_dim=self.film_hidden,
            hidden=self.film_hidden,
            film_mode=self.film_mode,
            gamma_scale=self.gamma_scale,
            beta_scale=self.beta_scale,
            gate_temp=self.gate_temp,
        )

        # tail calibrator (residual, starts as identity)
        self.tail_gate = nn.Sequential(
            nn.LayerNorm(self.film_hidden),
            nn.Linear(self.film_hidden, self.width),
            nn.Sigmoid(),
        )
        self.tail_conv = nn.Conv2d(self.width, self.width, kernel_size=1, bias=True)
        nn.init.zeros_(self.tail_conv.weight)
        nn.init.zeros_(self.tail_conv.bias)

        # Buffers for Y normalization (fallback 0..1 if not provided)
        self.register_buffer("y_min_buf", torch.tensor(0.0), persistent=False)
        self.register_buffer("y_max_buf", torch.tensor(1.0), persistent=False)
        if (y_min is not None) and (y_max is not None):
            self.y_min_buf.fill_(float(y_min))
            self.y_max_buf.fill_(float(y_max))

        # (Optional) precomputed RBF centers/widths (kept as buffers in case RBFEmbed consults them)
        t = torch.linspace(0, 1, self.rbf_K)
        centers = t**rbf_gamma
        edges = torch.cat([centers[:1], centers, centers[-1:]])
        spacings = torch.diff(edges)
        widths = 0.5 * spacings.clamp_min(rbf_min_width)
        self.register_buffer("rbf_centers", centers, persistent=False)
        self.register_buffer("rbf_widths", widths, persistent=False)

    # --- helpers ---
    def _y01_from_scalar(self, y_scalar: torch.Tensor) -> torch.Tensor:
        # Robust normalization to [0,1], using buffers if explicit min/max not provided.
        y_min = self.y_min_buf
        y_max = self.y_max_buf
        if (self.y_min is not None) and (self.y_max is not None):
            # Keep buffers in sync type/device-wise
            y_min = y_min.to(y_scalar.device).to(y_scalar.dtype)
            y_max = y_max.to(y_scalar.device).to(y_scalar.dtype)

        if self.y_map == "linear":
            denom = (y_max - y_min).clamp_min(1e-9)
            y01 = (y_scalar - y_min) / denom
            return y01.clamp(0.0, 1.0)

        elif self.y_map == "tanh":
            g = 1.0
            y_min = y_min.to(y_scalar.device).to(y_scalar.dtype)
            y_max = y_max.to(y_scalar.device).to(y_scalar.dtype)
            num = torch.tanh(g * (y_scalar - y_min))
            den = torch.tanh(g * (y_max - y_min)).clamp_min(1e-9)
            y01 = (num / den).clamp(0.0, 1.0)
            return y01

        else:
            raise ValueError(f"Unknown y_map: {self.y_map}")

    # --- trunk encoding (base18 + scalars Y,θ) ---
    def encode_trunk_from_components(
        self,
        base18: torch.Tensor,
        Y_scalar: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        # memory format nicety
        if self.channels_last:
            base18 = base18.contiguous(memory_format=torch.channels_last)

        # Trunk features
        h = self.lift(base18)                      # [B, C, H, W]
        B, C, H, W = h.shape

        # ----- build conditioning vector [B, *] -----
        y01 = self._y01_from_scalar(Y_scalar).to(h.device, h.dtype)

        # time embed expects [B]; flatten/broadcast safely
        y_flat = y01.reshape(B)  # works for [B], [B,1], [B,1,1,1], etc.
        eY = self.time_embed(y_flat)               # [B, KY]
        if eY.dim() > 2:
            eY = eY.view(B, -1)

        # theta embed → [B, KT]
        eT = self.theta_embed(theta.to(h.device, h.dtype))
        if eT.dim() == 1:
            eT = eT.unsqueeze(-1)
        if eT.dim() > 2:
            eT = eT.view(B, -1)

        h_cond_in = torch.cat([eY, eT], dim=1)    # [B, KY+KT]
        h_cond = self.pre_film(h_cond_in)         # [B, Hf]
        cond   = self.time_cond(h_cond)           # list of (gamma, beta, gate)

        # ----- FiLM residual blocks (pre-norm, FiLM on block delta, then gated residual add) -----
        for i, b in enumerate(self.blocks):
            h_norm = self.block_norm(h)
            h_delta = b(h_norm)                    # block sees normalized input

            gamma, beta, gate = cond[i]           # [B,W], [B,W] or None, gate [B,1] or [B,W]

            # gate -> per-channel
            if gate.dim() == 2 and gate.shape[1] == 1:
                gate = gate.expand(-1, self.width)
            elif gate.dim() == 1:
                gate = gate[:, None].expand(-1, self.width)

            # Apply FiLM to block delta only (stable); broadcast over H,W
            if gamma is not None:
                h_delta = h_delta * (1.0 + gamma.view(-1, self.width, 1, 1))
            if (beta is not None) and (self.film_mode != "scale_only"):
                h_delta = h_delta + beta.view(-1, self.width, 1, 1)

            # Residual update with per-channel gate
            h = h + gate.view(-1, self.width, 1, 1) * h_delta

        # Tail (starts at identity)
        tail_gate = self.tail_gate(h_cond).view(h.shape[0], self.width, 1, 1)
        h = h + tail_gate * self.tail_conv(h)
        return h

    def forward(self, base18: torch.Tensor,
                      Y_scalar: torch.Tensor,
                      theta: torch.Tensor,
                      *,
                      sample: bool | None = None,
                      nsamples: int = 1,
                      dY=None) -> torch.Tensor:
        # Allow accidental (out, extras)
        if isinstance(base18, (tuple, list)):
            base18 = base18[0]
        assert isinstance(base18, torch.Tensor), f"expected tensor, got {type(base18)}"

        B, _, H, W = base18.shape
        h = self.encode_trunk_from_components(base18, Y_scalar, theta)

        # Broadcast physical dY/Y to per-pixel maps for the head (dtype-safe)
        dtype = base18.dtype
        device = base18.device

        dYmap = None
        if dY is not None:
            dYmap = dY.to(device=device, dtype=dtype).view(B, 1, 1, 1).expand(B, 1, H, W)
        Ymap = Y_scalar.to(device=device, dtype=dtype).view(B, 1, 1, 1).expand(B, 1, H, W)

        return self.head(h, base18, Ymap, nsamples=nsamples, sample=sample, dY=dYmap)


def _reduce(x: torch.Tensor, reduction: str):
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none":
        return x
    raise ValueError(f"Unknown reduction: {reduction}")




class GroupLossWithQs(nn.Module):
    """
    Unified loss (formerly: GroupLossWithQs(GroupLoss)).
    Includes:
      - SU(3) packing/projection helpers
      - dipole / Q_s machinery
      - NLL/CRPS/spectral/moment pieces
      - optional full-cov α-NLL

    External helpers expected in the module:
      pack_to_complex, su3_gellmann_matrices, _ddp_world_pg_or_none,
      _q4_scalar, _iso_radial_power, _bch_alpha, _energy_distance_alpha
    """

    # class-level cache for radial binning
    _RADIAL_CACHE = {}

    def __init__(
        self,
        *,
        project_before_frob: bool = False,
        # existing extended terms
        dipole_weight: float = 0.0,
        dipole_slope_weight: float = 0.0,
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
        mono_weight: float = 0.0,
        qs_slope_weight: float = 0.0,
        qs_soft_beta: float = 0.,
        qs_soft_slope: float = 1.,
        nll_weight: float = 0.0,
        spec_weight: float = 0.0,
        kmin: float = 4,
        kmax: float = None,
        crps_weight: float = 0.,
        current_epoch: int = None,
        energy_weight: float = 0.,
        energy_grad_weight: float = 0.,
        moment_weight: float = 0.,
        spec_whiten_weight: float = 0.,
        spec_guard_weight: float = 0.,
        sigma_mode: str = "conv",         # "diag" | "conv" | "spectral"
        noise_core_fit_weight: float = 0,
    ):
        super().__init__()

        # ---- (formerly GroupLoss.__init__) ----
        self.project_before_frob = bool(project_before_frob)
        self.dipole_weight = float(dipole_weight)
        self.dip_offsets   = tuple(dipole_offsets)  # NOTE: code refers to self.dip_offsets
        self.register_buffer("I3", torch.eye(3, dtype=torch.cfloat), persistent=False)

        # ---- (formerly GroupLossWithQs.__init__) ----
        self.sigma_mode = sigma_mode
        self.dipole_slope_weight = float(dipole_slope_weight)
        self.energy_grad_weight = float(energy_grad_weight)
        self.qs_weight = float(qs_weight)
        self.qs_threshold = float(qs_threshold)
        self.qs_on = str(qs_on).upper()  # 'N' or 'S'
        assert self.qs_on in ("N", "S"), "qs_on must be 'N' or 'S'"
        self.qs_local = bool(qs_local)
        self.qs_scale = float(qs_scale)
        # higher-order
        self.quad_weight  = float(quad_weight)
        self.quad_pairs   = tuple((tuple(map(int, a)), tuple(map(int, b))) for (a,b) in quad_pairs)
        self.rough_weight = float(rough_weight)
        self.mono_weight = float(mono_weight)
        self.qs_slope_weight = float(qs_slope_weight)

        self.qs_soft_beta = float(qs_soft_beta)
        self.qs_soft_slope = float(qs_soft_slope)
        self.nll_weight = float(nll_weight)
        self.nll_target_mode = "none"
        self.energy_weight = float(energy_weight)

        self.current_epoch = int(current_epoch) if current_epoch is not None else None
        self.kmin = int(kmin)
        self.kmax = int(kmax) if kmax is not None else None

        # === feature flags (set from train() via setattr) ===
        self.use_fullcov   = getattr(self, 'use_fullcov', False)
        self.nll_compose   = getattr(self, 'nll_compose', 'brownian')
        self.bch_weight    = getattr(self, 'bch_weight', 0.0)
        self.energy_stride = getattr(self, 'energy_stride', 4)
        # Optional: uncertainty-based balancing
        self.auto_balance = getattr(self, 'auto_balance', False)
        if self.auto_balance:
            self.log_scales = nn.ParameterDict({
                'nll': nn.Parameter(torch.zeros(())),
                'bch': nn.Parameter(torch.zeros(())),
                'energy': nn.Parameter(torch.zeros(())),
            })
        self._lams_cached = None
        self.nll_stride = 1
        # local 3x3 identity (complex); I3 buffer already registered above
        self.metrics_pg = _ddp_world_pg_or_none() if '_ddp_world_pg_or_none' in globals() else None
        self.crps_weight = float(crps_weight)
        self.spec_weight = float(spec_weight)
        self.moment_weight = float(moment_weight)
        self.spec_whiten_weight  = float(spec_whiten_weight)
        self.spec_guard_weight = float(spec_guard_weight)
        self.noise_core_fit_weight = float(noise_core_fit_weight)

        print("[loss cfg] dipole_w=", self.dipole_weight, "qs_w=", self.qs_weight,
              "moment_w=", self.moment_weight, "quad_weight=", self.quad_weight,
              "nll_weight=", self.nll_weight, "energy_weight=", self.energy_weight,
              "spec_wighten_w=", self.spec_whiten_weight, "spec_guard_w=", self.spec_guard_weight)

        # --- Gell-Mann (λ/2) basis for projection (complex Hermitian) ---
        self.register_buffer("lambdas", su3_gellmann_matrices(), persistent=False)

        # cache for spectral white shape
        self._spec_white_cache = {}

    # ===== base helpers (formerly in GroupLoss) =====

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
        for _ in range(iters):
            XHX = X.conj().transpose(-1, -2) @ X
            X   = 0.5 * (X @ (3*I - XHX))
        det = torch.linalg.det(X)
        ang = torch.angle(det)
        phase_corr = torch.polar(torch.ones_like(ang), -ang/3.0).to(X.dtype)
        X = X * phase_corr[..., None, None]
        return X

    def _components(self, yhat: torch.Tensor, y: torch.Tensor, reduction: str = "mean"):
        # yhat, y: [B,18,H,W]
        Uh_raw = self._pack18_to_U(yhat)
        U_raw  = self._pack18_to_U(y)
        need_proj = self.project_before_frob
        Uh = self._su3_project(Uh_raw) if need_proj else Uh_raw
        U  = self._su3_project(U_raw)  if need_proj else U_raw
        return Uh, U, Uh_raw

    @staticmethod
    def _log_unitary(U: torch.Tensor, *, proj_iters: int = 1, log_dtype=torch.complex64) -> torch.Tensor:
        # project + eig in complex64 unless you *know* you need c128
        Uc = GroupLossWithQs._su3_project(U.to(log_dtype), iters=proj_iters)
        w, V = torch.linalg.eig(Uc)             # complex64 eigendecomp
        theta = torch.atan2(w.imag, w.real)
        Ldiag = 1j * theta
        L = V @ torch.diag_embed(Ldiag) @ torch.linalg.inv(V)
        return 0.5 * (L - L.conj().transpose(-1, -2))  # anti-Hermitian

    # ===== Qs/dipole/spectral helpers (already in your subclass) =====

    def _dipole_curve_from_U(self, U: torch.Tensor, stride: int) -> torch.Tensor:
        B,H,W = U.shape[:3]
        bins, counts = self._radial_bins(H, W, U.device)
        L = counts.numel()
        F = U.reshape(B,H,W,9).permute(0,3,1,2)
        Fk = torch.fft.fft2(F)
        corr = torch.fft.ifft2(Fk.conj() * Fk).real
        corr = corr.sum(dim=1) / (3.0*H*W)
        corr_flat = corr.view(B,-1); bin_ids = bins.view(-1)
        sums = torch.zeros(B, L, device=U.device, dtype=corr.dtype)
        sums.index_add_(1, bin_ids, corr_flat)
        S_rad = sums / counts.clamp_min(1).to(sums.dtype)
        if L>1: S_rad = S_rad[:,1:]
        N_rad = (1.0 - S_rad)
        return N_rad

    def _wb(self, name: str, loss_value: torch.Tensor, base_weight: float) -> torch.Tensor:
        if not getattr(self, 'auto_balance', False):
            return base_weight * loss_value
        s = self.log_scales[name]
        return torch.exp(-2*s) * (base_weight * loss_value) + s

    def _radial_power(self, X2d):
        """
        Radial power spectrum of a real field.
        Accepts X2d shaped [B,H,W], [H,W], or [B,C,H,W] (averages over C).
        Returns (spec[:,1:], k) with DC (r=0) dropped.
        """
        x = X2d
        if x.dim() == 4:
            x = x.mean(dim=1)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() != 3:
            raise ValueError(f"_radial_power expects [B,H,W]/[H,W]/[B,C,H,W], got {list(x.shape)}")
        B, H, W = x.shape
        F = torch.fft.fft2(x.to(torch.complex64), dim=(-2, -1))
        P = (F.abs() ** 2).to(torch.float32)
        bins, counts = self._radial_bins(H, W, x.device)
        K = counts.numel()
        spec = torch.zeros(B, K, device=x.device, dtype=P.dtype)
        spec.index_add_(1, bins.view(-1), P.view(B, -1))
        spec = spec / counts.to(spec.dtype)
        k = torch.arange(1, K, device=x.device, dtype=spec.dtype)
        return spec[:, 1:], k

    @classmethod
    def _radial_bins(cls, H: int, W: int, device):
        key = (H, W, device)
        cache = cls._RADIAL_CACHE
        if key in cache:
            return cache[key]
        dx = torch.arange(W, device=device)
        dy = torch.arange(H, device=device)
        dx = torch.minimum(dx, W - dx)[None, :].expand(H, W)
        dy = torch.minimum(dy, H - dy)[:, None].expand(H, W)
        r  = torch.sqrt(dx.to(torch.float32)**2 + dy.to(torch.float32)**2)
        bins = torch.round(r).to(torch.int64)
        L = int(round(math.hypot(W // 2, H // 2))) + 1
        counts = torch.bincount(bins.view(-1), minlength=L).to(torch.int64)
        cache[key] = (bins, counts)
        return cache[key]

    def _dipole_curve(self, U: 'torch.Tensor', local: bool, assume_su3: bool=False):
        if not assume_su3:
            U = self._su3_project(U)
        B, H, W = U.shape[:3]
        S_list, rs = [], []
        for dx, dy in self.dip_offsets:
            Us = torch.roll(U, shifts=(dy, dx), dims=(1, 2))
            prod = U @ Us.conj().transpose(-1, -2)
            Spr = torch.diagonal(prod, dim1=-2, dim2=-1).sum(-1).real / 3.0
            S_list.append(Spr if local else Spr.mean(dim=(1,2)))
            rs.append((dx*dx + dy*dy) ** 0.5)
        S = torch.stack(S_list, dim=-1)
        r = torch.tensor(rs, device=U.device, dtype=S.dtype)
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
        sumsq = torch.zeros(B, device=Uh.device, dtype=Uh.real.dtype)
        n = 0
        for (dx1,dy1),(dx2,dy2) in quad_pairs:
            qh = _q4_scalar(Uh, dx1, dy1, dx2, dy2)
            if sel is not None: qh = qh[sel]
            qh = qh.mean(dim=(-1)) if sel else qh.mean(dim=(1,2))
            with torch.no_grad():
                qt = _q4_scalar(Ut, dx1, dy1, dx2, dy2)
                if sel is not None: qt = qt[sel]
                qt = qt.mean(dim=(-1)) if sel else qt.mean(dim=(1,2))
            d = (qh - qt).to(dtype=sumsq.dtype)
            sumsq += d * d
            n += 1
        return (sumsq / max(n, 1)).mean()

    def _dipole_loss(self, Uh: torch.Tensor, U:  torch.Tensor,
                     *, local: bool = False, use_logN: bool = True,
                     per_radius_norm: bool = True, detach_target: bool = True) -> torch.Tensor:
        del local
        assert Uh.dtype.is_complex and U.dtype.is_complex
        B, H, W = Uh.shape[0], Uh.shape[1], Uh.shape[2]
        device = Uh.device
        bins, counts = self._radial_bins(H, W, device)
        L = counts.numel()

        def _radial_curve(Uc: torch.Tensor, grad: bool) -> torch.Tensor:
            F = Uc.reshape(B, H, W, 9).permute(0, 3, 1, 2).contiguous()
            ctx = (torch.enable_grad() if grad else torch.no_grad())
            with ctx:
                Fk   = torch.fft.fft2(F)
                corr = torch.fft.ifft2(Fk.conj() * Fk).real
                corr = corr.sum(dim=1)
                corr = corr / (3.0 * H * W)
                corr_flat = corr.view(B, -1)
                bin_ids   = bins.view(-1)
                sums = torch.zeros(B, L, device=device, dtype=corr.dtype)
                sums.index_add_(1, bin_ids, corr_flat)
                S_rad = sums / counts.clamp_min(1).to(sums.dtype)
                if L > 1: S_rad = S_rad[:, 1:]
                return S_rad

        S_pred = _radial_curve(Uh, grad=True)
        S_true = _radial_curve(U,  grad=not detach_target)
        N_pred, N_true = 1.0 - S_pred, 1.0 - S_true
        if per_radius_norm:
            w = 1.0 / N_true.abs().mean(dim=0).clamp_min(1e-3)
            N_pred = N_pred * w[None, :]
            N_true = N_true * w[None, :]
        if use_logN:
            N_pred = torch.log(N_pred.clamp_min(1e-6))
            N_true = torch.log(N_true.clamp_min(1e-6))
        return F.mse_loss(N_pred, N_true)

    @torch.no_grad()
    def _make_base_grid(self, H, W, device, dtype):
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        return yy, xx

    def _sample_shift_periodic(self, field, dx, dy, base=None):
        B, C, H, W = field.shape
        dtype, device = field.dtype, field.device
        yy, xx = base if base is not None else self._make_base_grid(H, W, device, dtype)
        x = (xx + dx) % W
        y = (yy + dy) % H
        gx = 2.0 * x / (W - 1) - 1.0
        gy = 2.0 * y / (H - 1) - 1.0
        grid = torch.stack((gx, gy), dim=-1).expand(B, H, W, 2)
        return F.grid_sample(field, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def _isotropic_dipole_loss(self, Uh: torch.Tensor, U:  torch.Tensor,
                               *, local: bool = False, use_logN: bool = True,
                               per_radius_norm: bool = True, detach_target: bool = True) -> torch.Tensor:
        del local
        assert Uh.dtype.is_complex and U.dtype.is_complex
        B, H, W = Uh.shape[0], Uh.shape[1], Uh.shape[2]
        device = Uh.device
        bins, counts = self._radial_bins(H, W, device)
        L = counts.numel()

        def _radial_curve(Uc: torch.Tensor, needs_grad: bool) -> torch.Tensor:
            F = Uc.reshape(B, H, W, 9).permute(0, 3, 1, 2).contiguous()
            ctx = (torch.enable_grad() if needs_grad else torch.no_grad())
            with ctx:
                Fk   = torch.fft.fft2(F)
                corr = torch.fft.ifft2(Fk.conj() * Fk).real
                corr = corr.sum(dim=1)
                corr = corr / (3.0 * H * W)
                corr_flat = corr.view(B, -1)
                bin_ids   = bins.view(-1)
                sums = torch.zeros(B, L, device=device, dtype=corr.dtype)
                sums.index_add_(1, bin_ids, corr_flat)
                S_rad = sums / counts.clamp_min(1).to(sums.dtype)
                if L > 1: S_rad = S_rad[:, 1:]
                return S_rad

        S_pred = _radial_curve(Uh, needs_grad=True)
        S_true = _radial_curve(U,  needs_grad=not detach_target)
        N_pred, N_true = 1.0 - S_pred, 1.0 - S_true
        if per_radius_norm:
            w = 1.0 / torch.sqrt((N_true**2).mean(dim=0).clamp_min(1e-8))
            w = (w / w.mean()).detach()
            N_pred = N_pred * w[None, :]
            N_true = N_true * w[None, :]
        if use_logN:
            N_pred = torch.log(N_pred.clamp_min(1e-6))
            N_true = torch.log(N_true.clamp_min(1e-6))
        return F.smooth_l1_loss(N_pred, N_true, beta=0.02)

    def _compute_Qs_from_U(self, U: 'torch.Tensor', *, local: bool):
        r, S = self._dipole_curve(U, local=local, assume_su3=True)
        X = 1.0 - S if self.qs_on == "N" else S
        thr = self.qs_threshold
        if getattr(self, "qs_soft_beta", 0.0) > 0.0:
            dX = X.diff(dim=-1, prepend=X[..., :1]).abs()
            scores = -self.qs_soft_beta * (X - thr).abs()
            if getattr(self, "qs_soft_slope", 1.0) != 0.0:
                eps = 1e-12
                scores = scores + self.qs_soft_slope * torch.log(dX.clamp_min(eps))
                w = torch.softmax(scores, dim=-1)
                r_star = (w * r).sum(dim=-1)
                return 1.0 / r_star.clamp_min(eps)

        X_mono, _ = torch.cummax(X, dim=-1)
        target = torch.tensor(self.qs_threshold, device=X.device, dtype=X.dtype)

        below = (X_mono < target).to(torch.int64)
        idx_hi = below.sum(dim=-1)
        K = X_mono.shape[-1]
        idx_hi = idx_hi.clamp(min=0, max=K-1)
        idx_lo = (idx_hi - 1).clamp(min=0, max=K-1)

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
        return Qs

    def _spec_loss_pack18(self, P18, T18, margin=0.08, high_over=0.05, eps=1e-12):
        import torch.nn.functional as F
        def ch18(x):  # channel-first
            return x if (x.ndim >= 4 and x.shape[1] == 18) else x.movedim(-1, 1)
        P = ch18(P18).float()
        T = ch18(T18).float()
        B, C, H, W = P.shape
        Lshape = P.new_tensor(0.0)
        Lamp   = P.new_tensor(0.0)
        Lhigh  = P.new_tensor(0.0)
        delta = math.log(1.0 + margin)
        overm = math.log(1.0 + high_over)
        for c in range(C):
            p = P[:, c] - P[:, c].mean(dim=(-2, -1), keepdim=True)
            t = T[:, c] - T[:, c].mean(dim=(-2, -1), keepdim=True)
            Pp, _ = self._radial_power(p)
            Pt, _ = self._radial_power(t)
            Tp = torch.log(Pp.sum(dim=-1) + eps)
            Tt = torch.log(Pt.sum(dim=-1) + eps)
            Lamp = Lamp + torch.clamp((Tp - Tt).abs() - delta, min=0.).mean()
            lp = torch.log(Pp + eps); lt = torch.log(Pt + eps)
            lp = lp - Tp.detach().unsqueeze(-1)
            lt = lt - Tt.detach().unsqueeze(-1)
            lp = lp - lp.mean(dim=-1, keepdim=True)
            lt = lt - lt.mean(dim=-1, keepdim=True)
            Lshape = Lshape + F.mse_loss(lp, lt)
            K  = Pp.shape[-1]
            k1 = max(1, (2 * K) // 3)
            Hp = torch.log(Pp[..., k1:].sum(dim=-1) + eps)
            Ht = torch.log(Pt[..., k1:].sum(dim=-1) + eps)
            Lhigh = Lhigh + torch.relu(Hp - Ht - overm).mean()
        C = float(C)
        return Lshape / C, Lamp / C, Lhigh / C

    def _quadrupole(self, U: torch.Tensor) -> torch.Tensor:
        U_q = U if self.project_before_frob else self._su3_project(U)
        outs = []
        for (dx1, dy1), (dx2, dy2) in self.quad_pairs:
            q4 = _q4_scalar(U_q, dx1, dy1, dx2, dy2)
            outs.append(q4.mean(dim=(1, 2)))
        return torch.stack(outs, dim=-1) if outs else U_q.new_zeros((U_q.shape[0], 0))

    def _get_lams(self, ref: torch.Tensor):
        if (self._lams_cached is None or
            self._lams_cached.device != ref.device or
            self._lams_cached.dtype  != ref.dtype):
            self._lams_cached = self.lambdas.to(ref.device, ref.dtype)
        return self._lams_cached

    def _alpha_map_from_pair(self, U0, U1, lams, *, stride=1, fast_thresh=0.15, proj_iters=1):
        if stride > 1:
            U0 = U0[:, ::stride, ::stride]
            U1 = U1[:, ::stride, ::stride]
        Delta = U1.mH.contiguous() @ U0
        I = self.I3.to(Delta.device, Delta.dtype)
        S = 0.5j * (Delta.mH.contiguous() - Delta)
        frob = torch.linalg.matrix_norm(Delta - I, ord='fro', dim=(-2,-1)) / (3.0**0.5)
        mask = (frob > fast_thresh)
        if mask.any():
            Dv = Delta.reshape(-1, 3, 3)
            mv = mask.reshape(-1)
            De = Dv[mv]
            De = self._su3_project(De, iters=proj_iters)
            w, V = torch.linalg.eig(De.to(torch.complex64))
            theta = torch.atan2(w.imag, w.real)
            L = V @ torch.diag_embed(1j * theta) @ torch.linalg.inv(V)
            L = L.contiguous(); Lh = L.mH.contiguous()
            S_exact = -0.5j * (L - Lh)
            S = S.reshape(-1, 3, 3); S[mv] = S_exact; S = S.reshape(Delta.shape)
        a = 2.0 * torch.real(torch.einsum('bhwij,aij->bahw', S, lams))
        return a

    def _get_white_shape(self, Hs, Ws, nbins, exclude_dc, window, device, dtype):
        key = (Hs, Ws, nbins, bool(exclude_dc), bool(window))
        if key in self._spec_white_cache:
            return self._spec_white_cache[key].to(device=device, dtype=dtype)
        with torch.no_grad():
            B, C = 4, 4
            z = torch.randn(B, C, Hs, Ws, device=device, dtype=dtype)
            Wk = _iso_radial_power(z, nbins=nbins, exclude_dc=exclude_dc, window=window)  # [B,C,K]
            Wk = Wk.mean(dim=(0,1))  # [K]
            Wk = torch.clamp(Wk, min=1e-12)
            self._spec_white_cache[key] = Wk.detach().clone().cpu()
            return Wk

    # ===== main loss =====
    def forward(self, yhat: 'torch.Tensor', y: 'torch.Tensor',
                base18: 'torch.Tensor' | None = None,
                dy_scalar: 'torch.Tensor' | None = None,
                theta=None, Y_final=None,
                mu_pred: 'torch.Tensor' | None = None,
                logsig_pred: 'torch.Tensor' | None = None,
                drift_pred: 'torch.Tensor' | None = None,
                amp_pred: 'torch.Tensor' | None = None,
                y_gate: 'torch.Tensor' | None = None,
                return_stats: bool = False,
                eta_core: torch.Tensor | None = None,
                ):

        Uh, U, Uh_raw = self._components(yhat, y, reduction="mean")

        # 2) Base loss (same weights you use in the base class)
        total = 0.0 
        stats = {}

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

        if getattr(self, "dipole_slope_weight", 0.0) > 0.0:
            # dipole curves [B,K]; drop DC bin if your helper returns it
            p = self._dipole_curve_from_U(Uh_q, stride=1)  # pred
            t = self._dipole_curve_from_U(U_q,  stride=1)  # target

            # center across radius so amplitude/offset don't dominate
            p = p - p.mean(dim=1, keepdim=True)            # [B,K]
            t = t - t.mean(dim=1, keepdim=True)

            # --- key fix: STOP-GRAD in the denominator ---
            # s = cov(p,t) / var(p_detached)  (only numerator drives grads)
            eps = 1e-12
            num = (p * t).mean(dim=1)                      # [B]
            den = (p.detach() * p.detach()).mean(dim=1) + eps
            s   = num / den                                # [B]

            # optional: emphasize radii where target curve changes fast
            w_r = (t[:,1:] - t[:,:-1]).abs()
            w_r = w_r / (w_r.mean(dim=1, keepdim=True) + eps)
            # project s to per-radius by local slopes (cheap proxy)
            # (alternatively just use s.mean() below)

            # one-sided hinge: encourage s >= 1 - margin (no pull if s is already high)
            margin = getattr(self, "dipole_slope_margin", 0.05)
            Ls = torch.relu((1.0 - margin) - s).mean()

            total = total + self._wb("dipole_slope", Ls, getattr(self, "dipole_slope_weight", 0.0))
        
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
            
        if self.rough_weight != 0.0:
            Pp, _ = self._radial_power(Uh_q)
            Pt, _ = self._radial_power(U_q)
            k0, k1 = self.kmin, (self.kmax if self.kmax is not None else Pp.shape[1])
            loss_rough = F.mse_loss(torch.log(Pp[:, k0:k1].clamp_min(1e-12)),
                                    torch.log(Pt[:, k0:k1].clamp_min(1e-12)))
            total = total + self.rough_weight * loss_rough
            
        ##----time----
        #if torch.cuda.is_available(): torch.cuda.synchronize();
        #t0=time.perf_counter()
        ##----end-time----


        a_true = None
        # --- NLL on α (projected generator) -----------------------------------
        if (self.nll_weight != 0.0) and (base18 is not None) \
           and (mu_pred is not None) and (logsig_pred is not None):
            with torch.no_grad():
                U0 = self._su3_project(self._pack18_to_U(base18), iters=1)
                U1 = self._su3_project(self._pack18_to_U(y),      iters=1)
                lams = self.lambdas.to(device=U0.device, dtype=U0.dtype)

                stride = getattr(self, "nll_map_stride", 1)
                a_true = self._alpha_map_from_pair(
                    U0, U1, lams,
                    stride=stride,
                    fast_thresh=getattr(self, "nll_fast_thresh", 0.15),
                    proj_iters=1
                )  # [B,C',H',W']

                if stride > 1:
                    # resize to match mu_pred H,W (not U0’s internal shape)
                    _, _, Ht, Wt = mu_pred.shape
                    a_true = torch.nn.functional.interpolate(
                        a_true, size=(Ht, Wt), mode='bilinear', align_corners=False
                    )

                # align channels to what you actually predict
                C = mu_pred.shape[1]
                if a_true.shape[1] != C:
                    a_true = a_true[:, :C]

            B, C, H, W = mu_pred.shape

            # ---- robust per-sample ΔY  -> [B,1,1,1] ----
            eps = 1e-12
            if isinstance(dy_scalar, torch.Tensor):
                s = dy_scalar.to(mu_pred.device, mu_pred.dtype).view(-1)
                if s.numel() == 1:
                    s = s.expand(B)
                elif s.numel() != B:
                    # fallback: average any trailing dims to B
                    s = s.reshape(B, -1).mean(dim=1)
            else:
                s = torch.tensor([float(dy_scalar)], device=mu_pred.device, dtype=mu_pred.dtype).expand(B)
            dY_map = s.view(B, 1, 1, 1).clamp_min(eps)

            # ---- drift & variance for THIS step ----
            # drift integrates over ΔY
            mu_step = mu_pred * dY_map  # [B,C,H,W]
    
            sigma_min, sigma_max = 1e-4, 0.20             # per unit-Y; tune to your system
            raw = logsig_pred
            sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(raw)
            log_sigma_unit = torch.log(sigma + eps)       # reuse in NLL
                  # parameterize log-σ safely:
            # if your head outputs "raw" (not logσ), convert to logσ via softplus
            #log_sigma_unit = torch.log(torch.nn.functional.softplus(logsig_pred) + eps)  # [B,C,H,W]
            # log variance for the step: log(σ^2 ΔY) = 2 logσ + log ΔY
            logvar_step = 2.0 * log_sigma_unit + torch.log(dY_map)

            # ---- canonical Gaussian NLL ----
            resid = mu_step - a_true  # [B,C,H,W]
            nll_elem = 0.5 * (resid.pow(2) * torch.exp(-logvar_step) + logvar_step)
            loss_nll = nll_elem.mean()  # invariant to batch/shape

            total = total + float(self.nll_weight) * loss_nll

            # optional sanity stats:
            stats['nll'] = loss_nll.detach()
            stats['nll_resid_mse'] = resid.pow(2).mean().detach()
            stats['nll_sigma2_mean'] = torch.exp(logvar_step).mean().detach()
            stats['nll_calib'] = (stats['nll_sigma2_mean'] /
                                  (stats['nll_resid_mse'] + eps)).detach()

        ##----time----
        #if torch.cuda.is_available(): torch.cuda.synchronize()
        #print(f"[timing] NLL loss: {(time.perf_counter()-t0)*1e3:.2f} ms")
        ##----end-time----
        ##----time----
        #if torch.cuda.is_available(): torch.cuda.synchronize();
        #t0=time.perf_counter()
        ##----end-time----

        # What is below:
        # Gaussian CRPS term that calibrates the magnitude of fluctuations (σ) on the α-grid (the Lie-algebra increment grid), while leaving μ untouched (you detach μ there).
        # Then a spectral matching term that teaches the noise core to reproduce the spatial correlation of those fluctuations by matching the radially averaged power spectrum of a teacher-forced residual.
        
        #Continuous Ranked Probability Score learns fluctuations: Proper and robust: encourages calibrated quantiles; much less sensitive to outliers than NLL (no big log σ penalty spikes).
        if (logsig_pred is not None) and (mu_pred is not None):
            eps = 1e-12
            dtype = mu_pred.dtype
            device = mu_pred.device

            with torch.no_grad():
                # Build a_true on the α-grid (low-res for stride > 1) 
                U0 = self._su3_project(self._pack18_to_U(base18), iters=1)
                U1 = self._su3_project(self._pack18_to_U(y),      iters=1)
                lams   = self._get_lams(U0)
                stride = int(getattr(self, "nll_map_stride", 1))
                a_true = self._alpha_map_from_pair(
                    U0, U1, lams,
                    stride=stride,
                    fast_thresh=getattr(self, "nll_fast_thresh", 0.15),
                    proj_iters=1
                )  # [B,Ca,Hs,Ws]

            # Downsample predictions to the α grid
            def _to_alpha_grid(t, s):
                if s <= 1: return t
                if t.dim() != 4: raise ValueError("expect [B,C,H,W]")
                return F.avg_pool2d(t, kernel_size=s, stride=s, ceil_mode=False)

            def _to_alpha_var_grid(var, s):
                if s <= 1: return var
                out = F.avg_pool2d(var, kernel_size=s, stride=s, ceil_mode=False)
                # variance of the average over s×s adds another / s^2
                return out / float(s*s)

            # Match channels
            mu_s    = _to_alpha_grid(mu_pred, stride)                     # [B,C?,Hs,Ws]
            sigma2  = F.softplus(logsig_pred).to(dtype=dtype).pow(2)      # σ^2 per unit Y
            sigma2s = _to_alpha_var_grid(sigma2, stride)                   # [B,C?,Hs,Ws]

            B,Ct,Hs,Ws = a_true.shape
            # Align batch & channels
            if mu_s.shape[0] != B:    mu_s    = mu_s[:B]
            if sigma2s.shape[0] != B: sigma2s = sigma2s[:B]
            Cmin = min(Ct, mu_s.shape[1], sigma2s.shape[1])
            a_true  = a_true[:,  :Cmin]
            mu_s    = mu_s[:,    :Cmin]
            sigma2s = sigma2s[:, :Cmin]

            # ΔY map on α grid (ensure positivity)
            if dy_scalar is None:
                dY = torch.ones(B, 1, 1, 1, device=device, dtype=dtype)
            else:
                sdy = dy_scalar.to(device=device, dtype=dtype).reshape(-1)[:B]
                dY  = (sdy.mean() if sdy.numel()==0 else sdy).view(-1,1,1,1)
                if dY.shape[0] == 1: dY = dY.expand(B,1,1,1)
            dY_abs = dY.abs().clamp_min(eps)

            # Step stats: μ_step and σ_step
            mu_step   = mu_s * dY                                         # [B,C,Hs,Ws]
            sigma2min = float(getattr(self, "moment_sigma2_min", 1e-8))
            sigma2max = float(getattr(self, "moment_sigma2_max", 1.0))    # give headroom
            sigma2s   = sigma2s.clamp(min=sigma2min, max=sigma2max)

            sig_step  = torch.sqrt(sigma2s * dY_abs).clamp_min(1e-6)      # [B,C,Hs,Ws]
            mu_anchor = mu_step.detach() #detach mu so this loss does not mess with it 
            z         = (a_true - mu_anchor) / sig_step

            # Gaussian CRPS (closed form)
            # Φ and φ; use erf or ndtr; both fine
            Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            phi = torch.exp(-0.5 * z*z) / math.sqrt(2.0*math.pi)
            crps = sig_step * (z * (2.0*Phi - 1.0) + 2.0*phi - 1.0/math.sqrt(math.pi))

            loss_crps = crps.mean()
            total = total + self._wb('crps', loss_crps, float(getattr(self, 'crps_weight', 0.0)))

            # Optional diagnostics
            with torch.no_grad():
                stats["crps/mean"]     = loss_crps.detach()
                stats["crps/|z|_mean"] = z.abs().mean().detach()
                stats["crps/sig_mean"] = sig_step.mean().detach()


            # --- teach the noise-core correlation (no effect on μ) ---

            # 1) Target: normalized residual on the α-grid (teacher-forced)
            eps = 1e-12
            resid = a_true - (mu_s * dY)                               # [B,C,Hs,Ws]
            r = resid / torch.sqrt(sigma2s.clamp_min(1e-8) * dY + eps) # ≈ N(0, Σ_space)

            # 3) Match spectra (or k-wise second moments). Isotropic radial power per channel:
            def iso_radial_power(x: torch.Tensor, nbins: int | None = None, eps: float = 1e-12) -> torch.Tensor:
                """
                Radial power spectrum averaged over batch and channels.
                x:     [B, C, H, W], real-valued field(s)
                nbins: number of radial bins; if None, uses min(H, W)//2
                returns: [nbins] tensor on x.device
                """
                assert x.dim() == 4, f"expected [B,C,H,W], got {x.shape}"
                B, C, H, W = x.shape
                if nbins is None:
                    nbins = int(min(H, W) // 2)

                # FFT and power (use rfft along width to save work)
                X = torch.fft.rfft2(x, dim=(-2, -1))              # [B,C,H, W//2+1], complex
                P = (X.conj() * X).real                           # power, [B,C,H, W//2+1]

                # Radial coordinates for the rfft grid
                ky = torch.fft.fftfreq(H, d=1.0, device=x.device)     # [-0.5..0.5) cycles/pixel
                kx = torch.fft.rfftfreq(W, d=1.0, device=x.device)    # [0..0.5]
                KY, KX = torch.meshgrid(ky, kx, indexing="ij")
                R = torch.sqrt(KY**2 + KX**2)                         # [H, W//2+1]

                # Bin edges and bin ids
                rmax = R.max()
                edges = torch.linspace(0, rmax + 1e-12, nbins + 1, device=x.device)
                idx = torch.bucketize(R.reshape(-1), edges) - 1       # [H*(W//2+1)], 0..nbins-1
                idx = idx.clamp_(0, nbins - 1)

                # Sum power per bin (average over B and C after)
                Pbc = P.reshape(B * C, -1)                            # [(B*C), HWr]
                idx_bc = idx.unsqueeze(0).expand(B * C, -1)           # match rows
                num = torch.zeros(B * C, nbins, device=x.device)
                num.scatter_add_(1, idx_bc, Pbc)

                # Pixel counts per bin (independent of B,C)
                ones = torch.ones_like(idx, dtype=P.dtype, device=x.device)
                counts = torch.zeros(nbins, device=x.device)
                counts.scatter_add_(0, idx, ones)
                counts = counts.clamp_min(1.0)

                Pr = num / counts                                    # [(B*C), nbins]
                return Pr.mean(dim=0)                                # [nbins]

            Pr = iso_radial_power(r.detach(), nbins=None) 

            # 2) Sample white and push through CURRENT noise-core (no ŷ, no exp/log)
            #epsn = torch.randn_like(mu_s)
#            eta  = noise_core_fn(epsn) if noise_core_fn is not None else epsn

            eta = eta_core
            
            Pe = iso_radial_power(eta, nbins=Pr.numel())             # same bin count
            L_core = F.mse_loss(torch.log(Pe.clamp_min(1e-8)),
                                torch.log(Pr.clamp_min(1e-8)))
            
            w = getattr(self, "noise_core_fit_weight", 0.1)            # add an argparse flag
            total = total + w * L_core
            stats["noise_core/L"] = L_core.detach()

                
        if getattr(self, "spec_weight", 0.0) > 0.0:
            Ls, La, Lhk = self._spec_loss_pack18(yhat, y,
                                                 margin=getattr(self, "spec_margin", 0.08),
                                                 high_over=getattr(self, "spec_high_over", 0.05))
            w = getattr(self, "spec_weight", 0.0) * getattr(self, "spec_ramp", 1.0)
            total = total + self._wb("spec_shape",      Ls,  w * 0.6)
            total = total + self._wb("spec_amp_band",   La,  w * 0.3)
            total = total + self._wb("spec_highk_over", Lhk, w * 0.1)

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


        tuner = None
        # --- MOMENT loss (NLL-consistent, per-pixel, resolution-aligned) ---
        if getattr(self, "moment_weight", 0.0) > -1.0: #always on
            if base18 is None:
                raise RuntimeError("moment_weight>0 but base18 is None. Pass U0 (18-ch) to criterion(...).")
            
            # 1) α supervision from (U0,U1) on a low-res grid; avoid bilinear up (it suppresses variance)
            with torch.no_grad():
                # stronger unitarization for supervision (more stable α)
                U0_full = self._su3_project(self._pack18_to_U(base18), iters=3)
                U1_full = self._su3_project(self._pack18_to_U(y),      iters=3)   # <-- y must be ground truth

                s       = int(getattr(self, "nll_map_stride", 1))
                lams    = self.lambdas.to(device=U0_full.device, dtype=U0_full.dtype)
                a_true  = self._alpha_map_from_pair(
                    U0_full, U1_full, lams,
                    stride=s,
                    fast_thresh=getattr(self, "nll_fast_thresh", 0.15),
                    proj_iters=1
                )  # [B, C, Hs, Ws]


            # 2) Choose predicted drift/amp sources (amp ≡ σ per unit Y)
            drift = drift_pred if (drift_pred is not None) else mu_pred
 
            if logsig_pred is not None:
                sigma = F.softplus(logsig_pred)
            else:
                sigma = None

            if (drift is not None) and (sigma is not None):
                # 3) Downsample to α grid
                def to_alpha_var_grid(var, stride):
                    if stride <= 1:
                        return var
                    # avg_pool gives mean(var) over s×s; variance of the average needs another / s^2
                    out = F.avg_pool2d(var, kernel_size=stride, stride=stride, ceil_mode=False)
                    return out / float(stride * stride)

                def to_alpha_grid(t, stride):
                    if stride <= 1: return t
                    if t.dim() != 4: raise ValueError("expect 4D [B,C,H,W] tensors")
                    return F.avg_pool2d(t, kernel_size=stride, stride=stride, ceil_mode=False)

                drift_s = to_alpha_grid(drift, s)  # [B, Cd, Hs, Ws]

                sigma2 = sigma.pow(2)
                sigma2_s = to_alpha_var_grid(sigma2, s)            # [B, Ca, Hs, Ws]
                # if sigma was broadcast-like, handle that path too:
                if sigma2_s.shape[1] == 1 and drift_s.shape[1] != 1:
                    sigma2_s = sigma2_s.expand(drift_s.shape[0], drift_s.shape[1], *sigma2_s.shape[-2:])

                # 4) Align batch & channels
                B, C, Hs, Ws = a_true.shape
                if drift_s.shape[0] != B: drift_s = drift_s[:B]
                if sigma2_s.shape[0] != B: sigma2_s = sigma2_s[:B]
                Cmin = min(C, drift_s.shape[1], sigma2_s.shape[1])
                if Cmin != C:
                    a_true   = a_true[:,  :Cmin]
                    drift_s  = drift_s[:, :Cmin]
                    sigma2_s = sigma2_s[:, :Cmin]

                # 5) Build dY and broadcast
                if dy_scalar is None:
                    dY = torch.ones(B, 1, 1, 1, device=a_true.device, dtype=a_true.dtype)
                else:
                    sdy = dy_scalar.to(device=a_true.device, dtype=a_true.dtype).reshape(-1)[:B]
                    dY  = (sdy.mean() if sdy.numel()==0 else sdy).view(-1,1,1,1)
                    if dY.shape[0] == 1: dY = dY.expand(B,1,1,1)

                eps = 1e-12

                # Optional cap (looser to avoid bias-low); or remove and parameterize σ in the head instead
                sigma2_min = float(getattr(self, "moment_sigma2_min", 1e-8))
                sigma2_max = float(getattr(self, "moment_sigma2_max", 0.50**2))  # allow more headroom
                sigma2_s = sigma2_s.clamp(min=sigma2_min, max=sigma2_max)

                # Step moments
                mu_step_s = drift_s * dY                         # [B,C,Hs,Ws]
                vpred     = sigma2_s * dY                        # [B,C,Hs,Ws]
                resid     = a_true - mu_step_s

                # >>> let a small fraction of grad hit σ to push it UP when needed <<<
                rho = float(getattr(self, "moment_whiten_grad_mix", 0.1))  # 0 (old detach) .. 1 (full grad)
                v_whiten = (1.0 - rho) * vpred.detach() + rho * vpred

                # 7) NLL-consistent pieces
                # m1: whitened residual (encourages correct mean); partial grad to σ via rho
                m1 = (resid.pow(2) / (v_whiten + eps)).mean()

                # m2: two-sided moment guard with deadzone, but underestimation gets a smaller weight
                m2_pred = (mu_step_s.detach().pow(2) + vpred)    # detach μ to not bias mean via m2
                m2_true = a_true.pow(2)

                tau_over  = float(getattr(self, "moment_over_tau", 0.10))
                tau_under = float(getattr(self, "moment_under_tau", 0.05))

                m2_over  = F.relu(m2_pred - (1.0 + tau_over)  * m2_true)
                m2_under = F.relu((1.0 - tau_under) * m2_true - m2_pred)

                w_over   = float(getattr(self, "moment_m2_over_weight", 0.25))  # stronger
                w_under  = float(getattr(self, "moment_m2_under_weight", 0.10)) # gentler
                m2 = (w_over * m2_over + w_under * m2_under).mean()

                # (Optional) smoothness on log σ to avoid speckle
                h1_w = float(getattr(self, "moment_amp_h1", 0.001))
                if h1_w > 0.0:
                    log_sigma_s = 0.5 * torch.log(sigma2_s + eps)               # log σ
                    dx = log_sigma_s[..., :, 1:] - log_sigma_s[..., :, :-1]
                    dy = log_sigma_s[..., 1:, :] - log_sigma_s[..., :-1, :]
                    m3 = (dx.pow(2).mean() + dy.pow(2).mean())
                else:
                    m3 = torch.zeros((), device=a_true.device, dtype=a_true.dtype)

                # (Optional) weak prior nudging σ toward init (raise if you still see bias-low)
                sigma0 = float(getattr(self, "moment_sigma0", 0.02))
                lambda_sigma = float(getattr(self, "moment_sigma_prior", 0.0))  # e.g., 1e-3
                m4 = lambda_sigma * (0.5*torch.log(sigma2_s + eps) - math.log(sigma0)).pow(2).mean()

                # 9) Weights
                m1_w = float(getattr(self, "moment_m1_weight", 1.0))
                m3_w = h1_w

                moment_loss = m1_w * m1 + m2 + m3_w * m3 + m4


                # ---- Spectral calibration (learn the spectrum of fluctuations) ----
                spec_bins        = int(getattr(self, "spec_bins", min(Hs, Ws)//2))
                spec_exclude_dc  = bool(getattr(self, "spec_exclude_dc", True))
                spec_window      = bool(getattr(self, "spec_window", False))  # set True if boundaries aren't periodic

                # (1) Whitened residual should be spectrally flat (white)
                r         = resid / torch.sqrt(v_whiten + eps)                # [B,C,Hs,Ws]
                Pr        = _iso_radial_power(r, nbins=spec_bins,
                                              exclude_dc=spec_exclude_dc, window=spec_window)  # [B,C,K]
                # For an orthonormal FFT, iid N(0,1) stays N(0,1) in k-space → target power ≈ 1
                spec_flat = F.smooth_l1_loss(Pr, torch.ones_like(Pr), reduction='mean')

                # (2) Match second moment in k-space (mean power + white-noise floor)
                S_true  = _iso_radial_power(a_true, nbins=spec_bins,
                                            exclude_dc=spec_exclude_dc, window=spec_window)
                S_mu    = _iso_radial_power(mu_step_s.detach(), nbins=spec_bins,
                                            exclude_dc=spec_exclude_dc, window=spec_window)


                Kbins = max(1, int(getattr(self, "spec_bins", min(Hs, Ws)//2)))
                spec_exclude_dc  = bool(getattr(self, "spec_exclude_dc", True))
                spec_window      = bool(getattr(self, "spec_window", False))

                # …compute S_true and S_mu exactly as you already do…
                
                white   = vpred.mean(dim=(-2, -1), keepdim=False)            # [B,C]
                Wk      = self._get_white_shape(Hs, Ws, Kbins, spec_exclude_dc, spec_window,
                                            device=a_true.device, dtype=a_true.dtype)  # [K]
                S_noise = white.unsqueeze(-1) * Wk.view(1,1,-1)               # [B,C,K]  <-- calibrated floor
                S_pred  = S_mu + S_noise


                
                # white   = vpred.mean(dim=(-2, -1), keepdim=False)             # [B,C]
                # S_noise = white.unsqueeze(-1).expand_as(S_mu)                 # [B,C,K]
                # S_pred  = S_mu + S_noise

                tau_k      = float(getattr(self, "spec_tau", 0.10))
                w_over_k   = float(getattr(self, "spec_over_weight", 0.25))
                w_under_k  = float(getattr(self, "spec_under_weight", 0.10))

                over_k   = F.relu(S_pred - (1.0 + tau_k)  * S_true)
                under_k  = F.relu((1.0 - tau_k) * S_true - S_pred)
                spec_guard = (w_over_k * over_k + w_under_k * under_k).mean()

                # Weights and combine
                spec_w_flat  = float(getattr(self, "spec_whiten_weight", 0.))   # e.g., 1e-2
                spec_w_guard = float(getattr(self, "spec_guard_weight", 0.))    # e.g., 1e-2


                moment_loss = moment_loss + spec_w_flat * spec_flat + spec_w_guard * spec_guard
                total = total + float(self.moment_weight) * moment_loss

                # ----- Per-sample fluctuation diagnostics for tuner -----
                # 1) Per-sample χ² of whitened residuals (should be ~1 if calibrated)
                #    Average over (C,H,W) for each sample.
                chi2_map = (resid.pow(2) / (v_whiten + eps))           # [B,C,Hs,Ws]
                fluct_chi2_per_sample = chi2_map.mean(dim=(1,2,3))     # [B]

                # 2) Per-sample spectral relative L2 error between predicted and true power
                #    S_* are [B, C, K]. Compute relL2 per channel then average over channels.
                num_k = (S_pred - S_true).pow(2).sum(dim=-1)           # [B, C]
                den_k = (S_true.pow(2).sum(dim=-1) + eps)              # [B, C]
                relL2_per_ch = torch.sqrt(num_k / den_k)               # [B, C]
                spec_relL2_per_sample = relL2_per_ch.mean(dim=1)       # [B]

                # 3) A single weighted fluctuation metric per sample (you can tune weights)
                w_chi2 = float(getattr(self, "tuner_w_chi2", 1.0))
                w_spec = float(getattr(self, "tuner_w_spec", 0.05))
                fluct_metric_per_sample = w_chi2 * fluct_chi2_per_sample + w_spec * spec_relL2_per_sample

                # 4) Save ΔY per sample so we can bin by it in validation
                #    (dy_scalar exists in your loss; if None we used ones above)
                if dy_scalar is None:
                    dy_batch = torch.ones(a_true.size(0), device=a_true.device, dtype=a_true.dtype)
                else:
                    dy_batch = dy_scalar.to(device=a_true.device, dtype=a_true.dtype).reshape(-1)[:a_true.size(0)]

                # 5) Package a small dict to return (detach + cpu to be safe)
                tuner = {
                    "fluct_chi2_per_sample": fluct_chi2_per_sample.detach().float().cpu(),
                    "spec_relL2_per_sample":  spec_relL2_per_sample.detach().float().cpu(),
                    "fluct_metric_per_sample": fluct_metric_per_sample.detach().float().cpu(),
                    "dy_per_sample":           dy_batch.detach().float().cpu(),
                }
                

                # Stats
                stats["moment/m1"] = m1.detach()
                stats["moment/m2_over"] = m2_over.mean().detach()
                stats["moment/m2_under"] = m2_under.mean().detach()
                if h1_w > 0.0: stats["moment/m3_tv"] = m3.detach()
                if lambda_sigma > 0.0: stats["moment/m4_sigma_prior"] = m4.detach()
                stats["moment/vpred_mean"] = vpred.mean().detach()
                stats["moment/m2_pred_mean"] = m2_pred.mean().detach()
                stats["moment/m2_true_mean"] = m2_true.mean().detach()
                stats["moment/calib_ratio"] = (m2_pred.mean() / (m2_true.mean() + eps)).detach()
                stats["spec/flat"]          = spec_flat.detach()
                stats["spec/guard"]         = spec_guard.detach()
                stats["spec/calib_ratio_k"] = (S_pred.mean() / (S_true.mean() + eps)).detach()


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
                    S_true = (-1j) * L_true
                    lams = self._get_lams(S_true)  
#lams = self.lambdas.to(device=S_true.device, dtype=S_true.dtype)  # or cached helper
                    B, H, W = S_true.shape[:3]
                    a_true_bar = 2.0 * torch.real(torch.einsum('bhwij,aij->ba', S_true, lams)) / (H * W)

            if need_a_pred:
                # Only needed for energy distance; compute directly at stride
                Upred = self._su3_project(self._pack18_to_U(yhat_s), iters=1)
                Delta_pred = Upred.mH @ U0
                L_pred = self._log_unitary(Delta_pred, proj_iters=2, log_dtype=torch.complex64)
                S_pred = (-1j) * L_pred
                lams = self._get_lams(S_pred)  
                #lams = self.lambdas.to(device=S_pred.device, dtype=S_pred.dtype)
                a_pred = 2.0 * torch.real(torch.einsum('bhwij,aij->bahw', S_pred, lams))  # [B,C,H',W']


#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize()
#        print(f"[timing] NLL 1: {(time.perf_counter()-t0)*1e3:.2f} ms")
#        #----end-time----
#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize();
#        t0=time.perf_counter()
#        #----end-time----

        # 2) Full-cov α-NLL on spatial average (requires composer & θ) — robust, no sigma_floor, no Cholesky
        if (self.nll_weight != 0.0) and getattr(self, 'use_fullcov', False) \
           and hasattr(self, 'param_nll') and (theta is not None) and (a_true_bar is not None):
            B, C = a_true_bar.shape

            # Match channels if model uses 16-ch α basis: duplicate 8->16
            if self.param_nll.C == 16 and C == 8:
                a_true_bar = torch.cat([a_true_bar, a_true_bar], dim=1)
                C = 16

            # Horizon
            Yv = (Y_final if Y_final is not None else dy_scalar).to(a_true_bar.dtype).view(-1)  # [B]

            # Parametric composer
            mu_unit, L, kappa = self.param_nll(theta)               # L: [B,C,C], kappa: [B,C]
            # (Ensure kappa > 0 inside param_nll with softplus+eps)
            muY  = self.param_nll.compose_mu(mu_unit, kappa, Yv)    # [B,C]
            CovY = self.param_nll.cov_to_Y(L, kappa, Yv)            # [B,C,C]

            # Enforce symmetry (important under AMP / numerical noise)
            CovY = 0.5 * (CovY + CovY.transpose(-1, -2))

            # Eigendecomposition in float32 for stability (keeps gradients)
            Cov32 = CovY.to(torch.float32)
            w, V = torch.linalg.eigh(Cov32)                         # w: [B,C] (ascending), V: [B,C,C]

            # Per-sample relative/absolute floors (NO fixed sigma floor)
            # floor ~ max(abs, rel * mean variance)
            mean_var = Cov32.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-12)  # [B,1]
            eps_abs = 1e-8
            eps_rel = 1e-6
            floor = torch.maximum(
                torch.full_like(mean_var, eps_abs),
                eps_rel * mean_var
            )                                                        # [B,1]
            w_clamped = torch.clamp_min(w, floor)                    # [B,C]  (strictly > 0)

            # Log-determinant: sum log eigenvalues
            logdet = torch.log(w_clamped).sum(dim=-1)               # [B]

            # Mahalanobis term: for resid = a_true_bar - muY
            resid = (a_true_bar - muY).to(torch.float32)            # [B,C]
            # z = V^T resid
            z = torch.matmul(V.transpose(-2, -1), resid.unsqueeze(-1)).squeeze(-1)  # [B,C]
            maha = (z.pow(2) / w_clamped).sum(dim=-1)               # [B]

            # Scale to match your previous recipe (scale_maha = 5.0)
            loss_nll_full = 0.5 * (5.0 * maha + logdet)             # [B]
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



            # Mahalanobis coverage (χ² check) using eigendecomposition (no Cholesky/solve)
            res_f32 = (a_true_bar - muY).to(torch.float32)                # [B,C]
            # z = V^T r
            z = torch.matmul(V.transpose(-2, -1), res_f32.unsqueeze(-1)).squeeze(-1)  # [B,C]
            # Σ^{-1} r = V diag(1/w_clamped) V^T r
            inv_res = torch.matmul(V, (z / w_clamped).unsqueeze(-1)).squeeze(-1)      # [B,C]
            mah2 = (res_f32 * inv_res).sum(dim=-1)                                    # [B]
            # (optional) log to stats dict if you track it:
            # stats['mah2_cov'] = mah2.detach().mean()

            # safety counters (how often we hit floors/caps)
            # unclamped smallest eigenvalue (diagnostic only)
            lam_min_raw = w.min(dim=-1).values     # [B]
            # stats['eigmin_raw'] = lam_min_raw.detach().mean()

            # ---- penalties / regularizers ----

            # Simple eigenvalue hinge to discourage indefiniteness (relative to mean variance)
            mean_var = Cov32.diagonal(dim1=-2, dim2=-1).mean(dim=-1)                 # [B]
            tau = (1e-3 * mean_var).detach()
            #tau = 1e-3 * mean_var                                                    # per-sample
            lambda_eig = 1e-2
            total = total + lambda_eig * F.relu(tau - lam_min_raw).mean()

            # Reconstruct a PSD covariance from clamped eigenpairs for ALL covariance-based regs
            # Σ_psd = V diag(w_clamped) V^T   (float32 for stability, then cast back)
            CovY_psd32 = (V * w_clamped.unsqueeze(-2)) @ V.transpose(-2, -1)         # [B,C,C]
            CovY_psd   = CovY_psd32.to(CovY.dtype)

            # Off-diagonal L2 penalty
            offdiag_weight = 0.01
            diag = torch.diagonal(CovY_psd, dim1=-2, dim2=-1)                       # [B,C]
            off  = CovY_psd - torch.diag_embed(diag)                                # [B,C,C]
            off_pen = (off**2).mean()
            total = total + self._wb('cov_offdiag', offdiag_weight * off_pen, 1.0)

            # “Cov barrier”: tr(Σ) - log|Σ| - C  (use logdet from eigenpairs above)
            cov_reg_weight = 0.002
            trY = diag.sum(-1)                                                      # [B]
            cov_barrier = (trY - logdet - C).mean()
            total = total + self._wb('cov_barrier', cov_reg_weight * cov_barrier, 1.0)

            # Correlation off-diagonal penalty
            offcorr_weight = 0.001
            std  = torch.sqrt(torch.clamp_min(diag, 1e-8))                          # [B,C]
            den  = std.unsqueeze(-1) * std.unsqueeze(-2) + 1e-8
            Corr = CovY_psd / den
            Corr = Corr - torch.diag_embed(torch.ones_like(std))                    # zero out diagonal
            corr_off = (Corr**2).mean()
            total = total + self._wb('cov_corr_off', offcorr_weight * corr_off, 1.0)

            # μ anchor (weak)
            mu_anchor_weight = 0.0001
            mu_anchor = F.mse_loss(muY, a_true_bar)                                 # both [B,C]
            total = total + self._wb('mu_anchor', mu_anchor_weight * mu_anchor, 1.0)


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
        
        if self.dipole_weight != 0.0:
            stats["dip_mse"] = dip.detach()

        if getattr(self, "energy_grad_weight", 0.0) > 0.0:
            def grads(x):
                gx = x[..., :, 1:] - x[..., :, :-1]
                gy = x[..., 1:, :] - x[..., :-1, :]
                return gx, gy
            gx_t, gy_t = grads(a_true); gx_p, gy_p = grads(a_pred)
            stride = max(int(getattr(self, 'energy_stride', 4)), 1)
            def flatten(v): return v[:, :, ::stride, ::stride].permute(0,2,3,1).reshape(-1, v.shape[1])
            T = torch.cat([flatten(gx_t), flatten(gy_t)], dim=0).detach()
            P = torch.cat([flatten(gx_p), flatten(gy_p)], dim=0)
            loss_grad = _energy_distance_alpha(T, P)
            total = total + self._wb('energy_grad', loss_grad, getattr(self, 'energy_grad_weight', 0.0))


        
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
                p = C
                stats['alpha_mah2_p']    = p                    # target mean ≈ p if calibrated

                stats['alpha_lammin_med'] = float(lam_min.median().detach())


        return total, stats, tuner

    @torch.no_grad()
    def components_dict(self, yhat: 'torch.Tensor', y: 'torch.Tensor', base18: 'torch.Tensor' | None = None):
        # No base class—start a fresh dict and compute the Qs/quad metrics this class adds.
        out = {}
        if self.qs_weight != 0.0 and len(self.dip_offsets) >= 2:
            Uh = self._pack18_to_U(yhat)
            U  = self._pack18_to_U(y)
            Qh = self._compute_Qs_from_U(Uh, local=self.qs_local)
            Qt = self._compute_Qs_from_U(U,  local=self.qs_local)
            out["qs_mse"] = torch.nn.functional.mse_loss(Qh, Qt).detach()
            out["qs_pred_mean"] = Qh.mean().detach()
            out["qs_true_mean"] = Qt.mean().detach()
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
        Rahleturns:
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

def rollout_predict(model, base18, dYin, k, theta,
                    *, track_grad=True):
    """
    Returns:
      yhat_roll:      k small steps from (random) start
      yhat_single_big: one big step of size k*ΔY from same (random) start
    """

    # --- main rollout used in losses (GRADS ON) ---
    ctx = contextlib.nullcontext() if track_grad else torch.no_grad()
    with ctx:
        # one big step of size k*ΔY from U_start
        yhat_single_big, _ = model(base18, dYin * k, theta, sample=False, dY=dYin * k)

        # k small steps from U_start
        y = base18
        for _ in range(k):
            y, _ = model(y, dYin, theta, sample=False, dY=dYin)
        yhat_roll = y

    return yhat_roll, yhat_single_big

def ramp(epoch, start, duration):
    # linear 0→1 starting at `start` over `duration` epochs
    if epoch < start: return 0.0
    if epoch >= start+duration: return 1.0
    return (epoch - start) / max(1, duration)



# ---- MINI a-FLUCTUATION MONITOR (pack-18 only) ----
@torch.no_grad()
def monitor_a_fluct(pred, truth, eps=1e-12):
    """
    pred, truth: pack-18 tensors shaped [B,18,H,W] or [B,H,W,18]
    Returns: (var_ratio_med, slope_med, grad_ratio_med)
    """
    def ch18(x):
        dims = list(x.shape)
        if 18 in dims:
            ch = dims.index(18)
            x = x if ch == 1 else x.movedim(ch, 1)   # -> [B,18,H,W]
            if x.dim() == 3: x = x.unsqueeze(0)      # [18,H,W] -> [1,18,H,W]
            return x.float()
        raise ValueError(f"monitor_a_fluct: expected pack-18, got shape {dims}")

    P = ch18(pred)
    T = ch18(truth)

    # center each sample/channel over space
    P = P - P.mean(dim=(-2,-1), keepdim=True)
    T = T - T.mean(dim=(-2,-1), keepdim=True)

    # 1) amplitude: variance ratio (median over B×C)
    vr = (P.var(dim=(-2,-1), unbiased=False) + eps) / (T.var(dim=(-2,-1), unbiased=False) + eps)
    var_ratio_med = vr.median().item()

    # 2) calibration: regression slope s = cov(P,T) / var(P)  (median over B×C)
    slope = ((P*T).mean(dim=(-2,-1)) / (P.pow(2).mean(dim=(-2,-1)) + eps)).median().item()

    # 3) length-scale proxy: gradient-power ratio (median over B×C)
    def gpow(X):
        dx = X[..., :, 1:] - X[..., :, :-1]
        dy = X[..., 1:, :] - X[..., :-1, :]
        return dx.pow(2).mean(dim=(-2,-1)) + dy.pow(2).mean(dim=(-2,-1))
    grad_ratio_med = ((gpow(P) + eps) / (gpow(T) + eps)).median().item()

    return var_ratio_med, slope, grad_ratio_med


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
        alpha_scale=1.0, clamp_alphas=getattr(args, "clamp_alphas", 2.),
        y_index=args.y_channel, film_mode=args.film_mode, rbf_K=args.rbf_K,
        film_hidden=args.film_hidden, gamma_scale=args.gamma_scale,
        beta_scale=args.beta_scale, gate_temp=args.gate_temp,
        y_min=(args.y_min if args.y_min is not None else y_min_data),
        y_max=(args.y_max if args.y_max is not None else y_max_data),
        y_map=args.y_map,
        sigma_mode=args.sigma_mode,
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
                _, extras = model(xb[:1], yb[:1], tb[:1], sample=False)    # materialize all Lazy params
                model.train()
                model = model.to(memory_format=torch.channels_last)

    def unwrap(m):
        return m.module if isinstance(m, DDP) else m

    
    mcore = unwrap(model)
    ema = EMA(mcore, decay=args.ema_decay) if args.ema_decay and args.ema_decay>0 else None
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            static_graph=False,
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
        print(epoch, "/", args.epochs)
        e = epoch
        w_qs    = args.qs_weight #* (1.0 - 0.5 * ramp(e, start=E1, duration=E2-E1))
        w_dip   = args.dipole_weight #* (1.0 - 0.5 * ramp(e, start=E1, duration=E2-E1))
        w_dip_slope = args.dipole_slope_weight
        w_quad  = args.quad_weight  * ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_nll   = args.nll_weight  * ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_energy= args.energy_weight * ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_eg    = args.energy_grad_weight *ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_spec  = args.spec_weight*ramp(e, start=E1, duration=max(1,(E2-E1)))
        # derive correlator geometry from args
        dipole_offsets = _parse_offsets_list(getattr(args, "dipole_offsets", "(1,0),(0,1),(2,0),(0,2)"))
        quad_pairs     = _parse_quad_pairs_list(getattr(args, "quad_pairs", "auto_from_dipole"), dipole_offsets=dipole_offsets)
        w_crps = args.crps_weight * ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_mom = args.moment_weight *ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_spec_w =  args.spec_whiten_weight * ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_spec_g =  args.spec_guard_weight * ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_noise_core = args.noise_core_fit_weight * ramp(e, start=E1, duration=max(1,(E2-E1)))
        
        # Optional cap for the number of quadrupole configs (broader sets can be large)
        max_qp = int(getattr(args, "quad_max_pairs", 0) or 0)
        if max_qp > 0 and len(quad_pairs) > max_qp:
            rnd = random.Random(1337)
            tmp = list(quad_pairs)
            rnd.shuffle(tmp)
            quad_pairs = tuple(tmp[:max_qp])

        criterion = GroupLossWithQs(
            dipole_weight=w_dip, dipole_slope_weight=w_dip_slope, dipole_offsets=dipole_offsets,
            qs_weight=w_qs, qs_threshold=0.5, qs_on='N', qs_local=True,
            qs_soft_beta=args.qs_soft_beta, spec_weight=w_spec,
            qs_soft_slope=args.qs_soft_slope,
            crps_weight=w_crps,
            moment_weight=w_mom,
            spec_whiten_weight=w_spec_w,
            spec_guard_weight=w_spec_g,
            quad_weight=w_quad, quad_pairs=quad_pairs,
            mono_weight=float(getattr(args,"mono_weight",0.0)),
            qs_slope_weight=float(getattr(args,"qs_slope_weight",0.0)),
            nll_weight = w_nll,
            sigma_mode = str(getattr(args,"sigma_mode","conv")),
            noise_core_fit_weight = w_noise_core,
            current_epoch = epoch, energy_weight = w_energy, energy_grad_weight = w_eg
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
                ##----time----
                #if torch.cuda.is_available(): torch.cuda.synchronize();
                #t0=time.perf_counter()
                ##----end-time----
                yhat_single, extras= model(base18, Y_scalar, theta, sample=False)
                yhat = yhat_single
                dy_scalar = Y_scalar  # replaces _y_from_batch(x, ...)

                ##----time----
                #if torch.cuda.is_available(): torch.cuda.synchronize()
                #print(f"[timing] forward: {(time.perf_counter()-t0)*1e3:.2f} ms")
                ##----end-time----

                # #----time----
                # if torch.cuda.is_available(): torch.cuda.synchronize();
                # t0=time.perf_counter()
                # #----end-time----

                core = unwrap(model)

                # (optional) if your code still keeps a legacy per-channel sigma head:
#                logsig_pred = getattr(core.head, "last_y_sigma", None)
                logsig_pred = extras.get("logsig")
                eta_core = extras.get("eta_core")
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
                else:
                    # fall back to whatever the model head produced
                    mu_pred     = extras.get("mu")
                
                if mu_pred is None or logsig_pred is None:
                    print("[warn] last_mu/logsig missing; NLL will be inactive.")


#                #----time----
#                if torch.cuda.is_available(): torch.cuda.synchronize();
#                t0=time.perf_counter()
#                #----end-time----

                
                loss, stats, tuner = criterion(
                    yhat, y, base18,
                    dy_scalar=Y_scalar,          # [B]
                    theta=theta,                 # [B,3]
                    Y_final=Y_scalar,            # alias as in your code
                    mu_pred=mu_pred,
                    logsig_pred=logsig_pred,
                    drift_pred=mu_pred,         # drift per channel (what head calls μ)
                    y_gate=y_gate,
                    return_stats=True,
                    eta_core=eta_core
                )

#                #----time----
#                if torch.cuda.is_available(): torch.cuda.synchronize()
#                print(f"[timing] loss: {(time.perf_counter()-t0)*1e3:.2f} ms")
#                #----end-time----


                rollout_k = int(rollout_k_curr)
                w_state   = float(getattr(args, "rollout_consistency", 0.0))
                w_single  = float(getattr(args, "rollout_single_consistency", 0.0))


                B = base18.size(0)
                
                def as_B1111(val, ref):
                    t = torch.as_tensor(val, device=ref.device, dtype=ref.dtype)
                    return t.view(1,1,1,1).expand(B,1,1,1)  # broadcast per-sample

                # one-step ΔY (float or tensor) -> batch tensor
                dY_step_val = steps_to_Y(1, ds_value.item() if torch.is_tensor(ds_value) else ds_value)
                Y_scalar_r = as_B1111(dY_step_val, base18)   # [B,1,1,1]
                #                dY = steps_to_Y(1, ds_value)
                Y_limit = core.y_max-rollout_k*dY_step_val
#                print(core.y_max, Y_limit)

                if (base18.device.type == "cpu") and bool(getattr(args, "skip_heavy_on_cpu", True)):
                    w_state = 0.0
                    w_single = 0.0

                if rollout_k > 1 and (w_state > 0.0 or w_single > 0.0 and Y_scalar < Y_limit):
                    #start from final U (called y) and evolve k steps
                    yhat_roll, yhat_single_big = rollout_predict(
                        model, y, Y_scalar_r, rollout_k, theta,
                        track_grad=True
                    )

                    sg_offsets = getattr(args, "sg_dipole_offsets", None)

                    if w_state > 0.0:
                        Nd = dipole_from_links18(yhat_single_big, offsets=sg_offsets)
                        Nc = dipole_from_links18(yhat_roll,       offsets=sg_offsets)
                        rollout_state_loss = F.mse_loss(Nd, Nc)
                        loss = loss + w_state * rollout_state_loss
                        meters.add("train/rollout_state_cons", rollout_state_loss.detach(), base18.size(0))
 
                    if w_single > 0.0:
                        # transitions from each predicted state (grads OK)
                        next_from_roll,  _ = model(yhat_roll.detach().clone(),       Y_scalar_r, theta, sample=False, dY=Y_scalar_r)
                        next_from_big,   _ = model(yhat_single_big.detach().clone(), Y_scalar_r, theta, sample=False, dY=Y_scalar_r)
                        Nr = dipole_from_links18(next_from_roll, offsets=sg_offsets)
                        Nb = dipole_from_links18(next_from_big,  offsets=sg_offsets)
                        rollout_single_loss = F.mse_loss(Nr, Nb)
                        loss = loss + w_single * rollout_single_loss
                        meters.add("train/rollout_single_cons", rollout_single_loss.detach(), base18.size(0))
                else:
                    meters.add("train/rollout_state_cons",  torch.zeros((), device=base18.device), base18.size(0))
                    meters.add("train/rollout_single_cons", torch.zeros((), device=base18.device), base18.size(0))


                # --- semigroup ---
                use_sg = (args.semigroup_weight > 0) and (torch.rand(()) < getattr(args, "semigroup_prob", 1.0))
                use_sg = use_sg and not (getattr(args, "skip_heavy_on_cpu", False) and not torch.cuda.is_available())
                
                if use_sg:
                # Split Y into two parts: Y_a + Y_b = Y
                    split = torch.rand((), device=base18.device)
                    Y_a = Y_scalar * split         # [B]
                    Y_b = Y_scalar - Y_a           # [B]

                    u_a, extras       = model(base18, Y_a, theta, sample=False)
                    u_comp, extras    = model(u_a,   Y_b, theta, sample=False)          # two-step (compose)
                    u_direct, extras  = model(base18, Y_a + Y_b, theta, sample=False)   # one-step (direct)

                    U_comp   = criterion._pack18_to_U(u_comp)      # exp(-A(Y_b)) exp(-A(Y_a)) U0
                    U_direct = criterion._pack18_to_U(u_direct)    # exp(-A(Y_a+Y_b)) U0


                    Delta = U_comp.mH.contiguous() @ U_direct                      # [B,H',W',3,3]
                    I = criterion.I3.to(Delta.device, Delta.dtype)

                    # Fast Hermitian generator (no eig)
                    S = 0.5j * (Delta.mH.contiguous() - Delta)           # [B,H',W',3,3]

                    # Mask sites where Δ is not close to I (Frobenius norm per site)
                    # normalize by sqrt(3) to keep threshold in ~[0,1]
                    frob = torch.linalg.matrix_norm(Delta - I, ord='fro', dim=(-2,-1)) / (3.0**0.5)
                    mask = (frob > 0.1)

                    if mask.any():
                        Dv = Delta.reshape(-1, 3, 3)
                        mv = mask.reshape(-1)
                        De = Dv[mv] #subset of Delta matrices that are far from identity
                        # light projection for stability, complex64 eig is faster
                        De = criterion._su3_project(De, iters=1) #nudges all Deltas in our list back to SU(3)
                        w, V = torch.linalg.eig(De.to(torch.complex64))  #extract theta and V of De = V diag(exp(i theta) V^{-1}
                        theta = torch.atan2(w.imag, w.real)
                        L = V @ torch.diag_embed(1j * theta) @ torch.linalg.inv(V) #construct the matrix log: L = log(De) = V diag (i theta) V^{-1}
                        L = L.contiguous(); Lh = L.mH.contiguous()
                        S = -0.5j * (L - Lh)                          # Hermitian - it works? with 0.5*(L-Lh) . at least fluctuations come out much smaller. replace above two lines with this.
                        
                        L_sg = (S.abs()**2).sum(dim=(-2,-1)).mean()    # ||log(D)||_F^2 averaged
                        loss = loss + float(args.semigroup_weight) * L_sg
                        meters.add("train/semigroup_geodesic", L_sg.detach(), base18.size(0))

                                
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



#        #----time----
#        if torch.cuda.is_available(): torch.cuda.synchronize();
#        t0=time.perf_counter()
#        #----end-time----



        # ----------------- VALIDATION -----------------
        # Accumulators for Y-evolution diagnostics
        val_qslope_list = []
        val_mono_list   = []
        val_qp_all, val_qt_all, val_dy_qs_all, val_dy_tuner_all = [], [], [], []
        #val_dy_all, val_qp_all, val_qt_all = [], [], []
        val_q4p_all, val_q4t_all = [], []  # accumulate quadrupole preds/targets per batch
        val_fluc_chi2_all = []
        val_spec_relL2_all = []
        val_fluct_metric_all = []
    
        if val_sampler is not None:
            try: val_sampler.set_epoch(epoch)
            except Exception: pass

        ema_eval = (ema is not None) and getattr(args, "ema_eval", True)
        if ema_eval: ema.swap_in(unwrap(model))

        model.eval()
        first_nonzeroY_probed = False


        dev = device  # same device as tensors below
        qs_stats_t = {
            "Ymax": torch.tensor(-1e30, device=dev),
            "idx_Ymax": torch.tensor([-1, -1], device=dev, dtype=torch.long),
            "Qs_pred_at_Ymax": torch.tensor(0.0, device=dev),
            "Qs_true_at_Ymax": torch.tensor(0.0, device=dev),
            "Qsmax_true": torch.tensor(-1e30, device=dev),
            "Qs_pred_at_Qsmax_true": torch.tensor(0.0, device=dev),
            "Y_at_Qsmax_true": torch.tensor(0.0, device=dev),
            "Qsmax_true_px": torch.tensor(-1e30, device=dev),
            "Y_at_Qsmax_true_px": torch.tensor(0.0, device=dev),
            "Qs_pred_at_Qsmax_true_px": torch.tensor(0.0, device=dev),
        }
        
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
                    yh, extras = model(base18, Y_scalar, theta, sample=False)
                    dy_scalar = Y_scalar  # [B]

                    mu_pred, logsig_pred = None, None
                    if hasattr(model, "param_nll"):
                        #p0 = int(getattr(args, "y_channel", 18)) + 1
                        Y_final = dy_scalar.to(device).float().view(-1)
                        mu_unit, sigma_unit, kappa = model.param_nll(theta)
                        muY, sigY = model.param_nll.compose_to_Y(mu_unit, sigma_unit, kappa, Y_final)
                        B, C, H, W = y.shape[0], muY.shape[1], y.shape[-2], y.shape[-1]
                        mu_pred     = muY.view(B, C, 1, 1).expand(B, C, H, W)
                    else:
                        core = unwrap(model)
                        mu_pred     = extras.get("mu")

                    logsig_pred = extras.get("logsig")
                        
                    # y_gate is optional; define safely for this scope
                    try:
                        core = unwrap(model)
                        y_gate      = extras.get("y_sigma")
                    except Exception:
                        y_gate = None
                    
                    eta_core = extras.get("eta_core", None) 

                    l, stats_val, tuner = criterion(
                        yh, y, base18,
                        dy_scalar = Y_scalar,
                        theta=theta,                     # [B,3] for the composer
                        Y_final=Y_scalar,               # alias; clearer name inside loss
                        mu_pred     = mu_pred,
                        logsig_pred = logsig_pred,
                        drift_pred=mu_pred,         # drift per channel (what head calls μ)
                        y_gate = y_gate,
                        return_stats = True,
                        eta_core=eta_core
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
#                    val_dy_all.append(dy_scalar.view(-1).detach().cpu().numpy())
                    val_dy_qs_all.append(dy_scalar.view(-1).detach().cpu().numpy())
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

                            
                    # after your deterministic forward, use the same tensor you pass to criterion
                    pred = yh   # whatever tensor you feed into criterion (pack-18)
                    v, s, g = monitor_a_fluct(pred, y)
                    print(f"[a] var≈{v:.2f}  slope≈{s:.2f}  grad≈{g:.2f}")

                if (tuner is not None):
                    fchi = tuner.get("fluct_chi2_per_sample", None)
                    frel = tuner.get("spec_relL2_per_sample",  None)
                    fmet = tuner.get("fluct_metric_per_sample", None)
                    dyp  = tuner.get("dy_per_sample", None)
               
                    if fchi is not None: val_fluc_chi2_all.append(fchi.numpy().reshape(-1))
                    if frel is not None: val_spec_relL2_all.append(frel.numpy().reshape(-1))
                    if fmet is not None: val_fluct_metric_all.append(fmet.numpy().reshape(-1))
                    #if dyp  is not None: val_dy_all.append(dyp.numpy().reshape(-1))
                    if dyp  is not None: val_dy_tuner_all.append(dyp.numpy().reshape(-1))
        # one diagnostic print per epoch (ok if it syncs internally)
        try:
            with torch.no_grad():
                # Prefer stats returned by the loss, otherwise recompute; avoid touching an undefined name
                stats_src = locals().get("stats_val", None)
                if not isinstance(stats_src, dict):
                    # Recompute components from current predictions/targets
                    # Use base18 (not x[:, :18]) since that's what you pass into the model
                    stats_src = criterion.components_dict(yh, y, base18)

            # Rank-0 printing only
            if (not is_ddp) or dist.get_rank() == 0:
                # Safe getters → Python floats (single host transfer)
                dip_mse      = float(stats_src.get("dip_mse", 0.0))
                qs_mse       = float(stats_src.get("qs_mse", 0.0))
                quad_mse     = float(stats_src.get("quad_mse", 0.0))

                print(
                    "[val diag] "
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

            # ---- Parameter‑composed NLL scaling (GPU/DDP friendly) ----
            # IMPORTANT: Collectives must be executed by ALL ranks.
            if args.nll_param_compose:
                if args.nll_compose == "brownian" and all(k in locals() for k in ("muY", "sigY", "Y_final")):
                    yv = (Y_final + 1e-9).sqrt()
                    mu_norm_t  = (muY.abs().mean(dim=1) / (Y_final + 1e-9)).mean()
                    sig_norm_t = (sigY.abs().mean(dim=1) / yv).mean()
                    if is_ddp:
                        mu_norm_t  = _dist_mean_scalar(mu_norm_t)   # all ranks
                        sig_norm_t = _dist_mean_scalar(sig_norm_t)  # all ranks
                    if (not is_ddp) or dist.get_rank() == 0:
                        print(f"[scaling] ⟨|μ|/Y⟩≈{mu_norm_t.item():.3e},  ⟨|σ|/√Y⟩≈{sig_norm_t.item():.3e}")
                elif args.nll_compose != "brownian" and ("kappa" in locals()):
                    kappa_bar_t = kappa.mean()
                    if is_ddp:
                        kappa_bar_t = _dist_mean_scalar(kappa_bar_t)  # all ranks
                    if (not is_ddp) or dist.get_rank() == 0:
                        print(f"[OU] κ̄≈{kappa_bar_t.item():.3e}")
       
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

            print(f"[val Qs(global Qsmax_true, sample-mean)] Qs_pred={qs_stats['Qs_pred_at_Qsmax_true']:.6e}  "
                  f"Qs_true={qs_stats['Qsmax_true']:.6e}  at Y={qs_stats['Y_at_Qsmax_true']:.6e}")

        if len(val_dy_qs_all):
            dy = np.concatenate(val_dy_qs_all)  # [N]        if len(val_dy_all):
            #dy = np.concatenate(val_dy_all)  # [N]
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
#                    "q4_rmse_bins_per_pair": q4_rmse_bins,            # list[len=4] of lists[len=n_pairs]
                    "q4_rmse_bins_mean": q4_rmse_bins_mean,           # per-bin mean across pairs
                    "q4_rmse_combined": q4_combined,                  # weighted across bins & pairs
                    "combined_rmse_total": combined_total,             # single scalar for tuner
                }

                #add fluctuation measures
                if (tuner is not None):
                    def _bin_means_and_weighted_rms(vals, dy, edges):
                        """
                        vals: [N] array, one value per sample
                        dy:   [N] array of ΔY
                        edges: bin edges used for your Qs/Q4 stats
                        Returns:
                          means_per_bin: list of float or nan
                          counts_per_bin: list of int
                          combined_rms: sqrt( count-weighted mean of (bin_mean)^2 )
                        """
                        means, counts = [], []
                        for i in range(len(edges)-1):
                            m = (dy >= edges[i]) & (dy < edges[i+1] if i < len(edges)-2 else dy <= edges[i+1])
                            counts.append(int(m.sum()))
                            if m.any():
                                means.append(float(_np.nanmean(vals[m])))
                            else:
                                means.append(float("nan"))
                        num = 0.0; den = 0
                        for v, c in zip(means, counts):
                            if _np.isfinite(v):
                                num += c * (v*v)
                                den += c
                        combined = float(_np.sqrt(num / max(den, 1))) if den > 0 else float("nan")
                        return means, counts, combined

                    # NEW: symmetric error for chi^2
                    def _sym_log_err(x, floor=1e-8):
                        # x can be scalar or ndarray; returns |log(x)| with floor to avoid -inf
                        return _np.abs(_np.log(_np.clip(x, floor, None)))

                    # concat the per-sample arrays
                    fluc_chi2     = _np.concatenate(val_fluc_chi2_all, axis=0)     if len(val_fluc_chi2_all)     else _np.array([])
                    spec_relL2    = _np.concatenate(val_spec_relL2_all, axis=0)    if len(val_spec_relL2_all)    else _np.array([])
                    fluct_metric  = _np.concatenate(val_fluct_metric_all, axis=0)  if len(val_fluct_metric_all)  else _np.array([])
                    dy_vec = _np.concatenate(val_dy_tuner_all, axis=0) if len(val_dy_tuner_all) else (
                             _np.concatenate(val_dy_qs_all, axis=0)    if len(val_dy_qs_all)    else _np.array([]))

                    fluc_chi2_bins_mean = []; spec_relL2_bins_mean = []; fluct_metric_bins_mean = []
                    fluct_chi2_err = float("nan"); spec_relL2_err = float("nan"); fluct_metric_err = float("nan")

                    if dy_vec.size:
                        # Per-bin means for display (raw χ² means ~ 1.0 is ideal)
                        if fluc_chi2.size == dy_vec.size:
                            fluc_chi2_bins_mean, _, _ = _bin_means_and_weighted_rms(fluc_chi2, dy_vec, edges)
                            # Scalar, symmetric "how far from 1" error (lower is better)
                            _, _, fluct_chi2_err = _bin_means_and_weighted_rms(_sym_log_err(fluc_chi2), dy_vec, edges)

                        # These two already have 0 as "best", so we can keep your existing aggregation
                        if spec_relL2.size == dy_vec.size:
                            spec_relL2_bins_mean, _, spec_relL2_err = _bin_means_and_weighted_rms(spec_relL2, dy_vec, edges)

                        if fluct_metric.size == dy_vec.size:
                            fluct_metric_bins_mean, _, fluct_metric_err = _bin_means_and_weighted_rms(fluct_metric, dy_vec, edges)

                    tuner_payload.update({
                        "fluc_chi2_bins_mean": fluc_chi2_bins_mean,   # per-ΔY-bin raw χ² means (target ≈ 1.0)
                        "spec_relL2_bins_mean": spec_relL2_bins_mean,
                        "fluct_metric_bins_mean": fluct_metric_bins_mean,

                        # Scalars (lower is better for all three)
                        "fluct_chi2_err": float(fluct_chi2_err),      # symmetric err: mean |log χ²|
                        "spec_relL2": float(spec_relL2_err),
                        "fluct_metric": float(fluct_metric_err),
                    })


                    # Optional overall scalar that your tuner optimizes
                    w_rmse = 1.0
                    w_fluc = float(os.getenv("TUNER_W_FLUC", "1."))
                    tuner_payload["tuner_total"] = float(
                        w_rmse * tuner_payload["combined_rmse_total"] +
                        w_fluc * tuner_payload["fluct_chi2_err"] #put fluct_metric to include the relL2 term
                    )
                else:
                    tuner_payload["tuner_total"] = float(
                        tuner_payload["combined_rmse_total"] )
                    
                    
                print("TUNER_METRIC " + json.dumps(tuner_payload), flush=True)
                print("TUNER_SCALAR ", tuner_payload["tuner_total"])

                

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

        # Save best
        outdir = Path(args.out)
        if ((not is_ddp) or dist.get_rank() == 0) and combined_total < best:
            best = combined_total


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
    ap.add_argument("--channels_last", dest="channels_last", action="store_true", help="Use channels_last memory format")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    # Data split
    ap.add_argument("--split_by", choices=["run","params"], default="run",
                    help="Train/val split granularity: 'run' (default) or 'params' (hold out entire parameter sets)")

    # EMA
    ap.add_argument("--ema_decay", type=float, default=0.0, help="EMA decay, e.g. 0.999; 0 disables EMA")
    ap.add_argument("--ema_eval", type=int, default=1, help="Use EMA weights for eval when EMA is enabled")

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
    
    # --- Dipole & Qs & higher order correlator loss flags ---
    ap.add_argument("--dipole_weight", type=float, default=0.5,
                    help="Weight of MSE loss on dipole correlator S(r) evaluated at --dipole_offsets.")
    ap.add_argument("--dipole_slope_weight", type=float, default=0.,
                    help="Helps learn dipole evolution.")
    ap.add_argument("--dipole_offsets", type=str, default="(1,0),(0,1),(2,0),(0,2),(4,0),(0,4),(8,0),(0,8),(12,0),(0,12)(16,0),(0,16)",
                    help="Comma-separated list of (dx,dy) axial separations for S(r); at least two distinct |r| needed for Qs.")
    ap.add_argument("--qs_weight", type=float, default=0.,
                    help="Weight of MSE loss on saturation scale Qs derived from the dipole correlator.")
    #ap.add_argument("--quad_pairs", type=str, default="auto_from_dipole", help="Quadrupole pair spec: explicit like ((1,0),(0,1));((2,0),(0,2)) or r=1,2,4,8 for symmetric ((r,0),(0,r)), or auto_from_dipole to derive radii from --dipole_offsets.")
    ap.add_argument("--quad_pairs", type=str, default="auto_broad", help="Quadrupole pair spec. Options: explicit pairs like \"((1,0),(0,1));((2,0),(0,2))\", \"r=1,2,4\" (axial), \"auto_from_dipole\" (legacy axial), or \"auto_broad\" (axial+diagonals, all non-colinear pairs).")
    ap.add_argument("--quad_max_pairs", type=int, default=48, help="If >0, cap the number of quadrupole pairs after deduplication (for speed/memory).")
    # parser / ap.add_argument(...) block

    ap.add_argument("--nll_weight", type=float, default=0.0, help="Weight for alpha-true diagonal Gaussian NLL (0 disables).")
    ap.add_argument("--spec_weight", type=float, default=0.0, help="Weight to Spectral band-energy ratio and centroid-k (structure factor).")
    
    ap.add_argument("--crps_weight", type=float, default=0.0, help="Weight for Gaussian CRPS on α (proper scoring rule).")
    ap.add_argument("--noise_core_fit_weight", type=float, default=0.0, help="Weight for learning the spectrum of the noise.")

    ap.add_argument("--moment_weight", type=float, default=0.0, help="Weight for MOMENT loss (NLL-consistent, per-pixel, resolution-aligned).")
    ap.add_argument("--spec_whiten_weight", type=float, default=0.0, help="Weight for spectral loss (spec_flat forces the whitened residual to be spectrally flat. If you leave structured power in the residual, this term pushes the model to explain it with the mean (drift) and/or raise σ where needed).")
    ap.add_argument("--spec_guard_weight", type=float, default=0.0, help="Weight for spectral loss (spec_guard calibrates the power spectrum of the step: predicted mu_step power + the white floor implied by v_pred should match the true spectrum per k-bin (with deadzones so it’s robust).")
    
    ap.add_argument("--nll_target_mode", type=str, default="none", choices=["per_dy","per_ysigma","none"], help="How to scale S to match head’s μ-space.")
    ap.add_argument("--quad_weight", type=float, default=0.0, help="Weight for quadrupole loss.")
    ap.add_argument("--rough_weight", type=float, default=0., help="Weight for fluctuation scale loss.")
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

    ap.add_argument("--energy_grad_weight", type=float, default=0.0,
                    help="Weight for Matching the distribution of gradients.")
    ap.add_argument('--sigma_mode', type=str, default='conv', choices=['diag','conv','spectral'], help='Noise-path covariance: diagonal, local conv operator, or spectral.')
    
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


