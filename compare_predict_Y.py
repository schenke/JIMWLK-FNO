#!/usr/bin/env python3
# Predict U(Y) and compare to ground truth (trace maps + dipole curves).
# Safe ckpt loading (PyTorch 2.6), signature-safe model build, and hardened
# post-processing of predictions (finite replace + adaptive clip + SVD polar).

import os, json, math, argparse, pathlib, inspect, cmath
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Must match your training run
OFFSETS = [(1,0),(0,1),(2,0),(0,2),(4,0),(0,4),(8,0),(0,8),(12,0),(0,12),(16,0),(0,16)]
QS_THRESHOLD = 0.5
YCH = 18  # <- set to your Y channel index in the model input


# -------- Fourier spectrum utilities --------
def scalar_field_from_U(U: np.ndarray, mode: str = "ReTrU") -> np.ndarray:
    """
    Extract scalar field from U[N,N,3,3] for spectra.
    mode in {"ReTrU","ImTrU","AbsTrU","ArgTrU"}.
    Returns [N,N].
    """
    tr = np.trace(U, axis1=-2, axis2=-1) / 3.0  # normalize by Nc
    if mode == "ReTrU":
        return tr.real
    elif mode == "ImTrU":
        return tr.imag
    elif mode == "AbsTrU":
        return np.abs(tr)
    elif mode == "ArgTrU":
        return np.angle(tr)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def radial_power_spectrum(field: np.ndarray, n_bins: int = 64, detrend: bool = True):
    """
    Isotropic (radially-averaged) 2D power spectrum of a real 2D field.
    Returns (k_centers, P_k) with k in cycles/pixel from FFT frequencies.
    """
    f = np.asarray(field, dtype=np.float64)
    if detrend:
        f = f - np.nanmean(f)

    F = np.fft.fft2(f)
    P = np.abs(F)**2 / f.size
    P = np.fft.fftshift(P)

    N = f.shape[0]
    assert f.shape[0] == f.shape[1], "Only square grids supported."
    freqs = np.fft.fftfreq(N, d=1.0)
    freqs = np.fft.fftshift(freqs)
    kx, ky = np.meshgrid(freqs, freqs, indexing="ij")
    kr = np.sqrt(kx**2 + ky**2)

    # bin by |k|
    kmax = float(kr.max())
    bins = np.linspace(0.0, kmax, int(n_bins) + 1)
    which = np.digitize(kr.ravel(), bins) - 1

    P_sum = np.zeros(int(n_bins), dtype=np.float64)
    P_cnt = np.zeros(int(n_bins), dtype=np.int64)
    mask = (which >= 0) & (which < int(n_bins))
    np.add.at(P_sum, which[mask], P.ravel()[mask])
    np.add.at(P_cnt, which[mask], 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        P_avg = np.where(P_cnt > 0, P_sum / P_cnt, np.nan)
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    return k_centers, P_avg

def spectra_from_U(U_true: np.ndarray, U_pred: np.ndarray, mode: str = "ReTrU", n_bins: int = 64):
    f_true = scalar_field_from_U(U_true, mode=mode)
    f_pred = scalar_field_from_U(U_pred, mode=mode)
    k, P_true = radial_power_spectrum(f_true, n_bins=n_bins)
    _, P_pred = radial_power_spectrum(f_pred, n_bins=n_bins)
    return k, P_true, P_pred

def plot_spectrum(k, P_true, P_pred, out_png: Path, title: str = ""):
    import matplotlib.pyplot as plt
    k = np.asarray(k); t = np.asarray(P_true); p = np.asarray(P_pred)
    m = np.isfinite(k) & np.isfinite(t) & np.isfinite(p) & (k > 0)
    k, t, p = k[m], t[m], p[m]

    fig = plt.figure(figsize=(6.4, 4.8), dpi=140)
    ax = fig.add_subplot(1,1,1)
    ax.loglog(k, t, label="truth")
    ax.loglog(k, p, label="prediction")
    ax.set_xlabel("|k| (cycles/pixel)")
    ax.set_ylabel("radial power spectrum")
    if title: ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_spectrum_ratio(k, P_true, P_pred, out_png: Path, title: str = ""):
    import matplotlib.pyplot as plt
    k = np.asarray(k); t = np.asarray(P_true); p = np.asarray(P_pred)
    m = np.isfinite(k) & np.isfinite(t) & np.isfinite(p) & (k > 0)
    k, t, p = k[m], t[m], p[m]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(t != 0, p / t, np.nan)

    fig = plt.figure(figsize=(6.4, 3.8), dpi=140)
    ax = fig.add_subplot(1,1,1)
    ax.semilogx(k, r)
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("|k| (cycles/pixel)")
    ax.set_ylabel("pred / truth")
    if title: ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --- Helper to interop with the new EvolverFNO API (base18, Y_scalar, theta) ---
def split_legacy_x(x: torch.Tensor):
    """
    Convert an old-style tensor x=[B,22,H,W] (base18 + Y map + 3 param maps)
    into (base18[B,18,H,W], Y_scalar[B], theta[B,3]).
    This mirrors train_evolver's `collate_base18_Y_theta`.
    """
    base18   = x[:, :18, ...]
    Y_scalar = x[:, 18, ...].mean(dim=(1, 2))      # spatially constant → scalar
    theta    = x[:, 19:22, 0, 0]                   # pick any pixel
    return base18, Y_scalar, theta


def _natsort_key(s):
    parts = re.findall(r'\d+|\D+', str(s))
    return [int(p) if p.isdigit() else p for p in parts]

def frob_mean(Ua: torch.Tensor, Ub: torch.Tensor) -> float:
    """Mean Frobenius distance per site between two SU(3) fields.
       Expects Ua, Ub with trailing (3,3) matrix dims, e.g. [B,H,W,3,3]."""
    return (Ua - Ub).norm(dim=(-1, -2)).mean().item()

def dipole_axial_torch(U: torch.Tensor, offsets=OFFSETS) -> tuple[torch.Tensor, torch.Tensor]:
    """Axial dipole S(r) like training. U: [B,H,W,3,3] complex64/complex128."""
    B, H, W = U.shape[:3]
    S_vals, r_vals = [], []
    # roll on spatial axes (1,2)
    for dx, dy in offsets:
        Ur = torch.roll(torch.roll(U, shifts=dy, dims=1), shifts=dx, dims=2)
        # prod: U(x) * U^\dagger(x+r)
        prod = U @ Ur.conj().transpose(-1, -2)
        tr = torch.diagonal(prod, dim1=-2, dim2=-1).sum(-1).real / 3.0  # [B,H,W]
        S_vals.append(tr.mean(dim=(1,2)))  # [B]
        r_vals.append(np.hypot(dx, dy))
    r = torch.tensor(r_vals, dtype=torch.float32, device=U.device)
    S = torch.stack(S_vals, dim=-1)  # [B, #offsets]
    # sort by r
    idx = torch.argsort(r)
    return r[idx], S.index_select(-1, idx)

def qs_from_axial_torch(U: torch.Tensor, thr=QS_THRESHOLD) -> torch.Tensor:
    """
    Q_s per item using axial samples and N(r)=1-S(r) @ threshold=thr.
    Returns: [B] tensor of Q_s.
    Robust to:
      - crossing left of the first sample (r★ < r[0])
      - no crossing within grid (right extrapolation)
      - plateaus after cummax (tiny-slope guarding)
    """
    # r:[K], S:[B,K]; ensure float dtypes and device alignment
    r, S = dipole_axial_torch(U)
    S = S.to(dtype=torch.float32)
    r = r.to(device=S.device, dtype=S.dtype)

    # N(r) and monotone repair
    N = (1.0 - S).clamp_(0.0, 1.0)          # [B,K]
    Nmono = torch.cummax(N, dim=-1).values  # non-decreasing in r

    B, K = Nmono.shape
    assert K >= 2, "Need at least two axial samples to interpolate"

    # Crossing mask and first crossing index (0 if none)
    crossing = (Nmono >= thr)
    any_cross = crossing.any(dim=-1)                 # [B] bool
    j_first = crossing.float().argmax(dim=-1).long() # [B], 0 if none

    # --------- Build indices for each regime ----------
    # Interior: first crossing at j > 0 → use segment (j-1, j)
    j_int = j_first.clamp(min=1)     # ensure ≥1 so (j-1) is valid
    j_prev = j_int - 1               # [B]

    # Left-of-grid: crossing flagged but j_first==0 → use segment (0,1)
    left_mask     = any_cross & (j_first == 0)
    interior_mask = any_cross & (j_first > 0)
    right_mask    = ~any_cross

    # For left-of-grid, force (j_prev, j_int) = (0, 1)
    j_prev = torch.where(left_mask, torch.zeros_like(j_prev), j_prev)
    j_int  = torch.where(left_mask, torch.ones_like(j_int), j_int)

    # For right-of-grid, we will use the last segment (K-2, K-1) below.

    # Gather helper
    def g(a, idx):
        return a.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    # Expand r to [B,K] for gather
    rB = r.expand(B, -1)

    # ---- Interior (also covers left-of-grid via forced indices (0,1)) ----
    n0 = g(Nmono, j_prev)   # [B]
    n1 = g(Nmono, j_int)    # [B]
    r0 = g(rB,    j_prev)   # [B]
    r1 = g(rB,    j_int)    # [B]

    # Linear interpolation with small-slope guard
    denom = (n1 - n0)
    eps = torch.finfo(S.dtype).eps
    denom_safe = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
    t_int = (thr - n0) / denom_safe
    # No need to clamp t for extrapolation; left/right handled by masks
    r_star_int = r0 + t_int * (r1 - r0)

    # ---- Right-of-grid (no crossing in-range): extrapolate from last segment) ----
    jL0 = torch.full((B,), K-2, device=S.device, dtype=torch.long)
    jL1 = torch.full((B,), K-1, device=S.device, dtype=torch.long)
    nL0, nL1 = g(Nmono, jL0), g(Nmono, jL1)
    rL0, rL1 = g(rB,    jL0), g(rB,    jL1)

    dN_last = (nL1 - nL0)
    dN_last_safe = torch.where(dN_last.abs() < eps, torch.full_like(dN_last, eps), dN_last)
    # r★ = rL1 + (thr - nL1) * (Δr / ΔN)  with Δr=(rL1-rL0), ΔN=(nL1-nL0)
    r_star_right = rL1 + (thr - nL1) * (rL1 - rL0) / dN_last_safe

    # ---- Stitch the three regimes together ----
    r_star = torch.empty(B, dtype=S.dtype, device=S.device)
    r_star[interior_mask] = r_star_int[interior_mask]
    r_star[left_mask]     = r_star_int[left_mask]      # left uses (0,1) segment
    r_star[right_mask]    = r_star_right[right_mask]   # right extrapolation

    # Safety: positive radii (allow very small to avoid div-by-zero)
    r_star = r_star.clamp(min=1e-8)

    return 1.0 / r_star


# def qs_from_axial_torch(U: torch.Tensor, thr=QS_THRESHOLD) -> torch.Tensor:
#     """Qs per item using axial samples and N(r)=1-S(r) @ threshold=0.5. Returns [B]."""
#     r, S = dipole_axial_torch(U)            # r:[K], S:[B,K]
#     N = (1.0 - S).clamp(min=0.0, max=1.0)   # [B,K]
#     # make monotone in r
#     #Nmono = torch.maximum.accumulate(N, dim=-1)
#     Nmono = torch.cummax(N, dim=-1).values
#     # for each batch item, find first index where N>=thr
#     B, K = Nmono.shape
#     j = (Nmono >= thr).float().argmax(dim=-1)  # [B], picks 0 if never meets; handle below
#     j = torch.clamp(j, 1, K-1)

#     # linear interp between (j-1) and j
#     j0 = (j-1).unsqueeze(-1)
#     j1 = j.unsqueeze(-1)
#     gather = lambda A, jj: A.gather(-1, jj).squeeze(-1)
#     n0, n1 = gather(Nmono, j0), gather(Nmono, j1)
#     r0, r1 = gather(r.expand(B, -1), j0), gather(r.expand(B, -1), j1)
#     denom = (n1 - n0).clamp(min=1e-12)
#     alpha = (thr - n0) / denom
#     r_star = r0 + alpha * (r1 - r0)
#     return 1.0 / r_star.clamp(min=1e-8)     # qs_scale=1.0

# # --- Try to import model + IO helpers from your training script(s)
EvolverFNO = None
read_wilson_binary = None
for modname in ("train_evolver_cuda_opt",""):
    try:
        mod = __import__(modname, fromlist=["EvolverFNO", "read_wilson_binary"])
        EvolverFNO = getattr(mod, "EvolverFNO")
        read_wilson_binary = getattr(mod, "read_wilson_binary")
        break
    except Exception:
        continue
if EvolverFNO is None or read_wilson_binary is None:
    raise ImportError("Could not import EvolverFNO/read_wilson_binary from training module.")


def steps_to_Y(steps: int, ds: float) -> float:
    return (math.pi**2) * ds * steps

def pick_target_snapshot(manifest, ds, target_steps=None, target_y=None, index=None):
    snaps = sorted(manifest["snapshots"], key=lambda s: int(s["steps"]))
    assert snaps, "No snapshots listed in manifest."
    if target_steps is not None:
        steps = int(target_steps)
        for s in snaps:
            if int(s["steps"]) == steps:
                return s, steps_to_Y(steps, ds)
        raise ValueError(f"No snapshot with steps={steps} in manifest.")
    if target_y is not None:
        y_target = float(target_y)
        best = min(snaps, key=lambda s: abs(steps_to_Y(int(s["steps"]), ds) - y_target))
        return best, steps_to_Y(int(best["steps"]), ds)
    if index is not None:
        idx = int(index)
        if not (0 <= idx < len(snaps)):
            raise ValueError(f"index {idx} out of range [0,{len(snaps)-1}]")
        s = snaps[idx]
        return s, steps_to_Y(int(s["steps"]), ds)
        s = snaps[-1]
    return s, steps_to_Y(int(s["steps"]), ds)


def _build_evolver_from_ckpt_args(in_ctor, ckpt_args: dict):
    # Allowed constructor keys discovered from the actual signature
    sig = inspect.signature(in_ctor)
    allowed = set(sig.parameters.keys())

    # Map historical arg names from training to current constructor names
    rename = {
        "y_channel": "y_index",
        "blocks": "n_blocks",
        "modes": "modes1",   # if only one modes was stored, use it for both
    }

    # Start with an empty dict and selectively copy args that exist in ckpt_args
    kwargs = {}
    for k_ckpt, v in ckpt_args.items():
        k = rename.get(k_ckpt, k_ckpt)
        if k in allowed:
            kwargs[k] = v

    # If only one modes value is present, mirror it
    if "modes1" in kwargs and "modes2" not in kwargs:
        kwargs["modes2"] = kwargs["modes1"]

    # Ensure required keys that might be missing
    # (use checkpoint values if present; otherwise use safe defaults)
    defaults = {
        "in_ch": 22,
        "width": 64,
        "modes1": ckpt_args.get("modes", 16),
        "modes2": ckpt_args.get("modes", 16),
        "n_blocks": ckpt_args.get("blocks", ckpt_args.get("n_blocks", 6)),
        # Time conditioning / head parameters
        "film_mode": "scale_only",
        "rbf_K": 12,
        "film_hidden": 64,
        "gamma_scale": 1.5,
        "beta_scale": 1.,
        "gate_temp": 1.0,          # keep training default if absent
        "y_gain": 1.0,
        "identity_eps": 0.0,
        "alpha_scale": 3.0,
        "clamp_alphas": None,
        "alpha_vec_cap": 15.0,
        # propagate Y normalization boundaries
        "y_min": ckpt_args.get("y_min", 0.0),
        "y_max": ckpt_args.get("y_max", 1.0),
    }
    for k, v in defaults.items():
        if k in allowed and k not in kwargs:
            kwargs[k] = v

    return kwargs


# def _build_evolver_from_ckpt_args(in_ctor, ckpt_args: dict):
#     sig = inspect.signature(in_ctor)
#     allowed = set(sig.parameters.keys())
#     defaults = {"in_ch": 22, "width": 64, "modes": 16, "blocks": 6}

#     in_ch  = ckpt_args.get("in_ch", defaults["in_ch"])
#     width  = ckpt_args.get("width", defaults["width"])
#     modes  = ckpt_args.get("modes", ckpt_args.get("modes1", defaults["modes"]))
#     modes1 = ckpt_args.get("modes1", modes)
#     modes2 = ckpt_args.get("modes2", modes)
#     blocks = ckpt_args.get("blocks", ckpt_args.get("n_blocks", defaults["blocks"]))
#     proj_iter = ckpt_args.get("proj_iter", None)

#     print(width, blocks, modes)
    
#     candidates = {
#         "in_ch": in_ch, "width": width,
#         "modes": modes, "modes1": modes1, "modes2": modes2,
#         "blocks": blocks, "n_blocks": blocks,
#         "proj_iter": proj_iter,
#     }

#     # >>> ADD THIS: hydrate the training-time conditioner/head knobs <<<
#     train_defaults = {
#         "y_index":     ckpt_args.get("y_channel", 18),
#         "film_mode":   ckpt_args.get("film_mode", "scale_only"),
#         "rbf_K":       ckpt_args.get("rbf_K", 12),
#         "film_hidden": ckpt_args.get("film_hidden", 64),
#         "gamma_scale": ckpt_args.get("gamma_scale", 1.5),
#         "beta_scale":  ckpt_args.get("beta_scale", 1.5),   # trainer default
#         "gate_temp":   ckpt_args.get("gate_temp", 1.0),    # trainer default
#         "y_gain":      ckpt_args.get("y_gain", 1.0),
#         "identity_eps":ckpt_args.get("identity_eps", 0.0),
#         "alpha_scale": ckpt_args.get("alpha_scale", 3.0),
#         "clamp_alphas":ckpt_args.get("clamp_alphas", 1.5), # trainer default
#         # optional if you want ctor to get these (you already fill the buffers later)
#         "y_min":       ckpt_args.get("y_min", None),
#         "y_max":       ckpt_args.get("y_max", None),
#     }
#     for k,v in train_defaults.items():
#         if (k in allowed) and (k not in candidates or candidates[k] is None):
#             candidates[k] = v
#     # <<< END ADD >>>

#     kwargs = {k: v for k, v in candidates.items() if (v is not None and k in allowed)}
#     if "in_ch" in allowed and "in_ch" not in kwargs:
#         kwargs["in_ch"] = defaults["in_ch"]
#     return kwargs



# # ---------- signature-safe builder for EvolverFNO ----------
# def _build_evolver_from_ckpt_args(in_ctor, ckpt_args: dict):
#     sig = inspect.signature(in_ctor)
#     allowed = set(sig.parameters.keys())
#     defaults = {"in_ch": 22, "width": 64, "modes": 16, "blocks": 6}
#     in_ch  = ckpt_args.get("in_ch", defaults["in_ch"])
#     width  = ckpt_args.get("width", defaults["width"])
#     modes  = ckpt_args.get("modes", ckpt_args.get("modes1", defaults["modes"]))
#     modes1 = ckpt_args.get("modes1", modes)
#     modes2 = ckpt_args.get("modes2", modes)
#     blocks = ckpt_args.get("blocks", ckpt_args.get("n_blocks", defaults["blocks"]))
#     proj_iter = ckpt_args.get("proj_iter", None)
#     candidates = {
#         "in_ch": in_ch, "width": width,
#         "modes": modes, "modes1": modes1, "modes2": modes2,
#         "blocks": blocks, "n_blocks": blocks,
#         "proj_iter": proj_iter,
#     }
#     kwargs = {k:v for k,v in candidates.items() if (v is not None and k in allowed)}
#     if "in_ch" in allowed and "in_ch" not in kwargs:
#         kwargs["in_ch"] = defaults["in_ch"]
#     return kwargs

#     defaults_train = {
#       "y_index":     ckpt_args.get("y_channel", 18),
#       "film_mode":   ckpt_args.get("film_mode", "scale_only"),
#       "rbf_K":       ckpt_args.get("rbf_K", 12),
#       "film_hidden": ckpt_args.get("film_hidden", 64),
#       "gamma_scale": ckpt_args.get("gamma_scale", 1.5),
#       "beta_scale":  ckpt_args.get("beta_scale", 1.5),   # trainer default
#       "gate_temp":   ckpt_args.get("gate_temp", 1.0),    # trainer default
#       "y_gain":      ckpt_args.get("y_gain", 1.0),
#       "identity_eps":ckpt_args.get("identity_eps", 0.0),
#       "alpha_scale": ckpt_args.get("alpha_scale", 3.0),
#       "clamp_alphas":ckpt_args.get("clamp_alphas", 1.5), # trainer default
#     }
#     for k,v in defaults_train.items():
#         if k in allowed and (k not in candidates or candidates[k] is None):
#             candidates[k] = v



def safe_load_model(ckpt_path: Path, device: str = "cpu"):
    import torch
    if "weights_only" in inspect.signature(torch.load).parameters:
        try:
            from torch.serialization import add_safe_globals
            add_safe_globals([pathlib.PosixPath])
        except Exception:
            pass
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location=device)

    kwargs = _build_evolver_from_ckpt_args(EvolverFNO.__init__, ckpt["args"])
    model = EvolverFNO(**kwargs).to(device)
    model.load_state_dict(ckpt["model"], strict=False) #model.load_state_dict(ckpt["model"])
    model.eval()

    
    # try:
    #     ident = getattr(getattr(model, "head", None), "identity_eps", None)
    #     ych   = getattr(model, "y_channel", None)
    #     print(f"[debug] identity_eps={ident}  y_channel={ych}")
    # except Exception:
    #     pass

    # try:
    #     head = getattr(model, "head", None)
    #     has_su3 = hasattr(head, "lambdas")
    #     print(f"[debug] model head: {type(head).__name__ if head is not None else None}  su3_head={has_su3}")
    # except Exception:
    #     pass

    return model, ckpt["args"]

# ---------- robust post-proc ----------
def finite_frac(Z):
    return (np.isfinite(Z.real) & np.isfinite(Z.imag)).mean()

def svd_polar_det1(U: np.ndarray) -> np.ndarray:
    """Unitary projection via SVD, then det=1. Works on complex64/128."""
    N = U.shape[0]
    X = U.reshape(-1,3,3).astype(np.complex128, copy=True)
    # replace non-finite with identity before SVD
    bad = ~np.isfinite(X.real) | ~np.isfinite(X.imag)
    if bad.any():
        mask = bad.reshape(X.shape[0], -1).any(axis=1)
        X[mask] = np.eye(3, dtype=np.complex128)
    Uu, s, Vh = np.linalg.svd(X, full_matrices=False)
    Uunit = Uu @ Vh
    det = np.linalg.det(Uunit)
    phase = np.exp(-1j * (np.angle(det) / 3.0))
    Uunit = Uunit * phase[:, None, None]
    return Uunit.reshape(N,N,3,3).astype(np.complex64, copy=False)

def y18_repair(y18: np.ndarray) -> np.ndarray:
    # only replace non-finite; do NOT shrink amplitudes
    y = y18.astype(np.float64, copy=True)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


# def y18_repair(y18: np.ndarray) -> np.ndarray:
#     """
#     1) replace non-finite with 0
#     2) adaptive clip by 99.9th percentile of |y|, capped to 10.0
#     """
#     y = y18.astype(np.float64, copy=True)
#     y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
#     q = np.percentile(np.abs(y), 99.9)
#     clip = float(np.clip(q, 1.0, 10.0))  # at least 1, at most 10
#     return np.clip(y, -clip, clip).astype(np.float32, copy=False)

# ---------- matrix-level metrics (finite-aware) ----------
def su3_metrics(U_pred: np.ndarray, U_true: np.ndarray):
    Up = U_pred.astype(np.complex128, copy=False)
    Ut = U_true.astype(np.complex128, copy=False)
    def fmean(x):
        x = np.asarray(x)
        m = np.isfinite(x)
        return float(np.mean(x[m])) if m.any() else float("nan")
    frob_mse = fmean(np.abs(Up - Ut)**2)
    tr_p = np.trace(Up, axis1=-2, axis2=-1) / 3.0
    tr_t = np.trace(Ut, axis1=-2, axis2=-1) / 3.0
    trace_mse_real = fmean((tr_p.real - tr_t.real)**2)
    trace_mse_imag = fmean((tr_p.imag - tr_t.imag)**2)
    I = np.eye(3, dtype=np.complex128)
    UU = np.matmul(np.conjugate(np.swapaxes(Up, -1, -2)), Up)
    d = np.abs(UU - I)
    d = np.clip(d, 0, 1e6)  # guard overflow in square
    unitarity_dev = fmean(d**2)
    return {"frob_mse": frob_mse, "trace_mse_real": trace_mse_real, "trace_mse_imag": trace_mse_imag, "unitarity_dev": unitarity_dev}

# def lattice_avg_trace_complex(U):
#     """Return complex scalar ⟨Tr U⟩ over all lattice sites (and any leading dims)."""
#     if torch.is_tensor(U):
#         tr = U.diagonal(dim1=-2, dim2=-1).sum(-1)  # ...,[H,W]
#         c = tr.mean().detach().cpu().numpy().astype(np.complex64).item()
#         return c
#     else:  # numpy
#         tr = np.trace(U, axis1=-2, axis2=-1)       # ...,[H,W]
#         c = tr.mean().astype(np.complex64).item()
#         return c

# def apply_z3_center(U, z):
#     """Multiply whole field by a Z3 center element z∈{1, e^{±2πi/3}}; preserves det=1."""
#     if torch.is_tensor(U):
#         # Broadcast a scalar complex to the whole tensor
#         zt = torch.tensor(z, dtype=U.dtype, device=U.device)
#         return U * zt
#     else:
#         return U * np.array(z, dtype=U.dtype)

# def z3_align_complex(c, ref_c):
#     """Pick z∈Z3 that best aligns complex number c to reference ref_c."""
#     zs = [1+0j, cmath.exp(2j*math.pi/3), cmath.exp(-2j*math.pi/3)]
#     # Maximize Re[(z*c) * conj(ref_c)]
#     scores = [((z*c) * np.conj(ref_c)).real for z in zs]
#     return zs[int(np.argmax(scores))]

# def align_prediction_to_truth(U_pred, U_truth):
#     """
#     Align the predicted SU(3) field to the truth by a global Z3 factor so that
#     ⟨Tr U_pred⟩ has the same center phase as ⟨Tr U_truth⟩.
#     Returns: U_pred_aligned, z_used (complex), c_pred_aligned (complex)
#     """
#     c_pred  = lattice_avg_trace_complex(U_pred)
#     c_truth = lattice_avg_trace_complex(U_truth)

#     # If truth’s average trace is ~0, alignment is ill-conditioned → skip
#     if abs(c_truth) < 1e-10:
#         return U_pred, 1+0j, c_pred

#     z = z3_align_complex(c_pred, c_truth)
#     U_aligned = apply_z3_center(U_pred, z)
#     c_pred_aligned = z * c_pred
#     return U_aligned, z, c_pred_aligned


def plot_trace_maps(U_pred: np.ndarray, U_true: np.ndarray, out_png: Path, title="",  gauge_side: str = "auto"):

#    U_pred_aligned = _align_global(U_true.astype(np.complex128), U_pred.astype(np.complex128), side=gauge_side)    
    U_pred_aligned = _align_global_conjugation(U_true, U_pred)

    tr_p = np.trace(U_pred_aligned, axis1=-2, axis2=-1) / 3.0
    tr_t = np.trace(U_true, axis1=-2, axis2=-1) / 3.0
    
    #tr_p1 = U_pred_aligned.diagonal(axis1=-2, axis2=-1).sum(-1)   # [H, W]
    #tr_p = np.abs(tr_p1) / 3.0                    # normalize to ~[0,1]
    #tr_t1 = U_true.diagonal(axis1=-2, axis2=-1).sum(-1)   # [H, W]
    #tr_t = np.abs(tr_t1) / 3.0                    # normalize to ~[0,1]
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=140, constrained_layout=True)
    im0 = axs[0].imshow(tr_t.real); axs[0].set_title("GT  |Tr(U)|/3"); axs[0].axis("off"); fig.colorbar(im0, ax=axs[0], fraction=0.046)
    im1 = axs[1].imshow(tr_p.real); axs[1].set_title("Pred |Tr(U)|/3"); axs[1].axis("off"); fig.colorbar(im1, ax=axs[1], fraction=0.046)
    im2 = axs[2].imshow((tr_p.real - tr_t.real)); axs[2].set_title("Diff (Pred - GT)"); axs[2].axis("off"); fig.colorbar(im2, ax=axs[2], fraction=0.046)
    if title: fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); plt.close(fig)

def plot_central_dipole_map(
    U_pred: np.ndarray,
    U_true: np.ndarray,
    out_png: Path,
    title: str = "",
    align_conjugation: bool = False,
    mode: str = "real",      # "real" or "abs"
):
    """
    Plot maps of Tr[ U(x0,y0) U(x,y)† ] / 3 for GT and Pred, plus their difference.
    Center (x0,y0) is (H//2, W//2). Arrays must be shape [H, W, 3, 3] complex.
    """
    U_pred = U_pred.astype(np.complex128, copy=False)
    U_true = U_true.astype(np.complex128, copy=False)

    if align_conjugation:
        # Optional: bring Pred into the same global gauge as True (doesn't change the dipole)
        U_pred = _align_global_conjugation(U_true, U_pred)

    H, W = U_true.shape[:2]
    cx, cy = H // 2, W // 2

    # Reference matrices at the center
    U0_t = U_true[cx, cy]                  # [3,3]
    U0_p = U_pred[cx, cy]                  # [3,3]

    # Conjugate-transposed fields
    Ud_t = np.conjugate(np.swapaxes(U_true, -1, -2))   # [H,W,3,3] (ji)
    Ud_p = np.conjugate(np.swapaxes(U_pred, -1, -2))   # [H,W,3,3]

    # Dipole maps: D(x,y) = Tr[ U0 * U(x,y)† ] / 3
    D_t = np.einsum('ij,xyji->xy', U0_t, Ud_t) / 3.0   # [H,W] complex
    D_p = np.einsum('ij,xyji->xy', U0_p, Ud_p) / 3.0   # [H,W] complex

    if mode == "abs":
        A_t = np.abs(D_t)
        A_p = np.abs(D_p)
        diff = A_p - A_t
        vmin, vmax = 0.0, 1.0
        label = r"$|\mathrm{Tr}[U_0 U^\dagger(x,y)]|/3$"
    else:  # "real"
        A_t = D_t.real
        A_p = D_p.real
        diff = (D_p - D_t).real
        vmin, vmax = -1.0, 1.0
        label = r"$\Re\,\mathrm{Tr}[U_0 U^\dagger(x,y)]/3$"

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=140, constrained_layout=True)
    im0 = axs[0].imshow(A_t, origin="lower", vmin=vmin, vmax=vmax)
    axs[0].set_title(f"GT  {label}"); axs[0].axis("off"); fig.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(A_p, origin="lower", vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Pred {label}"); axs[1].axis("off"); fig.colorbar(im1, ax=axs[1], fraction=0.046)

    # For the difference, use a symmetric range around 0
    dmax = np.max(np.abs(diff)) if np.isfinite(diff).any() else 1.0
    im2 = axs[2].imshow(diff, origin="lower", vmin=-dmax, vmax=dmax, cmap="coolwarm")
    axs[2].set_title("Diff (Pred - GT)"); axs[2].axis("off"); fig.colorbar(im2, ax=axs[2], fraction=0.046)

    if title:
        fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    
# ---------- dipole S(r) ----------
def _field9_from_U(U: np.ndarray) -> np.ndarray:
    return U.reshape(U.shape[0], U.shape[1], 9)

def dipole_axial(U: np.ndarray, offsets=((1,0),(0,1),(2,0),(0,2),(3,0),(0,3))):
    # mimic training: S(dx,dy) = Re⟨Tr[U(x) U†(x+r)]⟩/Nc, averaged over sites
    N = U.shape[0]
    S = []
    r = []
    for dx, dy in offsets:
        Ur = np.roll(np.roll(U, shift=dy, axis=0), shift=dx, axis=1)
        prod = U @ np.swapaxes(Ur.conj(), -1, -2)
        Spr  = np.trace(prod, axis1=-2, axis2=-1).real / 3.0
        S.append(Spr.mean())
        r.append((dx*dx + dy*dy)**0.5)
    idx = np.argsort(r)
    return np.asarray(r)[idx], np.asarray(S)[idx]

def dipole_map_autocorr(U: np.ndarray) -> np.ndarray:
    N = U.shape[0]
    F = _field9_from_U(U)
    Fk = np.fft.fft2(F, axes=(0,1))
    Pk = np.sum(Fk * np.conj(Fk), axis=2)
    corr = np.fft.ifft2(Pk)
    return (corr.real) / (N*N) / 3.0

import numpy as np


import numpy as np

def _align_global(Ua: np.ndarray, Ub: np.ndarray, side: str = "auto"):
    """
    Find a global SU(3) factor that best aligns Ub to Ua.
    side: "left", "right", or "auto" (choose the better of the two)
    Returns Ub_aligned (same shape as Ub).
    """
    H, W = Ua.shape[:2]

    def su3_polar_unitary(M):
        # unitary polar factor via SVD, projected to det=1
        U, _, Vh = np.linalg.svd(M)
        Uunit = U @ Vh
        det = np.linalg.det(Uunit)
        return Uunit * np.exp(-1j * np.angle(det) / 3.0)

    # --- Left candidate: minimize ||V Ub - Ua||_F  => maximize Re Tr(V <Ub Ua†>)
    C_left  = (Ub @ np.conjugate(np.swapaxes(Ua, -1, -2))).mean(axis=(0, 1))  # <Ub Ua†>
    V_left  = su3_polar_unitary(C_left)                                       # in SU(3)
    Ub_left = np.einsum('ij,xyjk->xyik', V_left, Ub)

    # --- Right candidate: minimize ||Ub W - Ua||_F => maximize Re Tr(W <Ub† Ua>)
    C_right  = (np.conjugate(np.swapaxes(Ub, -1, -2)) @ Ua).mean(axis=(0, 1)) # <Ub† Ua>
    W_right  = su3_polar_unitary(C_right)                                     # in SU(3)
    Ub_right = np.einsum('xyij,jk->xyik', Ub, W_right)

    # Score by r=0 cross (bigger is better)
    def r0_cross(Ua_, Ub_):
        M = (np.conjugate(np.swapaxes(Ua_, -1, -2)) @ Ub_).mean(axis=(0, 1))  # <Ua† Ub>
        return (np.real(np.trace(M)) / 3.0)

    if side == "left":
        return Ub_left
    if side == "right":
        return Ub_right

    # auto: pick the better global alignment
    return Ub_left if r0_cross(Ua, Ub_left) >= r0_cross(Ua, Ub_right) else Ub_right


def _align_global_conjugation(Ua: np.ndarray, Ub: np.ndarray) -> np.ndarray:
    """
    Find a single G ∈ SU(3) and return G Ub G† (conjugation).
    This preserves Tr per site exactly.
    We pick G from the polar factor of the correlation <Ub Ua†>.
    """
    # Correlation over the lattice
    C = (Ub @ np.conjugate(np.swapaxes(Ua, -1, -2))).mean(axis=(0,1))  # [3,3]

    # Unitary polar via SVD, then det=1 with phase-only
    U, _, Vh = np.linalg.svd(C, full_matrices=False)
    G = U @ Vh
    G *= np.exp(-1j * np.angle(np.linalg.det(G)) / 3.0)  # |.|=1

    # Conjugation (broadcast over H,W)
    Gdg = np.conjugate(G.T)
    return np.einsum('ij,xyjk,kl->xyil', G, Ub, Gdg)


def dipole_map_cross(Ua: np.ndarray, Ub: np.ndarray, *, gauge_side: str = "auto") -> np.ndarray:
    """
    S_cross(r) = (1/3) Re < Tr[ Ua^\dagger(x) * Ub(x+r) ] >_x,
    after first aligning Ub to Ua by an optimal global SU(3) factor.
    - Ua, Ub: [H, W, 3, 3] complex arrays (Wilson lines).
    - gauge_side: "left", "right", or "auto" (default).
    Returns: [H, W] real (map over r via FFT-based correlation).
    """
    assert Ua.shape == Ub.shape and Ua.shape[-2:] == (3, 3)
    H, W = Ua.shape[0], Ua.shape[1]

    # 1) Gauge-fix (align) Ub to Ua
    Ub_aligned = _align_global(Ua.astype(np.complex128), Ub.astype(np.complex128), side=gauge_side)

    # 2) Cross-correlator via FFT on the 9 complex components (no mean subtraction)
    Fa = _field9_from_U(Ua).astype(np.complex128)         # [H,W,9]
    Fb = _field9_from_U(Ub_aligned).astype(np.complex128) # [H,W,9]

    Fka = np.fft.fft2(Fa, axes=(0, 1))
    Fkb = np.fft.fft2(Fb, axes=(0, 1))
    Pk  = np.sum(Fka * np.conjugate(Fkb), axis=2)         # [H,W]
    corr = np.fft.ifft2(Pk)

    return (corr.real) / (H * W) / 3.0

# def dipole_map_cross(Ua: np.ndarray, Ub: np.ndarray) -> np.ndarray:
#     N = Ua.shape[0]
#     Fa = _field9_from_U(Ua); Fb = _field9_from_U(Ub)
#     Fka = np.fft.fft2(Fa, axes=(0,1)); Fkb = np.fft.fft2(Fb, axes=(0,1))
#     Pk = np.sum(Fka * np.conj(Fkb), axis=2)
#     corr = np.fft.ifft2(Pk)
#     return (corr.real) / (N*N) / 3.0

def radial_average_torus(Smap: np.ndarray):
    N = Smap.shape[0]
    rmax = N//2
    sums = np.zeros(rmax+1, dtype=np.float64)
    counts = np.zeros(rmax+1, dtype=np.int64)
    for dx in range(N):
        ddx = min(dx, N - dx)
        for dy in range(N):
            ddy = min(dy, N - dy)
            r = int(round(math.hypot(ddx, ddy)))
            if r <= rmax:
                sums[r] += Smap[dx, dy]
                counts[r] += 1
    valid = counts > 0
    radii = np.arange(rmax+1, dtype=np.int64)[valid]
    Sr = (sums[valid] / counts[valid]).astype(np.float64)
    return radii, Sr

def plot_dipole_curves(r_gt, S_gt, r_pr, S_pr, out_png: Path, title: str = "", r_cr=None, S_cr=None):
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2), dpi=140, constrained_layout=True)
    ax.plot(r_gt, S_gt, label="GT S(r)")
    ax.plot(r_pr, S_pr, label="Pred S(r)", linestyle="--")
    if r_cr is not None and S_cr is not None:
        ax.plot(r_cr, S_cr, label="Cross S_cross(r)", linestyle=":")
    ax.set_xlabel("r (lattice units)")
    ax.set_ylabel("S(r) = Re ⟨Tr[U(x)U†(x+r)]⟩ / Nc")
    ax.grid(True, alpha=0.3); ax.legend()
    if title: ax.set_title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); plt.close(fig)

# ---------- pack to complex ----------
def pack_to_complex(ch: np.ndarray) -> np.ndarray:
    re = ch[..., :9]; im = ch[..., 9:18]
    comp = re + 1j * im
    return comp.reshape(*comp.shape[:-1], 3, 3)

# ---------- generator microscope (su(3) direction & amplitude) ----------

def _gell_mann_8():
    L = np.zeros((8,3,3), dtype=np.complex128)
    L[0][0,1]=L[0][1,0]=1
    L[1][0,1]=-1j; L[1][1,0]=1j
    L[2][0,0]=1; L[2][1,1]=-1
    L[3][0,2]=L[3][2,0]=1
    L[4][0,2]=-1j; L[4][2,0]=1j
    L[5][1,2]=L[5][2,1]=1
    L[6][1,2]=-1j; L[6][2,1]=1j
    L[7][0,0]=1/np.sqrt(3); L[7][1,1]=1/np.sqrt(3); L[7][2,2]=-2/np.sqrt(3)
    return L

_GM = _gell_mann_8()

def _principal_log_unitary(U):
    w, V = np.linalg.eig(U)
    theta = np.angle(w)           # in (-pi, pi]
    return V @ np.diag(1j*theta) @ np.linalg.inv(V)

def _proj_su3(A):
    A = 0.5*(A - A.conj().T)      # anti-Hermitian numerically
    tr = np.trace(A)/3.0
    return A - tr*np.eye(3, dtype=A.dtype)

def _coeffs_gm(H):
    # H Hermitian, H = sum c_a lambda_a, with Tr(λ_a λ_b)=2 δ_ab
    c = np.empty(8, dtype=np.float64)
    for a in range(8):
        c[a] = 0.5*np.real(np.trace(H @ _GM[a]))
    return c

def generator_from_pair(UY, U0, Y):
    G = UY @ U0.conj().T
    A = (1.0/max(1e-12, float(Y))) * _principal_log_unitary(G)  # anti-Hermitian
    A = _proj_su3(A)
    H = (1j)*A                                                   # Hermitian
    nF = np.linalg.norm(A, 'fro')
    c8 = _coeffs_gm(H)
    return A, H, nF, c8

def avg_gen_stats(U_pred, Uy_true, U0, Y):
    H, W = U0.shape[:2]
    # sample a grid up to ~16x16 points for speed
    step_h = max(1, H//16); step_w = max(1, W//16)
    n_pred, n_true, cos_dir, scale = [], [], [], []
    for i in range(0, H, step_h):
        for j in range(0, W, step_w):
            Ap, Hp, npF, cp = generator_from_pair(U_pred[i,j], U0[i,j], Y)
            At, Ht, ntF, ct = generator_from_pair(Uy_true[i,j], U0[i,j], Y)
            n_pred.append(npF); n_true.append(ntF)
            denom = (np.linalg.norm(cp)*np.linalg.norm(ct) + 1e-12)
            cos_dir.append( float( (cp @ ct) / denom ) )
            scale.append( float( npF / (ntF + 1e-12) ) )
    return {
        "||A_pred||_F (mean)": float(np.mean(n_pred)),
        "||A_true||_F (mean)": float(np.mean(n_true)),
        "amplitude ratio pred/true (mean)": float(np.mean(scale)),
        "direction cosine (mean)": float(np.mean(cos_dir)),
        "direction cosine (p5..p95)": (
            float(np.percentile(cos_dir,5)), float(np.percentile(cos_dir,95))
        )
    }


# ---------- prediction ----------
def predict_Uy(model, U0, Y, params, device, sample = False):
    """
    Build [1,22,H,W] input for the model and return raw prediction as numpy [H,W,18].
    - U0: complex ndarray [H,W,3,3] (baseline field)
    - Y:  physical rapidity (or ΔY). At steps=0 this must be exactly 0.0
    - params: dict with m / Lambda_QCD / mu0 (any of *_GeV keys also ok)
    """
    import numpy as np
    import torch

    H, W = U0.shape[0], U0.shape[1]

    # ---- 18 channels from U0: Re and Im of 3x3 flattened to 9+9
    re9 = U0.real.reshape(H, W, 9)           # [H,W,9]
    im9 = U0.imag.reshape(H, W, 9)           # [H,W,9]

    x22 = np.empty((22, H, W), dtype=np.float32)
    x22[0:9]   = np.moveaxis(re9, -1, 0)     # [9,H,W]
    x22[9:18]  = np.moveaxis(im9, -1, 0)     # [9,H,W]

    # ---- scalar maps: put PHYSICAL Y (or ΔY) in channel 18 so gate sees exact 0 at steps=0
    def pget(d, *keys, default=0.0):
        for k in keys:
            if k in d:
                return float(d[k])
        return float(default)

    y_phys = float(Y)  # must be exactly 0.0 for steps=0
    m      = pget(params, "m_GeV", "m")
    lqcd   = pget(params, "Lambda_QCD_GeV", "Lambda_QCD", "lambda_qcd")
    mu0    = pget(params, "mu0_GeV", "mu0")

    print("m_GeV=", m, ", Lambda_QCD_GeV=", lqcd, ", mu0_GeV=", mu0)
    
    
    x22[18, :, :] = y_phys
    x22[19, :, :] = m
    x22[20, :, :] = lqcd
    x22[21, :, :] = mu0

    print("[head] alpha_scale=", getattr(getattr(model,"head",None), "alpha_scale", None),
          "clamp_alphas=", getattr(getattr(model,"head",None), "clamp_alphas", None),
          "identity_eps=", getattr(getattr(model,"head",None), "identity_eps", None))
    
    print("[time] film_mode=", getattr(model, "film_mode", None),
          "gamma_scale=", getattr(model, "time_cond", None).gamma_scale if hasattr(model,"time_cond") else None,
          "beta_scale=",  getattr(model, "time_cond", None).beta_scale  if hasattr(model,"time_cond") else None,
          "gate_temp=",   getattr(model, "time_cond", None).gate_temp   if hasattr(model,"time_cond") else None)

    
    # # ---- debug: what the model actually sees
    # print("[debug] input layout:",
    #       "base18->", float(np.abs(x22[:18]).mean()),
    #       "Ymin/max->", float(x22[18:19].min()), float(x22[18:19].max()))

    # # Detailed per-channel means (0..21): first 18 are U0 planes, 18=Y, 19..21=params
    # print("[debug] ch means(0..21): " + ", ".join(f"{float(x22[i].mean()):.3e}" for i in range(22)))
    # #print("hallo")
    
    # ---- to torch and run model
    x = torch.from_numpy(x22).unsqueeze(0)   # [1,22,H,W]
    x = x.to(device, non_blocking=True)

    # after building x22
    print("[debug] mean(Y map)=", float(x22[18].mean()))

    # assuming you already have the scalar Y as a Python/NumPy number
    Y_scalar = torch.as_tensor([float(Y)], device=device, dtype=torch.float32)  # shape [1]
    # normalize using the model’s buffers
    ymin = float(model.y_min_buf.item()); ymax = float(model.y_max_buf.item())
    if ymax > ymin:
        y01 = ((Y_scalar - ymin) / (ymax - ymin)).clamp_(0.0, 1.0).item()
    else:
        y01 = torch.sigmoid(Y_scalar).item()


#    y01 = model._y_scalar01(torch.from_numpy(x22).unsqueeze(0)).item()
    print("[debug] normalized y01=", y01)

    if sample == True:
        print("sampling!")
    model.eval()
    with torch.no_grad():
        base18, Y_scalar, theta = split_legacy_x(x)
        yhat = model(base18, Y_scalar, theta, sample=sample, dY=Y_scalar)  # [1,18,H,W]

        # inside main(), AFTER you build base18, Y_scalar, theta
        core = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            # either call the trunk explicitly...
            h = model.encode_trunk_from_components(base18, Y_scalar, theta)
            logsig = core.head.proj_logs(h)
            sigma_raw   = F.softplus(logsig)
            sigma_final = sigma_raw + core.head.sigma_floor
        print("infer σ_raw   mean/min/max:",
              float(sigma_raw.mean()), float(sigma_raw.min()), float(sigma_raw.max()))
        print("infer σ_final mean/min/max:",
              float(sigma_final.mean()), float(sigma_final.min()), float(sigma_final.max()))
        print("sigma_floor in predictor:", float(core.head.sigma_floor))
        
        #     # quick movement check before any projection/repair
    #     try:
    #         base18 = x22[:18]  # [18,H,W]
    #         moved = float(np.mean(np.abs(yhat[0].cpu().numpy() - base18)))
    #         print(f"[debug] |yhat-base18|mean={moved:.3e}")
    #     except Exception as _e:
    #         pass
        
    #     # movement diagnostic vs baseline channels
    #     base18 = x[0, :18].cpu().numpy()
    #     moved  = float(np.mean(np.abs(yhat[0].cpu().numpy())))
    #     print(f"[debug] |yhat|mean={moved:.3e}")

    
    y18 = yhat[0].detach().to("cpu").permute(1, 2, 0).numpy().astype(np.float32)  # [H,W,18]
    return y18


    

# ---------- quadrupole Q[(dx1,dy1),(dx2,dy2)] ----------
def _roll(U: np.ndarray, dy: int, dx: int) -> np.ndarray:
    # roll on torus: dy along axis 0 (y), dx along axis 1 (x)
    if (dy % U.shape[0] == 0) and (dx % U.shape[1] == 0):
        return U
    return np.roll(np.roll(U, shift=dy, axis=0), shift=dx, axis=1)

def quadrupole_autocorr(U: np.ndarray, pairs=((1,0),(0,1))) -> np.ndarray:
    """
    Q[(dx1,dy1),(dx2,dy2)] = ⟨ Re Tr[ U(x) U†(x+r1) U(x+r2) U†(x+r1+r2) ] ⟩ / Nc.
    Returns Q: [K] for K pairs; U is [N,N,3,3] complex, already SU(3)-projected.
    """
    N = U.shape[0]; assert U.shape[:2] == (N,N)
    Qs = []
    for (dx1,dy1),(dx2,dy2) in pairs:
        Ux  = U
        Ur1 = _roll(U, dy1, dx1)
        Ur2 = _roll(U, dy2, dx2)
        Ur12= _roll(U, dy1+dy2, dx1+dx2)

        prod = Ux @ np.swapaxes(Ur1.conj(), -1, -2) @ Ur2 @ np.swapaxes(Ur12.conj(), -1, -2)  # [N,N,3,3]
        tr = np.trace(prod, axis1=-2, axis2=-1).real / 3.0                                    # [N,N]
        Qs.append(tr.mean())
    return np.asarray(Qs, dtype=np.float64)

def quadrupole_cross(Ua: np.ndarray, Ub: np.ndarray, pairs=((1,0),(0,1))) -> np.ndarray:
    """Cross quadrupole mixing pred/true to probe alignment:
    Q_cross = ⟨ Re Tr[ Ua(x) Ub†(x+r1) Ua(x+r2) Ub†(x+r1+r2) ] ⟩ / Nc.
    """
    N = Ua.shape[0]; assert Ua.shape[:2] == (N,N) and Ub.shape[:2] == (N,N)
    Qs = []

    Ub_aligned = _align_global(Ua.astype(np.complex128), Ub.astype(np.complex128), side="auto")

    for (dx1,dy1),(dx2,dy2) in pairs:
        Ux_a   = Ua
        Ur1_b  = _roll(Ub_aligned, dy1, dx1)
        Ur2_a  = _roll(Ua, dy2, dx2)
        Ur12_b = _roll(Ub_aligned, dy1+dy2, dx1+dx2)

        prod = Ux_a @ np.swapaxes(Ur1_b.conj(), -1, -2) @ Ur2_a @ np.swapaxes(Ur12_b.conj(), -1, -2)
        tr = np.trace(prod, axis1=-2, axis2=-1).real / 3.0
        Qs.append(tr.mean())
    return np.asarray(Qs, dtype=np.float64)

def compare_quadrupole(U_pred: np.ndarray, Uy_true: np.ndarray,
                       pairs=(((1,0),(0,1)), ((2,0),(0,2)), ((1,1),(2,0))),
                       title: str = "", out_png: Path | None = None):
    """Compute quadrupole spectrum for pred/true (and cross) and report errors."""
    Q_true = quadrupole_autocorr(Uy_true, pairs=pairs)
    Q_pred = quadrupole_autocorr(U_pred,  pairs=pairs)
    Q_cross= quadrupole_cross(U_pred, Uy_true, pairs=pairs)

    # L2 error (finite-safe)
    def _l2(a, b):
        a = np.asarray(a); b = np.asarray(b)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() == 0: return np.nan
        d = a[m] - b[m]
        return float(np.sqrt(np.sum(d*d) / max(1, m.sum())))

    err = _l2(Q_pred, Q_true)
    print("[quad] pairs:", pairs)
    print(f"[quad] true: {np.round(Q_true, 6).tolist()}")
    print(f"[quad] pred: {np.round(Q_pred, 6).tolist()}")
    print(f"[quad] cross:{np.round(Q_cross,6).tolist()}")
    print(f"[quad] L2(pred,true)={err:.3e}")

    if out_png is not None:
        import matplotlib.pyplot as plt
        K = len(pairs)
        x = np.arange(K)
        w = 0.28
        fig, ax = plt.subplots(1,1, figsize=(7.2, 3.8), dpi=140, constrained_layout=True)
        ax.bar(x - w, Q_true, width=w, label="True")
        ax.bar(x,      Q_pred, width=w, label="Pred")
        ax.bar(x + w, Q_cross, width=w, label="Cross")
        ax.set_xticks(x); ax.set_xticklabels([f"{p[0]}|{p[1]}" for p in pairs], rotation=0)
        ax.set_ylabel("Quadrupole ⟨Re Tr[...]⟩ / Nc")
        if title: ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3); ax.legend()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png); plt.close(fig)


@torch.no_grad()
def forward_at_Y(x_one: torch.Tensor, Y: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (yhat, Uhat) at a specific Y.
    x_one: [1,C,H,W] float tensor on device, with channel YCH holding Y.
    """
    x = x_one.clone()
    x[:, YCH:YCH+1, :, :] = Y
    yhat = model(x)
    # --- REPLACE THIS with your script’s SU(3) reconstruction ---
    # Example possibilities:
    #   Uhat = criterion._pack18_to_U(yhat)
    #   Uhat = su3fix_from_18(yhat)           # <- your function
    Uhat = su3fix_from_18(yhat)               # <-- CHANGE THIS LINE
    # -------------------------------------------------------------
    return yhat, Uhat


# def predict_Uy(model, U0_np: np.ndarray, Y: float, params: dict, device="cpu"):
#     import torch
#     N = U0_np.shape[0]
#     x = np.concatenate([U0_np.real.reshape(N,N,9), U0_np.imag.reshape(N,N,9)], axis=-1)
#     cond = np.array([Y, params["m"], params["Lambda_QCD"], params["mu0"]], dtype=np.float32)
#     x = np.concatenate([x, np.broadcast_to(cond[None,None,:], (N,N,4))], axis=-1).astype(np.float32)    
#     xt = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(device)
#     with torch.no_grad():
#         y18 = model(xt).squeeze(0).permute(1,2,0).cpu().numpy()   # [N,N,18]
#     return y18  # raw 18 channels (float)

# ---------- CLI ----------
def main():
    import torch
    ap = argparse.ArgumentParser(description="Predict U(Y) and compare to ground truth, including dipole S(r).")
    ap.add_argument("--runs", type=Path, required=True, help="Root with run_00000/, run_00001/, ...")
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to evolver_best.pt (or last).")
    ap.add_argument("--ds", type=float, required=False, default=None,
                help="(deprecated/ignored) ds is read from the run's manifest.json")
    ap.add_argument("--run", type=str, required=True, help="Run folder name (e.g. run_00003) or integer index.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--steps", type=int, help="Choose target by exact cumulative steps.")
    group.add_argument("--target_y", type=float, help="Choose target by desired rapidity Y (closest match).")
    group.add_argument("--index", type=int, help="Choose target by snapshot index in manifest (0..).")
    ap.add_argument("--outdir", type=Path, default=Path("compare_out"))
    ap.add_argument("--device", type=str, default=None, help="cuda | mps | cpu (auto if omitted)")
    ap.add_argument("--plot_fft", action="store_true", help="Also compare Fourier spectra")
    ap.add_argument("--fft_quantity", type=str, default="ReTrU",
                    choices=["ReTrU","ImTrU","AbsTrU","ArgTrU"],
                    help="Scalar derived from U for the spectrum")
    ap.add_argument("--fft_bins", type=int, default=64, help="Radial bins for k-spectrum")


    args = ap.parse_args()

    # Resolve run dir and manifest
    run_dir = args.runs / (f"run_{int(args.run):05d}" if args.run.isdigit() else args.run)
    man = json.loads((run_dir / "manifest.json").read_text())
    # Read ds from manifest; CLI --ds is deprecated/ignored
    if args.ds is not None:
        print("[warn] --ds is ignored; reading ds from manifest.json instead.")
    ds = float(man.get("ds", man.get("params", {}).get("ds", float("nan"))))
    if not (ds == ds):
        raise ValueError("ds not found in " + str(run_dir / "manifest.json") + "; add a top-level 'ds'")

    # Device
    if args.device is None:
        if torch.cuda.is_available(): device = "cuda"
        #elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    else:
        device = args.device

    # Load model
    model, margs = safe_load_model(args.ckpt, device=device)
    ich = int(margs.get("in_ch", 22))
    print(f"[ckpt] in_ch={ich}  (22 expected for 4-parameter inputs)")
    if ich != 22:
        print("WARNING: checkpoint expects a different input channel count than this script feeds.\n"
              "         If the model was trained with g2mua (23 channels), predictions will be nonsense.")

    print("[head] alpha_scale=", getattr(model.head, "alpha_scale", None),
          "clamp_alphas=", getattr(model.head, "clamp_alphas", None),
          "identity_eps=", getattr(model.head, "identity_eps", None))

    print("ds=",ds)
    
    # compute a reasonable Y range from the run you’re evaluating
    y_vals = [steps_to_Y(int(s["steps"]), ds) for s in man["snapshots"]]
    y_min_data = float(min(y_vals))
    y_max_data = float(max(y_vals))
    
    # make the model use that normalization
    if hasattr(model, "y_min_buf"): model.y_min_buf.fill_(y_min_data)
    if hasattr(model, "y_max_buf"): model.y_max_buf.fill_(y_max_data)
    
    print(f"[debug] using y_min={y_min_data:.6g}, y_max={y_max_data:.6g}")


   
    # U0 (smallest steps) + params (NO g2mua)
    snaps = sorted(man["snapshots"], key=lambda s: int(s["steps"]))
    s0 = snaps[0]
    U0_path = run_dir / s0["path"] if not Path(s0["path"]).is_absolute() else Path(s0["path"])
    U0 = read_wilson_binary(U0_path)

    params = {
        "m": float(man["params"].get("m_GeV", man["params"].get("m", 0.0))),
        "Lambda_QCD": float(man["params"].get("Lambda_QCD_GeV", man["params"].get("Lambda_QCD", 0.0))),
        "mu0": float(man["params"].get("mu0_GeV", man["params"].get("mu0", 0.0))),
    }

    # Choose target
    target_snap, Y = pick_target_snapshot(man, ds, target_steps=args.steps,
                                          target_y=args.target_y, index=args.index)
    Uy_path = run_dir / target_snap["path"] if not Path(target_snap["path"]).is_absolute() else Path(target_snap["path"])
    Uy_true_raw = read_wilson_binary(Uy_path)
    Uy_true = svd_polar_det1(Uy_true_raw.copy())





    
    # # --- Generator-based calibration (single-shot; no rollout) ---
    # import numpy as np

    # def _mean_move(Ua, Ub):  # keep if you still want to log it
    #     return float(np.mean(np.abs(Ua - Ub)))

    # # Preliminary forward at target Y with current alpha_scale
    # y18_tmp = predict_Uy(model, U0, float(Y), params, device=device)
    # U_tmp   = svd_polar_det1(pack_to_complex(y18_tmp))

    # # Use generator microscope to get amplitude ratio (pred vs truth)
    # stats_tmp = avg_gen_stats(U_tmp, Uy_true, U0, Y)
    # amp_ratio = stats_tmp["amplitude ratio pred/true (mean)"]  # ~ < 1 if under-moving
    # dir_cos   = stats_tmp["direction cosine (mean)"]
    # print(f"[gen-calib] dir_cos={dir_cos:.3f}  amp_ratio={amp_ratio:.3e}")

    # # Compute gain; allow a larger cap (e.g., up to 64x)
    # if not np.isfinite(amp_ratio) or amp_ratio <= 1e-8:
    #     gain = 8.0
    # else:
    #     gain = 1.0 / amp_ratio

    # gain = float(np.clip(gain, 0.5, 64.0))   # <- raise cap from 8 to 64
    # print(f"[gen-calib] alpha_scale×={gain:.2f}")

    # # Apply: disable any clamp, then scale
    # if getattr(model.head, "clamp_alphas", None) is not None:
    #     model.head.clamp_alphas = None
    # model.head.alpha_scale = float(getattr(model.head, "alpha_scale", 1.0) * gain)
    # print("[head] adjusted alpha_scale ->", model.head.alpha_scale)

    # # Recompute prediction with calibrated gain
    # y18_raw = predict_Uy(model, U0, Y, params, device=device)


    # # --- Auto-calibrate head gain to match truth magnitude (single-shot; no rollout) ---

    # def _mean_move(Ua, Ub):
    #     return float(np.mean(np.abs(Ua - Ub)))

    # # 1) preliminary forward at target Y with current alpha_scale
    # y18_tmp = predict_Uy(model, U0, float(Y), params, device=device)
    # U_tmp   = svd_polar_det1(pack_to_complex(y18_tmp))

    # # 2) measure movement vs. U0 and vs. truth
    # m_pred = _mean_move(U_tmp, U0)
    # m_true = _mean_move(Uy_true, U0)

    # # 3) compute a conservative gain and apply it
    # gain = 4.0 if m_pred <= 1e-12 else float(np.clip(m_true / m_pred, 0.5, 8.0))
    # print(f"[calib] m_pred={m_pred:.3e}  m_true={m_true:.3e}  -> alpha_scale×={gain:.2f}")

    # # disable clamping if present, then scale the generator
    # if getattr(model.head, "clamp_alphas", None) is not None:
    #     model.head.clamp_alphas = None
    # model.head.alpha_scale = float(getattr(model.head, "alpha_scale", 1.0) * gain)
    # print("[head] adjusted alpha_scale ->", model.head.alpha_scale)

    # # 4) now run the REAL prediction at the calibrated gain
    y18_raw = predict_Uy(model, U0, Y, params, device=device, sample=True)

    # === Y-sensitivity at your chosen Y* versus Y=0 (single-sample sanity) ===

    # Forward once more at Y=0 using the same U0 and params
    y18_raw_0 = predict_Uy(model, U0, 0.0, params, device=device, sample=True)

    # Repair non-finite, pack to complex, and project to SU(3)
    Uy_pred = svd_polar_det1(pack_to_complex(y18_repair(y18_raw)))     # at Y*
    U0_pred = svd_polar_det1(pack_to_complex(y18_repair(y18_raw_0)))   # at Y=0

    # 1) How much does the predicted field move with Y?
    frob_per_site = np.linalg.norm(Uy_pred - U0_pred, axis=(-2, -1)).mean()
    print(f"[sanity] mean ||U_pred(Y*) - U_pred(0)||_F per site = {frob_per_site:.4e}")

    # 2) Trainer-style Qs using *axial* samples and N(r)=1-S(r), threshold 0.5
    #    (Override offsets to match your training run: 1,2,4,8,12,16 along axes.)
    OFFSETS_TRAIN = [(1,0),(0,1),(2,0),(0,2),(4,0),(0,4),(8,0),(0,8),(12,0),(0,12),(16,0),(0,16)]

    def qs_from_axial_np(U: np.ndarray, offsets=OFFSETS_TRAIN, thr=0.5) -> float:
        r_ax, S_ax = dipole_axial(U, offsets=offsets)    # uses your existing helper
        N_ax = 1.0 - S_ax
        N_mono = np.maximum.accumulate(N_ax)             # enforce monotone ↑ in r
        j = int(np.searchsorted(N_mono, thr, side="left"))
        j = max(1, min(j, len(r_ax)-1))
        n0, n1 = float(N_mono[j-1]), float(N_mono[j])
        r0, r1 = float(r_ax[j-1]), float(r_ax[j])
        if abs(n1 - n0) < 1e-12:
            r_star = r0
        else:
            a = (thr - n0) / (n1 - n0)
            r_star = r0 + a * (r1 - r0)
        return 1.0 / max(r_star, 1e-8)                   # qs_scale=1.0 in your trainer

    Qs_pred_Y  = qs_from_axial_np(Uy_pred)
    Qs_pred_0  = qs_from_axial_np(U0_pred)
    print(f"[sanity] Qs_pred (axial, N=1-S, thr=0.5):  Y=0 -> {Qs_pred_0:.5f}   Y* -> {Qs_pred_Y:.5f}   Δ={Qs_pred_Y - Qs_pred_0:+.5f}")

    # 3) (Optional) Show the axial samples on top of your radial curve for this Y*
    r_ax_Y, S_ax_Y = dipole_axial(Uy_pred, offsets=OFFSETS_TRAIN)
    r_ax_0, S_ax_0 = dipole_axial(U0_pred, offsets=OFFSETS_TRAIN)
    print(f"[sanity] axial N(r) at smallest nonzero r:  Y=0 -> {(1.0 - S_ax_0)[1]:.4e}   Y* -> {(1.0 - S_ax_Y)[1]:.4e}")


    # After Uy_pred / U0_pred are available
    def qs_from_axial_np_thr(U, thr=0.5):
        r_ax, S_ax = dipole_axial(U, offsets=OFFSETS_TRAIN)
        r_ax = np.asarray(r_ax); S_ax = np.asarray(S_ax); N_ax = 1.0 - S_ax
        N_mono = np.maximum.accumulate(N_ax)
        j = int(np.searchsorted(N_mono, thr, side="left"))
        j = max(1, min(j, len(r_ax)-1))
        n0, n1 = float(N_mono[j-1]), float(N_mono[j])
        r0, r1 = float(r_ax[j-1]),  float(r_ax[j])
        a = 0.0 if abs(n1-n0)<1e-12 else (thr - n0)/(n1 - n0)
        r_star = r0 + a*(r1 - r0)
        return 1.0/max(r_star, 1e-8), (j-1, j), (r0, r1), (n0, n1)

    Qs_pred_Y, seg_pred, rseg_pred, nseg_pred = qs_from_axial_np_thr(Uy_pred, thr=0.5)
    print(f"[pred] threshold seg {seg_pred}  r≈{rseg_pred}  N≈{nseg_pred}")
    # === Qs change: truth vs model; slope and sensitivity diagnostics ===

    # Reuse OFFSETS_TRAIN and qs_from_axial_np from earlier
    # OFFSETS_TRAIN = [(1,0),(0,1),(2,0),(0,2),(4,0),(0,4),(8,0),(0,8),(12,0),(0,12),(16,0),(0,16)]

    def qs_from_axial_np_thr(U, thr):
        r_ax, S_ax = dipole_axial(U, offsets=OFFSETS_TRAIN)
        r_ax = np.asarray(r_ax); S_ax = np.asarray(S_ax); N_ax = 1.0 - S_ax
        N_mono = np.maximum.accumulate(N_ax)
        j = int(np.searchsorted(N_mono, thr, side="left"))
        j = max(1, min(j, len(r_ax)-1))
        n0, n1 = float(N_mono[j-1]), float(N_mono[j])
        r0, r1 = float(r_ax[j-1]),  float(r_ax[j])
        a = 0.0 if abs(n1-n0)<1e-12 else (thr - n0)/(n1-n0)
        r_star = r0 + a*(r1-r0)
        return 1.0/max(r_star, 1e-8), (j-1, j), (r0, r1), (n0, n1)

    # 1) TRUTH delta Qs between Y=0 and Y*
    try:
        Qs_true_0, seg0_t, rseg0_t, nseg0_t = qs_from_axial_np_thr(U0, thr=0.5)
        Qs_true_Y, segY_t, rsegY_t, nsegY_t = qs_from_axial_np_thr(Uy_true, thr=0.5)
        print(f"[truth] Qs_true axial (thr=0.5):  Y=0 -> {Qs_true_0:.5f}   Y* -> {Qs_true_Y:.5f}   Δ={Qs_true_Y - Qs_true_0:+.5f}")
        print(f"[truth] threshold crossed between axial r indices {segY_t} (r≈{rsegY_t}) at N in {nsegY_t}")
    except Exception as e:
        print(f"[truth] Skipped: Uy_true/U0 not available here ({e}).")

    # 2) MODEL local slope at Y*: finite difference with ±eps around current Y
    eps = 0.04  # feel free to tweak smaller/larger; must fit in your Y-range
    Y_minus = max(0.0, Y - eps)
    Y_plus  = Y + eps
    y18_m = predict_Uy(model, U0, Y_minus, params, device=device, sample=True)
    y18_p = predict_Uy(model, U0, Y_plus,  params, device=device, sample=True)
    U_pred_m = svd_polar_det1(pack_to_complex(y18_repair(y18_m)))
    U_pred_p = svd_polar_det1(pack_to_complex(y18_repair(y18_p)))
    Qs_m, _, segm_r, segm_n = qs_from_axial_np_thr(U_pred_m, thr=0.5)
    Qs_p, _, segp_r, segp_n = qs_from_axial_np_thr(U_pred_p, thr=0.5)
    slope = (Qs_p - Qs_m) / max(1e-9, (Y_plus - Y_minus))
    print(f"[model] local slope dQs/dY @ Y≈{Y:.3f} ≈ {slope:.3f}  (Qs[Y-ε]={Qs_m:.5f}, Qs[Y+ε]={Qs_p:.5f})")

    # 3) Threshold sensitivity and axial-segment info
    for thr in (0.3, 0.5, 0.7):
        Qs0, seg0, rseg0, nseg0 = qs_from_axial_np_thr(U0_pred, thr)
        QsY, segY, rsegY, nsegY = qs_from_axial_np_thr(Uy_pred, thr)
        print(f"[thr={thr}] Qs_pred:  Y=0 -> {Qs0:.5f}  Y* -> {QsY:.5f}  Δ={QsY - Qs0:+.5f}  |  crossing seg {segY} (r≈{rsegY}, N≈{nsegY})")

    # --- Head activations probe (proj_mu output) at Y=0 and Y=target ---
    capt = {}
    def _hook_proj_mu(module, inputs, output):
        capt.setdefault('a', []).append(output.detach().cpu().numpy())  # [B,8,H,W]

    h = model.head.proj_mu.register_forward_hook(_hook_proj_mu)
    _ = predict_Uy(model, U0, 0.0,      params, device=device, sample=True)
    _ = predict_Uy(model, U0, float(Y), params, device=device, sample=True)
    h.remove()

    
    a0, aY = [np.abs(A).astype(np.float64) for A in capt['a']]
    a0 = a0[0]; aY = aY[0]
    def _summ(A):
        return dict(mean=float(A.mean()), p50=float(np.percentile(A,50)),
                    p90=float(np.percentile(A,90)), max=float(A.max()))
    print("[alphas] at Y=0  :", _summ(a0))
    print("[alphas] at Y=Y* :", _summ(aY))
    print("[alphas] ratio(Y/0):", {
        k: (_summ(aY)[k] / max(1e-12, _summ(a0)[k])) for k in ['mean','p90','max']
    })

    def _mean_move(Ua, Ub):
        return float(np.mean(np.abs(Ua - Ub)))

    Ys_test = [0.25*Y, 0.5*Y, 1.0*Y]
    moves = []
    for Yk in Ys_test:
        y18_k = predict_Uy(model, U0, float(Yk), params, device=device, sample = True)
        Uk = svd_polar_det1(pack_to_complex(y18_k))
        moves.append(_mean_move(Uk, U0))
    print("[linearity] (Y, mean|U(Y)-U0|):", list(zip([float(y) for y in Ys_test], moves)))

    
    # ---- Predict
    #y18_raw = predict_Uy(model, U0, Y, params, device=device)


    # after: y18_raw = predict_Uy(model, U0, Y, params, device=device)
#    y18_raw_0 = predict_Uy(model, U0, 0.0, params, device=device)
#    print("[debug] |yhat(Y)-yhat(0)|mean = %.3e" % np.mean(np.abs(y18_raw - y18_raw_0)))

    # diagnostics BEFORE repair
    tr_raw = np.trace((y18_raw[..., :9] + 1j*y18_raw[..., 9:18]).reshape(U0.shape[0], U0.shape[1], 3,3),
                      axis1=-2, axis2=-1) / 3.0
    print(f"Trace raw: min={tr_raw.real.min():.3e}+{tr_raw.imag.min():.3e}i  "
          f"max={tr_raw.real.max():.3e}+{tr_raw.imag.max():.3e}i")

    # repair the 18 channels
    y18 = y18_repair(y18_raw)
    U_pred = pack_to_complex(y18)
    U_pred = svd_polar_det1(U_pred)  # unitary, det=1
    tr_rep = np.trace(U_pred, axis1=-2, axis2=-1) / 3.0
    print(f"Trace repaired: min={tr_rep.real.min():.3e}+{tr_rep.imag.min():.3e}i  "
          f"max={tr_rep.real.max():.3e}+{tr_rep.imag.max():.3e}i")

    # --- place right after you have: U_pred (np.complex64 [N,N,3,3]), Uy_true, U0
    def dipole_curve(U):
        Smap = dipole_map_autocorr(U)  # uses Re⟨Tr U(x)U†(x+r)⟩/Nc
        return radial_average_torus(Smap)  # (r, S(r))

    def l2_S(Ua, Ub):
        ra, Sa = dipole_curve(Ua)
        rb, Sb = dipole_curve(Ub)
        # assume same r-grid; if not, interpolate
        return float(np.sqrt(np.mean((Sa - Sb)**2)))

    # Learned left-update (what you already plotted)
    G_pred = U_pred @ np.conjugate(np.swapaxes(U0, -1, -2))   # U_pred U0†
    U_left         = U_pred                                   # = exp(+) @ U0
    U_left_flip    = np.conjugate(np.swapaxes(G_pred, -1, -2)) @ U0          # = exp(−) @ U0
    U_right        = U0 @ G_pred                                               # = U0 @ exp(+)
    U_right_flip   = U0 @ np.conjugate(np.swapaxes(G_pred, -1, -2))            # = U0 @ exp(−)

    candidates = {
        "left":        U_left,
        "left_flip":   U_left_flip,
        "right":       U_right,
        "right_flip":  U_right_flip,
    }

    scores = {name: l2_S(Uc, Uy_true) for name, Uc in candidates.items()}
    print("[which-evo] dipole L2 vs GT:", {k: f"{v:.3e}" for k,v in scores.items()})
    best = min(scores, key=scores.get)
    print("[which-evo] best match:", best)

    
    print("[debug] y_min/y_max buffers:", 
          float(getattr(model, "y_min_buf", torch.tensor(float('nan'))).item()),
          float(getattr(model, "y_max_buf", torch.tensor(float('nan'))).item()))



    # ---- Metrics & plots
    mets = su3_metrics(U_pred, Uy_true)
    print("== Prediction vs Ground Truth ==")
    print(f"run: {run_dir.name}")
    print(f"target steps: {int(target_snap['steps'])}   target Y ~= {Y:.6f}")
    for k,v in mets.items():
        print(f"{k:>15}: {v:.6e}")

    out_trace = args.outdir / f"{run_dir.name}_steps_{int(target_snap['steps']):05d}_trace_compare.png"
    title = f"{run_dir.name}  steps={int(target_snap['steps'])}  Y≈{Y:.4f}"
    plot_trace_maps(U_pred, Uy_true, out_trace, title=title)
    print(f"Wrote figure: {out_trace}")

    out_central_dipole = args.outdir / f"{run_dir.name}_steps_{int(target_snap['steps']):05d}_centdip_compare.png"
    title = f"{run_dir.name}  steps={int(target_snap['steps'])}  Y≈{Y:.4f}"
    plot_central_dipole_map(U_pred, Uy_true, out_central_dipole, title=title)
    print(f"Wrote figure: {out_central_dipole}")

    if args.plot_fft:
        # choose scalar quantity and radial binning
        qty = args.fft_quantity      # "ReTrU" (default), "ImTrU", "AbsTrU", or "ArgTrU"
        nbin = int(args.fft_bins)

        # compute radial spectra (isotropic average of the 2D FFT power)
        k, P_true, P_pred = spectra_from_U(Uy_true, U_pred, mode=qty, n_bins=nbin)

        # consistent file naming with Y-tag (like your Qs/dipole plots)
        y_tag = f"Y{float(Y):.3f}".replace(".", "p")

        def _tagify(s: str) -> str:
            # keep it filesystem-friendly
            return s.replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")

        run_tag = run_dir.name                    # e.g. "run_00505_00000"
        
        fft_dir = args.outdir / f"fft_{qty}_{run_tag}"
        fft_dir.mkdir(parents=True, exist_ok=True)
        
        spectrum_png = fft_dir / f"{run_tag}_spectrum_{qty}_{y_tag}.png"
        ratio_png    = fft_dir / f"{run_tag}_spectrum_ratio_{qty}_{y_tag}.png"

        print("Wrote figure:",spectrum_png)
        print("Wrote figure:",ratio_png)
        
        plot_spectrum(k, P_true, P_pred, spectrum_png,
                      title=f"Power spectrum of {qty} at Y={Y:.3f} — {run_tag}")
        plot_spectrum_ratio(k, P_true, P_pred, ratio_png,
                            title=f"Prediction/Truth ratio of {qty} at Y={Y:.3f} — {run_tag}")
    
    # --- Dipole maps (make sure these lines already ran/are present above) ---
    Smap_true = dipole_map_autocorr(Uy_true)
    Smap_pred = dipole_map_autocorr(U_pred)

    # --- Baseline dipole from U0 (Y=0) ---
    Smap_base = dipole_map_autocorr(U0)

    # --- Radial averages (define the r,S arrays unconditionally) ---
    r_gt, S_gt = radial_average_torus(Smap_true)
    r_pr, S_pr = radial_average_torus(Smap_pred)
    r_ba, S_ba = radial_average_torus(Smap_base)

    # --- L2 diagnostics ---
    def l2(a, b):
        import numpy as np
        a = np.asarray(a); b = np.asarray(b)
        m = np.isfinite(a) & np.isfinite(b)
        if not m.any():
            return float("nan")
        return float(np.sqrt(np.mean((np.nan_to_num(a[m]) - np.nan_to_num(b[m]))**2)))

    print(f"[debug] dipole L2(pred,base)={l2(S_pr, S_ba):.3e}  "
          f"dipole L2(true,base)={l2(S_gt, S_ba):.3e}  "
          f"dipole L2(pred,true)={l2(S_pr, S_gt):.3e}")

    Smap_cross = dipole_map_cross(U_pred, Uy_true)
    r_cr, S_cr = radial_average_torus(Smap_cross)

    out_dip = args.outdir / f"{run_dir.name}_steps_{int(target_snap['steps']):05d}_dipole_compare.png"
    plot_dipole_curves(r_gt, S_gt, r_pr, S_pr, out_png=out_dip, title=title, r_cr=r_cr, S_cr=S_cr)
    print(f"Wrote figure: {out_dip}")

    pairs = (((1,0),(0,1)), ((2,0),(0,2)), ((1,1),(2,0)))  # tweak as you like
    out_quad = args.outdir / f"{run_dir.name}_steps_{int(target_snap['steps']):05d}_quadrupole_compare.png"
    compare_quadrupole(U_pred, Uy_true, pairs=pairs, title=title, out_png=out_quad)
    print(f"Wrote figure: {out_quad}")

    
if __name__ == "__main__":
    main()
