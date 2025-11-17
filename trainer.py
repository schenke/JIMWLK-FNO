import numpy as np
import collections
import torch
import time
import os, re, json, math, argparse, random
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Sequence
from torch.optim.lr_scheduler import (
    SequentialLR, LinearLR
)

#----------------------- Helpers -------------------------

# --- Canonical SU(3) Gell-Mann matrices ---
def su3_gellmann_matrices(
    *, dtype: torch.dtype = torch.complex64,
    device: torch.device | None = None
) -> torch.Tensor:
    """Return λ_a/2 (a=1..8) as a tensor of shape [8,3,3].
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
    L = L / 2.0
    # cast/move if needed; buffers will follow the module when .to(device) is called
    return L.to(dtype=dtype, device=device) if (dtype != torch.complex64 or device is not None) else L

def pack_to_complex(v18: 'torch.Tensor') -> 'torch.Tensor':
    """v18 [...,18] -> [...,3,3] complex"""
    v18 = v18.contiguous()
    real = v18[..., :9].view(*v18.shape[:-1], 3, 3)
    imag = v18[...,  9:].view(*v18.shape[:-1], 3, 3)
    return torch.complex(real.float(), imag.float())

def unpack_to_18(U: 'torch.Tensor') -> 'torch.Tensor':
    """U [...,3,3] complex -> [...,18] real"""
    real, imag = U.real, U.imag
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

#----------------------- Logging -------------------------

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
        self._count[name] = int(self._count.get(name, 0)) + int(n)

    def mean(self, name: str, default=float("nan")):
        s = self._sum.get(name); c = self._count.get(name)
        if s is None or c is None:
            return default
        # c may be a Python int (preferred) or a 0-dim tensor from old states
        if isinstance(c, torch.Tensor):
            try:
                c_val = int(c.detach().to('cpu').item())
            except Exception:
                return default
        else:
            c_val = int(c)
        if c_val == 0:
            return default
        return (s / c_val)  # returns a tensor; caller can convert when printing

    def reset(self):
        self._sum.clear(); self._count.clear()

def read_wilson_binary(path: Path, size: int | None = None) -> np.ndarray:
    """
    Read IP-Glasma/JIMWLK 'method 2' binary: 18 doubles/site (Re/Im for 9 entries).
    Supports a small header (0,2,4,8,16,32,64 doubles).
    Returns float32 array with shape [N, N, 18] = [9 real channels, 9 imag channels].
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
                N = int(round(math.isqrt(N2)))
                if N * N == N2:
                    return N, h
        return None, None

    if size is None:
        N, hdr = infer_from_any_header()
        if N is None:
            raise ValueError(f"{path}: cannot infer lattice size from {a.size} doubles (tried headers {tried_hdrs}).")
    else:
        hdr = None
        for h in tried_hdrs:
            rem = a.size - h
            if rem == 18 * size * size:
                hdr = h
                N = size
                break
        if hdr is None:
            raise ValueError(f"{path}: cannot match requested size {size}; total doubles={a.size}")

    payload = a[hdr: hdr + 18 * N * N]                 # length = 18*N*N
    M = payload.reshape(N, N, 9, 2)                    # (..., 9 entries, [Re, Im])

    out = np.empty((N, N, 18), dtype=np.float32)       # [N, N, 18] = [9 Re, 9 Im]
    out[..., :9] = M[..., 0]                           # real parts
    out[..., 9:] = M[..., 1]                           # imag parts
    return out

def steps_to_Y(steps: int, ds: float) -> float:
    # rcJIMWLK Langevin mstochaapping
    return (math.pi**2) * ds * steps

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
                 cache_initial: bool = True
                 ):
        super().__init__()
        
        self.root = Path(root)
        self.N = None
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

        # Take only the runs for the current split: train or val. The random seed needs to be the same in both calls.
        rng.shuffle(all_runs)
        cut = int(round(len(all_runs) * split_frac))
        self.runs = all_runs[:cut] if split == "train" else all_runs[cut:]
       
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
            A18 = read_wilson_binary(run_snaps[a0]["path"], size=None)   # [N, N, 18]
            A18 = np.asarray(A18)                                        # ensure ndarray
            # If your reader returns float64, uncomment next line
            # A18 = A18.astype(np.float32, copy=False)

            N_here = A18.shape[0]
            if self.N is None:
                self.N = N_here
            elif N_here != self.N:
                raise ValueError(f"Mixed lattice sizes detected: saw N={self.N} and N={N_here}")

            # Cache in canonical form or skip
            base18 = A18.copy() if self.cache_initial else None
            
            self.snapshots_by_run.append(run_snaps)
            self.params_by_run.append({
                "m": float(params.get("m_GeV") or params.get("m") or 0.0),
                "Lambda_QCD": float(params.get("Lambda_QCD_GeV") or params.get("Lambda_QCD") or 0.0),
                "mu0": float(params.get("mu0_GeV") or params.get("mu0") or 0.0),
            })
            self.anchor_idx_by_run.append(a0)
            self.anchor_cache_by_run.append(base18 if self.cache_initial else None)
            self._anchor_seen = set()     

        if self.N is None:
            raise ValueError("No valid runs/snapshots found.")

        # Build entries:
        self.entries: List[Dict[str, Any]] = []
        
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
        N = self.N
        e = self.entries[idx]

        # ---------- Two-time path: always SAME-RUN anchor -> target ----------
        run_idx = int(e["run_idx"])
        snaps = self.snapshots_by_run[run_idx]
        a0 = self.anchor_idx_by_run[run_idx]
        Ya = float(snaps[a0]["Y"])

        a = int(e["a_idx"])
        b = int(e["b_idx"])
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
            Ua18 = read_wilson_binary(snaps[a]["path"], size=N)
            # cache lazily per-worker for this run to avoid rereads next time
            if run_idx not in self._anchor_seen:
                self.anchor_cache_by_run[run_idx] = Ua18
                self._anchor_seen.add(run_idx)

        Ub18 = read_wilson_binary(snaps[b]["path"], size=N)

        Ua18 = np.array(Ua18, copy=True)  # [H,W,18], float32
        Ub18 = np.array(Ub18, copy=True)

        # Compose input channels: [U(Ya) 18ch] + [Y (1ch)] + [params 3ch]
        y_scalar = (Yb - Ya)
        base18 = torch.from_numpy(Ua18).permute(2,0,1)       # [18,H,W]
        y_s    = torch.tensor(y_scalar, dtype=torch.float32) # []
        theta  = torch.tensor([e["m"], e["Lambda_QCD"], e["mu0"]], dtype=torch.float32)  # [3] calling my three parameter vector theta
        target = torch.from_numpy(Ub18).permute(2,0,1)       # [18,H,W]
        return base18, y_s, theta, target

def make_loaders(root, batch_size=1, workers=2, seed=0, ddp: bool=False, **kwargs):
    #training dataset
    train_ds = JimwlkEvolverDataset(root, split="train", split_frac=0.9, seed=seed,
                                    cache_initial=(workers == 0))
    #validation dataset
    val_ds   = JimwlkEvolverDataset(root, split="val",   split_frac=0.9, seed=seed,
                                    cache_initial=(workers == 0))

    #if parallel on CUDA devices, use distributed samplers
    if ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   shuffle=False)
    else:
        train_sampler = None; val_sampler = None

    use_pin = torch.cuda.is_available()                # pin memory if using CUDA

    train_dl = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=workers, pin_memory=use_pin,
        persistent_workers=(workers > 0),         
        prefetch_factor=(4 if workers > 0 else None), 
        drop_last=True) #last partial batch dropped to keep all batches the same size
    
    val_dl   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler,
        num_workers=workers, pin_memory=use_pin,
        persistent_workers=(workers > 0),          
        prefetch_factor=(4 if workers > 0 else None), 
        drop_last=False) #do not drop last partial batch in validation
    
    #print some stats about Y in training/validation sets
    val_Y = [float(e["Y"]) for e in getattr(val_dl.dataset, "entries", [])]
    if val_Y:
        print(f"[val] Y range from data: y_min={min(val_Y):.6g}, y_max={max(val_Y):.6g}, "
              f"count={len(val_Y)}")

    return train_dl, val_dl, train_sampler, val_sampler

#----------------------- RBF Embedding -------------------------
#It turns a number in [0,1] into a smooth, K-dimensional vector using 
#Gaussian radial basis functions (RBFs)

#RBF features make it easy for a small MLP to represent nonlinear, smooth 
#dependence on Y and θ and to interpolate between regimes.

class RBFEmbed(nn.Module):
    def __init__(self, K: int, learnable: bool = True, init_sigma: float = 0.20):
        super().__init__()
        # centers in [0,1]; log-widths so widths stay positive
        centers = torch.linspace(0., 1., K) #Creates K centers evenly spaced in [0,1]
        log_widths = torch.log(torch.full((K,), init_sigma)) #Keeps log-widths so widths stay positive

        if learnable: #If learnable=True, both centers and widths get trained
            self.rbf_centers = nn.Parameter(centers)         # [K]
            self.rbf_log_widths = nn.Parameter(log_widths)   # [K]
        else:
            self.register_buffer("rbf_centers", centers)         # [K]
            self.register_buffer("rbf_log_widths", log_widths)   # [K]

    def forward(self, y01: torch.Tensor) -> torch.Tensor:
        # y01 is in [0,1]; shape [B] or [B,1]
        widths = self.rbf_log_widths.exp()                 # [K]
        z = (y01[..., None] - self.rbf_centers) / widths   # [B,K]  z_k = (y - c_k) / width_k
        time_feat = torch.exp(-0.5 * z * z)                # [B,K]  exp(-0.5 z*z)
        return time_feat

#----------------------- Time Conditioner -------------------------
#To work across many Y and θ values, FiLM lets the trunk retune itself 
#on the fly instead of learning a one-size-fits-all mapping. 
#That tends to improve accuracy and data efficiency.

#removed the shift (beta) term for simplicity - could be added again if needed

class TimeConditioner(nn.Module):
    def __init__(self, n_blocks: int, ch: int, emb_dim: int,
                 hidden: int = 64, 
                 gamma_scale: float = 1.5,
                 gate_temp: float = 2.0):
        super().__init__()
        self.n_blocks = n_blocks
        self.ch = ch
        self.gamma_scale = gamma_scale
        self.gate_temp = gate_temp
        self.shared = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.block_heads = nn.ModuleList()
        for _ in range(n_blocks):
            out_dim = (ch + 1)
            self.block_heads.append(nn.Linear(hidden, out_dim))
    
    def forward(self, t_emb: 'torch.Tensor'):
        h = self.shared(t_emb)
        outs = [head(h) for head in self.block_heads]
        parsed = []
        for o in outs:
            gamma = torch.tanh(o[:, :self.ch]) * self.gamma_scale
            beta  = None
            gate  = torch.sigmoid(self.gate_temp * o[:, -1:])
            parsed.append((gamma, beta, gate))
        return parsed

#----------------------- Head -------------------------

class SpecHead(torch.nn.Module):
    def __init__(self, in_ch, n_bins=48, hidden=128):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)  # global spatial pooling
        self.mlp  = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_ch, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, n_bins),
        )
    def forward(self, h):                       # h: (B, in_ch, H, W)
        return self.mlp(self.pool(h))           # (B, n_bins)

class SU3HeadGellMannStochastic(nn.Module):
    """
    Predict per-unit-Y drift μ and diagonal σ for 8 (or 16) Lie-algebra coeffs.
    Sampling: α_step = μ * ΔY + σ * √ΔY ∘ η, with optional spatially-correlated η.
    If dY is None, we use Ymap (so a single 0→Y step works out-of-the-box).
    """
    def __init__(self,
                 width: int,
                 *,
                 alpha_channels: int = 16,     
                 identity_eps: float = 0.0,
                 clamp_alphas: float | None = None,
                 alpha_vec_cap: float | None = 15.0,
                 A_cap: float | None = None,
                 spec_bins: int = 48,
                 sigma0: float = 0.03,
                 sigma_mode: str = "conv",       # "diag" | "conv" | "spectral"
                 noise_kernel: int = 9):
        super().__init__()

        self.C = alpha_channels
        self.identity_eps = float(identity_eps)
        self.clamp_alphas = clamp_alphas
        self.alpha_vec_cap = alpha_vec_cap
        self.A_cap = A_cap
        self.sigma_mode = sigma_mode
        self.spec_bins = spec_bins
        self.noise_kernel = int(noise_kernel)

        # Heads
        # Define the output “head(s)” that turn trunk features h into the quantities my loss/sampler needs
        self.proj_mu   = nn.Conv2d(width, self.C, kernel_size=1, bias=True) #the drift per Lie-algebra coefficient, per spatial location
        self.proj_logs = nn.Conv2d(width, self.C, kernel_size=1, bias=True) #the log of the diagonal sigma per Lie-algebra coefficient, per spatial location
        self.spec_head = SpecHead(width, n_bins=self.spec_bins, hidden=128) #the spectral head for spectral noise mode
        # Predicted power spectral density (PSD) radial bins for the noise core (used in "spectral" sigma mode to synthesize spatially correlated noise)


        # Set safe, sensible starting values for the two 1×1 conv “heads” that predict μ and logσ    
        nn.init.xavier_uniform_(self.proj_mu.weight, gain=1e-2)
        nn.init.constant_(self.proj_mu.bias, 0.0)
        nn.init.constant_(self.proj_logs.weight, 0.0)
        nn.init.constant_(self.proj_logs.bias, math.log(sigma0))  # σ≈sigma0 at start

        # Gell-Mann/2 (Hermitian, traceless)
        self.register_buffer("lambdas", su3_gellmann_matrices(), persistent=False)

        # Noise coloring for conv mode
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
        elif self.sigma_mode in ("diag", "spectral"):
            pass
        else:
            raise ValueError(f"Unknown sigma_mode: {self.sigma_mode}")

        # small monitors
        self.last_A_fro_mean = torch.tensor(0.0)
        self.last_sigma_mean = torch.tensor(0.0)

        # cache for rfft binning (spectral mode)
        self._bins_cache = {}

    # ---------------- Helpers ----------------
    @staticmethod
    def _whiten_per_batch(x: torch.Tensor) -> torch.Tensor:
        # zero mean + unit RMS per-sample, per-channel
        x = x - x.mean(dim=(2,3), keepdim=True)
        den = x.pow(2).mean(dim=(2,3), keepdim=True).add(1e-12).sqrt().detach()
        return x / den

    def _ensure_rfft_bins(self, H, W, device, n_bins):
        if not hasattr(self, "_bins_cache"): self._bins_cache = {}
        key = (H, W, device, n_bins, use_lattice_k, 1, 1)
        out = self._bins_cache.get(key)
        if out is not None:
            return out["bin_index"], out["counts"]

        # frequencies in cycles per unit length
        fy = torch.fft.fftfreq(H, d=1, device=device)      # vertical
        fx = torch.fft.rfftfreq(W, d=1, device=device)     # horizontal (RFFT)

        wy = 2 * torch.pi * fy * 1                         # angular * lattice spacing
        wx = 2 * torch.pi * fx * 1
        ky = 2.0 * torch.sin(0.5 * wy)                     # lattice momenta
        kx = 2.0 * torch.sin(0.5 * wx)

        rr = torch.sqrt(ky[:, None]**2 + kx[None, :]**2)     # (H, W//2+1)

        edges = torch.linspace(rr.min(), rr.max(), n_bins + 1, device=device)
        bin_index = torch.bucketize(rr, edges) - 1
        bin_index = bin_index.clamp_(0, n_bins - 1).to(torch.long)
        counts = torch.bincount(bin_index.view(-1), minlength=n_bins).clamp_min_(1)

        self._bins_cache[key] = {"bin_index": bin_index, "counts": counts}
        return bin_index, counts

    def _ensure_rfft_bins(self, H, W, device, n_bins):
        key = (H, W, device)
        out = self._bins_cache.get(key)
        if out is not None and out["n_bins"] == n_bins:
            return out["bin_index"], out["counts"]
        ky = torch.fft.fftfreq(H, d=1.0, device=device)
        kx = torch.fft.rfftfreq(W, d=1.0, device=device)
        rr = torch.sqrt(ky[:, None]**2 + kx[None, :]**2)           # (H, W//2+1)
        edges = torch.linspace(rr.min(), rr.max(), n_bins + 1, device=device)
        bin_index = torch.bucketize(rr, edges) - 1                 # [H, W//2+1]
        bin_index = bin_index.clamp_(0, n_bins - 1).to(torch.long)
        counts = torch.bincount(bin_index.view(-1), minlength=n_bins).clamp_min_(1)
        self._bins_cache[key] = {"bin_index": bin_index, "counts": counts, "n_bins": n_bins}
        return bin_index, counts

    def _synth_from_bins(self, logA_bins, C, H, W, device, zero_dc=True):
        B = logA_bins.size(0)
        W2 = W//2 + 1
        bin_index, _ = self._ensure_rfft_bins(H, W, device, logA_bins.size(1))
        A_bins = F.softplus(logA_bins) + 1e-8                       # (B, n_bins)
        idx    = bin_index.view(-1).unsqueeze(0).expand(B, -1)      # (B, H*W2)
        A_hw   = torch.gather(A_bins, 1, idx).view(B, H, W2)        # (B, H, W2)
        if zero_dc:
            A_hw = A_hw.clone(); A_hw[:, 0, 0] = 0.0
        A_map = A_hw.unsqueeze(1).expand(B, C, H, W2)               # (B,C,H,W2)
        z  = torch.randn(B, C, H, W2, dtype=torch.complex64, device=device)
        Yk = A_map * z
        eta = torch.fft.irfft2(Yk, s=(H, W), norm="ortho").real
        return self._whiten_per_batch(eta)

    def _noise_core(self, epsn: torch.Tensor) -> torch.Tensor:
        # diag: white; conv: depthwise separable coloring; always remove DC & whiten
        if self.sigma_mode == "diag":
            eta = epsn
        elif self.sigma_mode == "conv":
            eta = self.noise_pw(self.noise_dw(epsn))
        else:
            raise RuntimeError("Use _synth_from_bins for spectral mode.")
        return self._whiten_per_batch(eta)

    def _cap_alpha(self, a: torch.Tensor) -> torch.Tensor:
        if self.clamp_alphas is not None:
            a = torch.tanh(a) * float(self.clamp_alphas)
        if self.alpha_vec_cap is not None:
            vnorm = a.norm(dim=1, keepdim=True).clamp(min=1e-6)
            a = a * (float(self.alpha_vec_cap) / vnorm).clamp(max=1.0)
        return a

    def _assemble(self, alphas: torch.Tensor, device) -> torch.Tensor:
        # expects 8 channels
        if alphas.shape[1] != 8:
            raise ValueError(f"_assemble expects 8 channels, got {alphas.shape[1]}")
        T = self.lambdas.to(device=device)                           # [8,3,3] complex64
        S = torch.einsum("bchw,aij->bhwij", alphas.to(T.dtype), T)   # Hermitian
        if self.A_cap is not None:
            S_f = (S.real.square().sum(dim=(-2,-1)) + 1e-12).sqrt()
            S = S * (float(self.A_cap) / S_f).unsqueeze(-1).unsqueeze(-1).clamp(max=1.0)
        A = 1j * S                                                   # anti-Hermitian, traceless
        return A

    # ---------------- forward ----------------
    def forward(self,
                h: torch.Tensor,
                base18: torch.Tensor,
                Ymap: torch.Tensor,
                *,
                nsamples: int = 1,                 # (kept for API parity; not looped here)
                sample: bool | None = None,
                dY: torch.Tensor | None = None,
                detach_sigma_in_spec: bool = False,
                return_eta: bool = True,
                eta_stride: int = 1):

        if sample is None:
            sample = False

        B, _, H, W = base18.shape
        device = h.device
        dtype  = h.dtype

        # # Heads
        # mu        = self.proj_mu(h)               # [B,C,H,W]
        # logsig    = self.proj_logs(h)             # [B,C,H,W]
        # logA_bins = self.spec_head(h).float()     # [B, spec_bins] (always float for FFT path)


        # Heads
        mu_raw    = self.proj_mu(h)               # [B,C,H,W]

        # ---- make μ block-constant on 3×3 patches ----
        B, C, H, W = mu_raw.shape
        block = 3

        # downsample to block grid: [B,C,Hb,Wb], Hb≈H/3, Wb≈W/3
        mu_block = F.avg_pool2d(mu_raw, kernel_size=block, stride=block)

        # upsample back to full resolution (nearest or repeat)
        mu = F.interpolate(
            mu_block,
            size=(H, W),
            mode="nearest"
        )  # [B,C,H,W]

        # or, if you prefer exact 3×3 tiling:
        # mu = mu_block.repeat_interleave(block, dim=2).repeat_interleave(block, dim=3)
        # mu = mu[:, :, :H, :W]

        logsig    = self.proj_logs(h)             # [B,C,H,W]
        logA_bins = self.spec_head(h).float()     # [B, spec_bins]


        # Map logsig -> σ in [sigma_min, sigma_max]
        sigma_min, sigma_max = 1e-4, 1
        sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(logsig)

        # Build ΔY map and channel-broadcast
        # make dY in the right shape to multiply different objects
        if dY is None:
            dYt = Ymap.to(device=device, dtype=dtype)
        else:
            dYt = dY.to(device=device, dtype=dtype)
            if dYt.dim() <= 2:
                dYt = dYt.view(B, 1, 1, 1).expand(B, 1, H, W)
        dYt = dYt.clamp_min(0)
        C = mu.shape[1]
        dYc = dYt if dYt.shape[1] == C else dYt.expand(B, C, H, W)

        # α step: drift + (optional) stochastic term
        mu_step  = mu * dYc
        amp      = sigma * torch.sqrt(dYc)
        if detach_sigma_in_spec and sample:
            amp = amp.detach()

        eta_core_full = None
        if sample:
            mu_step = mu_step.detach()
            if self.sigma_mode == "spectral":
                #print("SAMPLING spectral")
                eta_core_full = self._synth_from_bins(logA_bins, C, H, W, device)
            else:
                #print("SAMPLING conv")
                epsn_full = torch.randn_like(mu)
                eta_core_full = self._noise_core(epsn_full)
            a_all = mu_step + amp * eta_core_full
        else:
            a_all = mu_step

        # Optionally produce eta_core for loss/metrics at lower res
        if return_eta:
            if eta_stride > 1:
                Hs, Ws = H // eta_stride, W // eta_stride
            else:
                Hs, Ws = H, W

            if self.sigma_mode == "spectral":
                eta_core_loss = self._synth_from_bins(logA_bins, C, Hs, Ws, device)
            else:
                epsn_loss = torch.randn(B, C, Hs, Ws, device=device, dtype=dtype)
                eta_core_loss = self._noise_core(epsn_loss)
        else:
            eta_core_loss = None

        # Assemble SU(3) update from α
        aL, aR = torch.split(a_all, 8, dim=1)  # left/right
        
        aL = self._cap_alpha(aL)
        aR = self._cap_alpha(aR)

        AL = self._assemble(aL, device=device)
        AR = self._assemble(aR, device=device)

        U0 = pack_to_complex(base18.permute(0, 2, 3, 1).to(torch.float32))
        GL = torch.linalg.matrix_exp(1.0 * AL)
        GR = torch.linalg.matrix_exp(-1.0 * AR)

        U = GL @ U0
        U = U @ GR

        self.last_A_fro_mean = (AL.abs().square().sum(dim=(-2,-1)).sqrt().mean()).detach()
        self.last_sigma_mean = sigma.detach().mean()

        out18 = unpack_to_18(U).permute(0, 3, 1, 2).to(dtype)

        # identity snap at very small |Y|
        if self.identity_eps > 0.0:
            y_abs0 = Ymap[:, 0, 0, 0].abs()
            if (y_abs0 <= self.identity_eps).any():
                mask = (y_abs0 <= self.identity_eps)
                out18 = out18.clone()
                out18[mask] = base18[mask]

        extras = {
            "mu": mu, "logsig": logsig, "sigma": sigma,
            "logA_bins": logA_bins, "dY": dYt, "alpha_step": a_all.detach(),
        }
        if return_eta and eta_core_loss is not None:
            extras["eta_core"] = eta_core_loss

        return out18, extras

#----------------------- Model -------------------------

# Simple channel-wise LayerNorm for NCHW (no spatial coupling)
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> LN over C only
        b, c, h, w = x.shape
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

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

    # For each frequency bin (x, y) and each output channel o, it takes the vector of all input 
    # channels at that bin and multiplies by a learned complex weight vector, then sums:
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
        # Think: look at the whole field at once. It FFTs the input, keeps only the lowest modes1×modes2 frequencies,
        # multiplies them by learned complex weights, then iFFTs back. Because FFT sees the whole image, 
        # this gives global receptive field with relatively few parameters.
        self.w        = nn.Conv2d(width, width, kernel_size=1)
        # Think: tiny local correction. A 1×1 conv just mixes channels at each pixel (no spatial neighborhood). 
        # It preserves high-freq details that might be lost when you truncate Fourier modes and gives an easy linear path for gradients.
        # Consider removing this if the per pixel fluctuations become too large.
        self.act      = nn.GELU()
        # Nonlinearity so the block can represent more than a linear transform. Nonlinearity makes depth useful.
        # "stick a nonlinear activation after a linear op so deeper stacks can learn truly complex mappings."
    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        return self.act(self.spectral(x) + self.w(x))

class EvolverFNO(nn.Module):
    def __init__(self,
                 in_ch=22, width=64, modes1=12, modes2=12, n_blocks=4,
                 identity_eps: float = 0.0,
                 clamp_alphas=None, 
                 alpha_vec_cap=15.0,
                 # conditioning
                 y_index: int = 18,
                 film_mode: str = "scale_only",
                 rbf_K: int = 12,
                 film_hidden: int = 64,
                 gamma_scale: float = 1.5,
                 gate_temp: float = 2.0,
                 y_map: str = "linear",
                 y_min: float | None = None,
                 y_max: float | None = None,
                 theta_min: Optional[Sequence[float]] = None,
                 theta_max: Optional[Sequence[float]] = None,
                 rbf_gamma: float = 1.0,
                 spec_bins: int = 48,
                 rbf_min_width: float = 0.,
                 sigma_mode: str = "conv",
                 channels_last: bool = True):

        super().__init__()
        self.width = int(width)
        self.y_index = int(y_index)
        self.film_mode = str(film_mode)
        self.rbf_K = int(rbf_K)
        self.film_hidden = int(film_hidden)
        self.gamma_scale = float(gamma_scale)
        self.gate_temp = float(gate_temp)
        self.y_map = y_map
        self.y_min = y_min
        self.y_max = y_max
        self.spec_bins = spec_bins  
        self.sigma_mode = sigma_mode
        self.channels_last = bool(channels_last)    

        # --- trunk ---
        self.lift = nn.Conv2d(18, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(n_blocks)])
        self.block_norm = LayerNorm2d(width)  # pre-norm (channel-only), more stable than GN(1)
        # Pre-norm tends to make deep residual stacks (like our FNO blocks with FiLM) more stable and easier to optimize.
               
        # --- SU(3) head ---
        self.head = SU3HeadGellMannStochastic(
            width,
            identity_eps=identity_eps,
            clamp_alphas=clamp_alphas,
            alpha_vec_cap=alpha_vec_cap,
            sigma_mode=self.sigma_mode,
            spec_bins=self.spec_bins
        )

        # --- Y & θ embeddings (RBF) ---
        self.time_embed  = RBFEmbed(K=rbf_K, learnable=True, init_sigma=0.20)
        self.theta_embed = RBFEmbed(K=rbf_K, learnable=True, init_sigma=0.20)

        # small “pre-FiLM” MLP to clean embeddings
        self.pre_film = nn.Sequential(
            nn.Linear(4 * self.rbf_K, self.film_hidden), 
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
            gamma_scale=self.gamma_scale,
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

        self.register_buffer("theta_min_buf", torch.zeros(3), persistent=False)
        self.register_buffer("theta_max_buf", torch.ones(3),  persistent=False)
        if theta_min is not None:
            self.theta_min_buf = torch.tensor(theta_min, dtype=torch.float32)
        if theta_max is not None:
            self.theta_max_buf = torch.tensor(theta_max, dtype=torch.float32)

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

    # --- helpers ---
    def _theta01_from_vector(self, theta: torch.Tensor) -> torch.Tensor:
        """
        theta: [B,3] containing (m, Lambda_QCD, mu0).
        Returns values scaled to [0,1] using the buffers.
        """
        tmin = self.theta_min_buf.to(theta.device, theta.dtype)
        tmax = self.theta_max_buf.to(theta.device, theta.dtype)
        denom = (tmax - tmin).clamp_min(1e-9)
        theta01 = (theta - tmin) / denom
        return theta01.clamp(0.0, 1.0)


    # --- trunk encoding (base18 + scalars Y,θ) ---
    def encode_trunk_from_components(
        self,
        base18: torch.Tensor,
        Y_scalar: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        # memory format nicety for CUDA
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
        theta01 = self._theta01_from_vector(theta.to(h.device, h.dtype))  # [B,3] in [0,1]
        eT = self.theta_embed(theta01)            # broadcasts → [B,3,K] if K center
        if eT.dim() == 1:
            eT = eT.unsqueeze(-1)
        if eT.dim() > 2:
            eT = eT.view(B, -1)

        # Only θ conditions FiLM now (Y is part of the trunk input)

        # FiLM conditions on Y and θ together (no Y tiling)
        if eY.dim() > 2: eY = eY.view(B, -1)     # safety, usually already [B,K]
        if eT.dim() > 2: eT = eT.view(B, -1)     # [B, 3K]
        h_cond_in = torch.cat([eY.view(B, -1), eT], dim=1)   # [B, 4K]
        h_cond    = self.pre_film(h_cond_in)                 # ✅ [B, film_hidden] (64)
        cond      = self.time_cond(h_cond)                   # OK

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
            if (beta is not None):
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
                      detach_sigma_in_spec: bool = False,
                      dY=None) -> torch.Tensor:
       
        if isinstance(base18, (tuple, list)):
            base18 = base18[0]
        assert isinstance(base18, torch.Tensor), f"expected tensor, got {type(base18)}"

        # 1) FiLM-conditioned trunk with Y & θ (no Y channel concatenation)
        h = self.encode_trunk_from_components(base18, Y_scalar, theta)

        # 2) Head uses ΔY as a scalar (broadcasted inside), so no H×W Y tiling
        B, _, _, _ = base18.shape
        Y_placeholder = Y_scalar.to(base18.device, base18.dtype).view(B, 1, 1, 1)
        return self.head(
            h, base18, Y_placeholder,
            nsamples=nsamples,
            sample=sample,
            dY=Y_scalar,                               # <— scalar [B], gets broadcast internally
            detach_sigma_in_spec=detach_sigma_in_spec,
        )


def mu_smoothness_loss(mu: torch.Tensor) -> torch.Tensor:
    # mu: [B, C, H, W]
    dx = mu[:, :, 1:, :] - mu[:, :, :-1, :]   # differences along H
    dy = mu[:, :, :, 1:] - mu[:, :, :, :-1]   # differences along W
    return (dx**2).mean() + (dy**2).mean()


class LossFunction(nn.Module):
    """
    Extended loss function
    """
    _RADIAL_CACHE = {}

    def __init__(
        self,
        *,
        dipole_weight: float = 0.0,
        # new Q_s controls
        qs_weight: float = 0.0,
        qs_threshold: float = 0.5,
        qs_on: str = "N",
        qs_local: bool = False,
        qs_scale: float = 1.0,
        shape_weight: float = 0.0,         # bin-to-bin S(r) shape
        spec_weight: float = 0.0,          # spatial spectrum (center field)
        spec_alpha: float = 1.0,           # ↑ weight for high-k (0..3 typical)
        spec_log: bool = True,             # match in log power
        tempo_weight: float = 0.0,         # spectrum of increments over ΔY
        fluct_weight: float = 0.0,         # turn on structure-function loss
        fluct_alpha: float = 1.0,          # how much to emphasize the smallest offsets
        patch_var_weight: float =0.0
    ):
        super().__init__()

        self.dipole_weight = float(dipole_weight)
        self.qs_weight = float(qs_weight)
        self.qs_threshold = float(qs_threshold)
        self.qs_on = str(qs_on).upper()  # 'N' or 'S'
        assert self.qs_on in ("N", "S"), "qs_on must be 'N' or 'S'"
        self.qs_local = bool(qs_local)
        self.qs_scale = float(qs_scale)
        self.shape_weight = float(shape_weight)
        self.spec_weight  = float(spec_weight)
        self.spec_alpha   = float(spec_alpha)
        self.spec_log     = bool(spec_log)
        self.tempo_weight = float(tempo_weight)
        self.fluct_weight = float(fluct_weight)
        self.fluct_alpha  = float(fluct_alpha)
        self.patch_var_weight = float(patch_var_weight)
        # --- Gell-Mann (λ/2) basis for projection (complex Hermitian) ---
        self.register_buffer("lambdas", su3_gellmann_matrices(), persistent=False)

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

    def _components(self, yhat: torch.Tensor, y: torch.Tensor):
        # yhat, y: [B,18,H,W]
        Uh_raw = self._pack18_to_U(yhat)
        U_raw  = self._pack18_to_U(y)
        Uh = self._su3_project(Uh_raw)
        U  = self._su3_project(U_raw)
        return Uh, U

    @classmethod
    def _radial_bins(cls, H: int, W: int, device):
        """
        Returns:
        bins   : [H,W] int64, radial bin index for each pixel (rounded radius)
        counts : [L]   int64, number of pixels in each bin
        rvals  : [L]   float32, mean radius per bin
        """
        key = (H, W, device)
        cache = cls._RADIAL_CACHE
        if key in cache:
            return cache[key]

        dx = torch.arange(W, device=device)
        dy = torch.arange(H, device=device)
        dx = torch.minimum(dx, W - dx)[None, :].expand(H, W)
        dy = torch.minimum(dy, H - dy)[:, None].expand(H, W)

        rmap = torch.sqrt(dx.to(torch.float32)**2 + dy.to(torch.float32)**2)  # [H,W]
        bins = torch.round(rmap).to(torch.int64)
        L = int(round(math.hypot(W // 2, H // 2))) + 1

        counts = torch.bincount(bins.view(-1), minlength=L).to(torch.int64)

        # mean radius per bin (nicer r-axis than 0,1,2,...)
        r_sums = torch.zeros(L, device=device, dtype=rmap.dtype)
        r_sums.index_add_(0, bins.view(-1), rmap.view(-1))
        rvals = r_sums / counts.clamp_min(1).to(rmap.dtype)

        cache[key] = (bins, counts, rvals)
        return cache[key]


    def _isotropic_curve(self, U: torch.Tensor, *, needs_grad: bool, assume_su3: bool = True, drop_r0: bool = True):
        """
        Radially averaged dipole S(r) via FFT autocorrelation (isotropic).
        Returns:
        r : [K]       mean radius per bin (excluding r=0)
        S : [B, K]    dipole vs radius (isotropic)
        """
        if not assume_su3:
            U = self._su3_project(U)

        B, H, W = U.shape[:3]
        device = U.device
        bins, counts, rvals = self._radial_bins(H, W, device)
        L = counts.numel()
        bin_ids = bins.view(-1)

        # Features: [B,9,H,W]
        feats = U.reshape(B, H, W, 9).permute(0, 3, 1, 2).contiguous()

        ctx = (torch.enable_grad() if needs_grad else torch.no_grad())
        with ctx:
            Fk   = torch.fft.fft2(feats)
            corr = torch.fft.ifft2(Fk.conj() * Fk).real          # real tensor (float)
            corr = (corr.sum(dim=1) / (3.0 * H * W)).view(B, -1) # [B, HW], real

            # IMPORTANT: match dtype to corr (real), not feats (complex)
            sums = torch.zeros(B, L, device=device, dtype=corr.dtype)
            sums.index_add_(1, bin_ids, corr)

            S_rad = sums / counts.clamp_min(1).to(corr.dtype)    # [B, L]

        # drop r=0 bin
        if drop_r0:
            r = rvals[1:]
            S = S_rad[:, 1:]
        else:
            r = rvals
            S = S_rad

        return r, S


    def _isotropic_dipole_loss(self, Uh: torch.Tensor, U: torch.Tensor,
                            *, local: bool = False, use_logN: bool = False,
                            per_radius_norm: bool = True, detach_target: bool = True,
                            dS_bias: torch.Tensor | None = None) -> torch.Tensor:
        del local  # isotropic by definition (no per-offset local curves)
        assert Uh.dtype.is_complex and U.dtype.is_complex

        # predicted curve WITH grads; target curve optionally detached
        r_pred, S_pred = self._isotropic_curve(Uh, needs_grad=True,  assume_su3=True)
        _,     S_true  = self._isotropic_curve(U,  needs_grad=not detach_target, assume_su3=True)

        if dS_bias is not None:
            # compensate for sampling-induced drop in S due to learned fluctuations
            S_pred = S_pred + dS_bias.to(S_pred.dtype)

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


    def _compute_Qs_from_U(self, U: torch.Tensor, *, local: bool):
        """
        Q_s from the *isotropic* dipole S(r). `local` is ignored (kept for API compatibility).
        """
        _ = local
        r, S = self._isotropic_curve(U, needs_grad=False, assume_su3=True)  # r:[K], S:[B,K]

        X = 1.0 - S if self.qs_on == "N" else S
        thr = self.qs_threshold

        # optional soft selection
        if getattr(self, "qs_soft_beta", 0.0) > 0.0:
            dX = X.diff(dim=-1, prepend=X[..., :1]).abs()
            scores = -self.qs_soft_beta * (X - thr).abs()
            if getattr(self, "qs_soft_slope", 1.0) != 0.0:
                eps = 1e-12
                scores = scores + self.qs_soft_slope * torch.log(dX.clamp_min(eps))
                w = torch.softmax(scores, dim=-1)
                r_star = (w * r).sum(dim=-1)
                return 1.0 / r_star.clamp_min(eps)

        # hard crossing with linear interpolation
        X_mono, _ = torch.cummax(X, dim=-1)
        target = torch.tensor(thr, device=X.device, dtype=X.dtype)

        below = (X_mono < target).to(torch.int64)
        idx_hi = below.sum(dim=-1)                 # [B]
        K = X_mono.shape[-1]
        idx_hi = idx_hi.clamp(min=0, max=K-1)
        idx_lo = (idx_hi - 1).clamp(min=0, max=K-1)

        r_b = r.view(*((1,) * (X_mono.ndim - 1)), -1).expand_as(X_mono)

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

    def _center_scalar(self, U: torch.Tensor) -> torch.Tensor:
        """
        Map an SU(3) Wilson line to a single real scalar per pixel: the (real) trace / 3.

        Args:
            U: Complex Wilson lines with shape [B, H, W, 3, 3].

        Returns:
            c: Real-valued "center field" with shape [B, 1, H, W].
            This is the scalar you then analyze spatially/spectrally.
        """
        # Diagonal elements have shape [B, H, W, 3]; sum over color to get trace
        # Take real part (trace should be complex in general; we use its real part)
        c = U.diagonal(dim1=-2, dim2=-1).sum(-1).real / 3.0  # [B, H, W]
        return c.unsqueeze(1)  # [B, 1, H, W]


    def _rfft2_power(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D real-to-complex FFT and power spectrum (squared modulus).

        Args:
            x: Real field [B, 1, H, W].

        Returns:
            P: Power spectrum |X(k)|^2 with shape [B, 1, H, W//2 + 1].
            rfft2 keeps non-negative frequencies along x, full along y.
        """
        X = torch.fft.rfft2(x, norm="ortho")    # complex tensor in frequency domain
        P = (X.real**2 + X.imag**2)             # squared magnitude
        return P


    def _k_radial_bins(self, H: int, W: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build radial bins in k-space that match the layout of rfft2(H, W).

        Returns:
            bins:   [H, W//2+1] int64  -> radial bin index at each k = (ky, kx)
            counts: [L] int64          -> number of (ky, kx) points in each radial bin
            kvals:  [L] float          -> mean |k| per bin (for plotting/weighting)
        """
        # Frequency coordinates: ky covers negative..positive; kx only non-negative for rfft
        wy = torch.fft.fftfreq(H, d=1.0, device=device)      # shape [H], in cycles/pixel
        wx = torch.fft.rfftfreq(W, d=1.0, device=device)     # shape [W//2+1]

        # Make broadcastable 2D grids of ky and kx
        ky = wy.view(H, 1).expand(H, wx.numel())             # [H, W//2+1]
        kx = wx.view(1, -1).expand(H, wx.numel())            # [H, W//2+1]

        # Radial frequency magnitude
        kr = torch.sqrt(kx**2 + ky**2)                       # [H, W//2+1]

        # Normalize by the maximum radius to make binning resolution-independent
        krn  = kr / (kr.max().clamp_min(1e-12))
        # Choose an integer number of rings, approximately up to Nyquist radius
        bins = torch.round(krn * (min(H, W)//2)).to(torch.int64)  # [H, W//2+1]
        L    = int(bins.max().item()) + 1

        # How many points in each ring
        counts = torch.bincount(bins.view(-1), minlength=L)       # [L]

        # Mean |k| in each ring (used as the k-axis for 1D PSD)
        ksum = torch.zeros(L, device=device, dtype=kr.dtype)
        ksum.index_add_(0, bins.view(-1), kr.view(-1))
        kvals = ksum / counts.clamp_min(1).to(kr.dtype)           # [L]
        return bins, counts, kvals


    def _psd1d_center(self, U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the 1D (radially averaged) power spectral density (PSD) of the center field.

        Steps:
        1) convert U -> scalar center field (real)
        2) remove spatial DC component (mean)
        3) take rfft2 power
        4) average power in radial rings in k-space

        Args:
            U: Complex Wilson lines [B, H, W, 3, 3].

        Returns:
            k:    [L]  mean |k| per radial bin
            P1d:  [B, L] radially averaged PSD for each sample in the batch
        """
        B, H, W = U.shape[:3]

        # Center scalar (trace/3), remove DC to keep pure fluctuation spectrum
        x = self._center_scalar(U)                               # [B, 1, H, W], real
        x = x - x.mean(dim=(2, 3), keepdim=True)                 # zero-mean per sample

        # 2D FFT power
        P = self._rfft2_power(x)                                 # [B, 1, H, W//2+1]

        # Build radial bins in k-space
        bins, counts, kvals = self._k_radial_bins(H, W, device=U.device)
        L = counts.numel()

        # Sum power per radial bin across the (ky, kx) grid
        # Reshape P so that spatial frequency axes are flattened per sample
        P_flat = P.view(B, -1)                                   # [B, H*(W//2+1)]

        # Accumulate into [B, L]
        sums = torch.zeros(B, L, device=U.device, dtype=P.dtype)
        sums.index_add_(1, bins.view(-1), P_flat)                # bin power by ring

        # Normalize by number of points per ring
        P1d = sums / counts.clamp_min(1).to(P.dtype)             # [B, L]
        return kvals, P1d


    def _shape_loss_from_S(
        self,
        r_pred: torch.Tensor,
        S_pred: torch.Tensor,
        S_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage correct *shape* of S(r): match slope and curvature across r-bins,
        with higher weight at small r.

        Args:
            r_pred: [B, K] or [K] radial grid.
            S_pred, S_true: [B, K] isotropic dipole curves (r already aligned).

        Returns:
            Scalar loss combining first- and second-difference SmoothL1 errors.
        """
        # Discrete slope (first differences) along r
        D1p = S_pred.diff(dim=-1)    # [B, K-1]
        D1t = S_true.diff(dim=-1)    # [B, K-1]

        # Discrete curvature (second differences)
        D2p = D1p.diff(dim=-1)       # [B, K-2]
        D2t = D1t.diff(dim=-1)       # [B, K-2]

        # ----- build radial weights: small r -> larger weight -----
        # Get a 1D r-grid [K]
        if r_pred.dim() == 2:
            # assume all batch elements share the same grid; average to be safe
            r = r_pred.mean(dim=0)   # [K]
        else:
            r = r_pred               # [K]

        eps = 1e-6
        # Example: w(r) ~ 1 / (r + eps)  (you can make this stronger with 1/(r**2+eps) etc.)
        w_r = 1.0 / (r + eps)        # [K]

        # Map to bins for D1 (between r_i and r_{i+1}) and D2 (around r_{i+1})
        # You could also use averages like 0.5*(w_r[i] + w_r[i+1]); this keeps it simple.
        w1 = w_r[:-1]                # [K-1]  for first differences
        w2 = w_r[1:-1]               # [K-2]  for second differences

        # Broadcast over batch
        w1 = w1.view(1, -1)          # [1, K-1]
        w2 = w2.view(1, -1)          # [1, K-2]

        # ----- elementwise SmoothL1, then weighted mean over r, then mean over batch -----
        l1_elem = F.smooth_l1_loss(D1p, D1t, beta=0.02, reduction="none")  # [B, K-1]
        l2_elem = F.smooth_l1_loss(D2p, D2t, beta=0.02, reduction="none")  # [B, K-2]

        # weighted mean over r-bins
        L1_per_batch = (l1_elem * w1).sum(dim=-1) / w1.sum()   # [B]
        L2_per_batch = (l2_elem * w2).sum(dim=-1) / w2.sum()   # [B]

        L1 = L1_per_batch.mean()
        L2 = L2_per_batch.mean()

        return 0.5 * L1 + 0.5 * L2


    def _spectrum_loss(self, U_pred: torch.Tensor, U_true: torch.Tensor) -> torch.Tensor:
        """
        Match the *shape* of the radial PSD of the center field (spatial spectrum).

        - We normalize each spectrum to unit area so the loss compares shapes, not amplitudes.
        - Optional log transform (self.spec_log) to emphasize multiplicative differences.
        - Optional high-k emphasis via exponent self.spec_alpha.

        Args:
            U_pred, U_true: [B, H, W, 3, 3] complex tensors.

        Returns:
            Scalar SmoothL1 loss between normalized, (optionally log'd) 1D PSDs.
        """
        k, Pp = self._psd1d_center(U_pred)  # k:[L], Pp:[B, L]
        _, Pt = self._psd1d_center(U_true)  # Pt:[B, L]

        # Normalize to probability densities over k (compare shapes)
        eps = 1e-12
        Pp = Pp / (Pp.sum(dim=-1, keepdim=True) + eps)
        Pt = Pt / (Pt.sum(dim=-1, keepdim=True) + eps)

        # (Optional) log spectra to de-emphasize big peaks
        if getattr(self, "spec_log", False):
            Pp = torch.log(Pp.clamp_min(1e-10))
            Pt = torch.log(Pt.clamp_min(1e-10))

        # Weight higher k if desired (spec_alpha >= 0)
        alpha = float(getattr(self, "spec_alpha", 0.0))
        w = (k / k.max().clamp_min(1e-12)) ** alpha  # [L]
        w = w.to(Pp.dtype).unsqueeze(0)              # [1, L] broadcast across batch

        return F.smooth_l1_loss(Pp * w, Pt * w, beta=0.02)


    def _temporal_spectrum_loss(self,
                                U_base: torch.Tensor,
                                U_pred: torch.Tensor,
                                U_true: torch.Tensor) -> torch.Tensor:
        """
        Match the spectrum of *temporal increments* over ΔY, i.e., the change from the base state.

        We:
        - project each Wilson line to a scalar center field,
        - compute increments db = x_pred - x_base and dt = x_true - x_base,
        - compare their (normalized) 2D rFFT power spectra with optional high-k emphasis.

        Args:
            U_base: [B, H, W, 3, 3] (state at Y_a)
            U_pred: [B, H, W, 3, 3] (model output at Y_c)
            U_true: [B, H, W, 3, 3] (target at Y_c)

        Returns:
            Scalar SmoothL1 loss between normalized increment spectra.
        """
        # Convert to scalar center fields
        xb = self._center_scalar(U_base)  # [B, 1, H, W]
        xp = self._center_scalar(U_pred)  # [B, 1, H, W]
        xt = self._center_scalar(U_true)  # [B, 1, H, W]

        # Temporal increments over ΔY
        db = xp - xb                      # model's increment
        dt = xt - xb                      # true increment

        # 2D power spectra
        Pp = self._rfft2_power(db)        # [B, 1, H, W//2+1]
        Pt = self._rfft2_power(dt)        # [B, 1, H, W//2+1]

        # Normalize per-sample so we compare shapes not total energy
        eps = 1e-12
        Pp = Pp / (Pp.sum(dim=(2, 3), keepdim=True) + eps)
        Pt = Pt / (Pt.sum(dim=(2, 3), keepdim=True) + eps)

        # (Optional) log spectra
        if getattr(self, "spec_log", False):
            Pp = torch.log(Pp.clamp_min(1e-10))
            Pt = torch.log(Pt.clamp_min(1e-10))

        # Frequency weighting: emphasize high-|k| if self.spec_alpha > 0
        H, Wr = Pp.shape[-2:]                              # Wr = W//2+1
        # Build ky, kx grids consistent with rFFT layout
        ky = torch.fft.fftfreq(H, d=1.0, device=Pp.device).view(1, 1, H, 1).abs()
        kx = torch.fft.rfftfreq(2 * Wr - 2, d=1.0, device=Pp.device).view(1, 1, 1, Wr).abs()
        kr = torch.sqrt(kx**2 + ky**2)
        alpha = float(getattr(self, "spec_alpha", 0.0))
        w = (kr / kr.max().clamp_min(1e-12)) ** alpha

        return F.smooth_l1_loss(Pp * w, Pt * w, beta=0.02)


    def _high_k_power(self, x_bchw: torch.Tensor, frac: float = 0.5, norm: str = "ortho") -> torch.Tensor:
        """
        Gentle anti-jitter regularizer: penalize energy at high spatial frequencies.

        Args:
            x_bchw: Real map(s) [B, C, H, W] — typically the raw 18 channels before packing.
            frac:   Keep fraction of lowest |k|; penalize the complement (top 1-frac).
                    e.g., frac=0.5 keeps the lowest half of the radial spectrum.
            norm:   FFT normalization (use "ortho" for scale stability).

        Returns:
            Scalar mean power in the masked high-|k| region.
            (If you want scale-invariance, divide by total power first.)
        """
        B, C, H, W = x_bchw.shape

        # Complex 2D FFT per channel
        X = torch.fft.fft2(x_bchw, norm=norm)  # [B, C, H, W], complex

        # Build radial |k| grid (full FFT layout for both axes here)
        fx = torch.fft.fftfreq(W, d=1.0, device=x_bchw.device).view(1, 1, 1, W)  # [-0.5..0.5)
        fy = torch.fft.fftfreq(H, d=1.0, device=x_bchw.device).view(1, 1, H, 1)
        kx = fx.expand(B, C, H, W)
        ky = fy.expand(B, C, H, W)
        kr = torch.sqrt(kx * kx + ky * ky)

        # Radial cutoff: penalize only frequencies with |k| >= kcut
        kmax = kr.max().clamp_min(1e-8)
        kcut = frac * kmax
        mask = (kr >= kcut).to(x_bchw.dtype)

        # Mean masked power across batch, channels, and spatial frequencies
        power = (X.real**2 + X.imag**2)
        return (mask * power).mean()

    def _structure_function_loss(self, Uh: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Match D(Δ)=E[(C(x+Δ)-C(x))^2] over a few small offsets Δ.
        This suppresses pixel-level jitter and sets the short-range correlation.
        """
        Cp = self._center_scalar(Uh)  # [B,1,H,W], real
        Ct = self._center_scalar(U)   # [B,1,H,W]

        # remove DC so D reflects fluctuations (not mean)
        Cp = Cp - Cp.mean(dim=(2,3), keepdim=True)
        Ct = Ct - Ct.mean(dim=(2,3), keepdim=True)

        B,_,H,W = Cp.shape

        # small set of offsets; you can tweak, but keep it tiny
        offsets = [(0,1), (1,0), (1,1), (2,0)]  # favor shortest separations

        Dp_list, Dt_list, w_list = [], [], []
        for dy, dx in offsets:
            Cp_s = torch.roll(Cp, shifts=(dy, dx), dims=(2,3))
            Ct_s = torch.roll(Ct, shifts=(dy, dx), dims=(2,3))

            # Structure function at this Δ: average over space (and channel)
            Dp = ((Cp_s - Cp) ** 2).mean(dim=(1,2,3))  # [B]
            Dt = ((Ct_s - Ct) ** 2).mean(dim=(1,2,3))  # [B]
            Dp_list.append(Dp)
            Dt_list.append(Dt)

            # Heavier weight for smaller |Δ| to hit jitter hardest
            dist = (dy**2 + dx**2) ** 0.5
            w = math.exp(-self.fluct_alpha * max(0.0, dist - 1.0))
            w_list.append(w)

        Dp = torch.stack(Dp_list, dim=-1)  # [B, M]
        Dt = torch.stack(Dt_list, dim=-1)  # [B, M]
        w  = torch.tensor(w_list, device=Dp.device, dtype=Dp.dtype)  # [M]
        w  = w / (w.mean() + 1e-12)

        # SmoothL1 on weighted structure functions
        return F.smooth_l1_loss(Dp * w, Dt * w, beta=0.02)

    def _patch_var_loss(self, Uh, U, patch=3):
        # Uh, U: [B,H,W,3,3] complex
        Cp = self._center_scalar(Uh)  # [B,1,H,W]
        Ct = self._center_scalar(U)   # [B,1,H,W]

        B, _, H, W = Cp.shape

        # unfold into patches: [B, C*patch*patch, num_patches]
        Cp_patches = F.unfold(Cp, kernel_size=patch, stride=1, padding=patch//2)
        Ct_patches = F.unfold(Ct, kernel_size=patch, stride=1, padding=patch//2)
        # shapes: [B, patch², N]

        # variance across the patch dimension (dim=1)
        Cp_mean = Cp_patches.mean(dim=1, keepdim=True)
        Ct_mean = Ct_patches.mean(dim=1, keepdim=True)

        Cp_var = ((Cp_patches - Cp_mean) ** 2).mean(dim=1)  # [B, N]
        Ct_var = ((Ct_patches - Ct_mean) ** 2).mean(dim=1)  # [B, N]

        # match patch-wise variances (SmoothL1 or MSE)
        return F.smooth_l1_loss(Cp_var, Ct_var, beta=0.02)

    # ===== main loss =====
    def forward(self, yhat: torch.Tensor, y: torch.Tensor,
                base18: torch.Tensor | None = None
                ):

        Uh, U = self._components(yhat, y)

        # 2) Base loss (same weights you use in the base class)
        total = 0.0 
        stats = {}

        # ---- Q_s loss (global) ----
        if self.qs_weight != 0.0:
            # Q_s from the same isotropic S(r) used in validation
            Qh = self._compute_Qs_from_U(Uh, local=self.qs_local)   # [B]
            Qt = self._compute_Qs_from_U(U,  local=self.qs_local).detach()  # [B], no grads through target

            # robust, scale-stable loss (log-space SmoothL1); switch to MSE if you prefer
            eps = 1e-6
            L_qs = F.smooth_l1_loss(torch.log(Qh.clamp_min(eps)),
                                    torch.log(Qt.clamp_min(eps)),
                                    beta=0.02)

            total = total + self.qs_weight * L_qs

            # stats
            qs_mse = F.mse_loss(Qh, Qt)
            qs_rel = torch.mean(torch.abs(Qh - Qt) / (torch.abs(Qt) + eps))
            stats["qs_loss"]   = L_qs.detach()
            stats["qs_mse"]    = qs_mse.detach()
            stats["qs_rel"]    = qs_rel.detach()
            stats["qs_pred"]   = Qh.detach().mean()
            stats["qs_true"]   = Qt.detach().mean()


            #----time----
            #if torch.cuda.is_available(): torch.cuda.synchronize();
            #t0=time.perf_counter()
            #----end-time----

        if self.dipole_weight != 0.0:
            dS_bias = None
            dip = self._isotropic_dipole_loss(
                Uh, U,
                local=False,
                use_logN=False,
                per_radius_norm=True,
                detach_target=True,
                dS_bias=dS_bias
            )
            total = total + self.dipole_weight * dip
     
            diplog = self._isotropic_dipole_loss(
                Uh, U,
                local=False,
                use_logN=True,
                per_radius_norm=False,
                detach_target=True,
                dS_bias=dS_bias
            )
            total = total + self.dipole_weight * diplog 
            stats["dipole_loss"] = diplog.detach() + dip.detach()
   
            #----time----
            #if torch.cuda.is_available(): torch.cuda.synchronize()
            #print(f"[timing] Dipole block: {(time.perf_counter()-t0)*1e3:.2f} ms")
            #----end-time----

        # --- (A) bin-to-bin S(r) shape ---
        if self.shape_weight != 0.0:
            r_pred, S_pred = self._isotropic_curve(Uh, needs_grad=True,  assume_su3=True)
            _,      S_true = self._isotropic_curve(U,  needs_grad=False, assume_su3=True)
            L_shape = self._shape_loss_from_S(r_pred, S_pred, S_true)
            total = total + self.shape_weight * L_shape
            stats["shape_loss"] = L_shape.detach()

        # --- (B) spatial PSD (center field) ---
        if self.spec_weight != 0.0:
            L_spec = self._spectrum_loss(Uh, U)
            total = total + self.spec_weight * L_spec
            stats["spec_loss"] = L_spec.detach()

        # --- (C) temporal PSD of increments (needs base18) ---
        if self.tempo_weight != 0.0 and base18 is not None:
            Ub = self._su3_project(self._pack18_to_U(base18))
            L_temp = self._temporal_spectrum_loss(Ub, Uh, U)
            total = total + self.tempo_weight * L_temp
            stats["tempo_loss"] = L_temp.detach()

        if self.fluct_weight != 0.0:  
            L_fluct = self._structure_function_loss(Uh, U)
            total = total + self.fluct_weight * L_fluct
            stats["fluct_loss"] = L_fluct.detach()

        if self.patch_var_weight != 0.0:  
            L_patch = self._patch_var_loss(Uh, U, patch=3)
            total = total + self.patch_var_weight * L_patch
            stats["patch_var_loss"] = L_patch.detach()


        return total, stats, None


# ---- VALIDATION METRICS: curve_RMSE over full val set ----
def eval_curve_rmse_full(model, criterion, val_dl, device, *, K_samples=0,
                         channels_last=False, device_type="cuda", amp_dtype=torch.float16, use_amp=False):
    model.eval()
    mse_sum_det = torch.zeros((), device=device, dtype=torch.float64)
    mse_cnt_det = 0
    mse_sum_samp = 0.0
    mse_cnt_samp = 0
    r_ref = None  # cache a reference r so weights are consistent across batches

    with torch.no_grad():
        for base18, Y_scalar, theta, y in val_dl:
            base18   = base18.to(device, non_blocking=True)
            Y_scalar = Y_scalar.to(device, dtype=torch.float32, non_blocking=True)
            theta    = theta.to(device, non_blocking=True)
            y        = y.to(device, non_blocking=True)

            if channels_last and device_type == "cuda":
                base18 = base18.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                # --- deterministic path ---
                y_det, _ = model(base18, Y_scalar, theta, sample=False)
                U_det = criterion._pack18_to_U(y_det)   
                U_det = criterion._su3_project(U_det)
                r, S_det = criterion._isotropic_curve(U_det, needs_grad=False, assume_su3=True)

                U_true = criterion._pack18_to_U(y)
                U_true = criterion._su3_project(U_true)
                _, S_true = criterion._isotropic_curve(U_true, needs_grad=False, assume_su3=True)

                N_pred, N_true = 1.0 - S_det, 1.0 - S_true

                err_det = (N_pred - N_true)**2
                mse_sum_det = mse_sum_det + err_det.mean().to(torch.float64)
                mse_cnt_det += 1

                # # --- optional sampling path ---
                # if K_samples and K_samples > 0:
                #     S_sum = 0.0
                #     for _ in range(K_samples):
                #         yk, _ = model(base18, Y_scalar, theta, sample=True, dY=Y_scalar)
                #         Uk = criterion._pack18_to_U(yk)
                #         if not getattr(criterion, "project_before_frob", False):
                #             Uk = criterion._su3_project(Uk)
                #         _, Sk = criterion._isotropic_curve(Uk, needs_grad=False, assume_su3=True)
                #         S_sum = S_sum + Sk
                #     S_samp = S_sum / K_samples

                #     err_samp = (w_r * (S_samp - S_true))**2
                #     mse_sum_samp += err_samp.mean().item()
                #     mse_cnt_samp += 1

    curve_rmse_det_t  = (mse_sum_det / max(1, mse_cnt_det)).sqrt()
    #curve_rmse_samp = ((mse_sum_samp / max(1, mse_cnt_samp))**0.5) if mse_cnt_samp > 0 else None
    return float(curve_rmse_det_t.detach().to('cpu').item()), None

def tv_center_loss(x18_bchw: torch.Tensor, criterion) -> torch.Tensor:
    # x18: [B,18,H,W]
    U = criterion._pack18_to_U(x18_bchw).to(torch.complex64)
    # (optional but usually helps:) project before measuring smoothness
    U = criterion._su3_project(U)  # [B,H,W,3,3] complex
    center = U.diagonal(dim1=-2, dim2=-1).sum(-1).real / 3.0  # [B,H,W]
    center = center.unsqueeze(1)  # [B,1,H,W]
    dx = center - center.roll(-1, 2)
    dy = center - center.roll(-1, 3)
    return (dx.pow(2).mean() + dy.pow(2).mean())

def ramp(epoch, start, duration):
    # linear 0→1 starting at `start` over `duration` epochs
    if epoch < start: return 0.0
    if epoch >= start+duration: return 1.0
    return (epoch - start) / max(1, duration)



#----------------------- Training -------------------------

def train(args):
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

    base_seed = int(getattr(args, "seed", 0))

    # mix in current time (in seconds) – or use time.time_ns() for finer resolution
    time_seed = int(time.time())  # or int(time.time_ns() % 2**31)

    seed = (base_seed + time_seed) % (2**31 - 1)

    # ---- Data ----
    train_dl, val_dl, train_sampler, val_sampler = make_loaders(
        args.data_root, batch_size=args.batch, workers=args.workers,
        seed=seed, ddp=is_ddp
    )
   
    inferred_N = train_dl.dataset.N
    ds_value = train_dl.dataset.ds

    y_values = [float(e["Y"]) for e in train_dl.dataset.entries]
    y_min_data = float(min(y_values)) if y_values else 0.0
    y_max_data = float(max(y_values)) if y_values else 1.0
    if (not is_ddp) or dist.get_rank() == 0:
        print(f"[train] Y range from data: y_min={y_min_data:.6g}, y_max={y_max_data:.6g}")

    def _init_ybins():
        edges = np.linspace(y_min_data, y_max_data, 7, dtype=np.float64)
        return torch.tensor(edges, device=device, dtype=torch.float64)

    theta_rows = np.array([
        [float(e.get("m", 0.0)),
        float(e.get("Lambda_QCD", 0.0)),
        float(e.get("mu0", 0.0))]
        for e in train_dl.dataset.entries
    ], dtype=np.float64)

    #Determine range in data for θ = (m, Lambda_QCD, mu0)
    if theta_rows.size == 0:
        theta_min_data = np.zeros(3, dtype=np.float64)
        theta_max_data = np.ones(3, dtype=np.float64)
    else:
        theta_min_data = theta_rows.min(axis=0)
        theta_max_data = theta_rows.max(axis=0)

    if (not is_ddp) or dist.get_rank() == 0:
        print("[train] θ ranges (m, Lambda_QCD, mu0):")
        print(f"        min = {theta_min_data.tolist()}")
        print(f"        max = {theta_max_data.tolist()}")

    # ---- CUDA specific ----
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
    
    channels_last = getattr(args, "channels_last", True)

    # ---- Model ----
    model = EvolverFNO(
        in_ch=22, width=args.width, modes1=args.modes, modes2=args.modes,
        n_blocks=args.blocks, identity_eps=args.identity_eps,
        clamp_alphas=getattr(args, "clamp_alphas", 2.),
        y_index=18, rbf_K=args.rbf_K,
        film_hidden=args.film_hidden, gamma_scale=args.gamma_scale,
        gate_temp=args.gate_temp,
        y_min=y_min_data, y_max=y_max_data,
        theta_min=theta_min_data.tolist(),
        theta_max=theta_max_data.tolist(),
        y_map=args.y_map,
        spec_bins=args.spec_bins,
        sigma_mode=args.sigma_mode,
        channels_last=channels_last
        ).to(device)

    # Optional NHWC for speed on convs
    if getattr(args, "channels_last", True):
        model = model.to(memory_format=torch.channels_last)

    def unwrap(m):
        return m.module if isinstance(m, DDP) else m

    def _dist_sum_(x: torch.Tensor) -> torch.Tensor:
        if is_ddp:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    core = unwrap(model)

    ema = EMA(core, decay=args.ema_decay) if args.ema_decay and args.ema_decay>0 else None

    # Wrap into DDP for CUDA parallelism
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
    
    ## Test print parameters
    # for n, p in core.named_parameters():
    #     print(f"{n:50s}  {tuple(p.shape)}  requires_grad={p.requires_grad}")

    # ---- Optimizer / Sched / AMP ----
    opt = torch.optim.AdamW(
        [
            {"params": trunk, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": head,  "lr": args.lr * 8.0, "weight_decay": 0.0}
        ],  
        betas=(0.9, 0.98), eps=1e-8, 
        fused=(device_type == "cuda")
    )

    # ---- LR scheduler (configurable) ----
    w_epochs = max(int(args.warmup_epochs), 0)
    main_epochs = max(args.epochs - w_epochs, 1)

    if args.scheduler == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=main_epochs, eta_min=args.min_lr)
    elif args.scheduler == "cosine_wr":
        main = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=args.t0, T_mult=args.tmult, eta_min=args.min_lr)
    elif args.scheduler == "step":
        main = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.scheduler == "multistep":
        milestones = [int(m) for m in args.lr_milestones.split(",") if m.strip()]
        main = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=args.lr_gamma)
    elif args.scheduler == "exponential":
        main = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_gamma)
    elif args.scheduler == "plateau":
        main = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=args.lr_gamma, patience=args.lr_patience, min_lr=args.min_lr, threshold=1e-4
        )
    elif args.scheduler == "poly":
        base_lr = args.lr
        lr_span = max(base_lr - args.min_lr, 0.0)
        def _lambda(e):
            t = min(max(e, 0), main_epochs) / float(max(main_epochs, 1))
            return (args.min_lr + lr_span * (1.0 - t) ** args.poly_power) / base_lr
        main = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lambda)
    elif args.scheduler == "constant":
        main = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0)
    else:
        raise ValueError(f"Unknown scheduler {args.scheduler!r}")
    
    if w_epochs > 0:
        warmup_start = args.warmup_start_factor if args.warmup_start_factor is not None else min(0.1, args.min_lr / args.lr)
        warmup = LinearLR(opt, start_factor=warmup_start, end_factor=1.0, total_iters=w_epochs)
        sched = SequentialLR(opt, [warmup, main], milestones=[w_epochs])
    else:
        sched = main

    # Initialize the best validation criterion as infinity
    best = float("inf")
    # Make sure output directory exists
    os.makedirs(args.out, exist_ok=True)

    # Define logger that only prints on rank 0
    def log(msg: str):
        if (not is_ddp) or dist.get_rank() == 0:
            print(msg, flush=True)

    #Print the device
    if (not is_ddp) or dist.get_rank() == 0:
        print("Device=", device_type)

    # Get the E1 and E2 values for the epochs at which ramping starts and ends
    E1, E2 = args.E1, args.E2
    meters = EpochMeters(device=device_type)

    alpha_channels = getattr(core.head, 'C')

    # ---- Loss function ----
    criterion = LossFunction(
        dipole_weight=args.dipole_weight,
        qs_weight=args.qs_weight, qs_threshold=0.5, qs_on='N', qs_local=True,
        shape_weight=0.,          # 0.05–0.2
        spec_weight=0.,          # 0.02–0.1
        spec_alpha=args.spec_alpha,            # 0.5–2.0 (↑ weights high-k)
        spec_log=True,
        tempo_weight=0.,         # 0.02–0.1 (needs base18 passed in forward)
        fluct_weight=0.,
        patch_var_weight=0.
    ).to(device) 

    # ---- Training loop ----

    for epoch in range(1, args.epochs+1):
        # ---- staged weights ----
        print(epoch, "/", args.epochs)

        e = epoch
        w_qs    = args.qs_weight #* ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_shape = args.shape_weight 
        w_spec  = args.spec_weight * ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_tempo = args.tempo_weight * ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_fluc  = args.fluct_weight * ramp(e, start=E1, duration=max(1,(E2-E1)))
        w_pv    = args.patch_var_weight * ramp(e, start=E1, duration=max(1,(E2-E1)))

        mu_smooth_weight = args.mu_smooth_weight

        criterion.current_epoch = epoch; 
        criterion.dipole_weight = args.dipole_weight
        criterion.qs_weight=w_qs
        criterion.shape_weight = w_shape
        criterion.spec_weight = w_spec
        criterion.tempo_weight = w_tempo
        criterion.fluct_weight = w_fluc
        criterion.fluct_weight = w_pv

        print("[loss cfg] dipole_w=", args.dipole_weight, "qs_w=", w_qs, "shape_w=", w_shape, "spec_w=", w_spec, "tempo_w=", w_tempo, "w_fluc=", w_fluc, "w_patch_var=", w_pv)

        p_mean = float(getattr(args, "mean_branch_frac", 0.5))
        p_mean = max(0.0, min(1.0, p_mean))  # clamp

        model.train()

        #Use epoch to seed the sapmler for shuffling in DDP
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ---- device-side running sums ----
        tr_sum = torch.zeros((), device=device, dtype=torch.float64)
        tr_cnt = torch.zeros((), device=device, dtype=torch.float64)
        val_sum = torch.zeros((), device=device, dtype=torch.float64)
        val_cnt = torch.zeros((), device=device, dtype=torch.float64)

        opt.zero_grad(set_to_none=True)

        # for it, (base18, Y_scalar, theta, y) in enumerate(train_dl, 1):
        #     base18   = base18.to(device, non_blocking=True)                 # [B,18,H,W]
        #     Y_scalar = Y_scalar.to(device, dtype=torch.float32, non_blocking=True)  # [B]
        #     theta    = theta.to(device, non_blocking=True)                   # [B,3]
        #     y        = y.to(device, non_blocking=True)

        #     if channels_last and device_type == "cuda":
        #         base18 = base18.contiguous(memory_format=torch.channels_last)

        #     global_step = (epoch - 1) * max(1, len(train_dl)) + (it - 1)

        #     # enable semigroup on a fraction of steps (expected fraction = semi_frac_iters)
        #     gen = torch.Generator(device=Y_scalar.device)
        #     gen.manual_seed(global_step)  # deterministic across ranks
        #     do_semi = (args.semigroup_weight > 0.0) and (torch.rand((), generator=gen) < args.semigroup_frac)

        #     with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
        #         # --- direct a→c ---
        #         yhat_ac, extras = model(base18, Y_scalar, theta, sample=False)
        #         loss, stats, _ = criterion(yhat_ac, y, base18)   # your existing supervised loss

        #         mu = extras["mu"]
        #         total = loss

                               
        #         # replace TV with gentler high-k penalty on the raw 18 channels
        #         lambda_highk = getattr(args, "highk_weight", 1e-3)  # start tiny
        #         frac_keep    = getattr(args, "highk_keep_frac", 0.5)
        #         if lambda_highk > 0.0:
        #             L_highk = criterion._high_k_power(yhat_ac, frac=frac_keep)
        #             total = total + lambda_highk * L_highk
        #             stats["highk_loss"] = L_highk.detach()

        #         loss_mu_smooth = mu_smoothness_loss(mu)
        #         total = total + mu_smooth_weight * loss_mu_smooth

        #         # --- semigroup ---
        #         if do_semi:
        #             DY_THRESH = y_max_data/2.0 
        #             # mask big-ΔY samples
        #             mask = (Y_scalar > DY_THRESH)
        #             if mask.any():
        #                 idx = mask.nonzero(as_tuple=False).squeeze(1)

        #                 base18_big = base18.index_select(0, idx)
        #                 theta_big  = theta.index_select(0, idx)
        #                 dY_ac_big  = Y_scalar.index_select(0, idx)

        #                 # random split ΔY_ac = ΔY_ab + ΔY_bc (avoid extremes)
        #                 low, high = 0.25, 0.75
        #                 u = low + (high - low) * torch.rand_like(dY_ac_big)
        #                 dY_ab = u * dY_ac_big
        #                 dY_bc = dY_ac_big - dY_ab

        #                 # a→b (mean path)
        #                 yhat_ab, _ = model(base18_big, dY_ab, theta_big, sample=False)
        #                 # b→c (compose)
        #                 yhat_ac_comp, _ = model(yhat_ab, dY_bc, theta_big, sample=False)

        #                 # direct (a→c) for the same masked indices
        #                 yhat_ac_big = yhat_ac.index_select(0, idx)

        #                 # label-free consistency on big-ΔY only
        #                 L_semi = torch.mean((yhat_ac_comp - yhat_ac_big)**2)
        #                 total = total + args.semigroup_weight * L_semi
        #             # else: no big-ΔY in this batch → skip semigroup this iter



        #     if use_amp and getattr(scaler, "is_enabled", lambda: False)():
        #         scaler.scale(total).backward()
        #         scaler.step(opt)
        #         scaler.update()
        #     else:
        #         total.backward()
        #         opt.step()

        #     opt.zero_grad(set_to_none=True)

        #     if ema is not None:
        #         ema.update(unwrap(model)) 

        for it, (base18, Y_scalar, theta, y) in enumerate(train_dl, 1):
            base18   = base18.to(device, non_blocking=True)
            Y_scalar = Y_scalar.to(device, dtype=torch.float32, non_blocking=True)
            theta    = theta.to(device, non_blocking=True)
            y        = y.to(device, non_blocking=True)

            if channels_last and device_type == "cuda":
                base18 = base18.contiguous(memory_format=torch.channels_last)

            global_step = (epoch - 1) * max(1, len(train_dl)) + (it - 1)

            # deterministic per-step RNG for DDP
            gen = torch.Generator(device=Y_scalar.device)
            gen.manual_seed(global_step)

            # semigroup toggle (unchanged)
            do_semi = (args.semigroup_weight > 0.0) and (
                torch.rand((), generator=gen) < args.semigroup_frac
            )

            # NEW: choose mean vs fluctuation branch
            do_mean_branch = torch.rand((), generator=gen) < p_mean
            if epoch < E1: do_mean_branch = True

            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):

                if do_mean_branch:
                    # ========= MEAN / DETERMINISTIC BRANCH =========
                    # use mean-focused losses: dipole, Qs, maybe shape
                    criterion.dipole_weight = args.dipole_weight
                    criterion.qs_weight     = w_qs
                    criterion.shape_weight  = w_shape
                    criterion.spec_weight   = 0.0       # or w_spec if you want here too
                    criterion.tempo_weight  = 0.0
                    criterion.fluct_weight  = 0.0
                    criterion.patch_var_weight  = 0.0

                    # deterministic forward (no sampling)
                    yhat_ac, extras = model(base18, Y_scalar, theta, sample=False)

                    loss, stats, _ = criterion(yhat_ac, y, base18)
                    mu = extras["mu"]
                    total = loss

                    # high-k regularization on raw output
                    lambda_highk = getattr(args, "highk_weight", 1e-3)
                    frac_keep    = getattr(args, "highk_keep_frac", 0.5)
                    if lambda_highk > 0.0:
                        L_highk = criterion._high_k_power(yhat_ac, frac=frac_keep)
                        total = total + lambda_highk * L_highk
                        stats["highk_loss"] = L_highk.detach()

                    # μ smoothness always helps
                    loss_mu_smooth = mu_smoothness_loss(mu)
                    total = total + mu_smooth_weight * loss_mu_smooth

                    # --- semigroup loss only in mean branch ---
                    if do_semi:
                        DY_THRESH = y_max_data / 2.0
                        mask = (Y_scalar > DY_THRESH)
                        if mask.any():
                            idx = mask.nonzero(as_tuple=False).squeeze(1)

                            base18_big = base18.index_select(0, idx)
                            theta_big  = theta.index_select(0, idx)
                            dY_ac_big  = Y_scalar.index_select(0, idx)

                            low, high = 0.25, 0.75
                            u = low + (high - low) * torch.rand_like(dY_ac_big)
                            dY_ab = u * dY_ac_big
                            dY_bc = dY_ac_big - dY_ab

                            yhat_ab, _      = model(base18_big, dY_ab, theta_big, sample=False)
                            yhat_ac_comp, _ = model(yhat_ab,   dY_bc, theta_big, sample=False)

                            yhat_ac_big = yhat_ac.index_select(0, idx)

                            L_semi = torch.mean((yhat_ac_comp - yhat_ac_big) ** 2)
                            total = total + args.semigroup_weight * L_semi

                else:
                    # ========= FLUCTUATION / STOCHASTIC BRANCH =========
                    # use fluctuation-focused losses: spec, tempo, struct
                    criterion.dipole_weight = 0
                    criterion.qs_weight     = 0
                    criterion.shape_weight  = 0
                    criterion.spec_weight   = w_spec
                    criterion.tempo_weight  = w_tempo
                    criterion.fluct_weight  = w_fluc
                    criterion.patch_var_weight  = w_pv

                    # stochastic forward (sampling ON)
                    yhat_ac, extras = model(base18, Y_scalar, theta, sample=True, dY=Y_scalar)
                    loss, stats, _ = criterion(yhat_ac, y, base18)
                    mu = extras["mu"]
                    total = loss

                    # still regularize high-k in output
                    lambda_highk = getattr(args, "highk_weight", 1e-3)
                    frac_keep    = getattr(args, "highk_keep_frac", 0.5)
                    if lambda_highk > 0.0:
                        L_highk = criterion._high_k_power(yhat_ac, frac=frac_keep)
                        total = total + lambda_highk * L_highk
                        stats["highk_loss"] = L_highk.detach()

                    # still keep μ smooth
                    loss_mu_smooth = mu_smoothness_loss(mu)
                    total = total + mu_smooth_weight * loss_mu_smooth

                    # NOTE: we do NOT apply semigroup loss in the stochastic branch
                    # (composition with noise is ill-defined w.r.t. a single target)


            # --- backprop & optimizer step (unchanged) ---
            if use_amp and getattr(scaler, "is_enabled", lambda: False)():
                scaler.scale(total).backward()
                scaler.step(opt)
                scaler.update()
            else:
                total.backward()
                opt.step()

            opt.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(unwrap(model))

        # ---- VALIDATION ----

        val_qp_all, val_qt_all, val_dy_qs_all, val_dy_tuner_all = [], [], [], []

        if val_sampler is not None:
            try: val_sampler.set_epoch(epoch)
            except Exception: pass

        ema_eval = (ema is not None) and getattr(args, "ema_eval", True)
        if ema_eval: ema.swap_in(unwrap(model))

        model.eval()

        # Restore full loss configuration for validation
        criterion.dipole_weight = args.dipole_weight
        criterion.qs_weight     = w_qs
        criterion.shape_weight  = w_shape
        criterion.spec_weight   = w_spec
        criterion.tempo_weight  = w_tempo
        criterion.fluct_weight  = w_fluc
        criterion.patch_var_weight  = w_pv


        val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        val_loss_cnt = torch.zeros((), device=device, dtype=torch.float64)

        # --- fluctuation stats accumulators (device-side) ---
        shape_sum = torch.zeros((), device=device, dtype=torch.float64)
        shape_cnt = torch.zeros((), device=device, dtype=torch.float64)

        spec_sum  = torch.zeros((), device=device, dtype=torch.float64)
        spec_cnt  = torch.zeros((), device=device, dtype=torch.float64)

        fluc_sum  = torch.zeros((), device=device, dtype=torch.float64)
        fluc_cnt  = torch.zeros((), device=device, dtype=torch.float64)

        pv_sum  = torch.zeros((), device=device, dtype=torch.float64)
        pv_cnt  = torch.zeros((), device=device, dtype=torch.float64)

        tempo_sum = torch.zeros((), device=device, dtype=torch.float64)
        tempo_cnt = torch.zeros((), device=device, dtype=torch.float64)

        val_sigma_sum = torch.zeros((), device=device, dtype=torch.float64)
        val_sigma_cnt = torch.zeros((), device=device, dtype=torch.float64)


        with torch.no_grad():
            for i, (base18, Y_scalar, theta, y) in enumerate(val_dl):
                base18   = base18.to(device, non_blocking=True)
                Y_scalar = Y_scalar.to(device,  dtype=torch.float32, non_blocking=True)
                theta    = theta.to(device, non_blocking=True)
                y        = y.to(device, non_blocking=True)
                
                if getattr(args, "channels_last", False) and device_type == "cuda":
                    base18 = base18.contiguous(memory_format=torch.channels_last)
                
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
#                    yh = model.forward(base18, Y_scalar, theta)
                    yh, extras = model(base18, Y_scalar, theta, sample=False, dY=Y_scalar)
                    dy_scalar = Y_scalar  # [B]

                    ## === spectral residual-shape metric (output space, robust) ===
                    #y_mean = yh
                    #y_samp, _ = model(base18, Y_scalar, theta, sample=True, dY=Y_scalar)

                    loss_val, stats_val, tuner = criterion(
                        yh, y, base18,
                    )

                    if "sigma" in extras:
                        sigma = extras["sigma"]                   # [B,C,H,W]
                        batch_sigma_mean = sigma.detach().mean()  # scalar
                        bs = y.shape[0]
                        val_sigma_sum += batch_sigma_mean.to(torch.float64) * bs
                        val_sigma_cnt += bs

                    # --- semigroup metric (label-free) ---
                    if args.semigroup_weight > 0.0:
                        u = torch.rand_like(Y_scalar)
                        dY_ab = u * Y_scalar
                        dY_bc = Y_scalar - dY_ab

                        yhat_ab, _     = model(base18, dY_ab, theta, sample=False)
                        yhat_ac_cmp, _ = model(yhat_ab, dY_bc, theta, sample=False)

                        L_semi_direct = torch.mean((yhat_ac_cmp - yh)**2)   # comp vs direct

                        # accumulate
                        if i == 0:
                            val_semi_sum = torch.zeros((), device=device, dtype=torch.float64)
                            val_semi_cnt = torch.zeros((), device=device, dtype=torch.float64)
                        val_semi_sum += L_semi_direct.detach().to(torch.float64) * y.shape[0]
                        val_semi_cnt += y.shape[0]

                    bs = y.shape[0]
                    val_loss_sum += loss_val.detach().to(torch.float64) * bs
                    val_loss_cnt += bs

                    sl = stats_val.get("shape_loss", None)
                    if sl is not None:
                        shape_sum += sl.to(torch.float64) * bs
                        shape_cnt += bs

                    sp = stats_val.get("spec_loss", None)
                    if sp is not None:
                        spec_sum  += sp.to(torch.float64) * bs
                        spec_cnt  += bs
                    
                    fl = stats_val.get("fluct_loss", None)
                    if fl is not None:
                        fluc_sum  += fl.to(torch.float64) * bs
                        fluc_cnt  += bs

                    pl = stats_val.get("patch_var_loss", None)
                    if pl is not None:
                        pv_sum  += pl.to(torch.float64) * bs
                        pv_cnt  += bs

                    tl = stats_val.get("tempo_loss", None)
                    if tl is not None:
                        tempo_sum += tl.to(torch.float64) * bs
                        tempo_cnt += bs

                    # --- Per-sample Qs for ΔY-binned diagnostics ---
                    try:
                        # Use the same helpers as the loss for consistency
                        Uh = criterion._pack18_to_U(yh)  # complex [B,H,W,3,3]
                        Ut = criterion._pack18_to_U(y)
                        Uh = criterion._su3_project(Uh)
                        Ut = criterion._su3_project(Ut)
                        qp = criterion._compute_Qs_from_U(Uh, local=False)  # [B]
                        qt = criterion._compute_Qs_from_U(Ut, local=False)  # [B]
                        val_qp_all.append(qp.detach())
                        val_qt_all.append(qt.detach())
                        val_dy_qs_all.append(dy_scalar.view(-1).detach())
                    except Exception:
                        pass

            _dist_sum_(val_loss_sum)
            _dist_sum_(val_loss_cnt)

            val_loss_mean = (val_loss_sum / val_loss_cnt.clamp_min(1)).item()
            if (not is_ddp) or dist.get_rank() == 0:
                print(f"[val] mean_loss = {val_loss_mean:.6e}")

            if len(val_dy_qs_all):
                dy_t = torch.cat(val_dy_qs_all, dim=0)
                qp_t = torch.cat(val_qp_all, dim=0)
                qt_t = torch.cat(val_qt_all, dim=0)
                dy = dy_t.to('cpu', non_blocking=True).numpy()
                qp = qp_t.to('cpu', non_blocking=True).numpy()
                qt = qt_t.to('cpu', non_blocking=True).numpy()
                # ΔY quantile bins (4 bins: quartiles)
                edges = np.quantile(dy, [0.0, 0.25, 0.5, 0.75, 1.0])
                rmse_bins, mape_bins = [], []
                eps = 1e-8
                for i in range(len(edges) - 1):
                    m = (dy >= edges[i]) & (dy < edges[i+1] if i < len(edges)-2 else dy <= edges[i+1])
                    if m.any():
                        rmse = float(np.sqrt(np.mean((qp[m] - qt[m])**2)))
                        mape = float(np.mean(np.abs((qp[m] - qt[m]) / (np.abs(qt[m]) + qp[m] + eps))))
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

                print(f"[val Qs(ΔY bins)] edges={edges}  rmse={np.array(rmse_bins)}  smape={np.array(mape_bins)}")
                print(f"[val Qs(ΔY top quartile)] rmse={top_rmse:.3e} smape={top_mape:.3e} "
                    f"Qs_pred_mean={top_pred:.3e} Qs_true_mean={top_true:.3e}")


            # DDP reductions
            _dist_sum_(shape_sum); _dist_sum_(shape_cnt)
            _dist_sum_(spec_sum);  _dist_sum_(spec_cnt)
            _dist_sum_(fluc_sum);  _dist_sum_(fluc_cnt)
            _dist_sum_(pv_sum);  _dist_sum_(pv_cnt)
            _dist_sum_(tempo_sum); _dist_sum_(tempo_cnt)
            _dist_sum_(val_sigma_sum); _dist_sum_(val_sigma_cnt)

            if (not is_ddp) or dist.get_rank() == 0:
                def _mean_or_nan(s, c):
                    return (s / c.clamp_min(1)).item() if c.item() > 0 else float("nan")

                mean_shape = _mean_or_nan(shape_sum, shape_cnt)
                mean_spec  = _mean_or_nan(spec_sum,  spec_cnt)
                mean_fluc  = _mean_or_nan(fluc_sum,  fluc_cnt)
                mean_pv  = _mean_or_nan(pv_sum,  pv_cnt)
                mean_tempo = _mean_or_nan(tempo_sum, tempo_cnt)

                print("[val fluctuations]")
                if shape_cnt.item() > 0:
                    print(f"  shape_loss  (bin-to-bin S(r))   = {mean_shape:.6e}")
                if spec_cnt.item()  > 0:
                    print(f"  spec_loss   (spatial PSD)       = {mean_spec:.6e}")
                if fluc_cnt.item()  > 0:
                    print(f"  fluct_loss                      = {mean_fluc:.6e}")
                if pv_cnt.item()  > 0:
                    print(f"  patch_var_loss                  = {mean_pv:.6e}")
                if tempo_cnt.item() > 0:
                    print(f"  tempo_loss  (ΔY increment PSD)  = {mean_tempo:.6e}")
                mean_sigma = _mean_or_nan(val_sigma_sum, val_sigma_cnt)
                print(f"[val] mean sigma = {mean_sigma:.3e}")


            if args.semigroup_weight > 0.0 and 'val_semi_cnt' in locals() and val_semi_cnt.item() > 0:
                _dist_sum_(val_semi_sum); _dist_sum_(val_semi_cnt)
                val_semi_mean = (val_semi_sum / val_semi_cnt.clamp_min(1)).item()
                semi_ratio = val_semi_mean / (val_loss_mean + 1e-12)
                if (not is_ddp) or dist.get_rank() == 0:
                    print(f"[val] L_semi_direct = {val_semi_mean:.6e}   semigroup_ratio = {semi_ratio:.3f}")


        if ema_eval: 
            ema.swap_out(unwrap(model))

        if args.scheduler == "plateau":
            sched.step(val_loss_mean)
        else:
            sched.step()

        try:
            lr = sched.get_last_lr()[0]
        except Exception:
            lr = opt.param_groups[0]["lr"]

        print("lr=",lr)

        with torch.no_grad():
            if ema is not None and getattr(args, "ema_eval", True):
                ema.swap_in(unwrap(model))
            curve_rmse_det, curve_rmse_samp = eval_curve_rmse_full(
                model, criterion, val_dl, device,
                K_samples=4, channels_last=getattr(args, "channels_last", False),
                device_type=device_type, amp_dtype=amp_dtype, use_amp=use_amp
            )
            if ema is not None and getattr(args, "ema_eval", True):
                ema.swap_out(unwrap(model))

            if (not is_ddp) or dist.get_rank() == 0:
                #if curve_rmse_samp is None:
                print(f"[epoch {epoch}] curve_RMSE(det)≈{curve_rmse_det:.3e}")
                #else:
                #    print(f"[epoch {epoch}] curve_RMSE(det)≈{curve_rmse_det:.3e}  curve_RMSE(samp,K=4)≈{curve_rmse_samp:.3e}")


        # Save best
        outdir = Path(args.out)

        if ((not is_ddp) or dist.get_rank() == 0) and (curve_rmse_det < best):
            best = curve_rmse_det

            core = (model.module if is_ddp else model)  # unwrap once

            save_ema = (ema is not None) and bool(getattr(args, "ema_eval", True))
            if save_ema:
                # ---- primary "best": EMA weights ----
                # Prefer a robust snapshot by temporarily swapping EMA weights in.
                ema.swap_in(core)
                try:
                    state_to_save = {k: v.detach().clone() for k, v in core.state_dict().items()}
                finally:
                    ema.swap_out(core)
            else:
                # ---- primary "best": base weights ----
                state_to_save = core.state_dict()

            ckpt = {
                "model": state_to_save,
                "args": {
                    "in_ch": 22, "width": args.width, "modes": args.modes, "blocks": args.blocks,
                    "proj_iter": 8, "gate_temp": args.gate_temp, "alpha_vec_cap": 15, "rbf_K": args.rbf_K,
                    "sigma_mode": args.sigma_mode, "spec_bins": args.spec_bins,
                },
                "meta": {"N": inferred_N, "ds": ds_value, "epoch": epoch, "is_ema": save_ema}
            }
            torch.save(ckpt, outdir / "evolver_best.pt")            


def main():
    ap = argparse.ArgumentParser(description="Train FNO evolver: (U0, Y, params) -> U_Y")

    ap.add_argument("--data_root", type=Path, required=True,
                    help="Training root dir with run_*/ (each has manifest.json + evolved_wilson_lines/)")
    ap.add_argument("--out", type=str, default="outputs_evolver", help="Output dir for checkpoints")
    ap.add_argument("--epochs", type=int, default=1, help='Number of training epochs to run.')
    ap.add_argument("--batch", type=int, default=1, help="Per-device batch size")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    ap.add_argument("--seed", type=int, default=0, help='Random seed for reproducibility (Python/NumPy/PyTorch).')
    ap.add_argument("--identity_eps", type=float, default=0.0, help="If |Y|<=eps, return U0 exactly (skip update)")
    ap.add_argument("--clamp_alphas", type=float, default=2.0, help="Clamp alpha values")

    # Model size
    ap.add_argument("--width", type=int, default=32, help='Base channel width of the FNO trunk.')
    ap.add_argument("--modes", type=int, default=8, help='Number of retained Fourier modes per spatial dimension in spectral convs.')
    ap.add_argument("--blocks", type=int, default=6, help='Number of FNO residual blocks (depth of the trunk).')

    # Device / precision
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA)")
    ap.add_argument("--channels_last", dest="channels_last", action="store_true", help="Use channels_last memory format")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    #FiLM / time conditioning
    ap.add_argument("--rbf_K", type=int, default=12, help='Number of RBF centers used to embed Y (time) before FiLM.')
    ap.add_argument("--film_hidden", type=int, default=64, help='Hidden size of the FiLM MLP used in the time conditioner.')
    ap.add_argument("--gamma_scale", type=float, default=1.5, help='Initial scale for FiLM gamma (multiplicative) parameters.')
    ap.add_argument("--gate_temp", type=float, default=1.0, help='Temperature for gating nonlinearity in the time conditioner; higher = sharper gates.')

    ap.add_argument('--y_map', type=str, default='linear', choices=['tanh','linear'], help='Map from physical Y to internal Y_eff')

    ap.add_argument('--sigma_mode', type=str, default='conv', choices=['diag','conv','spectral'], help='Noise-path covariance: diagonal, local conv operator, or spectral.')
    ap.add_argument('--spec_bins', type=int, default=48)    

    # EMA
    ap.add_argument("--ema_decay", type=float, default=0, help="EMA decay, e.g. 0.999; 0 disables EMA")
    ap.add_argument("--ema_eval", type=int, default=1, help="Use EMA weights for eval when EMA is enabled")

    # Optimizer / LR
    ap.add_argument("--lr", type=float, default=1e-3, help='Initial learning rate for the optimizer.')
    ap.add_argument("--weight_decay", type=float, default=1e-7, help='Weight decay (L2 regularization) applied by the optimizer.')
    ap.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs")
    ap.add_argument("--warmup_start_factor", type=float, default=0.1, help="Warmup factor (relative to lr)")
    ap.add_argument("--min_lr", type=float, default=1e-5, help="Eta_min for cosine phase")
    ap.add_argument("--scheduler", type=str, default="cosine",
                    choices=["cosine","cosine_wr","step","multistep","exponential","plateau","poly","constant"],
                    help="LR schedule used after the warmup phase.")
    ap.add_argument("--E1", type=int, default=0, help="Epoch when to start ramping up loss functions other than the dipole.")
    ap.add_argument("--E2", type=int, default=0, help="Epoch when to finish ramping up loss functions other than the dipole.")

    # Weights for loss terms
    ap.add_argument("--dipole_weight", type=float, default=0.5,
                    help="Weight of MSE loss on dipole correlator S(r) evaluated at --dipole_offsets.")
    ap.add_argument("--qs_weight", type=float, default=0.,
                    help="Weight of MSE loss on saturation scale Qs derived from the dipole correlator.")

    ap.add_argument("--semigroup_weight", type=float, default=0.0, help="weight for semigroup loss (0 disables)")
    ap.add_argument("--semigroup_frac", type=float, default=0.0, help="fraction of iterations to apply semigroup loss (0 disables)")
    ap.add_argument("--highk_weight", type=float, default=0.0, help="reduces cell by cell jitter")

    ap.add_argument("--shape_weight", type=float, default=0.0, help="weight for shape: Shape loss focuses the model on bin-to-bin variation and curvature of S(r), not just its coarse level")
    ap.add_argument("--spec_weight", type=float, default=0.0, help="weight for fluctuation spectrum: Spatial spectrum loss makes the predicted center field carry the same fluctuation spectrum as truth")
    ap.add_argument("--spec_alpha", type=float, default=1.0, help="larger weighs larger k")
    ap.add_argument("--tempo_weight", type=float, default=0.0, help="Temporal spectrum loss makes the increment over ΔY have the right spectrum")
    ap.add_argument("--fluct_weight", type=float, default=0.0, help="weight for structure-function loss on the center field")
    ap.add_argument("--patch_var_weight", type=float, default=0.0, help="weight for patch variance loss")
  
    ap.add_argument("--mu_smooth_weight", type=float, default=0.0, help="make mu (the average of the operator) fluctuate less frm cell to cell")

    ap.add_argument("--mean_branch_frac", type=float, default=1.0, help="fraction of times to train ony the mean vs. the fluctuations")


    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()