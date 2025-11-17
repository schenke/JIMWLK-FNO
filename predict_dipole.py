#!/usr/bin/env python3
"""
Predict and compare dipole S(r) and also plot the per-site correlator with the center
cell for both prediction and truth, for a specific run and target rapidity.

Output:
- If --out ends with .pdf: a single multi-page PDF
  Page 1: S(r) comparison (truth vs prediction)
  Page 2: 2D correlator maps to the center (truth vs prediction, side by side)
- Else: saves S(r) to --out, and also writes "<stem>_center_maps.pdf" for the maps.

Correlator map definition (for center (N/2+1,N/2+1) 1-based index → N//2 zero-based):
  C(x) = (1/Nc) Re Tr[ U(x) U(center)^\dagger ]
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch.nn.functional as F


# ------------------------- utils -------------------------
def load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location("trainer_module", str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _run_dirs(root: Path) -> List[Path]:
    items = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("run_") and (p / "manifest.json").exists():
            items.append(p)
    return items


def _read_manifest(run_dir: Path) -> Dict:
    man_path = run_dir / "manifest.json"
    with open(man_path, "r") as f:
        man = json.load(f)
    # Normalize relative paths to absolute on disk
    snaps = []
    for s in man.get("snapshots", []):
        sp = Path(s["path"])
        if not sp.is_absolute():
            sp = (run_dir / sp).resolve()
        snaps.append({"steps": int(s["steps"]), "path": str(sp)})
    man["snapshots"] = snaps
    # Harmonize parameter keys
    p = man.get("params", {})
    man["params_norm"] = {
        "m": float(p.get("m_GeV") or p.get("m") or 0.0),
        "Lambda_QCD": float(p.get("Lambda_QCD_GeV") or p.get("Lambda_QCD") or 0.0),
        "mu0": float(p.get("mu0_GeV") or p.get("mu0") or 0.0),
    }
    man["ds"] = float(man["ds"])
    return man


def scan_root_for_normalizers(data_root: Path) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
    """
    Scan *all* run_*/manifest.json once (no heavy file reads) to compute:
      y_min_data, y_max_data, theta_min[3], theta_max[3], and a consistent ds.
    Y is computed as steps * ds * π^2, and we only consider Y > 0 (targets beyond anchor).
    """
    rdirs = _run_dirs(data_root)
    if not rdirs:
        raise FileNotFoundError(f"No run_*/manifest.json found under {data_root}")

    ds_vals = []
    y_values_pos = []
    theta_rows = []

    for rd in rdirs:
        man = _read_manifest(rd)
        ds_vals.append(man["ds"])
        # θ
        t = man["params_norm"]
        theta_rows.append([t["m"], t["Lambda_QCD"], t["mu0"]])
        # Y per snapshot
        ys = [s["steps"] * man["ds"] * (math.pi ** 2) for s in man.get("snapshots", [])]
        if not ys:
            continue
        Ya = min(ys)
        for y in ys:
            if y <= Ya:
                continue
            y_values_pos.append(y - Ya)

    if not ds_vals:
        raise ValueError("Could not infer ds from manifests.")
    if abs(max(ds_vals) - min(ds_vals)) > 1e-12:
        raise ValueError(f"Inconsistent ds across runs: {sorted(set(ds_vals))}")
    ds_value = ds_vals[0]

    y_min_data = float(min(y_values_pos)) if y_values_pos else 0.0
    y_max_data = float(max(y_values_pos)) if y_values_pos else 1.0

    theta_arr = np.array(theta_rows, dtype=np.float64) if theta_rows else np.zeros((0,3))
    if theta_arr.size == 0:
        theta_min = np.zeros(3, dtype=np.float32)
        theta_max = np.ones(3, dtype=np.float32)
    else:
        theta_min = theta_arr.min(axis=0).astype(np.float32)
        theta_max = theta_arr.max(axis=0).astype(np.float32)

    return y_min_data, y_max_data, theta_min, theta_max, ds_value


def _resolve_run_dir(data_root: Path, run_id: str) -> Path:
    """
    Resolve a run directory given a run_id (5 digits), expecting the name:
      run_{run_id}_00000
    Accepts either '00037' or '37' etc.
    """
    # Zero pad
    try:
        rid5 = f"{int(run_id):05d}"
    except Exception:
        rid5 = run_id  # if already formatted, keep as-is

    candidates = [
        data_root / f"run_{rid5}_00000",
        data_root / f"run_{run_id}_00000",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    raise FileNotFoundError(f"Could not find run directory for id={run_id!r}. Tried: {', '.join(str(c) for c in candidates)}")


def load_anchor_info_for_run(data_root: Path, run_id: str) -> Tuple[Dict, int, float, str]:
    """
    Return (manifest, anchor_steps, ds, anchor_path) for the specific run.
    """
    run_dir = _resolve_run_dir(data_root, run_id)
    man = _read_manifest(run_dir)
    if not man.get("snapshots"):
        raise ValueError(f"{run_dir}: manifest has no snapshots.")

    steps_list = [int(s["steps"]) for s in man["snapshots"]]
    a_idx = int(np.argmin(steps_list))
    anchor_steps = steps_list[a_idx]
    anchor_path = Path(man["snapshots"][a_idx]["path"])

    return man, anchor_steps, float(man["ds"]), str(anchor_path)


def select_snapshot_closest_to_Y(man: Dict, anchor_steps: int, Y_target: float) -> Tuple[int, str, float]:
    """
    Given a manifest, the anchor step count, and Y_target (distance from anchor),
    pick the snapshot index closest to that Y, returning (idx, path, Y_actual).
    """
    ds = float(man["ds"])
    best = None
    for i, s in enumerate(man["snapshots"]):
        dsteps = int(s["steps"]) - anchor_steps
        Y_i = dsteps * ds * (math.pi ** 2)
        if Y_i < 0:
            continue  # only consider beyond anchor
        err = abs(Y_i - Y_target)
        if best is None or err < best[0]:
            best = (err, i, s["path"], Y_i)
    if best is None:
        raise SystemExit("No snapshot beyond anchor to compare against; please provide a positive --Y/--steps.")
    _, idx, path, Y_actual = best
    return idx, path, Y_actual

def build_model(trainer, y_min, y_max, theta_min, theta_max, ckpt_path: Optional[Path], device: torch.device):
    """
    Build EvolverFNO. If a checkpoint is provided, use its hyperparams and weights.
    Provide normalization ranges so inference matches the training scaling.
    """
    width, modes, blocks = 64, 12, 4
    gate_temp = 2.0
    alpha_vec_cap = 15.0
    state = None
    rbf_K = 12
    sigma_mode = "conv"
    spec_bins = 48

    if ckpt_path is not None and ckpt_path.exists():
        chk = torch.load(str(ckpt_path), map_location="cpu")
        argsd = chk.get("args", {}) if isinstance(chk, dict) else {}
        width = int(argsd.get("width", width))
        modes = int(argsd.get("modes", modes))
        blocks = int(argsd.get("blocks", blocks))
        rbf_K = int(argsd.get("rbf_K", rbf_K))
        gate_temp = float(argsd.get("gate_temp", gate_temp))
        alpha_vec_cap = float(argsd.get("alpha_vec_cap", alpha_vec_cap))
        state = chk.get("model", chk)
        sigma_mode = argsd.get("sigma_mode", sigma_mode)
        spec_bins=int(argsd.get("spec_bins", spec_bins))

    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("model", sd.get("state_dict", sd))
    hparams = sd.get("hparams", {})  # may be empty on old ckpts

    model = trainer.EvolverFNO(
        width=width, modes1=modes, modes2=modes, n_blocks=blocks,
        gate_temp=gate_temp, alpha_vec_cap=alpha_vec_cap,
        y_min=y_min, y_max=y_max, rbf_K=rbf_K,
        theta_min=theta_min.tolist(), theta_max=theta_max.tolist(),
        sigma_mode=sigma_mode, spec_bins=spec_bins
    ).to(device).eval()

    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
            if missing: print("  missing:", missing[:10], "..." if len(missing) > 10 else "")
            if unexpected: print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    return model


def isotropic_curve_from_wilson(U_complex: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute isotropic S(r) via FFT-based correlation on a periodic lattice.

    Args:
        U_complex: [1,H,W,3,3] complex tensor (batch=1).

    Returns:
        r_centers (np.ndarray), S_r (np.ndarray)
    """
    assert U_complex.ndim == 5 and U_complex.size(0) == 1, "Expected [1,H,W,3,3]"
    if not torch.is_complex(U_complex):
        raise TypeError("U_complex must be complex dtype [1,H,W,3,3].")

    U = U_complex[0]                     # [H,W,3,3]
    H, W = U.shape[0], U.shape[1]
    Nc = 3.0

    # Sum of auto-correlations over all matrix elements:
    S_map = torch.zeros((H, W), dtype=torch.float32, device=U.device)
    for i in range(3):
        for k in range(3):
            X = U[..., i, k]                            # [H,W] complex
            F = torch.fft.fft2(X)                       # complex
            acf = torch.fft.ifft2(F * torch.conj(F))    # [H,W] complex
            S_map += acf.real

    # Normalize: average over sites and divide by Nc
    S_map = S_map / (H * W * Nc)                        # [H,W], real

    # Build minimal-image radial grid
    xs = torch.arange(H, device=U.device)
    ys = torch.arange(W, device=U.device)
    dx = torch.minimum(xs, H - xs)[:, None].expand(H, W)
    dy = torch.minimum(ys, W - ys)[None, :].expand(H, W)
    rgrid = torch.sqrt(dx.to(torch.float32)**2 + dy.to(torch.float32)**2)  # [H,W]

    r_np = rgrid.detach().cpu().numpy().ravel()
    S_np = S_map.detach().cpu().numpy().ravel()

    # Bin by integer radii
    r_max = float(r_np.max())
    edges = np.arange(0.0, math.floor(r_max) + 1.5, 1.0)   # width=1 bins
    if len(edges) < 2:
        return np.array([0.0], dtype=np.float32), np.array([S_np.mean()], dtype=np.float32)

    bin_idx = np.digitize(r_np, edges) - 1
    nb = len(edges) - 1
    sums = np.bincount(bin_idx, weights=S_np, minlength=nb)
    cnts = np.bincount(bin_idx, minlength=nb).astype(np.float64)
    mask = cnts > 0
    S_r = np.zeros(nb, dtype=np.float64)
    S_r[mask] = sums[mask] / cnts[mask]
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    return r_centers.astype(np.float32), S_r.astype(np.float32)


def center_correlator_map(U_complex: torch.Tensor) -> np.ndarray:
    """
    Correlator with fixed center cell (N/2+1,N/2+1) in 1-based indexing
    → zero-based (N//2, N//2).

    C(x) = (1/Nc) Re Tr[ U(x) U(center)^\dagger ]

    Args:
        U_complex: [1,H,W,3,3] complex tensor

    Returns:
        map: [H,W] numpy array (float32)
    """
    assert U_complex.ndim == 5 and U_complex.size(0) == 1, "Expected [1,H,W,3,3]"
    U = U_complex[0]
    H, W = U.shape[0], U.shape[1]
    Nc = 3.0
    cx, cy = H // 2, W // 2  # 1-based (N/2+1) → zero-based N//2

    Uc = U[cx, cy]                         # [3,3] complex
    Uc_H = Uc.conj().transpose(-1, -2)     # [3,3]

    # Broadcasted matmul over (H,W,3,3) with (3,3) → (H,W,3,3)
    M = torch.matmul(U, Uc_H)
    tr = torch.real(torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)) / Nc   # [H,W] real
    return tr.detach().cpu().numpy().astype(np.float32)

def per_sample_val_metrics(out18_bchw, tgt18_bchw, crit):
    """
    Recompute the validation dipole loss *for this one sample*,
    exactly like _isotropic_dipole_loss (no dS_bias), and also compute
    an unweighted RMSE over S(r) for your plot.
    Returns: dict with L_val, rmse_S, r, S_pred, S_true, (optional) w
    """
    with torch.no_grad():
        # pack & project exactly like validation
        U_pred = crit._su3_project(crit._pack18_to_U(out18_bchw))
        U_true = crit._su3_project(crit._pack18_to_U(tgt18_bchw))

        # isotropic curves (drops r=0 internally)
        r_t, S_pred_t = crit._isotropic_curve(U_pred, needs_grad=False, assume_su3=True, drop_r0=True)
        _,   S_true_t = crit._isotropic_curve(U_true, needs_grad=False, assume_su3=True, drop_r0=True)

        # shapes: S_*_t: [B, R]; for B=1 take index 0
        S_pred = S_pred_t[0]
        S_true = S_true_t[0]

        # validation loss is on N = 1 - S, with per-radius norm and SmoothL1
        N_pred, N_true = (1.0 - S_pred), (1.0 - S_true)

        # per-radius normalization (same as your loss)
        w = 1.0 / torch.sqrt((N_true**2).mean(dim=0, keepdim=False).clamp_min(1e-8))
        w = (w / w.mean()).detach()
        N_pred_n = N_pred * w
        N_true_n = N_true * w

        L_val = F.smooth_l1_loss(N_pred_n, N_true_n, beta=0.02)

        # raw curve RMSE in S-space (what your eye sees)
        rmse_S = torch.sqrt(torch.mean((S_pred - S_true) ** 2))

        return {
            "L_val": float(L_val.cpu()),
            "rmse_S": float(rmse_S.cpu()),
            "r": r_t[0].detach().cpu().numpy(),
            "S_pred": S_pred.detach().cpu().numpy(),
            "S_true": S_true.detach().cpu().numpy(),
            "w": w.detach().cpu().numpy(),  # per-radius weights used by val loss
        }

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Predict dipole S(r) and plot center correlator maps (truth vs prediction) for a specific run.")
    ap.add_argument("--trainer", type=Path, default=Path("trainer.py"),
                    help="Path to trainer.py that defines EvolverFNO and helpers like read_wilson_binary().")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Root folder containing run_*/ subfolders (each with manifest.json).")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Optional path to model checkpoint (evolver_best.pt).")
    ap.add_argument("--run", type=str, required=True,
                    help="5-digit run id (e.g. 00037). Integers also accepted (will be zero-padded).")
    ap.add_argument("--Y", type=float, default=None,
                    help="Target rapidity distance from the anchor (overrides --steps).")
    ap.add_argument("--steps", type=int, default=None,
                    help="Alternative to --Y: integer steps converted via ds with π^2 factor (Y = steps * ds * π^2).")
    ap.add_argument("--out", type=Path, default=Path("dipole.pdf"),
                    help="Output filename. If .pdf, both S(r) and maps go to a single multi-page PDF.")
    ap.add_argument("--device", type=str, default=None,
                    help="Force device: 'cuda', 'cpu', or leave empty for auto.")
    args = ap.parse_args()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.Y is not None and args.steps is not None:
        raise SystemExit("Please provide only one of --Y or --steps, not both.")

    # Load the provided trainer module (for model + binary reader)
    trainer = load_module_from_path(args.trainer)

    # Use the exact helper used in validation
    crit = trainer.LossFunction(dipole_weight=0.0, qs_weight=0.0).to(device)

    # Compute global normalization from manifests (no heavy reads)
    y_min, y_max, theta_min, theta_max, _ = scan_root_for_normalizers(args.data_root)

    # Load run info + anchor
    man, anchor_steps, ds_run, anchor_path = load_anchor_info_for_run(args.data_root, args.run)

    # Decide the Y to use
    if args.steps is not None:
        Y_target = float(args.steps) * ds_run * (math.pi ** 2)
        print(f"[info] Converted steps→Y with π^2: steps={args.steps}, ds={ds_run:.6g}, Y={Y_target:.6g}")
    elif args.Y is not None:
        Y_target = float(args.Y)
    else:
        # Use the smallest positive Y available in this run
        Ys = [(int(s["steps"]) - anchor_steps) * ds_run * (math.pi ** 2) for s in man["snapshots"]]
        posYs = [y for y in Ys if y > 0]
        if not posYs:
            raise SystemExit("No target snapshots beyond anchor in this run; please provide --Y or --steps explicitly.")
        Y_target = float(min(posYs))
        print(f"[info] Using smallest positive Y from run {_resolve_run_dir(args.data_root, args.run).name}: Y={Y_target:.6g}")

    if Y_target < 0:
        raise SystemExit(f"Y must be non-negative (got {Y_target}).")

    # Pick the truth snapshot closest to Y_target
    idx_truth, truth_path, Y_truth = select_snapshot_closest_to_Y(man, anchor_steps, Y_target)
    if abs(Y_truth - Y_target) > 1e-8:
        print(f"[info] Truth uses closest Y available in run: requested Y={Y_target:.6g}, actual Y={Y_truth:.6g}")

    # Read anchor (input) and truth (reference) as 18-ch
    Ua18 = trainer.read_wilson_binary(Path(anchor_path), size=None)        # [N,N,18]
    Ut18 = trainer.read_wilson_binary(Path(truth_path), size=None)         # [N,N,18]

    # Prepare tensors
    base18 = torch.from_numpy(np.asarray(Ua18)).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,18,H,W]
    theta_vec = torch.tensor([man["params_norm"]["m"], man["params_norm"]["Lambda_QCD"], man["params_norm"]["mu0"]],
                             dtype=torch.float32, device=device).reshape(1, 3)             # [1,3]

    # Build & load model (with normalization from manifests)
    model = build_model(trainer, y_min, y_max, theta_min, theta_max, args.ckpt, device)

    # Predict at Y_target
    y_scalar = torch.tensor([Y_target], dtype=torch.float32, device=device)  # [1]
    with torch.no_grad():
        out18, extras = model(base18, y_scalar, theta_vec, sample=True, nsamples=1)
        U_pred = crit._pack18_to_U(out18)  # -> [B,H,W,3,3] complex64

    # Prepare truth Wilson lines as complex
    Ut18_bchw = torch.from_numpy(np.asarray(Ut18)).to(torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    U_true = crit._pack18_to_U(Ut18_bchw)  # -> [B,H,W,3,3] complex64                                               # [1,H,W,3,3] complex

    # Compute isotropic curves
    with torch.no_grad():
        # match validation: project both to SU(3)
        U_pred_su3 = crit._su3_project(U_pred.to(torch.complex64))
        U_true_su3 = crit._su3_project(U_true.to(torch.complex64))

        # match validation: same FFT-based radial average and binning; drops r=0 internally
        r_pred_t, S_pred_t = crit._isotropic_curve(U_pred_su3, needs_grad=False, assume_su3=True, drop_r0=False)
        r_true_t, S_true_t = crit._isotropic_curve(U_true_su3, needs_grad=False, assume_su3=True, drop_r0=False)

    # convert to numpy (batch=1)
    r_pred = r_pred_t.detach().cpu().numpy()
    S_pred = S_pred_t[0].detach().cpu().numpy()
    r_true = r_true_t.detach().cpu().numpy()
    S_true = S_true_t[0].detach().cpu().numpy()

    m = per_sample_val_metrics(out18, Ut18_bchw, crit)
    print(f"[this sample] val_dipole_loss={m['L_val']:.4e}   curve_RMSE(S)={m['rmse_S']:.4e}")

    # (optional) see why the first bins “look worse”
    K = 5
    rmse_head = np.sqrt(np.mean((m["S_pred"][:K] - m["S_true"][:K])**2))
    rmse_tail = np.sqrt(np.mean((m["S_pred"][K:] - m["S_true"][K:])**2))
    print(f"RMSE first {K} bins: {rmse_head:.4e}   remaining: {rmse_tail:.4e}")
    print("first 5 S_pred:", m["S_pred"][:5])
    print("first 5 S_true:", m["S_true"][:5])

    mu    = extras["mu"][0].detach().cpu().numpy()     # [C,H,W]
    sigma = extras["sigma"][0].detach().cpu().numpy()  # [C,H,W]   
    print("mu mean/std:", mu.mean(), mu.std())
    print("sigma mean/std:", sigma.mean(), sigma.std())


    # Compute center correlator maps
    map_pred = center_correlator_map(U_pred)    # [H,W]
    map_true = center_correlator_map(U_true)    # [H,W]
    vmin = float(min(map_pred.min(), map_true.min()))
    vmax = float(max(map_pred.max(), map_true.max()))

    rid5 = f"{int(args.run):05d}" if str(args.run).isdigit() else str(args.run)

    # Decide output(s)
    out_path = args.out
    if out_path.suffix.lower() == ".pdf":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(out_path) as pdf:
            # Page 1: S(r) comparison
            fig1 = plt.figure()
            plt.plot(r_true, S_true, label=f"Truth S(r) @ Y={Y_truth:.3g}", linestyle='--')
            plt.plot(r_pred, S_pred, label=f"Prediction S(r) @ Y={Y_target:.3g}")
            plt.xlabel("r (lattice units)")
            plt.ylabel("S(r)")
            plt.title(f"Run {rid5} — Dipole S(r) comparison")
            plt.legend()
            plt.tight_layout()
            pdf.savefig(fig1)
            plt.close(fig1)

            # Page 2: maps side by side
            fig2, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            im0 = axs[0].imshow(map_true, origin="lower", vmin=vmin, vmax=vmax)
            axs[0].set_title(f"Truth center correlator @ Y={Y_truth:.3g}")
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("y")
            im1 = axs[1].imshow(map_pred, origin="lower", vmin=vmin, vmax=vmax)
            axs[1].set_title(f"Prediction center correlator @ Y={Y_target:.3g}")
            axs[1].set_xlabel("x")
            cbar = fig2.colorbar(im1, ax=axs, fraction=0.046, pad=0.04)
            cbar.set_label("C(x) = Re \mathrm{Tr}[U_0 U^\dagger(x,y)]/3$")
            pdf.savefig(fig2)
            plt.close(fig2)
        print(f"[ok] Wrote multi-page PDF: {out_path}")
    else:
        # Save S(r) figure to requested path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure()
        plt.plot(r_true, S_true, label=f"Truth S(r) @ Y={Y_truth:.3g}", linestyle='--')
        plt.plot(r_pred, S_pred, label=f"Prediction S(r) @ Y={Y_target:.3g}")
        plt.xlabel("r (lattice units)")
        plt.ylabel("S(r)")
        plt.title(f"Run {rid5} — Dipole S(r) comparison")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[ok] Wrote {out_path}")

        # Also save the maps as a separate PDF
        maps_pdf = out_path.with_name(out_path.stem + "_center_maps.pdf")
        with PdfPages(maps_pdf) as pdf:
            fig2, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            im0 = axs[0].imshow(map_true, origin="lower", vmin=vmin, vmax=vmax)
            axs[0].set_title(f"Truth center correlator @ Y={Y_truth:.3g}")
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("y")
            im1 = axs[1].imshow(map_pred, origin="lower", vmin=vmin, vmax=vmax)
            axs[1].set_title(f"Prediction center correlator @ Y={Y_target:.3g}")
            axs[1].set_xlabel("x")
            cbar = fig2.colorbar(im1, ax=axs, fraction=0.046, pad=0.04)
            cbar.set_label("C(x) = Re \mathrm{Tr}[U_0 U^\dagger(x,y)]/3$")
            pdf.savefig(fig2)
            plt.close(fig2)
        print(f"[ok] Wrote maps PDF: {maps_pdf}")

if __name__ == "__main__":
    main()
