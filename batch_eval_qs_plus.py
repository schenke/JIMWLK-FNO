#!/usr/bin/env python3
"""
Batch-evaluate Q_s **and** quadrupole operators at a target rapidity Y across many runs.

- Scans a folder for run subfolders containing `manifest.json`.
- For each run:
    * Loads U0 (earliest snapshot) and the "truth" Uy at the snapshot
      whose Y is closest to the requested --target_y.
    * Uses the same input packing and postprocessing as `compare_predict_Y.py`:
      predict 18 channels, repair, pack to complex SU(3), polar/SVD det=1.
    * Computes:
        - Q_s (axial dipole, threshold 0.5 by default) for prediction and truth.
        - Quadrupole Q[(dx1,dy1),(dx2,dy2)] for a selection of offset-pair configurations.
- Produces:
    * Scatter plot Q_s(pred) vs Q_s(true) with diagonal.
    * Scatter plots for quadrupoles (one per pair + a combined figure).
    * CSV/JSON of per-run results that include Q_s and the chosen quadrupoles.
    * Aggregate metrics JSON: RMSE/MAE/R2 for Q_s and each quadrupole pair.

Requires `compare_predict_Y.py` in PYTHONPATH (same folder is fine).

Example
-------
python batch_eval_qs_plus.py \
  --runs /path/to/runs_root \
  --ckpt /path/to/evolver_best.pt \
  --target_y 1.0 \
  --outdir batch_eval_out \
  --quad_pairs "(1,0);(0,1)|(2,0);(0,2)|(1,1);(2,0)"
"""
from __future__ import annotations

from pathlib import Path
import json, argparse, itertools
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt, numpy as np, itertools

import numpy as np
import torch

# --- import helpers from your pipeline ---
try:
    from compare_predict_Y import (
        safe_load_model, predict_Uy,
        y18_repair, pack_to_complex, svd_polar_det1,
        read_wilson_binary, pick_target_snapshot,
        qs_from_axial_torch,
        quadrupole_autocorr,
    )
except Exception as e:
    print("[fatal] Could not import required helpers from compare_predict_Y.py")
    raise

# ----------------- utilities -----------------
def find_runs(root: Path) -> List[Path]:
    root = Path(root)
    runs = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "manifest.json").exists():
            runs.append(p)
    return runs

def manifest_load(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text())

def manifest_params(man: dict) -> dict:
    params = man.get("params", {})
    return {
        "m": float(params.get("m_GeV", params.get("m", 0.0))),
        "Lambda_QCD": float(params.get("Lambda_QCD_GeV", params.get("Lambda_QCD", 0.0))),
        "mu0": float(params.get("mu0_GeV", params.get("mu0", 0.0))),
    }

def _ensure_dir(p: Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _project_su3(U: np.ndarray) -> np.ndarray:
    """SVD polar projection + det=1 safeguard."""
    return svd_polar_det1(U.astype(np.complex128, copy=True))

def _metrics(trues: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    trues = np.asarray(trues, dtype=float)
    preds = np.asarray(preds, dtype=float)
    m = np.isfinite(trues) & np.isfinite(preds)
    if m.sum() == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
    e = preds[m] - trues[m]
    rmse = float(np.sqrt(np.mean(e**2)))
    mae  = float(np.mean(np.abs(e)))
    tbar = float(np.mean(trues[m]))
    ss_res = float(np.sum((preds[m]-trues[m])**2))
    ss_tot = float(np.sum((trues[m]-tbar)**2)) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

# ----------------- quadrupole pairs parsing -----------------
Pair = Tuple[Tuple[int,int], Tuple[int,int]]

def parse_pairs(pairs_str: str) -> List[Pair]:
    """
    Parse string like "(1,0);(0,1)|(2,0);(0,2)" into
    [((1,0),(0,1)), ((2,0),(0,2))].
    Default selection is small offsets that are typically informative.
    """
    if not pairs_str:
        return [((1,0),(0,1)), ((2,0),(0,2)), ((1,1),(2,0))]
    groups: List[Pair] = []
    for grp in pairs_str.split("|"):
        grp = grp.strip().strip("[]")
        if not grp: 
            continue
        a, b = grp.split(";")
        def _p(s: str) -> Tuple[int,int]:
            s = s.strip().strip("()")
            x, y = s.split(",")
            return (int(x), int(y))
        groups.append((_p(a), _p(b)))
    return groups

# ----------------- core per-run evaluation -----------------
def compute_qs_scalar(U_np: np.ndarray, device: str = "cpu") -> float:
    """Compute Q_s from an SU(3) field U_np [H,W,3,3] (complex)."""
    U = torch.from_numpy(U_np).to(torch.complex64)  # [H,W,3,3]
    U = U.unsqueeze(0)  # [1,H,W,3,3]
    with torch.no_grad():
        qs = qs_from_axial_torch(U)  # [1]
    return float(qs.item())

def eval_one_run(run_dir: Path, model, device: str, target_y: float,
                 quad_pairs: List[Pair]) -> Dict:
    """Load U0 and closest-to-target_y Uy(true); predict Uy(pred); compute Q_s + quadrupoles."""
    run_dir = Path(run_dir)
    man = manifest_load(run_dir)

    # ds
    ds = float(man.get("ds", man.get("params", {}).get("ds", float("nan"))))
    if not (ds == ds):
        raise ValueError(f"[{run_dir.name}] ds not found in manifest.json; add a top-level 'ds'.")

    snaps = sorted(man["snapshots"], key=lambda s: int(s["steps"]))
    if not snaps:
        raise ValueError(f"[{run_dir.name}] manifest has no snapshots.")

    # U0 (earliest)
    s0 = snaps[0]
    U0_path = Path(s0["path"]); 
    if not U0_path.is_absolute(): U0_path = run_dir / U0_path
    U0 = read_wilson_binary(U0_path)

    # Target truth (closest Y to requested)
    tgt_snap, Y = pick_target_snapshot(man, ds, target_y=target_y)
    Uy_path = Path(tgt_snap["path"])
    if not Uy_path.is_absolute(): Uy_path = run_dir / Uy_path
    Uy_true_raw = read_wilson_binary(Uy_path)
    Uy_true = _project_su3(Uy_true_raw.copy())

    # Predict
    params = manifest_params(man)
    y18 = predict_Uy(model, U0, float(Y), params, device=device)
    U_pred = _project_su3(pack_to_complex(y18_repair(y18)))

    # Q_s
    Qs_true = compute_qs_scalar(Uy_true, device=device)
    Qs_pred = compute_qs_scalar(U_pred,   device=device)

    # Quadrupoles
    quad_true = {}
    quad_pred = {}
    for (a,b) in quad_pairs:
        q_t = float(quadrupole_autocorr(Uy_true, pairs=[(a,b)])[0])
        q_p = float(quadrupole_autocorr(U_pred,  pairs=[(a,b)])[0])
        quad_true[str((a,b))] = q_t
        quad_pred[str((a,b))] = q_p

    return {
        "run": run_dir.name,
        "steps": int(tgt_snap["steps"]),
        "Y": float(Y),
        "Qs_true": float(Qs_true),
        "Qs_pred": float(Qs_pred),
        "quadrupole_true": quad_true,
        "quadrupole_pred": quad_pred,
    }

# ----------------- plotting -----------------
def _scatter(ax, x, y, label=None):
    import numpy as np
    lo = np.nanmin([np.nanmin(x), np.nanmin(y)])
    hi = np.nanmax([np.nanmax(x), np.nanmax(y)])
    pad = 0.05 * (hi - lo if np.isfinite(hi - lo) else 1.0)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], linestyle=":", linewidth=1.25)
    ax.scatter(x, y, s=18, alpha=0.5, label=label)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Truth"); ax.set_ylabel("Prediction")
    if label: ax.legend(loc="best", frameon=False)

def scatter_qs(
    results: List[Dict],
    out_png: Path,
    title: str | None = None,
    target_y: float | None = None,
    point_alpha: float = 0.6,
):
    if not results:
        return

    import matplotlib.pyplot as plt
    x = np.array([r["Qs_true"] for r in results], dtype=np.float64)
    y = np.array([r["Qs_pred"] for r in results], dtype=np.float64)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4), dpi=140, constrained_layout=True)
    ax.scatter(x, y, s=36, alpha=point_alpha)

    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    pad = 0.05 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", alpha=0.6)

    ax.set_xlabel(r"$Q_s$ (truth)")
    ax.set_ylabel(r"$Q_s$ (prediction)")

    base = r"$Q_s$ prediction vs. truth" if title is None else title
    if target_y is not None:
        base = rf"{base} — $Y={target_y:.3f}$"
    ax.set_title(base)

    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)                              # PNG
    fig.savefig(out_png.with_suffix(".pdf"))          # PDF
    plt.close(fig)


def scatter_quadrupoles(results: List[Dict], pairs: List[Pair], outdir: Path, combined: bool = True):
    """Per-pair scatter plots + optional combined figure."""
    import matplotlib.pyplot as plt, numpy as np, itertools
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Build arrays per pair
    by_pair = {}
    for r in results:
        for p in pairs:
            key = str(p)
            by_pair.setdefault(key, {"t": [], "p": []})
            by_pair[key]["t"].append(r["quadrupole_true"][key])
            by_pair[key]["p"].append(r["quadrupole_pred"][key])

    pngs = []
    # individual
    for key, data in by_pair.items():
        x = np.array(data["t"], dtype=np.float64)
        y = np.array(data["p"], dtype=np.float64)
        fig = plt.figure(figsize=(5.8, 4.6), dpi=140); ax = fig.add_subplot(1,1,1)
        _scatter(ax, x, y, label=f"{key}")
        ax.set_title(f"Quadrupole {key}: prediction vs truth")
        out_png = outdir / f"quad_pred_vs_true_{key.replace(' ', '').replace('(', '').replace(')', '').replace(',', '_')}.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        pngs.append(out_png)

    combined_png = None
    if combined and by_pair:
        markers = itertools.cycle(["o","s","^","D","v","*","P","X"])
        fig = plt.figure(figsize=(6.4, 5.2), dpi=140); ax = fig.add_subplot(1,1,1)
        allx, ally = [], []
        for key, data in by_pair.items():
            x = np.asarray(data["t"], float); y = np.asarray(data["p"], float)
            m = next(markers)
            ax.scatter(x, y, s=22, alpha=0.5, marker=m, label=key)
            allx.append(x); ally.append(y)
        allx = np.concatenate(allx); ally = np.concatenate(ally)
        lo = np.nanmin([np.nanmin(allx), np.nanmin(ally)])
        hi = np.nanmax([np.nanmax(allx), np.nanmax(ally)])
        pad = 0.05 * (hi - lo if np.isfinite(hi - lo) else 1.0)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], linestyle=":", linewidth=1.25)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Truth"); ax.set_ylabel("Prediction")
        ax.set_title("Quadrupole: prediction vs truth (combined)")
        ax.legend(loc="best", frameon=False, fontsize=9)
        combined_png = outdir / "quad_pred_vs_true_combined.png"
        fig.savefig(combined_png, bbox_inches="tight")                 # PNG
        fig.savefig(combined_png.with_suffix(".pdf"), bbox_inches="tight")  # PDF
        plt.close(fig)

    return [str(p) for p in pngs], (str(combined_png) if combined_png else None)



# def scatter_qs(
#     results: List[Dict],
#     out_png: Path,
#     title: str | None = None,
#     target_y: float | None = None,
#     point_alpha: float = 0.6,
# ):
#     if not results:
#         return

#     x = np.array([r["Qs_true"] for r in results], dtype=np.float64)
#     y = np.array([r["Qs_pred"] for r in results], dtype=np.float64)

#     fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4), dpi=140, constrained_layout=True)
#     ax.scatter(x, y, s=36, alpha=point_alpha)

#     lo = float(np.nanmin([x.min(), y.min()]))
#     hi = float(np.nanmax([x.max(), y.max()]))
#     pad = 0.05 * (hi - lo + 1e-12)
#     ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", alpha=0.6)

#     ax.set_xlabel(r"$Q_s$ (truth)")
#     ax.set_ylabel(r"$Q_s$ (prediction)")

#     # Build title and always append Y if provided                                                                                                                                                                  
#     base = r"$Q_s$ prediction vs. truth" if title is None else title
#     if target_y is not None:
#         base = rf"{base} — $Y={target_y:.3f}$"
#     ax.set_title(base)

#     ax.grid(True, alpha=0.3)
#     out_png.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_png)
#     plt.close(fig)

    
# # def scatter_qs(results: List[Dict], out_png: Path, title: str | None = None, target_y: float | None = None):
# #     if not results: return
# #     import matplotlib.pyplot as plt, numpy as np
# #     x = np.array([r["Qs_true"] for r in results], dtype=np.float64)
# #     y = np.array([r["Qs_pred"] for r in results], dtype=np.float64)
# #     fig = plt.figure(figsize=(5.8, 4.6), dpi=140); ax = fig.add_subplot(1,1,1)
# #     _scatter(ax, x, y, label="Q_s")
# #     if title is None: title = "Q_s: prediction vs truth"
# #     if target_y is not None: title += f"  (target Y≈{target_y:g})"
# #     ax.set_title(title)
# #     out_png.parent.mkdir(parents=True, exist_ok=True)
# #     fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

# def scatter_quadrupoles(results: List[Dict], pairs: List[Pair], outdir: Path, combined: bool = True):
#     """Per-pair scatter plots + optional combined figure."""
#     outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

#     # Build arrays per pair
#     by_pair = {}
#     for r in results:
#         for p in pairs:
#             key = str(p)
#             by_pair.setdefault(key, {"t": [], "p": []})
#             by_pair[key]["t"].append(r["quadrupole_true"][key])
#             by_pair[key]["p"].append(r["quadrupole_pred"][key])

#     pngs = []
#     # individual
#     for key, data in by_pair.items():
#         x = np.array(data["t"], dtype=np.float64)
#         y = np.array(data["p"], dtype=np.float64)
#         fig = plt.figure(figsize=(5.8, 4.6), dpi=140); ax = fig.add_subplot(1,1,1)
#         _scatter(ax, x, y, label=f"{key}")
#         ax.set_title(f"Quadrupole {key}: prediction vs truth")
#         out_png = outdir / f"quad_pred_vs_true_{key.replace(' ', '').replace('(', '').replace(')', '').replace(',', '_')}.png"
#         fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
#         pngs.append(out_png)

#     combined_png = None
#     if combined and by_pair:
#         markers = itertools.cycle(["o","s","^","D","v","*","P","X"])
#         fig = plt.figure(figsize=(6.4, 5.2), dpi=140); ax = fig.add_subplot(1,1,1)
#         allx, ally = [], []
#         for key, data in by_pair.items():
#             x = np.asarray(data["t"], float); y = np.asarray(data["p"], float)
#             m = next(markers)
#             ax.scatter(x, y, s=22, alpha=0.5, marker=m, label=key)
#             allx.append(x); ally.append(y)
#         allx = np.concatenate(allx); ally = np.concatenate(ally)
#         lo = np.nanmin([np.nanmin(allx), np.nanmin(ally)])
#         hi = np.nanmax([np.nanmax(allx), np.nanmax(ally)])
#         pad = 0.05 * (hi - lo if np.isfinite(hi - lo) else 1.0)
#         lo, hi = lo - pad, hi + pad
#         ax.plot([lo, hi], [lo, hi], linestyle=":", linewidth=1.25)
#         ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
#         ax.set_xlabel("Truth"); ax.set_ylabel("Prediction")
#         ax.set_title("Quadrupole: prediction vs truth (combined)")
#         ax.legend(loc="best", frameon=False, fontsize=9)
#         combined_png = outdir / "quad_pred_vs_true_combined.png"
#         fig.savefig(combined_png, bbox_inches="tight"); plt.close(fig)

#     return [str(p) for p in pngs], (str(combined_png) if combined_png else None)

# ----------------- aggregate + IO -----------------
def aggregate_metrics(results: List[Dict], pairs: List[Pair]) -> Dict:
    if not results:
        return {}
    out = {}
    # Qs metrics
    tq = np.array([r["Qs_true"] for r in results], dtype=np.float64)
    pq = np.array([r["Qs_pred"] for r in results], dtype=np.float64)
    out["Qs"] = _metrics(tq, pq)
    # per-pair
    for p in pairs:
        key = str(p)
        t = np.array([r["quadrupole_true"][key] for r in results], dtype=np.float64)
        y = np.array([r["quadrupole_pred"][key] for r in results], dtype=np.float64)
        out[key] = _metrics(t, y)
    return out

def save_table(results: List[Dict], out_csv: Path, pairs: List[Pair]):
    import csv
    out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)

    # flatten quadrupoles into columns
    pair_keys = [str(p) for p in pairs]
    cols = ["run","steps","Y","Qs_true","Qs_pred"] + \
           [f"{k}_true" for k in pair_keys] + [f"{k}_pred" for k in pair_keys]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            row = {
                "run": r["run"], "steps": r["steps"], "Y": r["Y"],
                "Qs_true": r["Qs_true"], "Qs_pred": r["Qs_pred"],
            }
            for k in pair_keys:
                row[f"{k}_true"] = r["quadrupole_true"][k]
                row[f"{k}_pred"] = r["quadrupole_pred"][k]
            w.writerow(row)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Batch-evaluate Q_s and quadrupoles over runs.")
    ap.add_argument("--runs", type=Path, required=True, help="Folder containing run subfolders with manifest.json")
    ap.add_argument("--ckpt", type=Path, required=True, help="Model checkpoint (.pt)")
    ap.add_argument("--target_y", type=float, required=True, help="Target rapidity Y to compare against (closest snapshot chosen per run)")
    ap.add_argument("--outdir", type=Path, default=Path("batch_eval_out"))
    ap.add_argument("--device", type=str, default="auto", help="cuda | cpu | auto")
    ap.add_argument("--quad_pairs", type=str, default="", help="e.g. \"(1,0);(0,1)|(2,0);(0,2)|(1,1);(2,0)|(4,1);(2,8)\"")
    ap.add_argument("--no_quad_combined", action="store_true", help="Do not write combined quadrupole scatter")

    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model, margs = safe_load_model(args.ckpt, device=device)

    # parse pairs
    pairs = parse_pairs(args.quad_pairs)

    results = []
    for run_dir in find_runs(args.runs):
        try:
            r = eval_one_run(run_dir, model, device=device, target_y=float(args.target_y), quad_pairs=pairs)
            print(f"[ok] {run_dir.name}: steps={r['steps']}  Y≈{r['Y']:.6g}  Qs_pred={r['Qs_pred']:.4g}")
            results.append(r)
        except Exception as e:
            print(f"[fail] {Path(run_dir).name}: {e!r}")

    # Save tables
    _ensure_dir(args.outdir)
    save_table(results, args.outdir / "per_run_qs_and_quadrupoles.csv", pairs)
    (args.outdir / "per_run_qs_and_quadrupoles.json").write_text(json.dumps(results, indent=2))


    # Plots
    y_tag = f"Y{float(args.target_y):.3f}".replace(".", "p")

    # Qs scatter with Y in the filename
    qs_path = args.outdir / f"qs_pred_vs_true_{y_tag}.png"
    scatter_qs(results, qs_path, target_y=float(args.target_y))

    # Quadrupoles: put outputs in a Y-tagged subfolder (or adjust if your function takes a filename)
    quad_outdir = args.outdir / f"quadrupoles_{y_tag}"
    pngs, combined = scatter_quadrupoles(
    results, pairs, quad_outdir, combined=(not args.no_quad_combined))
    
    # scatter_qs(results, args.outdir / "qs_pred_vs_true.png", target_y=float(args.target_y))
    # pngs, combined = scatter_quadrupoles(results, pairs, args.outdir, combined=(not args.no_quad_combined))

    # Metrics
    metrics = aggregate_metrics(results, pairs)
    (args.outdir / "aggregate_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nAggregate metrics:")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}: " + ", ".join(f"{m}={val:.4g}" for m,val in v.items()))
        else:
            print(f"  {k}: {v}")

    # Summary
    print("\nWrote:")
    print(f"  - {args.outdir / 'per_run_qs_and_quadrupoles.csv'}")
    print(f"  - {args.outdir / 'per_run_qs_and_quadrupoles.json'}")
    print(f"  - {qs_path}")
    if pngs:
        for p in pngs: print(f"  - {p}")
    if combined: print(f"  - {combined}")
    print(f"  - {args.outdir / 'aggregate_metrics.json'}")

if __name__ == "__main__":
    main()
