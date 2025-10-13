#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import optuna

def spike_slab(trial: optuna.trial.Trial, name: str, low: float, high: float, log: bool = True, zero_ok: bool = False) -> float:
    """Return 0.0 with some probability; otherwise sample log/uniform in [low, high]."""
    if zero_ok and trial.suggest_categorical(f"{name}__on", [True, False]) is False:
        return 0.0
    return trial.suggest_float(name, low, high, log=log)

LOG_LINE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]\s*train\s+([0-9.]+)\s+val\s+([0-9.]+)")
TUNER_SCALAR_RE = re.compile(r"\bTUNER_SCALAR\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
# (deprecated in this tuner) QS_BINS_RE = re.compile(r"Qs\(.+?bins\).*rmse=\[([0-9eE+.\s,-]+)\]")

def build_cmd(args, trial_outdir: Path, trial: optuna.trial.Trial):
    # ---- Suggest hyperparameters (compact, safe search space) ----
    p = {}
    p["lr"]              = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    p["weight_decay"]    = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    p["width"]           = trial.suggest_categorical("width", [48])
    p["modes"]           = trial.suggest_categorical("modes", [8, 10, 12, 14, 16])
    p["blocks"]          = trial.suggest_int("blocks", 8, 16)
    p["warmup_epochs"]   = trial.suggest_int("warmup_epochs", 0, 10)
    p["accum"]           = trial.suggest_categorical("accum", [4, 6, 8])
    p["rollout_k"]       = trial.suggest_int("rollout_k", 1, 4)
    p["rollout_consistency"] = trial.suggest_float("rollout_consistency", 0.0, 0.2)
    p["semigroup_weight"]    = trial.suggest_float("semigroup_weight", 0.0, 0.2)
    p["semigroup_prob"]      = trial.suggest_float("semigroup_prob", 0.0, 0.5)

    # Optional booleans
    if trial.suggest_categorical("amp", [True, False]):
        p["amp"] = True
    if trial.suggest_categorical("channels_last", [True, False]):
        p["channels_last"] = True
        
    # ---- Extra loss/physics weights & scalars ----
    p["dipole_weight"] = trial.suggest_float("dipole_weight", 1e-3, 1.)  # linear, no spike
    p["trace_weight"]   = spike_slab(trial, "trace_weight", 1e-3, 0.2, log=False)
    p["quad_weight"]    = spike_slab(trial, "quad_weight", 1e-3, 1., log=False)
    p["loops_weight"]   = spike_slab(trial, "loops_weight", 0, 0.2, log=False, zero_ok=True)
    p["geo_weight"]     = spike_slab(trial, "geo_weight", 0, 0.2, log=False, zero_ok=True)
    p["nll_weight"]     = spike_slab(trial, "nll_weight", 1e-3, 0.2, log=False)
    p["moment_weight"]  = spike_slab(trial, "moment_weight", 1e-3, 0.3, log=False)
    p["y_gain"]         = trial.suggest_float("y_gain", 1, 2, log=False)
    p["E1"]             = trial.suggest_int("E1", 0, 10, log=False)
    p["E2"]             = trial.suggest_categorical("E2", [10, 15, 20])


    # ---- Base command ----
    base = args.base_cmd.strip() or f"{shlex.quote(sys.executable)} {shlex.quote(str(args.train_script))}"
    cmd = [part for part in shlex.split(base) if part]

    # Core required & fixed args
    cmd += [
        "--data_root", str(args.data_root),
        "--out", str(trial_outdir),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--workers", str(args.workers),
        "--seed", str(args.seed + trial.number),  # vary seed a bit per trial
        "--nll_fullcov",
        "--nll_param_compose"
    ]

    
    # Apply suggested hyperparameters with a flexible flag-name map.
    # Default assumes CLI flags match keys; override with --flag_map_json '{"logical":"cli_name",...}' if needed.
    flag_name_map = {k: k for k in p.keys()}
    if getattr(args, "flag_map_json", ""):
        import json
        try:
            user_map = json.loads(args.flag_map_json)
            assert isinstance(user_map, dict)
            flag_name_map.update(user_map)
        except Exception as e:
            print(f"[WARN] Could not parse --flag_map_json: {e}", file=sys.stderr)

    # DEBUG: save sampled params for this trial
    import json

    for k, v in p.items():
        name = flag_name_map.get(k, k)
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{name}")
        else:
            cmd += [f"--{name}", str(v)]

    # Any passthrough extras
    if args.extra:
        cmd += shlex.split(args.extra)

    with open(trial_outdir / "suggested_params.json", "w") as f:
        json.dump(p, f, indent=2)
    print(f"[trial {trial.number:04d}] CMD: " + " ".join(shlex.quote(c) for c in cmd), flush=True)

    return cmd

def run_trial(args, trial: optuna.trial.Trial) -> float:
    # Create a unique outdir for the trial
    trial_outdir = Path(args.out_root) / f"trial_{trial.number:04d}"
    trial_outdir.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(args, trial_outdir, trial)
    
    # Stream stdout to parse val losses
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)

    best_val = float("inf")
    last_seen_epoch = -1
    last_rmse_bins = None
    try:
        start = time.time()
        for line in proc.stdout:

            # --- Only metric: TUNER_SCALAR <number> ---
            m_scalar = TUNER_SCALAR_RE.search(line)
            if m_scalar:
                try:
                    val = float(m_scalar.group(1))
                    # report to Optuna (step is the count of scalars seen)
                    last_seen_epoch += 1
                    trial.report(val, step=last_seen_epoch)
                    if val < best_val:
                        best_val = val
                    if args.prune and trial.should_prune():
                        raise optuna.TrialPruned(f"Pruned on TUNER_SCALAR at step {last_seen_epoch} with value={val:.6g}")
                except Exception:
                    pass
                continue  # ignore all other parsing for this line
            # stream trainer logs through
            print(f"[trial {trial.number:04d}] {line}", end="")

            # 2) When an epoch summary arrives, build a composite objective
            m = LOG_LINE.search(line)
            if not m:
                continue

            epoch    = int(m.group(1))
            # train_loss = float(m.group(3))  # if you ever want it
            val_loss = float(m.group(4))      # your existing validation scalar

            # Compose: val + λ * (worst-bin RMSE normalized by a target)
            obj_qs = 0.0
            if last_rmse_bins:
                worst = max(last_rmse_bins)
                target = 0.10   # <- pick a sensible target for worst-bin RMSE
                obj_qs = max(0.0, worst / max(target, 1e-12))

            lam = 1.0           # <- adjust or expose as an argument if you like
            val_total = val_loss + lam * obj_qs

            # Report composite objective to Optuna (so pruning/selection uses it)
            trial.report(val_total, step=epoch)
            if val_total < best_val:
                best_val = val_total

            if args.prune and trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch {epoch} with val_total={val_total:.5f}")

    

        # for line in proc.stdout:
        #     print(f"[trial {trial.number:04d}] {line}", end="")  # echo for visibility
        #     m = LOG_LINE.search(line)
        #     if m:
        #         epoch = int(m.group(1))
        #         total_epochs = int(m.group(2))
        #         train_loss = float(m.group(3))
        #         val_loss = float(m.group(4))
        #         last_seen_epoch = epoch
        #         # Report intermediate value for pruning
        #         trial.report(val_loss, step=epoch)
        #         if val_loss < best_val:
        #             best_val = val_loss
        #         # Early prune?
        #         if args.prune and trial.should_prune():
        #             raise optuna.TrialPruned(f"Pruned at epoch {epoch} with val={val_loss:.5f}")

        #     # Timeout per trial
        #     if args.time_limit_sec and (time.time() - start) > args.time_limit_sec:
        #         raise TimeoutError(f"Trial time limit ({args.time_limit_sec}s) exceeded")

        # proc.wait()
        # if proc.returncode != 0:
        #     raise RuntimeError(f"Training process exited with code {proc.returncode}.")

    except (optuna.TrialPruned, TimeoutError) as e:
        # Kill the process nicely, then force if needed
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            raise

    except Exception as e:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            raise

    # Fallback if nothing was parsed
    if best_val == float("inf"):
        # Try to read a hparams/metrics if present in outdir; else, conservatively return a large number
        best_val = 1e9

    # Optionally write a marker with the best val we saw
    try:
        with open(trial_outdir / "best_val.txt", "w") as f:
            f.write(f"{best_val}\n")
    except Exception:
        pass

    return best_val

def main():
    ap = argparse.ArgumentParser(description="Optuna tuner for train_evolver_cuda_opt.py")
    ap.add_argument("--train_script", type=Path, default=Path("./NERSC/scripts/train_evolver_cuda_opt.py"))
    ap.add_argument("--data_root", type=Path, required=True, help="Path to training data root (contains run_* dirs).")
    ap.add_argument("--out_root", type=Path, default=Path("tuning_runs"), help="Where to put trial outputs.")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30, help="Epochs per trial (keep small for speed).")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time_limit_sec", type=int, default=0, help="Optional per-trial timeout (0 = no limit).")
    ap.add_argument("--prune", action="store_true", help="Enable median-pruning based on val loss.")
    ap.add_argument("--base_cmd", type=str, default="", help="Override full base command (e.g., 'torchrun --standalone --nproc_per_node=1 ...').")
    ap.add_argument("--extra", type=str, default="", help="Any extra CLI flags to append to the training run.")
    ap.add_argument("--study_name", type=str, default="evolver_tuning")
    ap.add_argument("--storage", type=str, default="", help="Optuna storage URL for persistence (e.g., sqlite:///study.db).")

    args = ap.parse_args()
    args.out_root = Path(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("SLURM_PROCID",
          os.environ.get("PMI_RANK",
          os.environ.get("LOCAL_RANK", "0"))))
    
    # Create study
    sampler = optuna.samplers.TPESampler(n_startup_trials=min(10, max(2, args.n_trials // 5)), seed=args.seed + rank)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10, interval_steps=1) if args.prune else optuna.pruners.NopPruner()

    if args.storage:
        study = optuna.create_study(direction="minimize", study_name=args.study_name, sampler=sampler, pruner=pruner, storage=args.storage, load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize", study_name=args.study_name, sampler=sampler, pruner=pruner)

    try:
        study.optimize(lambda tr: run_trial(args, tr), n_trials=args.n_trials, gc_after_trial=True)
    except KeyboardInterrupt:
        print("Interrupted by user. Returning best-so-far.")

    print("\n=== Best Trial ===")
    bt = study.best_trial
    print(f"Value (min val loss): {bt.value}")
    print("Params:")
    for k, v in bt.params.items():
        print(f"  {k}: {v}")

    # Locate its outdir
    best_out = Path(args.out_root) / f"trial_{bt.number:04d}"
    ckpt = best_out / "evolver_best.pt"
    if ckpt.exists():
        print(f"\nBest checkpoint: {ckpt}")
    else:
        print("\n(evolver_best.pt not found—check logs; the training script should have saved it when val improved.)")

if __name__ == "__main__":
    main()

    
