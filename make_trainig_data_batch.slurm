#!/bin/bash -l
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -A m1820
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00 
#SBATCH -J jimwlk_cpu
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -L scratch

set -euo pipefail
umask 0022

# ---- User paths ----------------------------------------------------------------
ORCH="$HOME/ML/jimwlk/jimwlk_rc_lhs.py"
REPO="$HOME/ML/jimwlk/jimwlk"
JIMWLK_BIN="$HOME/ML/jimwlk/jimwlk/build/bin/jimwlk"
TEMPLATE="$HOME/ML/jimwlk/jimwlk_input_template"

# One job workspace; single OUT_ROOT for all passes
OUT_PARENT="$SCRATCH/jimwlk_runs_128_test_2"
OUT_ROOT="${OUT_PARENT}/${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT"

# Seeds: unique per pass to avoid reuse
BASE_SEED0=2025
SEED_STRIDE_PER_PASS=100000

# How many tasks per pass (matches srun -–ntasks)
NTASKS_PER_PASS=128

ulimit -s unlimited
export OMP_STACKSIZE=1G

export OMP_NUM_THREADS=1

# ---- Per-task runner: orchestrator only ----------------------------------------
RUNNER="$OUT_ROOT/.runner_${SLURM_JOB_ID}.sh"
cat > "$RUNNER" <<'EOF'
#!/bin/bash
set -euo pipefail

: "${ORCH:?}" "${REPO:?}" "${JIMWLK_BIN:?}" "${TEMPLATE:?}" "${OUT_ROOT:?}" \
  "${BASE_SEED:?}" "${PID_OFFSET:?}"

TASK_ID=${SLURM_PROCID:-0}
GLOBAL_PID=$(( PID_OFFSET + TASK_ID ))
PID=$(printf '%05d' "${GLOBAL_PID}")

RUN_DIR="${OUT_ROOT}/run_${PID}"
mkdir -p "${RUN_DIR}"

TPL="${RUN_DIR}/template.input"
cp "${TEMPLATE}" "${TPL}"
sed -Ei 's/^[[:space:]]*output_dir[[:space:]]+\S+/output_dir evolved_wilson_lines/' "${TPL}" || true
grep -q '^EndOfFile' "${TPL}" || echo 'EndOfFile' >> "${TPL}"

SEED=$(( BASE_SEED + TASK_ID ))

python -u "${ORCH}" \
  --repo "${REPO}" \
  --bin "${JIMWLK_BIN}" \
  --template "${TPL}" \
  --out "${RUN_DIR}" \
  --samples 1 \
  --runs_per_param 1 \
  --seed "${SEED}" \
  --omp 1
EOF
chmod +x "$RUNNER"

export ORCH REPO JIMWLK_BIN TEMPLATE OUT_ROOT

# ---- Helpers for flatten/export -------------------------------------------------
shopt -s nullglob dotglob

merge_dir() {
  # merge_dir SRC DEST (idempotent: keep existing files in DEST)
  local src="$1" dest="$2"
  if [ -d "$dest" ]; then
    echo "[merge] $src -> $dest"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --ignore-existing "${src}/" "${dest}/"
      rm -rf "${src}"
    else
      mkdir -p "${dest}"
      cp -a "${src}/." "${dest}/"
      rm -rf "${src}"
    fi
  else
    mv "${src}" "${dest}"
  fi
}

rewrite_manifest_child() {
  # .../run_PID/run_CID/... -> .../run_PID_CID/...
  local dest="$1" pid="$2" cid="$3" man="${dest}/manifest.json"
  [ -f "$man" ] || return 0
  sed -i -e "s|run_${pid}/run_${cid}|run_${pid}_${cid}|g" "$man" || true
}
rewrite_manifest_parent() {
  # .../run_PID/... -> .../run_PID_00000/...
  local dest="$1" pid="$2" man="${dest}/manifest.json"
  [ -f "$man" ] || return 0
  sed -i -E "s|(run_${pid})(/|$)|\1_00000\2|g" "$man" || true
  sed -i -E "s|(run_${pid})/|\1_00000/|g" "$man" || true
}

update_manifest_paths() {
  local man="$1"
  [ -f "$man" ] || return 0
  # Replace any absolute/relative prefix pointing into the per-job workspace
  perl -0777 -pe '
    my $o = $ENV{OLD};
    my $n = $ENV{NEW};
    # OLD/ -> NEW/  (handles when a slash follows)
    s!\Q$o\E/!$n/!g;
    # OLD  -> NEW   (handles end-of-string or when no slash follows)
    s!\Q$o\E!$n!g;
  ' -i -- "$man" || true
}

# update_manifest_paths() {
#   local man="$1"
#   [ -f "$man" ] || return 0
#   perl -0777 -pe '
#     my $o = $ENV{OLD};
#     my $n = $ENV{NEW};
#     s!\Q$o\E/!$n/!g;   # OLD/ -> NEW/
#     s!\Q$o\E!$n!g;     # OLD  -> NEW
#   ' -i -- "$man" || true
# }

flatten_and_export() {
  echo "[flatten] scanning ${OUT_ROOT} for run_XXXXX/run_YYYYY children..."

  # 1) Promote child runs: OUT_ROOT/run_XXXXX/run_YYYYY -> OUT_PARENT/run_XXXXX_YYYYY
  mapfile -d '' CHILDREN < <(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -type d \
    -name 'run_[0-9][0-9][0-9][0-9][0-9]' -print0)
  echo "[flatten] found ${#CHILDREN[@]} child run directories"

  for child in "${CHILDREN[@]}"; do
    parent_dir=$(dirname "$child")
    parent_base=$(basename "$parent_dir")      # run_XXXXX
    [[ "$parent_base" == run_[0-9][0-9][0-9][0-9][0-9] ]] || continue
    pid=${parent_base#run_}
    child_base=$(basename "$child")            # run_YYYYY
    cid=${child_base#run_}
    dest="${OUT_PARENT}/run_${pid}_${cid}"

    echo "[flatten] PLACE $child -> $dest"
    merge_dir "$child" "$dest"
    rewrite_manifest_child "$dest" "$pid" "$cid"
  done

  # 2) Sweep leftovers from parent run_XXXXX into run_PID_00000
  while IFS= read -r -d '' run_dir; do
    pid=${run_dir##*/}; pid=${pid#run_}
    dest0="${OUT_PARENT}/run_${pid}_00000"
    mkdir -p "$dest0"

    for item in "$run_dir"/*; do
      [ -e "$item" ] || continue
      base=$(basename "$item")
      if [[ "$base" == run_[0-9][0-9][0-9][0-9][0-9] && -d "$item" ]]; then
        cid=${base#run_}
        dest="${OUT_PARENT}/run_${pid}_${cid}"
        echo "[flatten] (late) $item -> $dest"
        merge_dir "$item" "$dest"
        rewrite_manifest_child "$dest" "$pid" "$cid"
        continue
      fi
      echo "[flatten] leftover $item -> $dest0/"
      mv "$item" "$dest0/"
    done
    rewrite_manifest_parent "$dest0" "$pid"
  done < <(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'run_[0-9][0-9][0-9][0-9][0-9]' -print0)

  # 3) Rewrite manifest.json paths to strip the JOBID directory
  echo "[manifest] stripping JOBID ($SLURM_JOB_ID) from manifest paths"
  OLD="$OUT_ROOT"
  NEW="$OUT_PARENT"
  export OLD NEW


  # 3) Rewrite manifest.json paths to strip the JOBID directory
  echo "[manifest] stripping JOBID ($SLURM_JOB_ID) from manifest paths"
  OLD="$OUT_ROOT"   # e.g., .../jimwlk_runs_few16/1234567
  NEW="$OUT_PARENT" # e.g., .../jimwlk_runs_few16
  export OLD NEW

  # Rewrite EVERY manifest under OUT_PARENT (no leftover job-id paths)
  while IFS= read -r -d '' man; do
    update_manifest_paths "$man"
  done < <(find "$OUT_PARENT" -type f -name 'manifest.json' -print0)


  # while IFS= read -r -d '' man; do
  #   update_manifest_paths "$man"
  # done < <(find "$OUT_PARENT" -mindepth 2 -maxdepth 2 -type f -name 'manifest.json' \
  #          -path "$OUT_PARENT/run_*_*/*" -print0)

  # # Optionally also fix any manifests still under OUT_ROOT (if any)
  # while IFS= read -r -d '' man; do
  #   update_manifest_paths "$man"
  # done < <(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -type f -name 'manifest.json' -print0)

  echo "[flatten] export completed; results in ${OUT_PARENT}"
}

# ---- Run PASSES passes sequentially on the same node --------------------------------
PASSES=1
for PASS in $(seq 0 $((PASSES-1))); do
  printf "\n===== PASS %d/%d =====\n" "$((PASS+1))" "$PASSES"

  BASE_SEED=$(( BASE_SEED0 + PASS * SEED_STRIDE_PER_PASS ))
  PID_OFFSET=$(( PASS * NTASKS_PER_PASS ))

  export BASE_SEED PID_OFFSET

  # Launch the usual 128-way srun on the same allocation
  srun --cpu-bind=cores --mem-bind=local --ntasks=${NTASKS_PER_PASS} --ntasks-per-node=${NTASKS_PER_PASS} \
       --cpus-per-task=${SLURM_CPUS_PER_TASK} "$RUNNER"

  # Flatten/export into the SAME destination folder (no CID offsetting)
  flatten_and_export
done

echo "[all done] All runs exported under: ${OUT_PARENT}"
