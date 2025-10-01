#!/bin/bash -l
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -A m1820
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH -t 00:20:00          # was 00:05:00 → give the export time to run
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

# Per-job output root
OUT_PARENT="$SCRATCH/jimwlk_runs_256"
OUT_ROOT="${OUT_PARENT}/${SLURM_JOB_ID}"

BASE_SEED=2025
mkdir -p "$OUT_ROOT"
export OMP_NUM_THREADS=1
ulimit -s unlimited
export OMP_STACKSIZE=1G

# ---- Per-task runner: orchestrator only (no flattening here) -------------------
RUNNER="$OUT_ROOT/.runner_${SLURM_JOB_ID}.sh"
cat > "$RUNNER" <<'EOF'
#!/bin/bash
set -euo pipefail

: "${ORCH:?}" "${REPO:?}" "${JIMWLK_BIN:?}" "${TEMPLATE:?}" "${OUT_ROOT:?}" "${BASE_SEED:?}"

TASK_ID=${SLURM_PROCID:-0}
PID=$(printf '%05d' "${TASK_ID}")
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
  --runs_per_param 10 \
  --seed "${SEED}" \
  --omp 1
EOF
chmod +x "$RUNNER"

export ORCH REPO JIMWLK_BIN TEMPLATE OUT_ROOT BASE_SEED

# ---- Launch tasks --------------------------------------------------------------
srun --cpu-bind=cores --mem-bind=local --ntasks=128 --ntasks-per-node=128 --cpus-per-task=${SLURM_CPUS_PER_TASK} "$RUNNER"

# ---- Global flatten (regex-free) ----------------------------------------------
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
  perl -0777 -pe "s|(run_${pid})/run_(${cid})|\$1_\$2|g" -i -- "$man" || true
}
rewrite_manifest_parent() {
  # .../run_PID/... -> .../run_PID_00000/...
  local dest="$1" pid="$2" man="${dest}/manifest.json"
  [ -f "$man" ] || return 0
  perl -0777 -pe "s|(run_${pid})(/|$)|\${1}_00000\2|g; s|(run_${pid})/|\${1}_00000/|g" -i -- "$man" || true
}

echo "[flatten] scanning ${OUT_ROOT} for run_XXXXX/run_YYYYY children..."

# 1) Promote ALL child runs found at depth 2: OUT_ROOT/run_XXXXX/run_YYYYY
mapfile -d '' CHILDREN < <(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -type d \
  -name 'run_[0-9][0-9][0-9][0-9][0-9]' -print0)

echo "[flatten] found ${#CHILDREN[@]} child run directories"
for child in "${CHILDREN[@]}"; do
  parent_dir=$(dirname "$child")
  parent_base=$(basename "$parent_dir")          # run_XXXXX
  [[ "$parent_base" == run_[0-9][0-9][0-9][0-9][0-9] ]] || continue
  pid=${parent_base#run_}
  child_base=$(basename "$child")                # run_YYYYY
  cid=${child_base#run_}
  dest="${OUT_ROOT}/run_${pid}_${cid}"

  echo "[flatten] PLACE $child -> $dest"
  merge_dir "$child" "$dest"
  rewrite_manifest_child "$dest" "$pid" "$cid"
done

# 2) For each remaining run_XXXXX parent, sweep leftovers into run_PID_00000
while IFS= read -r -d '' run_dir; do
  pid=${run_dir##*/}; pid=${pid#run_}
  dest0="${OUT_ROOT}/run_${pid}_00000"
  mkdir -p "$dest0"

  for item in "$run_dir"/*; do
    [ -e "$item" ] || continue
    base=$(basename "$item")
    # Promote any late run_YYYYY too
    if [[ "$base" == run_[0-9][0-9][0-9][0-9][0-9] && -d "$item" ]]; then
      cid=${base#run_}
      dest="${OUT_ROOT}/run_${pid}_${cid}"
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

# 3) MOVE run_XXXXX_YYYYY UP one level into OUT_PARENT, leave run_XXXXX in OUT_ROOT
echo "[export] moving run_XXXXX_YYYYY from ${OUT_ROOT} -> ${OUT_PARENT}"

while IFS= read -r -d '' subrun; do
  base=$(basename "$subrun")               # run_XXXXX_YYYYY
  dest="${OUT_PARENT}/${base}"
  echo "[export] ${subrun} -> ${dest}"
  merge_dir "$subrun" "$dest"
done < <(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'run_[0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9]' -print0)

echo "[done] Exports are in ${OUT_PARENT}; per-task parents remain in ${OUT_ROOT}"

# 4) Rewrite manifest.json paths to strip the JOBID directory
echo "[manifest] stripping JOBID ($SLURM_JOB_ID) from manifest paths"

OLD="$OUT_ROOT"       # .../jimwlk_runs/$SLURM_JOB_ID
NEW="$OUT_PARENT"     # .../jimwlk_runs
export OLD NEW

update_manifest() {
  local man="$1"
  [ -f "$man" ] || return 0
  # Replace any occurrence of $OLD (with or without trailing slash) with $NEW
  perl -0777 -pe '
    my $o = $ENV{OLD};
    my $n = $ENV{NEW};
    s!\Q$o\E/!$n/!g;   # OLD/ -> NEW/
    s!\Q$o\E!$n!g;     # OLD  -> NEW  (in case it wasn’t followed by /)
  ' -i -- "$man" || true
}

# Fix manifests in the exported locations: $OUT_PARENT/run_XXXXX_YYYYY/**
while IFS= read -r -d '' man; do
  update_manifest "$man"
done < <(find "$OUT_PARENT" -mindepth 2 -maxdepth 2 -type f -name 'manifest.json' \
         -path "$OUT_PARENT/run_*_*/*" -print0)

# (Optional) also fix any manifests that remain under the job workspace
while IFS= read -r -d '' man; do
  update_manifest "$man"
done < <(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -type f -name 'manifest.json' -print0)

echo "[manifest] path rewrite complete"
