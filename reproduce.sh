#!/usr/bin/env bash
set -euo pipefail

DATA_REPO=Xkev/BVSR-data
DATA=data
RESULTS=results
RUNS=runs
CKPT=ckpt
PY=${PY:-python}

TASKS="svg labelproj pertpred ccc mls_anomaly mls_calibration sl_lrbsz sl_ushape astro_cassini astro_voyager2 zap_h16 zap_h32"
PROPOSERS="kimi glm glmflash dsflash dsv4"
STES_RUNS="svg:plain svg:bvsr pertpred:plain pertpred:bvsr mls_anomaly:plain mls_anomaly:bvsr sl_lrbsz:plain sl_lrbsz:bvsr zap_h16:plain zap_h16:bvsr"

SIMPLETES_URL=${SIMPLETES_URL:-https://github.com/wq-will/SimpleTES.git}
SIMPLETES_COMMIT=a19a54b109db6185ab1f13dd59dd150074b24136
SIMPLETES_ASTRO_COMMIT=6f5091d3fd1c6c36b4a35a825d3a3bdfe1cb75cc
ZAPBENCH_URL=https://github.com/google-research/zapbench.git
ZAPBENCH_COMMIT=b08d584e3ba80125788ac915eec63a2e4e11467b
ADBENCH_URL=${ADBENCH_URL:-https://raw.githubusercontent.com/Minqi824/ADBench/3dac8221081e190f157d78e93bfa8867f90d0965/adbench/datasets/Classical}
LIANA_VERSION=${LIANA_VERSION:-1.8.1}
OP_BUCKET=${OP_BUCKET:-s3://openproblems-data/resources}
SVG_SRC=${SVG_SRC:-$OP_BUCKET/task_spatially_variable_genes/datasets/zenodo_spatial/dbitseq/mouse_e10_eye}
LABELPROJ_SRC=${LABELPROJ_SRC:-$OP_BUCKET/task_label_projection/datasets/cellxgene_census/gtex_v9/log_cp10k}
PERTPRED_SRC=${PERTPRED_SRC:-$OP_BUCKET/task_perturbation_prediction/datasets/neurips-2023-data}
CCC_SRC=${CCC_SRC:-$OP_BUCKET/datasets/openproblems_v1/tnbc_wu2021/log_cp10k/dataset.h5ad}
OP_RESULTS=${OP_RESULTS:-https://raw.githubusercontent.com/openproblems-bio/website/76ce7f288da591b1b19c32cbfe8ce50bc3706ece/results}
SVG_RESULTS=${SVG_RESULTS:-$OP_RESULTS/spatially_variable_genes/data/results.json}
LABELPROJ_RESULTS=${LABELPROJ_RESULTS:-$OP_RESULTS/label_projection/data/results.json}
PERTPRED_RESULTS=${PERTPRED_RESULTS:-$OP_RESULTS/perturbation_prediction/data/results.json}
PERTPRED_METHOD_INFO=${PERTPRED_METHOD_INFO:-$OP_RESULTS/perturbation_prediction/data/method_info.json}

SR_GPU=${SR_GPU:-0}
SR_PORT=${SR_PORT:-8765}
GPU_ENV=(env TORCH_COMPILE_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)

usage() {
  cat <<EOF
usage: bash reproduce.sh            download the released data ($DATA_REPO) and draw Table 1, Fig. 2, 3, 4
       bash reproduce.sh --rerun    rerun blind variation, selective retention and SimpleTES, then draw them
Results go to $RESULTS/. A finished --rerun step is recorded in $RUNS/.done/ and skipped next time.
EOF
}

MODE=pull
case "${1:-}" in
  "") ;;
  --rerun) MODE=rerun ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[ $# -le 1 ] || { usage >&2; exit 2; }

cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=$PWD
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
DONE=$RUNS/.done
USER_GPUS=${BVSR_GPUS:-}
SR_PID=

log() { printf '\n[reproduce] %s\n' "$*"; }

step() {
  local name=$1
  shift
  if [ -e "$DONE/$name" ]; then
    log "$name: done before, skipped"
    return 0
  fi
  log "$name"
  "$@"
  mkdir -p "$DONE"
  touch "$DONE/$name"
}

fetch_released() {
  if [ -d "$DATA/bv_results" ] && [ -d "$DATA/sr_data" ] && [ -d "$DATA/bvsr_simpletes" ]; then
    log "released data: using $DATA/bv_results, $DATA/sr_data, $DATA/bvsr_simpletes"
  else
    log "released data: downloading $DATA_REPO into $DATA/"
    hf download "$DATA_REPO" --repo-type dataset --local-dir "$DATA"
  fi
}

table1_fig2() {
  log "Table 1 and Fig. 2 from $1"
  "$PY" -m bvsr.bv.table1 --candidates "$1" --out "$RESULTS/table1.md"
  "$PY" -m bvsr.bv.fig2 --candidates "$1" --bits "$2" --out "$RESULTS/fig2.pdf"
}

fig3() {
  log "Fig. 3 from $*"
  "$PY" -m bvsr.sr.evaluate --data "$DATA/sr_data" --rankings "$@" --out "$RESULTS/fig3.json"
  "$PY" -m bvsr.sr.fig3 --curves "$RESULTS/fig3.json" --out "$RESULTS/fig3.pdf"
}

pull() {
  fetch_released
  mkdir -p "$RESULTS"
  table1_fig2 "$DATA/bv_results/candidates.csv.gz" "$DATA/bv_results/info_bits.csv"
  fig3 "$DATA/sr_data/eval_rankings.jsonl.gz"
  log "Fig. 4 from $DATA/bvsr_simpletes"
  "$PY" -m bvsr.stes.fig4 --data "$DATA/bvsr_simpletes" --out "$RESULTS/fig4.pdf"
  log "done: $RESULTS/table1.md $RESULTS/fig2.pdf $RESULTS/fig3.pdf $RESULTS/fig4.pdf"
}

need() {
  local v
  for v in "$@"; do
    [ -n "${!v:-}" ] || { echo "reproduce.sh --rerun needs $v in the environment" >&2; exit 1; }
  done
}

need_cmd() {
  local c
  for c in "$@"; do
    command -v "$c" >/dev/null 2>&1 || { echo "reproduce.sh --rerun needs the command $c" >&2; exit 1; }
  done
}

gpu_list() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi --query-gpu=index --format=csv,noheader | grep -vx "${1:-none}" | paste -sd, - || true
}

setup_simpletes() {
  local patch=$REPO/third_party/simpletes/simpletes_a19a54b.patch
  [ -d "$SIMPLETES_HOME/.git" ] || git clone "$SIMPLETES_URL" "$SIMPLETES_HOME"
  git -C "$SIMPLETES_HOME" fetch --quiet origin
  git -C "$SIMPLETES_HOME" checkout --quiet "$SIMPLETES_COMMIT"
  if git -C "$SIMPLETES_HOME" apply --reverse --check "$patch" 2>/dev/null; then
    echo "  patch already applied"
  else
    git -C "$SIMPLETES_HOME" apply --check "$patch"
    git -C "$SIMPLETES_HOME" apply "$patch"
    echo "  patch applied: $(git -C "$SIMPLETES_HOME" diff --stat | tail -1)"
  fi
  "$PY" -m pip install --no-deps -e "$SIMPLETES_HOME"
}

fetch() {
  mkdir -p "$(dirname "$2")"
  if [ -s "$2" ]; then echo "  have $2"; return; fi
  case "$1" in
    s3://*) need_cmd aws; aws s3 cp --no-sign-request "$1" "$2" ;;
    *) curl -fL --retry 3 -o "$2" "$1" ;;
  esac
  echo "  got  $2"
}

data_svg() {
  local d=${SVG_DATA:-$BVSR_DATA/svg}
  fetch "$SVG_SRC/dataset.h5ad" "$d/dataset.h5ad"
  fetch "$SVG_SRC/solution.h5ad" "$d/solution.h5ad"
  fetch "$SVG_RESULTS" "$d/svg_results.json"
}

data_labelproj() {
  local d=${LABELPROJ_DATA:-$BVSR_DATA/labelproj} f
  for f in train.h5ad test.h5ad solution.h5ad; do
    fetch "$LABELPROJ_SRC/$f" "$d/$f"
  done
  fetch "$LABELPROJ_RESULTS" "$d/lp_results.json"
}

data_pertpred() {
  local d=${PERTPRED_DATA:-$BVSR_DATA/pertpred} f
  for f in de_train.h5ad de_test.h5ad id_map.csv; do
    fetch "$PERTPRED_SRC/$f" "$d/$f"
  done
  fetch "$PERTPRED_RESULTS" "$d/pp_results.json"
  fetch "$PERTPRED_METHOD_INFO" "$d/pp_method_info.json"
}

data_ccc() {
  local d=${CCC_DATA:-$BVSR_DATA/ccc}
  local raw=$d/raw
  local resource=$raw/liana_${LIANA_VERSION}_omni_resource.csv
  fetch "$CCC_SRC" "$raw/tnbc_wu2021_log_cp10k_dataset.h5ad"
  if [ ! -s "$resource" ]; then
    "$PY" -m pip download --no-deps --only-binary=:all: --python-version 3.12 \
        "liana==$LIANA_VERSION" -d "$raw/liana_wheel"
    "$PY" - "$raw/liana_wheel" "$resource" <<'EOF'
import glob, sys, zipfile
wheel = sorted(glob.glob(f"{sys.argv[1]}/liana-*.whl"))[-1]
with zipfile.ZipFile(wheel) as z, open(sys.argv[2], "wb") as out:
    out.write(z.read("liana/resource/omni_resource.csv"))
print("  got ", sys.argv[2])
EOF
  fi
  "$PY" -m bvsr.tasks.prepare_ccc --src "$raw/tnbc_wu2021_log_cp10k_dataset.h5ad" \
      --resource "$resource" --out "$d"
}

data_mls_anomaly() {
  local d=${MLS_ANOMALY_DATA:-$BVSR_DATA/adbench} lib=${MLS_ANOMALY_PYLIBS:-$BVSR_DATA/mls_anomaly_pylibs} f
  mkdir -p "$d"
  for f in 6_cardio.npz 38_thyroid.npz 30_satellite.npz 32_shuttle.npz; do
    [ -s "$d/$f" ] || curl -fL --retry 3 -o "$d/$f" "$ADBENCH_URL/$f"
  done
  mkdir -p "$lib/_cache/mpl" "$lib/_cache/numba"
  "$PY" -m pip install --no-deps --target "$lib" pyod==3.6.5 numba==0.67.0 llvmlite==0.49.0
}

data_mls_calibration() {
  MLSCAL_CACHE=${MLSCAL_CACHE:-$BVSR_DATA/mls_calibration_cache} \
      "$PY" "$REPO/bvsr/tasks/eval_mls_calibration.py" --build-cache
}

data_astro() {
  [ -d "$SIMPLETES_ASTRO_HOME/.git" ] || git clone "$SIMPLETES_URL" "$SIMPLETES_ASTRO_HOME"
  git -C "$SIMPLETES_ASTRO_HOME" fetch --quiet origin "$SIMPLETES_ASTRO_COMMIT"
  git -C "$SIMPLETES_ASTRO_HOME" checkout --quiet "$SIMPLETES_ASTRO_COMMIT"
  ( cd "$SIMPLETES_ASTRO_HOME/datasets/astrodynamics" && "$PY" download_ephemerides.py )
}

data_zap() {
  local wbf=$SIMPLETES_HOME/datasets/zapbench/whole_brain_forecasting
  local zsrc=$SIMPLETES_HOME/third_party/zapbench
  local cache=${ZAP_CACHE:-$BVSR_DATA/zapbench/cache}
  if [ ! -d "$zsrc/.git" ]; then
    mkdir -p "$SIMPLETES_HOME/third_party"
    git clone "$ZAPBENCH_URL" "$zsrc"
  fi
  git -C "$zsrc" checkout --quiet "$ZAPBENCH_COMMIT"
  ( cd "$wbf" && bash setup.sh )
  mkdir -p "$cache"
  "$wbf/.venv/bin/python" "$wbf/prepare_data.py" --context-len 4 --out-dir "$(cd "$cache" && pwd)"
}

bv_effort() {
  case "$1:$2" in
    glm:svg|glm:labelproj|glm:pertpred|glm:ccc|glm:sl_lrbsz|glm:sl_ushape) echo low ;;
    *) echo auto ;;
  esac
}

blind_variation() {
  local task model dir
  export BVSR_GPUS=${USER_GPUS:-$(gpu_list)}
  for task in $TASKS; do
    for model in $PROPOSERS; do
      dir=$RUNS/bv/${task}_${model}
      step "bv_${task}_${model}" env TOGETHER_MIN_INTERVAL="${TOGETHER_MIN_INTERVAL:-8}" \
          "$PY" -m bvsr.bv.run --task "$task" --model "$model" --run_dir "$dir" \
          --workers 3 --max_tokens 120000 --timeout 600 --reasoning_effort "$(bv_effort "$model" "$task")"
      step "verify_${task}_${model}" "$PY" -m bvsr.bv.verify --run_dir "$dir" --workers 4
    done
  done
  step bv_creativity "$PY" -m bvsr.bv.label_creativity --runs "$RUNS"/bv/* --out "$RESULTS/creativity" \
      --baselines "$DATA/bv_results/candidates.csv.gz"
  step bv_info_bits "$PY" -m bvsr.bv.info_bits --out "$RESULTS/info_bits.csv"
  table1_fig2 "$RESULTS/creativity/candidates.csv" "$RESULTS/info_bits.csv"
}

check_fast_path() {
  "$PY" - <<'EOF'
import sys
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
if not m.is_fast_path_available:
    sys.exit("Qwen3.5 fast path unavailable: install flash-linear-attention and, with "
             "--no-build-isolation, causal-conv1d==1.7.0")
EOF
}

selective_retention() {
  local s
  for s in sr_train sr_score sft_train sft_score; do
    [ -e "$DONE/$s" ] || { check_fast_path; break; }
  done
  mkdir -p "$RESULTS/sr"
  step sr_train "${GPU_ENV[@]}" "$PY" -m bvsr.sr.train_sr --data "$DATA/sr_data" --save_dir "$CKPT/sr_selector"
  step sr_score "${GPU_ENV[@]}" "$PY" -m bvsr.sr.score --data "$DATA/sr_data" --ckpt "$CKPT/sr_selector" --method SR \
      --out "$RESULTS/sr/rankings_SR.jsonl.gz"
  step sft_train "${GPU_ENV[@]}" "$PY" -m bvsr.sr.train_sft --data "$DATA/sr_data" --save_dir "$CKPT/sft_selector"
  step sft_score "${GPU_ENV[@]}" "$PY" -m bvsr.sr.score --data "$DATA/sr_data" --ckpt "$CKPT/sft_selector" --method SFT \
      --out "$RESULTS/sr/rankings_SFT.jsonl.gz"
  fig3 "$DATA/sr_data/eval_rankings.jsonl.gz" "$RESULTS/sr/rankings_SR.jsonl.gz" "$RESULTS/sr/rankings_SFT.jsonl.gz"
}

stop_sr_server() {
  if [ -n "$SR_PID" ]; then
    kill "$SR_PID" 2>/dev/null || true
    wait "$SR_PID" 2>/dev/null || true
    SR_PID=
  fi
}

start_sr_server() {
  local key=$RUNS/stes/sr_key.txt
  check_fast_path
  mkdir -p "$RUNS/stes"
  export BVSR_SR_URL=http://127.0.0.1:$SR_PORT
  log "SR server: $CKPT/sr_selector on GPU $SR_GPU at $BVSR_SR_URL (log $RUNS/stes/sr_server.log)"
  "${GPU_ENV[@]}" CUDA_VISIBLE_DEVICES="$SR_GPU" "$PY" -m bvsr.stes.sr_server --ckpt "$CKPT/sr_selector" \
      --host 127.0.0.1 --port "$SR_PORT" --api_key_file "$key" >> "$RUNS/stes/sr_server.log" 2>&1 &
  SR_PID=$!
  trap stop_sr_server EXIT
  until "$PY" -c 'import sys; from bvsr.stes.sr_client import SRClient; sys.exit(SRClient.health(sys.argv[1]) is None)' \
      "$BVSR_SR_URL"; do
    kill -0 "$SR_PID" 2>/dev/null || { echo "SR server stopped; see $RUNS/stes/sr_server.log" >&2; exit 1; }
    sleep 20
  done
  BVSR_SR_KEY=$(cat "$key")
  export BVSR_SR_KEY
}

newest_checkpoint() {
  [ -d "$1" ] || return 0
  find "$1" -mindepth 4 -maxdepth 4 -path '*/instance-*/db_state_*/nodes.json' -printf '%T@ %h\n' \
      | sort -n | tail -n 1 | cut -d' ' -f2-
}

stes_run() {
  local resume
  resume=$(newest_checkpoint "$3")
  "$PY" -m bvsr.stes.run --task "$1" --arm "$2" --out "$3" ${resume:+--resume "$resume"}
}

simpletes() {
  local spec task arm figruns=()
  export BVSR_GPUS=${USER_GPUS:-$(gpu_list "$SR_GPU")}
  for spec in $STES_RUNS; do
    task=${spec%%:*}
    arm=${spec##*:}
    if [ "$arm" != plain ] && [ ! -e "$DONE/stes_${task}_${arm}" ] && [ -z "$SR_PID" ]; then
      start_sr_server
    fi
    step "stes_${task}_${arm}" stes_run "$task" "$arm" "$RUNS/stes/${task}_${arm}"
    figruns+=(--run "$task:$arm:$RUNS/stes/${task}_${arm}")
  done
  stop_sr_server
  log "Fig. 4 from $RUNS/stes"
  "$PY" -m bvsr.stes.fig4 "${figruns[@]}" --out "$RESULTS/fig4.pdf"
}

rerun() {
  need TOGETHER_API_KEY OPENAI_API_KEY
  need_cmd git curl
  export BVSR_DATA=${BVSR_DATA:-$REPO/$DATA}
  export SIMPLETES_HOME=${SIMPLETES_HOME:-$REPO/third_party/SimpleTES}
  export SIMPLETES_ASTRO_HOME=${SIMPLETES_ASTRO_HOME:-$REPO/third_party/SimpleTES_astro}
  fetch_released
  mkdir -p "$RESULTS"
  step simpletes_checkout setup_simpletes
  step data_svg data_svg
  step data_labelproj data_labelproj
  step data_pertpred data_pertpred
  step data_ccc data_ccc
  step data_mls_anomaly data_mls_anomaly
  step data_mls_calibration data_mls_calibration
  step data_astro data_astro
  step data_zap data_zap
  blind_variation
  selective_retention
  simpletes
  log "done: $RESULTS/table1.md $RESULTS/fig2.pdf $RESULTS/fig3.pdf $RESULTS/fig4.pdf"
}

"$MODE"
