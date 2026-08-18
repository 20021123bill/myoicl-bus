#!/usr/bin/env bash
# =============================================================================
# MyoICL bus runner -- run this ONCE on amax, inside tmux, and then forget it.
#
# WHAT IT DOES
#   every POLL seconds:
#     1. commit + push whatever is currently in bus/results and bus/status
#        (partial logs included, so training curves stream out live)
#     2. pull the repo (new code + new job scripts)
#     3. launch any job in bus/jobs/*.sh that has not been started yet,
#        in the BACKGROUND, output tee'd to bus/results/<name>.log
#     4. refresh bus/status/heartbeat.md (nvidia-smi, tmux, running jobs,
#        tail of every live log)
#
# WHAT IT DOES NOT DO
#   It never deletes anything, never touches runs/, never kills a job.
#   To stop it: create bus/jobs/STOP (or Ctrl-C in its tmux window).
#   Every job it runs is a plain shell script you can read in bus/jobs/
#   before it runs -- nothing is hidden.
#
# USAGE
#   cd /data2/chenyuxiang/code/myoicl
#   tmux new -s bus
#   bash runner.sh
#   # Ctrl-b then d to detach
# =============================================================================
set -uo pipefail

REPO="${REPO:-/data2/chenyuxiang/code/myoicl}"
POLL="${POLL:-30}"
BRANCH="${BRANCH:-main}"
CONDA_ENV="${CONDA_ENV:-/data2/chenyuxiang/conda_envs/qwerty}"

cd "$REPO" || { echo "no such repo: $REPO"; exit 1; }
mkdir -p bus/jobs bus/results bus/status

# Make conda available to every job without each script re-deriving it.
if [ -z "${CONDA_EXE:-}" ] && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -z "${CONDA_EXE:-}" ] && [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
export CONDA_ENV

git config --local user.email "runner@amax" 2>/dev/null
git config --local user.name  "myoicl-runner" 2>/dev/null

echo "[bus] repo=$REPO branch=$BRANCH poll=${POLL}s env=$CONDA_ENV"
echo "[bus] started $(date -Is)"

sync_out() {
  git add -A bus/results bus/status >/dev/null 2>&1
  if ! git diff --cached --quiet 2>/dev/null; then
    git commit -q -m "results $(date -Is)" >/dev/null 2>&1
  fi
  git pull -q --rebase --autostash origin "$BRANCH" >/dev/null 2>&1 \
    || { git rebase --abort >/dev/null 2>&1; git pull -q --no-rebase origin "$BRANCH" >/dev/null 2>&1; }
  git push -q origin "$BRANCH" >/dev/null 2>&1
}

copy_external_logs() {
  # Training processes write to /data2/chenyuxiang/runs/joblogs/ which git does
  # not track. We copy, never the reverse, so a pull cannot roll a live log
  # back the way it did on 2026-08-18.
  if [ -d /data2/chenyuxiang/runs/joblogs ]; then
    cp -f /data2/chenyuxiang/runs/joblogs/*.log bus/results/ 2>/dev/null || true
  fi
}

heartbeat() {
  local f=bus/status/heartbeat.md
  {
    echo "# heartbeat $(date -Is)"
    echo
    echo '## gpu'
    echo '```'
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader 2>/dev/null
    echo '```'
    echo
    echo '## jobs'
    echo '```'
    for m in bus/results/*.started; do
      [ -e "$m" ] || continue
      local n pid state
      n=$(basename "$m" .started); pid=$(cat "$m" 2>/dev/null)
      if [ -e "bus/results/$n.done" ]; then
        state="DONE rc=$(cat "bus/results/$n.done")"
      elif kill -0 "$pid" 2>/dev/null; then
        state="RUNNING pid=$pid"
      else
        state="DEAD (no .done -- crashed or killed)"
      fi
      printf '%-40s %s\n' "$n" "$state"
    done
    echo '```'
    echo
    echo '## tail of each log (last 25 lines)'
    for l in bus/results/*.log; do
      [ -e "$l" ] || continue
      echo
      echo "### $(basename "$l")"
      echo '```'
      tail -n 25 "$l" 2>/dev/null
      echo '```'
    done
  } > "$f".tmp && mv "$f".tmp "$f"
}

launch_new_jobs() {
  for j in bus/jobs/*.sh; do
    [ -e "$j" ] || continue
    local n; n=$(basename "$j" .sh)
    [ -e "bus/results/$n.started" ] && continue
    echo "[bus] launching $n at $(date -Is)"
    # Each job gets a clean login-ish shell with conda on PATH. The job script
    # is responsible for `conda activate` and for picking CUDA_VISIBLE_DEVICES.
    setsid bash -c 'bash "$0" > "$1" 2>&1; echo $? > "$2"' \
        "$j" "bus/results/$n.log" "bus/results/$n.done" &
    local pid=$!
    echo "$pid" > "bus/results/$n.started"
  done
}

while true; do
  if [ -e bus/jobs/STOP ]; then
    echo "[bus] STOP file present -- exiting at $(date -Is)"
    heartbeat; sync_out; exit 0
  fi
  copy_external_logs
  sync_out
  launch_new_jobs
  heartbeat
  sleep "$POLL"
done
