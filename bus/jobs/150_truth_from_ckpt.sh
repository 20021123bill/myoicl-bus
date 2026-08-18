set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH

echo "########## the log is unreliable -- ask the checkpoint and the process ##########"
# bus/results/*.log are tracked files that a live process writes while git
# pulls rewrite them. Lines can and did disappear. The checkpoint's own step
# field and the process table cannot be rolled back by git.
python - <<'PYEOF'
import torch, os, time
for tag, p in [("D1", "/data2/chenyuxiang/runs/myoicl_d1_spawn/last.pt"),
               ("D2", "/data2/chenyuxiang/runs/myoicl_d2_spawn/last.pt")]:
    if not os.path.exists(p):
        print(f"{tag}: no checkpoint yet"); continue
    st = torch.load(p, map_location='cpu').get('step')
    age = time.time() - os.path.getmtime(p)
    print(f"{tag}: checkpoint step={st}  written {age/60:.1f} min ago")
PYEOF

echo
echo "--- processes and how much CPU time they have burned ---"
ps -eo pid,etime,time,pcpu,args --sort=-pcpu | grep "[t]rain_qwerty" | cut -c1-160

echo
echo "--- true step numbers straight from the live logs on disk ---"
for f in bus/results/036_d1_spawn2.log bus/results/034_d2_spawn.log; do
  echo "$f: $(grep -cE '^step ' "$f" 2>/dev/null) step lines, last = $(grep -E '^step ' "$f" 2>/dev/null | tail -1)"
done

echo
echo "########## keep future logs OUT of git's reach ##########"
# From now on, jobs tee to /data2/chenyuxiang/runs/joblogs/ (untracked) and the
# runner copies them into bus/results one-way. A pull can no longer revert a
# log that git is not the author of.
mkdir -p /data2/chenyuxiang/runs/joblogs
python - <<'PYEOF'
import pathlib
p = pathlib.Path("runner.sh"); s = p.read_text()
anchor = "heartbeat() {"
inject = '''copy_external_logs() {
  # Training processes write to /data2/chenyuxiang/runs/joblogs/ which git does
  # not track. We copy, never the reverse, so a pull cannot roll a live log
  # back the way it did on 2026-08-18.
  if [ -d /data2/chenyuxiang/runs/joblogs ]; then
    cp -f /data2/chenyuxiang/runs/joblogs/*.log bus/results/ 2>/dev/null || true
  fi
}

heartbeat() {'''
if "copy_external_logs" in s:
    print("runner already copies external logs")
else:
    s = s.replace(anchor, inject, 1)
    s = s.replace("  sync_out\n  launch_new_jobs", "  copy_external_logs\n  sync_out\n  launch_new_jobs", 1)
    p.write_text(s)
    print("runner.sh patched (effective on next runner restart)")
PYEOF
git add -A runner.sh && git commit -q -m "runner: one-way copy of external job logs" 2>&1 | tail -1 || true
