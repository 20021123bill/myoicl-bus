set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
python - <<'PYEOF'
import pathlib
p = pathlib.Path("runner.sh"); s = p.read_text()
old = '''    setsid bash "$j" > "bus/results/$n.log" 2>&1 &
    local pid=$!
    echo "$pid" > "bus/results/$n.started"
    # Record the exit code when it finishes, without blocking this loop.
    ( wait "$pid"; echo "$?" > "bus/results/$n.done" ) &'''
new = '''    setsid bash -c 'bash "$0" > "$1" 2>&1; echo $? > "$2"' \\
        "$j" "bus/results/$n.log" "bus/results/$n.done" &
    local pid=$!
    echo "$pid" > "bus/results/$n.started"'''
if "bash -c 'bash \"$0\"" in s: print("already patched")
elif old in s: p.write_text(s.replace(old,new)); print("runner.sh patched; effective on next restart")
else: print("ANCHOR NOT FOUND; left untouched")
PYEOF
