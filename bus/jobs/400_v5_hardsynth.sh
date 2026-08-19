set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
exec > >(tee -a "$L/400_v5.log") 2>&1

# =============================================================================
# V5 -- THE TRAINING-DISTRIBUTION FIX
#
# WHY (2026-08-19, from the teacher fleet's first 10 users):
#   The released backbone was trained on our 96 meta-training users, so its
#   zero-shot CER on a HELD-OUT SESSION of those users is 2.5-15, not 55.
#   Fitting a per-user teacher there gains +0.00 .. +0.15 -- there is no
#   headroom, hence nothing to distil and, in v1-v3.2, nothing for the context
#   module to learn.  Meanwhile deployment (8 unseen users) sits at 55.4.
#   With p_synth=0.30 / strength U(0.15,0.55), the MEDIAN training episode was
#   ~18 CER against a 55 CER deployment condition: the meta-training
#   distribution never contained the meta-test problem.
#
#   diagnose_signal.py already calibrated the fix: strength 0.4 -> 40.7 CER,
#   0.5 -> 64.6 CER, so strength ~0.45 reproduces the real 55.4 gap.
#   V5 sets p_synth=1.0 and strength ~ U(0.35, 0.60).
#
# Kill the teacher fleet first: it is fitting teachers on zero-headroom users.
# Keep its output as the diagnostic that motivates all of this.
# =============================================================================

echo "=== 0. freeze the teacher-headroom diagnostic, then stop the fleet ==="
grep -h "zero-shot .* -> best" "$L"/teachers_shard*.log \
  > /data2/chenyuxiang/runs/teacher_headroom.txt 2>/dev/null
wc -l /data2/chenyuxiang/runs/teacher_headroom.txt
cp -f /data2/chenyuxiang/runs/teacher_headroom.txt bus/results/archive/ 2>/dev/null
pkill -f "myoicl.teachers" && echo "teacher fleet stopped" || echo "(no fleet running)"
sleep 20
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo "=== 1. write V5 configs ==="
python - <<'PY'
import re, pathlib
src = pathlib.Path("myoicl/configs/qwerty_v31_kvsplit.yaml").read_text()

def variant(name, out_dir, seed, p_synth, strength, mode_probs,
            freeze, steps=9000):
    t = src
    t = t.replace("out_dir: /data2/chenyuxiang/runs/myoicl_v31_kvsplit",
                  f"out_dir: {out_dir}")
    t = re.sub(r"^seed: .*$", f"seed: {seed}", t, flags=re.M)
    t = re.sub(r"^freeze_backbone: .*$", f"freeze_backbone: {freeze}", t,
               flags=re.M)
    t = re.sub(r"^  p_synth: .*$", f"  p_synth: {p_synth}", t, flags=re.M)
    t = re.sub(r"^  synth_strength: .*$", f"  synth_strength: {strength}", t,
               flags=re.M)
    t = re.sub(r"^  mode_probs: .*$", f"  mode_probs: {mode_probs}", t,
               flags=re.M)
    t = re.sub(r"^  max_steps: .*$", f"  max_steps: {steps}", t, flags=re.M)
    t = re.sub(r"^  val_every: .*$", "  val_every: 500", t, flags=re.M)
    t = re.sub(r"^  save_every: .*$", "  save_every: 500", t, flags=re.M)
    p = pathlib.Path(f"myoicl/configs/{name}.yaml")
    p.write_text(t)
    print("wrote", p)

# A1  frozen backbone, labelled context -- pure test of "can context be used"
variant("qwerty_v5_frozen", "/data2/chenyuxiang/runs/v5_frozen", 5101,
        1.0, "[0.35, 0.60]", "[0.15, 0.15, 0.70]", "true")
# A2  joint (deployable form); 15% clean episodes keep the backbone grounded
variant("qwerty_v5_joint", "/data2/chenyuxiang/runs/v5_joint", 5102,
        0.85, "[0.35, 0.60]", "[0.15, 0.15, 0.70]", "false")
# A3  CONTROL: unlabeled context only. If A3 == A1 the labels add nothing and
#     the claim must be "context helps", not "labelled context helps".
variant("qwerty_v5_modeB", "/data2/chenyuxiang/runs/v5_modeB", 5103,
        1.0, "[0.35, 0.60]", "[0.15, 0.70, 0.15]", "true")
PY

echo "=== 2. launch A1/A2/A3 on GPU 0/1/2 ==="
for v in frozen:0 joint:1 modeB:2; do
  n=${v%%:*}; g=${v##*:}
  CUDA_VISIBLE_DEVICES=$g nohup python -m myoicl.train_qwerty \
    --config myoicl/configs/qwerty_v5_$n.yaml \
    > "$L/v5_$n.log" 2>&1 &
  echo "launched v5_$n on GPU$g pid=$!"
  sleep 20
done

echo "=== 3. GPU3: zero-shot CER over ALL 96 training users (the figure) ==="
CUDA_VISIBLE_DEVICES=3 nohup python - > "$L/v5_zeroshot_scan.log" 2>&1 <<'PY' &
# No fitting -- just the released backbone's CER on a HELD-OUT SESSION of each
# user it was trained on. Paired against the 8 official test users' 55.39 this
# is the whole diagnosis in one figure.
import json, torch
from emg2qwerty.charset import charset as charset_fn
from myoicl.ceiling_probe import eval_user
from myoicl.model import build_model
from myoicl.pretrained import (backbone_eval_mode, freeze_backbone,
                               load_official_backbone)
from myoicl.teachers import training_users

cs = charset_fn()
dev = torch.device("cuda")
model = build_model({"model": {"d_ctx": 128, "d_bneck": 128, "film_rank": 32}},
                    num_classes=cs.num_classes).to(dev)
load_official_backbone(model,
    "/data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt", verbose=False)
freeze_backbone(model, verbose=False); model.eval(); backbone_eval_mode(model)

out = {}
for k, (u, paths) in enumerate(training_users(
        "/data2/chenyuxiang/code/emg2qwerty")):
    held = paths[-1:]
    cer = eval_user(model, None, None, held, cs, dev, True)
    out[u] = {"cer": cer, "n_sessions": len(paths)}
    print(f"[{k+1}/96] {u}: held-out-session CER {cer:.2f} "
          f"({len(paths)} sessions)", flush=True)
    json.dump(out, open("/data2/chenyuxiang/runs/zeroshot_train_users.json",
                        "w"), indent=1)
v = [r["cer"] for r in out.values()]
v.sort()
print(f"\n=== SEEN users (n={len(v)}): median {v[len(v)//2]:.2f}  "
      f"p10 {v[len(v)//10]:.2f}  p90 {v[9*len(v)//10]:.2f} ===")
print("=== UNSEEN users (8 official test): 55.39 (published) ===")
PY
echo "launched zero-shot scan on GPU3 pid=$!"

echo "=== 4. stream progress for 12 h ==="
for k in $(seq 1 144); do
  sleep 300
  cp -f "$L"/v5_*.log bus/results/ 2>/dev/null
  echo "--- $(date +%H:%M) ---"
  for n in frozen joint modeB; do
    echo "[v5_$n] $(grep -E '^\[val\]' "$L/v5_$n.log" 2>/dev/null | tail -1)"
  done
  echo "[scan] $(tail -1 "$L/v5_zeroshot_scan.log" 2>/dev/null)"
  pgrep -f "myoicl.train_qwerty" >/dev/null || { echo "all v5 runs ended"; break; }
done
echo "=== 400 done ==="
