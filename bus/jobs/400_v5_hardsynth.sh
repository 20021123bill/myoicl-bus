set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/400_v5.log") 2>&1

# =============================================================================
# V5 -- THE TRAINING-DISTRIBUTION FIX  (2026-08-19)
#
# WHY.  The V4 teacher fleet's first 13 users showed the released backbone's
# CER on a HELD-OUT SESSION of a *training* user is a median 6.96 (min 2.46),
# and gradient-fitting a per-user interface there gains +0.00.  Deployment (8
# unseen users) is 55.39; published per-user fine-tuning on unseen users is
# 11.28.  So our meta-training users have LESS than zero adaptation headroom --
# their zero-shot beats the fine-tuned number on unseen users.  With
# p_synth=0.30 / strength U(0.15,0.55) the median training episode sat at ~18
# CER against a 55 CER deployment condition.  The context module was never
# once trained on a task that needed it.  That, not the injection mechanism,
# is why v1-v3.2 all returned gain <= 0.
#
# THE LADDER RUN HERE (all config-only; no code is changed):
#   A0  positive control.  Synthetic subject = per-channel GAIN only (no
#       rotation, no mixing, no noise).  A raw-EMG channel gain is an ADDITIVE
#       per-(band,channel) constant on the log-spectrogram, so the per-unit
#       affine head (ctx_version 2, input_conditioning) can invert it EXACTLY.
#       Well-posed problem + exactly-matched interface.  If this does not
#       learn, the training loop itself is broken and no architecture will
#       save us.  (Honest caveat: a gain is recoverable from UNLABELLED
#       statistics, so A0 validates forward adaptation, not labelled ICL.)
#   A1  same synthetic subject, but the v3.1 frame-biasing interface.  Asks
#       whether cross-attention biasing can also express the correction, or
#       whether conditioning must act before the frontend mixes channels.
#   A2  realistic difficulty: sample_calibrated at strength U(0.35,0.60),
#       which includes integer electrode ROTATION.  Rotation is the part that
#       genuinely requires labels (unlabelled statistics are permutation
#       -invariant), and also the part no diagonal affine can undo -- expect
#       this to expose the need for a channel-remix head.  That is the point.
#   scan  zero-shot CER for all 96 training users -> the diagnosis figure.
#
# Difficulty is CALIBRATED FIRST (phase 1) instead of guessed: sweep the gain
# sigma and pick the one whose degradation lands nearest the real 55.39 gap.
# =============================================================================

echo "=== phase 0: freeze the teacher-headroom diagnostic, stop the fleet ==="
grep -h "zero-shot .* -> best" "$L"/teachers_shard*.log > "$R/teacher_headroom.txt" 2>/dev/null
wc -l "$R/teacher_headroom.txt"
mkdir -p bus/results/archive && cp -f "$R/teacher_headroom.txt" bus/results/archive/
pkill -f "myoicl.teachers" && echo "teacher fleet stopped" || echo "(no fleet running)"
sleep 20
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo
echo "=== phase 1: calibrate the gain-only synthetic subject (GPU0, ~15 min) ==="
for g in 0.5 0.7 0.9 1.2; do
  echo "--- gain_log_std $g ---"
  CUDA_VISIBLE_DEVICES=0 python -m myoicl.diagnose_signal \
    --n-episodes 8 --rotations 0 --gain-log-std $g --p-mix 0.0 \
    --snr 60 60 --out "$R/gain_probe_$g.json" 2>&1 | grep -E "synthetic shift|ratio ="
done

SIGMA=$(python - <<'PY'
import glob, json
best, bs = None, None
for f in glob.glob('/data2/chenyuxiang/runs/gain_probe_*.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    s = float(f.rsplit('_', 1)[1][:-5])
    err = abs(d['train_users_synth'] - 55.39)
    print(f"  sigma {s}: CER {d['train_users_synth']:.2f} (|d-55.4| {err:.2f})")
    if best is None or err < best:
        best, bs = err, s
print(f"CHOSEN {bs}")
PY
)
echo "$SIGMA"
G=$(echo "$SIGMA" | sed -n 's/^CHOSEN //p')
[ -z "$G" ] && G=0.9
echo "=== using gain_log_std = $G ==="

echo
echo "=== phase 2: write the three V5 configs ==="
GAIN_SIGMA="$G" python - <<'PY'
import os, re, pathlib
G = os.environ["GAIN_SIGMA"]
SYNTH = ("  synth:\n"
         "    rotation_choices: [0]\n"
         f"    gain_log_std: {G}\n"
         "    p_noise: 0.0\n"
         "    p_mix: 0.0\n")

def build(base, name, out_dir, seed, synth_block, steps=8000, set_ctx_lr=False):
    t = pathlib.Path(f"myoicl/configs/{base}.yaml").read_text()
    if set_ctx_lr:  # inputcond config has no ctx_lr; frozen backbone needs one
        t = re.sub(r"^  lr: .*$", "  lr: 1.0e-3\n  ctx_lr: 1.0e-3", t,
                   count=1, flags=re.M)
    t = re.sub(r"^out_dir: .*$", f"out_dir: {out_dir}", t, flags=re.M)
    t = re.sub(r"^seed: .*$", f"seed: {seed}", t, flags=re.M)
    t = re.sub(r"^freeze_backbone: .*$", "freeze_backbone: true", t, flags=re.M)
    t = re.sub(r"^  p_synth: .*$", "  p_synth: 1.0", t, flags=re.M)
    t = re.sub(r"^  synth_strength: .*$", synth_block.rstrip("\n"), t, flags=re.M)
    t = re.sub(r"^  num_workers: .*$", "  num_workers: 4", t, flags=re.M)
    t = re.sub(r"^  max_steps: .*$", f"  max_steps: {steps}", t, flags=re.M)
    t = re.sub(r"^  val_every: .*$", "  val_every: 500", t, flags=re.M)
    t = re.sub(r"^  save_every: .*$", "  save_every: 500", t, flags=re.M)
    t = re.sub(r"^  log_every: .*$", "  log_every: 100", t, flags=re.M)
    p = pathlib.Path(f"myoicl/configs/{name}.yaml")
    p.write_text(t)
    import yaml
    c = yaml.safe_load(p.read_text())
    e = c["episodes"]
    print(f"[ok] {p}  p_synth={e['p_synth']} synth={e.get('synth')} "
          f"strength={e.get('synth_strength')} frozen={c['freeze_backbone']}")

# A0: per-unit input affine (ctx_version 2) -- the exactly-matched inverse
build("qwerty_joint_inputcond", "qwerty_v5_a0_gain_affine",
      "/data2/chenyuxiang/runs/v5_a0", 5101, SYNTH, set_ctx_lr=True)
# A1: v3.1 frame biasing on the SAME synthetic subject
build("qwerty_v31_kvsplit", "qwerty_v5_a1_gain_v31",
      "/data2/chenyuxiang/runs/v5_a1", 5102, SYNTH)
# A2: realistic difficulty (rotation + tilt + mixing + noise), affine interface
build("qwerty_joint_inputcond", "qwerty_v5_a2_realistic",
      "/data2/chenyuxiang/runs/v5_a2", 5103,
      "  synth_strength: [0.35, 0.60]", set_ctx_lr=True)
PY
[ $? -ne 0 ] && { echo "CONFIG WRITE FAILED -- not launching"; exit 1; }

echo
echo "=== phase 3: launch A0/A1/A2 on GPU0/1/2 ==="
i=0
for n in a0_gain_affine a1_gain_v31 a2_realistic; do
  CUDA_VISIBLE_DEVICES=$i nohup python -m myoicl.train_qwerty \
    --config "myoicl/configs/qwerty_v5_$n.yaml" > "$L/v5_$n.log" 2>&1 &
  echo "launched v5_$n on GPU$i pid=$!"
  i=$((i+1)); sleep 25
done

echo "=== phase 3b: GPU3 -- zero-shot CER over all 96 training users ==="
CUDA_VISIBLE_DEVICES=3 nohup python - > "$L/v5_zeroshot_scan.log" 2>&1 <<'PY' &
# No fitting. Just the released backbone's CER on a HELD-OUT SESSION of every
# user it was trained on. Paired against the 8 test users' 55.39, this one
# figure is the whole diagnosis.
import json, torch
from emg2qwerty.charset import charset as charset_fn
from myoicl.ceiling_probe import eval_user
from myoicl.model import build_model
from myoicl.pretrained import (backbone_eval_mode, freeze_backbone,
                               load_official_backbone)
from myoicl.teachers import training_users

cs = charset_fn(); dev = torch.device("cuda")
model = build_model({"model": {"d_ctx": 128, "d_bneck": 128, "film_rank": 32}},
                    num_classes=cs.num_classes).to(dev)
load_official_backbone(
    model, "/data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt",
    verbose=False)
freeze_backbone(model, verbose=False); model.eval(); backbone_eval_mode(model)

out = {}
for k, (u, paths) in enumerate(
        training_users("/data2/chenyuxiang/code/emg2qwerty")):
    cer = eval_user(model, None, None, paths[-1:], cs, dev, True)
    out[u] = {"cer": cer, "n_sessions": len(paths)}
    print(f"[{k+1}/96] {u}: held-out-session CER {cer:.2f} "
          f"({len(paths)} sessions)", flush=True)
    json.dump(out, open("/data2/chenyuxiang/runs/zeroshot_train_users.json",
                        "w"), indent=1)
v = sorted(r["cer"] for r in out.values())
print(f"\n=== SEEN users (n={len(v)}): median {v[len(v)//2]:.2f} "
      f"p10 {v[len(v)//10]:.2f} p90 {v[9*len(v)//10]:.2f} ===")
print("=== UNSEEN users (8 official test): 55.39 published ===")
PY
echo "launched zero-shot scan on GPU3 pid=$!"

echo
echo "=== phase 4: stream progress (12 h) ==="
for k in $(seq 1 144); do
  sleep 300
  cp -f "$L"/v5_*.log bus/results/ 2>/dev/null
  echo "--- $(date +%H:%M) ---"
  for n in a0_gain_affine a1_gain_v31 a2_realistic; do
    echo "[$n] $(grep -E '^\[val\] step' "$L/v5_$n.log" 2>/dev/null | tail -1)"
    tail -3 "$L/v5_$n.log" 2>/dev/null | grep -Ei "error|traceback|assert" && \
      echo "  ^^ $n LOOKS BROKEN"
  done
  echo "[scan] $(tail -1 "$L/v5_zeroshot_scan.log" 2>/dev/null)"
  pgrep -f "myoicl.train_qwerty" >/dev/null || { echo "all v5 runs ended"; break; }
done
echo "=== 400 done ==="
