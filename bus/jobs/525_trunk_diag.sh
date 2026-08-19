set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/525_diag.log") 2>&1

# =============================================================================
# CPU-only diagnostic while the fleet trains: greedy CER is stuck at ~84 from
# step 4k to 10k while loss declines monotonically (2.98 -> 2.13). Before
# deciding whether this is the normal CTC blank-plateau or a structural
# problem, measure three things on the CURRENT ref checkpoint:
#   1. input scale into the first conv (raw sEMG is not audio-scaled; GroupNorm
#      sits after conv1, but a pathological scale would still distort conv1)
#   2. blank fraction of greedy output -- the plateau signature: if the model
#      emits blank at ~every frame, CER ~= 100 * (1 - blanks that survive
#      collapse); a falling blank fraction means the plateau is breaking
#   3. per-position entropy -- is the model actually sharpening?
# =============================================================================

CK=$R/tf_ref_lr1e3/last.pt
[ -f "$CK" ] || { echo "no checkpoint yet"; exit 0; }

CUDA_VISIBLE_DEVICES="" python - <<'PY'
import torch
from emg2qwerty.charset import charset as charset_fn
from myoicl.episodes import build_windowed_dataset, windowed_collate
from myoicl.qwerty_data import load_user_sessions
from myoicl.trunk_tf import build_trunk

cs = charset_fn()
ck = torch.load('/data2/chenyuxiang/runs/tf_ref_lr1e3/last.pt',
                map_location='cpu')
m = build_trunk({"model": {"tf_size": ck["args"]["size"],
                           "conv_strides": ck["args"].get("conv_strides",
                                                          [5, 2, 2])}},
                num_classes=cs.num_classes)
m.load_state_dict(ck["model"]); m.eval()
print(f"checkpoint step {ck['step']}")

sess = load_user_sessions('/data2/chenyuxiang/code/emg2qwerty', 'generic')
ds = build_windowed_dataset(sess['test'][:2], train=False,
                            window_length=8000, padding=(1800, 200), raw=True)
b = windowed_collate([ds[i] for i in range(6)])
raw = b["inputs"].permute(1, 2, 3, 0).flatten(1, 2).float()   # (N, 32, T)

print(f"\n1. INPUT SCALE: mean {raw.mean():.1f}  std {raw.std():.1f}  "
      f"absmax {raw.abs().max():.0f}")
with torch.no_grad():
    h1 = m.featurizer.net[0](raw)                  # conv1 pre-norm
    print(f"   post-conv1  std {h1.std():.3f}  absmax {h1.abs().max():.1f}")
    hf = m.featurizer(raw)
    print(f"   post-featurizer  std {hf.std():.3f}")

    em = m(raw)                                    # (T, N, K) log-probs
p = em.exp()
blank = p[..., cs.null_class]
top = em.argmax(-1)
frac_blank = float((top == cs.null_class).float().mean())
ent = float(-(p * em).sum(-1).mean())
print(f"\n2. BLANK: argmax==blank on {frac_blank*100:.1f}% of frames | "
      f"mean p(blank) {float(blank.mean()):.3f}")
print(f"3. ENTROPY: {ent:.3f} nats (uniform would be {torch.log(torch.tensor(float(cs.num_classes))):.3f})")

from myoicl.metrics import greedy_ctc_decode
from emg2qwerty.data import LabelData
lens = m.output_length(b["input_lengths"])
preds = greedy_ctc_decode(em, lens, blank=cs.null_class)
tg, tl = b["targets"].numpy(), b["target_lengths"].numpy()
for n in range(3):
    hyp = LabelData.from_labels(preds[n]).text
    ref = LabelData.from_labels(tg[: tl[n], n]).text
    print(f"\n[{n}] ref({len(ref)}): {ref[:70]!r}")
    print(f"    hyp({len(hyp)}): {hyp[:70]!r}")
PY
echo "=== 525 done ==="
