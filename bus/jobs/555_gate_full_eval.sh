set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/555_gate_full.log") 2>&1

# =============================================================================
# THE OFFICIAL GATE NUMBERS: full 5s-window evaluation of the 200-epoch-budget
# trunks (the training monitor is a 512-window subsample that reads ~5 CER
# high). Waits for ref_full and fold1_full to finish their last ~10k steps,
# then scores every *_full last.pt. ref_full vs the paper's 35.9 is the
# reproduction verdict the paper table will cite.
# =============================================================================

echo "=== wait for ref_full / fold1_full to finish ==="
for i in $(seq 1 90); do
  a=$(pgrep -f "tf_ref_full" | wc -l); b=$(pgrep -f "tf_fold1_full" | wc -l)
  [ "$a" = "0" ] && [ "$b" = "0" ] && break
  sleep 120
done

one() {  # one <ckpt-path> <label>
CKPT="$1" LABEL="$2" python - <<'PY'
import json, os, torch
from emg2qwerty.charset import charset as charset_fn
from emg2qwerty.data import LabelData
from myoicl.episodes import build_windowed_dataset, windowed_collate
from myoicl.folds import split_for_fold
from myoicl.metrics import CERAccumulator, greedy_ctc_decode
from myoicl.qwerty_data import load_user_sessions
from myoicl.trunk_tf import build_trunk
from torch.utils.data import DataLoader

path, label = os.environ["CKPT"], os.environ["LABEL"]
cs = charset_fn(); dev = torch.device("cuda:0")
sess = load_user_sessions('/data2/chenyuxiang/code/emg2qwerty', 'generic')

def ev(model, pairs, wlen=10000):
    ds = build_windowed_dataset(pairs, train=False, window_length=wlen,
                                padding=(1800, 200), raw=True)
    dl = DataLoader(ds, batch_size=16, num_workers=2,
                    collate_fn=windowed_collate)
    acc = CERAccumulator()
    with torch.no_grad():
        for b in dl:
            raw = b["inputs"].permute(1, 2, 3, 0).flatten(1, 2).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                em = model(raw)
            lens = model.output_length(b["input_lengths"].to(dev))
            for n, p in enumerate(greedy_ctc_decode(em.float(), lens.cpu(),
                                                    blank=cs.null_class)):
                tg, tl = b["targets"].numpy(), b["target_lengths"].numpy()
                acc.update(LabelData.from_labels(p).text,
                           LabelData.from_labels(tg[: tl[n], n]).text)
    return acc.cer

ck = torch.load(path, map_location='cpu')
m = build_trunk({"model": {"tf_size": ck["args"]["size"],
                           "conv_strides": ck["args"].get("conv_strides",
                                                          [5, 2, 2])}},
                num_classes=cs.num_classes).to(dev)
m.load_state_dict(ck["model"]); m.eval()
row = {"step": ck["step"], "test_5s": ev(m, sess['test'])}
fold = ck["args"].get("fold", -1)
if fold is not None and fold >= 0:
    _, held, _ = split_for_fold('/data2/chenyuxiang/code/emg2qwerty', fold, 4)
    row["heldout_5s"] = ev(m, held[:24])
print(f"[{label}] step {row['step']} | 8-test 5s CER {row['test_5s']:.2f}"
      + (f" | fold-heldout {row['heldout_5s']:.2f}" if 'heldout_5s' in row
         else ""), flush=True)
out = '/data2/chenyuxiang/runs/gate_full.json'
d = json.load(open(out)) if os.path.exists(out) else {}
d[label] = row
json.dump(d, open(out, 'w'), indent=1)
PY
echo "  (rc=$? for $2)"
}

for spec in \
  "$R/tf_ref_full/last.pt:ref_full143k" \
  "$R/tf_fold0_full/last.pt:fold0_full143k" \
  "$R/tf_fold1_full/last.pt:fold1_full143k" \
  "$R/tf_fold3/last.pt:fold3_40k" ; do
  p=${spec%%:*}; lab=${spec##*:}
  [ -f "$p" ] && one "$p" "$lab" || echo "  (missing $p)"
done

echo
cp -f "$R/gate_full.json" bus/results/archive/ 2>/dev/null
echo "reference: paper Tiny 35.9 | verdict: <=45 in-class (train longer to"
echo "close the rest); 45-60 partial; >80 re-implementation bug -> run fairemg"
echo "=== 555 done ==="
