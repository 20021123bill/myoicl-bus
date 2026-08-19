set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONUNBUFFERED=1
L=/data2/chenyuxiang/runs/joblogs; mkdir -p "$L"
R=/data2/chenyuxiang/runs
exec > >(tee -a "$L/530_gate.log") 2>&1

# =============================================================================
# THE DEFINITIVE REPRODUCTION-GATE READING.
#
# The fleet finished at monitor-CER ~87, far from the paper's 35.9. But the
# monitor was measuring OUT OF DISTRIBUTION: training used 5 s windows
# (10000 samples) while build_eval_set used 30 s windows (60000) -- a 6x
# length extrapolation for a causal transformer trained on ~500-frame
# sequences. The paper evaluates on the SAME 4 s + 1 s-context windowing it
# trains on (section 3.1). And the 525 "99.2% blank" probe drew the first
# windows of test sessions, whose references are EMPTY (session-start
# silence) -- blank was the correct answer there.
#
# This job measures what the paper measures: greedy CER over ALL 10000-sample
# windows of every test session (both sessions of each of the 8 test users),
# plus the same for fold-heldout users, plus a window-length sweep on one
# checkpoint to quantify how much the 30 s monitor inflated CER.
# =============================================================================

python - <<'PY'
import glob, json, torch
from emg2qwerty.charset import charset as charset_fn
from emg2qwerty.data import LabelData
from myoicl.episodes import build_windowed_dataset, windowed_collate
from myoicl.folds import split_for_fold
from myoicl.metrics import CERAccumulator, greedy_ctc_decode
from myoicl.qwerty_data import load_user_sessions
from myoicl.trunk_tf import build_trunk
from torch.utils.data import DataLoader

cs = charset_fn()
dev = torch.device("cuda:0")
sess = load_user_sessions('/data2/chenyuxiang/code/emg2qwerty', 'generic')

def eval_windows(model, pairs, wlen):
    ds = build_windowed_dataset(pairs, train=False, window_length=wlen,
                                padding=(1800, 200), raw=True)
    dl = DataLoader(ds, batch_size=16, num_workers=3,
                    collate_fn=windowed_collate)
    acc = CERAccumulator()
    with torch.no_grad():
        for b in dl:
            raw = b["inputs"].permute(1, 2, 3, 0).flatten(1, 2).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                em = model(raw)
            lens = model.output_length(b["input_lengths"].to(dev))
            preds = greedy_ctc_decode(em.float(), lens.cpu(),
                                      blank=cs.null_class)
            tg, tl = b["targets"].numpy(), b["target_lengths"].numpy()
            for n, p in enumerate(preds):
                acc.update(LabelData.from_labels(p).text,
                           LabelData.from_labels(tg[: tl[n], n]).text)
    return acc.cer, acc.total

results = {}
for d in sorted(glob.glob('/data2/chenyuxiang/runs/tf_*/best.pt')):
    if "superseded" in d or "discarded" in d:
        continue
    name = d.split('/')[-2]
    ck = torch.load(d, map_location='cpu')
    m = build_trunk({"model": {"tf_size": ck["args"]["size"],
                               "conv_strides": ck["args"].get("conv_strides",
                                                              [5, 2, 2])}},
                    num_classes=cs.num_classes).to(dev)
    m.load_state_dict(ck["model"]); m.eval()

    cer_test, chars = eval_windows(m, sess['test'], 10000)
    row = {"step": ck["step"], "test_5s": cer_test, "test_chars": chars}
    fold = ck["args"].get("fold", -1)
    if fold is not None and fold >= 0:
        _, held, _ = split_for_fold('/data2/chenyuxiang/code/emg2qwerty',
                                    fold, 4)
        cer_h, _ = eval_windows(m, held[:24], 10000)
        row["heldout_5s"] = cer_h
    results[name] = row
    print(f"[{name}] step {ck['step']} | 8-test-user 5s-window CER "
          f"{cer_test:.2f} ({chars} chars)"
          + (f" | fold-heldout {row.get('heldout_5s', float('nan')):.2f}"
             if 'heldout_5s' in row else ""), flush=True)
    del m; torch.cuda.empty_cache()

print("\n=== window-length sweep on tf_ref_lr1e3 (quantify the monitor bias) ===")
d = '/data2/chenyuxiang/runs/tf_ref_lr1e3/best.pt'
ck = torch.load(d, map_location='cpu')
m = build_trunk({"model": {"tf_size": ck["args"]["size"],
                           "conv_strides": ck["args"].get("conv_strides",
                                                          [5, 2, 2])}},
                num_classes=cs.num_classes).to(dev)
m.load_state_dict(ck["model"]); m.eval()
for wlen in (10000, 20000, 40000, 60000):
    cer, chars = eval_windows(m, sess['test'][:4], wlen)
    print(f"  window {wlen/2000:4.0f}s: CER {cer:6.2f}  ({chars} chars)",
          flush=True)

json.dump(results, open('/data2/chenyuxiang/runs/gate_5s_eval.json', 'w'),
          indent=1)
print("\nreference: paper Tiny 35.9 (4 s windows, same 8 test users)")
print("verdict guide: <=45 -> reproduction roughly holds, monitor was the "
      "artifact; 60-75 -> partial, dig into encoder internals; >80 -> "
      "re-implementation genuinely fails, switch to running fairemg itself.")
PY
cp -f /data2/chenyuxiang/runs/gate_5s_eval.json bus/results/archive/ 2>/dev/null
echo "=== 530 done ==="
