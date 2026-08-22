set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1

# =============================================================================
# 608 -- L_char is silently inactive: segs/step 0. Stop the arms, diagnose.
#
# 606's smoke ran 60 steps with segs/step 0.0 throughout, so the contrastive
# term contributed nothing and all three arms -- including the two controls --
# are currently just CTC. Worse, I cannot see WHY, because
# forced_align_segments wraps the call in `except Exception: return []`. That
# is the same class of defect as Part B's confidence filter that kept 100% of
# windows: no crash, no error, plausible-looking logs, and a loss that never
# fires. Swallowing the exception is the actual bug; the alignment failure is
# only its symptom.
#
# This job removes the swallow, prints the first failure verbatim, and probes
# the alignment on one real batch with the real shapes and dtypes.
# =============================================================================

echo "=== stop the three align arms (they are pure CTC right now) ==="
pkill -f "myoicl.train_align" && echo "  stopped" || echo "  none running"
sleep 3

echo
echo "=== patch: surface the exception instead of swallowing it ==="
$PY - <<'PY'
import sys
p = "myoicl/align_char.py"
src = open(p).read()
A = '''    except Exception:
        return []'''
B = '''    except Exception as e:                                    # noqa: BLE001
        # SILENT FAILURE WAS THE BUG. A bare `return []` here turned a hard
        # error into "no segments", which looks exactly like "the model
        # cannot align yet" and cost a full round of GPU time before the
        # segs/step counter gave it away. Report once, loudly, then degrade.
        global _ALIGN_ERR_SHOWN
        if not _ALIGN_ERR_SHOWN:
            _ALIGN_ERR_SHOWN = True
            print(f"[align] forced_align FAILED: {type(e).__name__}: {e}\\n"
                  f"        lp{tuple(lp.shape)} {lp.dtype} tg{tuple(tg.shape)}"
                  f" {tg.dtype} T={T} L={L} blank={blank}", flush=True)
        return []'''
n = src.count(A)
if n != 1:
    sys.exit(f"[FATAL] anchor found {n} times, expected 1")
src = src.replace(A, B)
src = src.replace("from torch import nn",
                  "from torch import nn\n\n_ALIGN_ERR_SHOWN = False")
open(p, "w").write(src)
print("[patched] align_char.py: exception surfaced")
PY
$PY -c "import ast;ast.parse(open('myoicl/align_char.py').read())" || exit 2
grep -q "_ALIGN_ERR_SHOWN" myoicl/align_char.py || { echo "[FATAL]"; exit 2; }
echo "  verified"

echo
echo "=== probe forced_align on ONE REAL batch, real shapes and dtypes ==="
CUDA_VISIBLE_DEVICES=2 $PY - <<'PY'
import torch
from torch.utils.data import ConcatDataset, DataLoader
from emg2qwerty.charset import charset as charset_fn
from emg2qwerty.data import WindowedEMGDataset
from myoicl.align_char import forced_align_segments
from myoicl.model import build_model
from myoicl.qwerty_data import (group_by_user, load_user_sessions,
                                official_train_transform)
from myoicl.train_align import WithUser, collate_with_user

dev = torch.device("cuda")
cs = charset_fn()
print(f"  charset: num_classes={cs.num_classes} blank/null={cs.null_class}")

sess = load_user_sessions("/data2/chenyuxiang/code/emg2qwerty", "generic")
by = group_by_user(sess["train"])
u0 = sorted(by)[0]
ds = ConcatDataset([WithUser(WindowedEMGDataset(
    by[u0][0], window_length=8000, padding=(1800, 200),
    transform=official_train_transform(), jitter=True), 0)])
b = next(iter(DataLoader(ds, batch_size=4, collate_fn=collate_with_user)))
print(f"  inputs {tuple(b['inputs'].shape)} targets "
      f"{tuple(b['targets'].shape)} {b['targets'].dtype} "
      f"lengths {b['target_lengths'].tolist()}")

cfg = {"model": {"version": 1, "frontend": "official",
                 "official_mlp_features": [384], "freq_bins": 33,
                 "num_bands": 2, "channels_per_band": 16, "d_model": 768,
                 "tds_block_channels": [24, 24, 24, 24],
                 "tds_kernel_width": 32, "use_residual_context": False}}
m = build_model(cfg, num_classes=cs.num_classes).to(dev).eval()
with torch.no_grad():
    h = m.tds(m.frontend(b["inputs"].to(dev)))
    lp = m.classifier(h).float().log_softmax(-1)
print(f"  hidden {tuple(h.shape)} logprobs {tuple(lp.shape)}")
print(f"  target ids min={int(b['targets'].min())} "
      f"max={int(b['targets'].max())}  (blank must NOT appear: "
      f"{cs.null_class})")

tg = b["targets"].to(dev)
tl = b["target_lengths"].to(dev)
for n in range(min(3, lp.shape[1])):
    segs = forced_align_segments(lp[:, n], tg[:, n], lp.shape[0],
                                 int(tl[n]), cs.null_class)
    lens = [e - s for (s, e, _c) in segs]
    print(f"  utt {n}: T={lp.shape[0]} L={int(tl[n])} -> {len(segs)} segments"
          f" | seg lengths min/median/max = "
          f"{min(lens) if lens else '-'}/"
          f"{sorted(lens)[len(lens)//2] if lens else '-'}/"
          f"{max(lens) if lens else '-'}")
    if segs and n == 0:
        print(f"    first 5: {segs[:5]}")
print("\n  if segment counts are non-zero here but min_len=2 drops them all,")
print("  the filter is the problem; if they are zero, the printed")
print("  forced_align error above names the real cause.")
PY

echo "=== 608 done -- arms stay stopped until the cause is named ==="
