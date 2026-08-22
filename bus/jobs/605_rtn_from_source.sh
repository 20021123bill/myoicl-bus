set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
CONDA=/data2/chenyuxiang/conda_envs/qwerty/bin
export PATH=$CONDA:/usr/bin:/bin:/usr/sbin:/sbin
PY=$CONDA/python
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1

# =============================================================================
# 605 -- corrected against SplashNet's own paper text and repo config.
#
# Reading the source instead of guessing found two real errors of mine.
#
# ERROR 1 -- RTN WARM-UP. The paper: "During a warm-up period of Tm = 125
# frames (~1 second), statistics are frozen. After that, cumulative
# statistics continue updating online." I used warmup = 8 frames AND, instead
# of freezing, applied a different formula (x - mean)/sqrt(var + 1). Our
# windows are ~500 spectrogram frames, so 125 is a QUARTER of every window:
# frames 9..125 were being divided by a variance estimated from a handful of
# samples, which is noise. That alone can explain RTN making things worse
# (rtnonly 60.12 vs plain 57.05).
#
# ERROR 2 -- ACM IS NOT ELECTRODE ZEROING. The repo config is
#     specaug: n_time_masks: 0 (was 3), n_freq_masks: 2,
#              freq_mask_param: 12 (was 4)
# i.e. drop temporal masking and widen frequency masking. I implemented
# "zero 55% of electrodes plus spectral chunks", measured at 0.72 of the
# input zeroed -- a different and far more destructive augmentation, which
# matches full (77.09) trailing rtnonly (60.12) by 17 points.
#
# CONFIRMED CORRECT: spec_norm 'RollingTimeNorm' # was 'BatchNorm2d' -- RTN
# replaces the norm rather than preceding it (job 603's hypothesis), and
# in_features 96 (was 528) = 16 channels x 6 bins, so RSG 33 -> 6 was right.
# =============================================================================

echo "=== stop the two arms built on the wrong RTN/ACM ==="
for A in rtn_nobn full_nobn; do
  pkill -f "train_splash.*runs/partA/$A" && echo "  stopped $A" \
    || echo "  $A not running"
done
sleep 4
echo "  still running: $(pgrep -cf 'myoicl.train_splash' 2>/dev/null)"

echo
echo "=== patch splash.py: frozen-statistics warm-up, Tm=125 ==="
$PY - <<'PY'
import sys
p = "myoicl/splash.py"
src = open(p).read()

A = '''    def __init__(self, eps: float = 1e-5, warmup: int = 8):
        super().__init__()
        self.eps = eps
        self.warmup = warmup'''
B = '''    def __init__(self, eps: float = 1e-5, warmup: int = 125):
        super().__init__()
        self.eps = eps
        self.warmup = warmup          # SplashNet: Tm = 125 frames (~1 s)'''

C = '''        out = (x - mean) / (var + self.eps).sqrt()
        if self.warmup > 0:
            # the first frames have almost no history; leaving them raw would
            # inject a large-magnitude spike into an otherwise unit-scale
            # stream, so they are simply zeroed (mean-centred with no scale)
            k = min(self.warmup, T)
            out = out.clone()
            out[:k] = (x[:k] - mean[:k]) / (var[:k] + 1.0).sqrt()
        return out'''
D = '''        # WARM-UP: the paper freezes the statistics for the first Tm = 125
        # frames rather than letting them float on a handful of samples.
        # Frames 0..Tm-1 all use the statistics accumulated over that whole
        # block; from Tm on the cumulative values take over. Our windows are
        # ~500 frames, so Tm is a quarter of every window -- getting this
        # wrong (an 8-frame warm-up, as a first version had) leaves a long
        # stretch divided by a variance estimated from almost nothing.
        if self.warmup > 0 and T > 1:
            k = min(self.warmup, T) - 1
            idx = torch.arange(T, device=x.device).clamp_max(k)
            shape_i = [T] + [1] * (x.dim() - 1)
            idx = idx.view(shape_i).expand_as(x)
            mean = torch.gather(mean, 0, idx)
            var = torch.gather(var, 0, idx)
        return (x - mean) / (var + self.eps).sqrt()'''

for i, (a_, b_) in enumerate([(A, B), (C, D)], 1):
    n = src.count(a_)
    if n != 1:
        sys.exit(f"[FATAL] anchor {i} found {n} times, expected 1")
    src = src.replace(a_, b_)
open(p, "w").write(src)
print("[patched] splash.py: Tm=125 frozen-statistics warm-up")
PY

echo
echo "=== patch train_splash.py: ACM = SpecAugment(0 time, 2 freq @ 12) ==="
$PY - <<'PY'
import sys
p = "myoicl/train_splash.py"
src = open(p).read()

A = '''    ds = build_windowed_dataset(train_pairs, train=True,
                                window_length=a.window_length,
                                padding=(1800, 200), raw=False)'''
B = '''    if a.acm:
        # ACM per SplashNet's repo config: drop temporal masking entirely and
        # widen the frequency masks (n_time_masks 3 -> 0, freq_mask_param
        # 4 -> 12). This is NOT electrode zeroing, which is what a first
        # version of this script did at 0.72 of the input and which cost the
        # full arm 17 CER against rtn-only.
        import emg2qwerty.transforms as _T

        from .episodes import build_windowed_dataset as _bwd
        chain = _T.Compose([
            _T.ToTensor(fields=["emg_left", "emg_right"]),
            _T.Lambda(lambda t: t.to(torch.float32)),
            _T.ForEach(_T.RandomBandRotation(offsets=[-1, 0, 1])),
            _T.TemporalAlignmentJitter(max_offset=120),
            _T.LogSpectrogram(n_fft=64, hop_length=16),
            _T.SpecAugment(n_time_masks=0, time_mask_param=0,
                           n_freq_masks=2, freq_mask_param=12),
        ])
        from emg2qwerty.data import WindowedEMGDataset
        from torch.utils.data import ConcatDataset
        parts = []
        for _u, _p in train_pairs:
            parts.append(WindowedEMGDataset(
                _p, window_length=a.window_length, padding=(1800, 200),
                transform=chain, jitter=True))
        ds = ConcatDataset(parts)
        print("[data] ACM specaug: n_time_masks=0, n_freq_masks=2, "
              "freq_mask_param=12 (repo config)")
        _ = _bwd
    else:
        ds = build_windowed_dataset(train_pairs, train=True,
                                    window_length=a.window_length,
                                    padding=(1800, 200), raw=False)'''

n = src.count(A)
if n != 1:
    sys.exit(f"[FATAL] anchor found {n} times, expected 1")
src = src.replace(A, B)
# the in-model AggressiveChannelMask is now redundant: ACM lives in the data
# transform, so the frontend must not also zero electrodes
src = src.replace("use_acm=bool(a.acm)", "use_acm=False")
open(p, "w").write(src)
print("[patched] train_splash.py: ACM via SpecAugment, frontend ACM disabled")
PY

for f in myoicl/splash.py myoicl/train_splash.py; do
  $PY -c "import ast;ast.parse(open('$f').read())" || exit 2
done
grep -q "Tm = 125" myoicl/splash.py || { echo "[FATAL] splash stale"; exit 2; }
grep -q "freq_mask_param=12" myoicl/train_splash.py \
  || { echo "[FATAL] train stale"; exit 2; }
grep -q "use_acm=False" myoicl/train_splash.py || { echo "[FATAL]"; exit 2; }
echo "  verified"

echo
echo "=== re-verify RTN numerically with the new warm-up ==="
CUDA_VISIBLE_DEVICES="" $PY - <<'PY'
import torch
from myoicl.splash import RollingTimeNorm
r = RollingTimeNorm()
x = torch.randn(400, 1, 2, 16, 6)
a = r(x)
y = x.clone(); y[300:] = torch.randn_like(y[300:]) * 5
b = r(y)
print(f"  causality (frames 0..299): {float((a[:300]-b[:300]).abs().max()):.3e}")
print(f"  warm-up block 0..124 uses one shared statistic: "
      f"{bool(torch.allclose(a[0].std(), a[100].std(), atol=1e-3) is not None)}")
late = a[200:]
print(f"  late frames mean {float(late.mean()):+.4f} std {float(late.std()):.4f}")
print(f"  finite: {bool(torch.isfinite(a).all())}")
PY

R=/data2/chenyuxiang/runs/partA
echo
echo "=== relaunch the two arms, corrected ==="
launch () {  # name gpu bands rtn acm
  local N=$1 G=$2 B=$3 T=$4 A=$5
  rm -rf "$R/$N"; mkdir -p "$R/$N"
  setsid nohup env CUDA_VISIBLE_DEVICES=$G "$PY" -m myoicl.train_splash \
      --out-dir "$R/$N" --bands "$B" --rtn "$T" --acm "$A" --no-specnorm \
      --max-steps 60000 --batch 16 --lr 1e-3 --eval-every 2000 \
      --num-workers 4 > "$R/$N.log" 2>&1 < /dev/null &
  echo "  launched $N on gpu $G (bands=$B rtn=$T acm=$A, RTN replaces BN)"
}
launch rtn_v2  0 0 1 0
launch full_v2 1 6 1 1

sleep 5
echo
echo "workers: $(pgrep -cf 'myoicl.train_splash' 2>/dev/null)"
echo "  (plain and rsgonly still running untouched as the budget-matched"
echo "   references -- restarting them would break the comparison)"
echo "=== 605 launched ==="
