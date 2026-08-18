set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH

echo "########## patch: fork -> spawn for episode dataloader workers ##########"
python - <<'PYEOF'
import pathlib, ast
p = pathlib.Path("myoicl/train_qwerty.py"); s = p.read_text()
old = """        train_loader = DataLoader(
            episode_ds, batch_size=None, num_workers=num_workers,
            pin_memory=True, persistent_workers=num_workers > 0,
        )"""
new = """        # Workers are FORKED by default. The episode dataset opens HDF5
        # files and libhdf5 is not fork-safe: a handle created in the parent
        # and inherited by a forked child can wedge both. Measured 2026-08-18:
        # every run with num_workers>0 stopped advancing after 100-400 steps
        # with no error, no exit, ~0% CPU and 0% GPU, while num_workers=0 ran
        # fine (just 8x slower). "spawn" starts each worker as a fresh
        # interpreter that inherits no file handles.
        _mpctx = dcfg.get("mp_context", "spawn") if num_workers > 0 else None
        train_loader = DataLoader(
            episode_ds, batch_size=None, num_workers=num_workers,
            pin_memory=True, persistent_workers=num_workers > 0,
            multiprocessing_context=_mpctx,
        )"""
if "multiprocessing_context=_mpctx" in s:
    print("already patched")
elif old in s:
    s = s.replace(old, new, 1); ast.parse(s); p.write_text(s); print("patched to spawn")
else:
    print("ANCHOR NOT FOUND -- left untouched")
PYEOF

echo
echo "########## stop the slow workers=0 run, relaunch with spawn ##########"
pkill -f "myoicl.train_qwerty" 2>/dev/null
sleep 8
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=3 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_forcectx.yaml \
  --set data.num_workers=4 \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_d2_spawn \
  > bus/results/034_d2_spawn.log 2>&1 &
sleep 20
CUDA_VISIBLE_DEVICES=2 nohup python -m myoicl.train_qwerty \
  --config myoicl/configs/qwerty_gatefix.yaml \
  --set data.num_workers=4 \
  --set out_dir=/data2/chenyuxiang/runs/myoicl_d1_spawn \
  > bus/results/035_d1_spawn.log 2>&1 &

echo "launched D2 (GPU3) and D1 (GPU2) with spawn workers; watching 6 minutes"
sleep 360
echo "########## after 6 minutes ##########"
echo "--- D2 ---"; tail -6 bus/results/034_d2_spawn.log
echo "--- D1 ---"; tail -6 bus/results/035_d1_spawn.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
