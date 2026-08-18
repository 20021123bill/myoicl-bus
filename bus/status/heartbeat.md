# heartbeat 2026-08-18T20:59:51+08:00

## gpu
```
0, 16 MiB, 24576 MiB, 0 %
1, 12 MiB, 24576 MiB, 0 %
2, 2441 MiB, 24576 MiB, 0 %
3, 2751 MiB, 24576 MiB, 43 %
```

## jobs
```
001_apply_v41                            DONE rc=127
002_apply_v41                            DONE rc=127
010_archive                              DONE rc=127
020_d1_gatefix                           DONE rc=127
030_d2_forcectx                          DONE rc=127
040_fix_runner                           DONE rc=127
050_eval_e1_then_d0a                     DONE rc=127
060_d0b_gate0_forced                     DONE rc=127
070_diag_relaunch                        DONE rc=127
```

## tail of each log (last 25 lines)

### 001_apply_v41.log
```
tar (child): tools/myoicl_v41_drop.tar.gz：无法 open: 没有那个文件或目录
tar (child): Error is not recoverable: exiting now
tar: Child returned status 2
tar: Error is not recoverable: exiting now
python: can't open file '/tmp/v41/patch_v41.py': [Errno 2] No such file or directory
PATCH FAILED rc=2 -- refusing to queue D1/D2
```

### 002_apply_v41.log
```
=== v4.1 patch report ===
  CHANGED  context.py:signature
  CHANGED  context.py:zero-init-matrix
  CHANGED  model.py:signature
  CHANGED  model.py:build_model
  CHANGED  model.py:cross-attn-callsites(2)
  CHANGED  train_qwerty.py:optimizer-nodecay-groups
  CHANGED  gate_report.py:gate-threshold
  CHANGED  gate_report.py:matrix-threshold
  CHANGED  gate_report.py:three-state-flag
  CHANGED  configs/qwerty_forcectx.yaml:gate_init
  CHANGED  configs/qwerty_gatefix.yaml:created
  AST OK   context.py
  AST OK   model.py
  AST OK   train_qwerty.py
  AST OK   gate_report.py
  YAML OK  qwerty_gatefix.yaml | p_synth=0.15 gate_init=1.0 lr=3e-05/0.001 out=myoicl_d1_gatefix
  YAML OK  qwerty_forcectx.yaml | p_synth=0.85 gate_init=1.0 lr=3e-05/0.001 out=myoicl_d2_forcectx
v4.1 applied; 010/020/030 queued for the next runner cycle
```

### 010_archive.log
```
  all CLOSED  -> the context pathway never opened. A zero
                conditioning gain is expected and says nothing
                about whether context is informative. Fix the
                incentive (curriculum / p_synth / ctx_lr), not
                the architecture.
  some OPEN   -> context does reach the decoder. A zero gain then
                means the model looked and found nothing useful,
                which is a real (negative) result about the
                training signal.
=== checkpoints on disk ===
-rw-rw-r-- 1 chenyuxiang chenyuxiang 75167599 8月  18 02:53 /data2/chenyuxiang/runs/myoicl_joint/best.pt
-rw-rw-r-- 1 chenyuxiang chenyuxiang 75167599 8月  18 03:19 /data2/chenyuxiang/runs/myoicl_joint/last.pt
-rw-rw-r-- 1 chenyuxiang chenyuxiang 75167535 8月  18 06:16 /data2/chenyuxiang/runs/myoicl_scratch/best.pt
-rw-rw-r-- 1 chenyuxiang chenyuxiang 75167535 8月  18 08:13 /data2/chenyuxiang/runs/myoicl_scratch/last.pt
=== archived ===
总用量 164
drwxrwxr-x 2 chenyuxiang chenyuxiang  4096 8月  18 20:26 .
drwxrwxr-x 3 chenyuxiang chenyuxiang  4096 8月  18 20:26 ..
-rw-rw-r-- 1 chenyuxiang chenyuxiang  3616 8月  18 20:26 e2_ABC_k256.json
-rw-rw-r-- 1 chenyuxiang chenyuxiang   298 8月  18 20:26 e2_probe_user0.log
-rw-rw-r-- 1 chenyuxiang chenyuxiang  1514 8月  18 20:26 gate_report.txt
-rw-rw-r-- 1 chenyuxiang chenyuxiang 24047 8月  18 20:26 joint.log
-rw-rw-r-- 1 chenyuxiang chenyuxiang 27808 8月  18 20:26 joint_log.csv
-rw-rw-r-- 1 chenyuxiang chenyuxiang 42932 8月  18 20:26 scratch.log
-rw-rw-r-- 1 chenyuxiang chenyuxiang 48841 8月  18 20:26 scratch_log.csv
```

### 020_d1_gatefix.log
```
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
step 100/8000 | loss 1.2474 | lr 1.01e-05 | 2.92 it/s
step 200/8000 | loss 1.1509 | lr 2.01e-05 | 2.47 it/s
step 300/8000 | loss 1.0646 | lr 3.00e-05 | 2.39 it/s
step 400/8000 | loss 1.1144 | lr 3.00e-05 | 3.05 it/s
```

### 021_d1_retry.log
```
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
step 100/8000 | loss 1.0837 | lr 1.01e-05 | 1.07 it/s
step 200/8000 | loss 1.0618 | lr 2.01e-05 | 1.32 it/s
```

### 030_d2_forcectx.log
```
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
step 100/8000 | loss 3.1743 | lr 1.01e-05 | 1.18 it/s
step 200/8000 | loss 2.8405 | lr 2.01e-05 | 1.48 it/s
```

### 031_d2_retry.log
```
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
```

### 040_fix_runner.log
```
runner.sh patched; effective on next restart
```

### 050_eval_e1_then_d0a.log
```
=== E1 scratch: 8-user eval A/B/C K=256 ===
[ckpt] /data2/chenyuxiang/runs/myoicl_scratch/best.pt (v1, step 50000)
[A] user0: CER 65.64
[A] user1: CER 63.57
[A] user2: CER 43.79
[A] user3: CER 54.54
[A] user4: CER 52.89
[A] user5: CER 42.05
[A] user6: CER 58.39
```

### 060_d0b_gate0_forced.log
```
=== D0b: gate_init 0.0, p_synth 0.85 (deadlock control for D2) ===
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
step 100/8000 | loss 3.2248 | lr 1.01e-05 | 0.94 it/s
step 200/8000 | loss 2.7142 | lr 2.01e-05 | 1.12 it/s
```

### 070_diag_relaunch.log
```
[五 5月 15 17:19:17 2026] GTP_0 invoked oom-killer: gfp_mask=0x1100cca(GFP_HIGHUSER_MOVABLE), order=0, oom_score_adj=0
[五 5月 15 17:19:17 2026] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0-1,global_oom,task_memcg=/user.slice/user-1021.slice/session-963.scope,task=MATLAB,pid=4167877,uid=1021
[五 5月 15 17:19:17 2026] Out of memory: Killed process 4167877 (MATLAB) total-vm:78782016kB, anon-rss:49186560kB, file-rss:0kB, shmem-rss:0kB, UID:1021 pgtables:99276kB oom_score_adj:0
[五 5月 15 17:19:25 2026] MCR 0 interpret invoked oom-killer: gfp_mask=0x1100cca(GFP_HIGHUSER_MOVABLE), order=0, oom_score_adj=0
[五 5月 15 17:19:25 2026] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0-1,global_oom,task_memcg=/user.slice/user-1021.slice/session-963.scope,task=MATLAB,pid=4166554,uid=1021
[五 5月 15 17:19:25 2026] Out of memory: Killed process 4166554 (MATLAB) total-vm:74016844kB, anon-rss:50961616kB, file-rss:0kB, shmem-rss:0kB, UID:1021 pgtables:101960kB oom_score_adj:0
[五 5月 15 17:19:36 2026] GTP_1 invoked oom-killer: gfp_mask=0x1100cca(GFP_HIGHUSER_MOVABLE), order=0, oom_score_adj=0
[五 5月 15 17:19:36 2026] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0-1,global_oom,task_memcg=/user.slice/user-1021.slice/session-963.scope,task=MATLAB,pid=4158571,uid=1021
[五 5月 15 17:19:36 2026] Out of memory: Killed process 4158571 (MATLAB) total-vm:78520472kB, anon-rss:52567212kB, file-rss:0kB, shmem-rss:0kB, UID:1021 pgtables:106796kB oom_score_adj:0
=== last lines of the two dead runs ===
tail: 在无效上下文中使用选项 -- 3

=== killing all training procs, relaunching D1/D2 with num_workers=2 ===
=== live training procs AFTER ===
3425974 python -m myoicl.train_qwerty --config myoicl/configs/qwerty_gatefix.yaml --set data.num_workers=2 --set out_dir=/data2/chenyuxiang/runs/myoic
3426208 python -m myoicl.train_qwerty --config myoicl/configs/qwerty_gatefix.yaml --set data.num_workers=2 --set out_dir=/data2/chenyuxiang/runs/myoic
3426271 python -m myoicl.train_qwerty --config myoicl/configs/qwerty_gatefix.yaml --set data.num_workers=2 --set out_dir=/data2/chenyuxiang/runs/myoic
3426829 python -m myoicl.train_qwerty --config myoicl/configs/qwerty_forcectx.yaml --set data.num_workers=2 --set out_dir=/data2/chenyuxiang/runs/myoi
3427068 python -m myoicl.train_qwerty --config myoicl/configs/qwerty_forcectx.yaml --set data.num_workers=2 --set out_dir=/data2/chenyuxiang/runs/myoi
3427131 python -m myoicl.train_qwerty --config myoicl/configs/qwerty_forcectx.yaml --set data.num_workers=2 --set out_dir=/data2/chenyuxiang/runs/myoi
=== first lines of the retries ===
tail: 在无效上下文中使用选项 -- 4
              总计         已用        空闲      共享    缓冲/缓存    可用
内存：        2015          20         172           0        1822        1984
交换：           7           0           7
```
