# heartbeat 2026-08-19T07:37:03+08:00

## gpu
```
0, 16 MiB, 24576 MiB, 0 %
1, 12 MiB, 24576 MiB, 0 %
2, 12 MiB, 24576 MiB, 0 %
3, 12 MiB, 24576 MiB, 0 %
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
080_stackdump_and_solo                   DONE rc=127
090_stall_hunt                           DONE rc=127
100_io_throughput                        DONE rc=127
110_spawn_fix                            DONE rc=127
120_clean_home                           DONE rc=127
130_commit_and_eval_d1                   DONE rc=127
140_gates_and_later_evals                DONE rc=127
150_truth_from_ckpt                      DONE rc=127
160_eval_curve_by_ckpt                   DONE rc=127
170_effective_injection                  DONE rc=127
180_pretrain_units                       DONE rc=127
190_init_units_from                      DONE rc=127
200_d3_from_pretrained_units             DONE rc=127
210_encoding_beta_d4                     DONE rc=127
220_v3_deploy_smoke_train                DONE rc=127
230_v3_fix_smoke_and_train               DONE rc=127
240_v3_fix2_smoke_train                  DONE rc=127
250_v3_smoke_train                       DONE rc=127
260_v3_smoke3_train                      DONE rc=127
270_v3_fullwin_train                     DONE rc=127
280_v3_padfix_train                      DONE rc=127
290_v3_cheavy                            DONE rc=127
300_v3_frozen                            DONE rc=127
310_v31_kvsplit                          DONE rc=127
320_v31_relaunch                         DONE rc=127
330_v32_filmonly                         DONE rc=127
340_ctxsource_diag                       DONE rc=127
350_samesession_diag                     DONE rc=127
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
step 100/8000 | loss 3.2712 | lr 1.01e-05 | 0.57 it/s
```

### 032_d2_solo.log
```
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
step 100/8000 | loss 3.2712 | lr 1.01e-05 | 0.56 it/s
```

### 033_d2_w0.log
```
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
[watchdog] armed
step 100/8000 | loss 3.1789 | lr 1.01e-05 | 0.36 it/s
step 200/8000 | loss 2.9469 | lr 2.01e-05 | 0.34 it/s
```

### 034_d2_spawn.log
```
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
[watchdog] armed
step 100/8000 | loss 3.3284 | lr 1.01e-05 | 0.82 it/s
step 200/8000 | loss 2.7280 | lr 2.01e-05 | 0.86 it/s
```

### 035_d1_spawn.log
```
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
[watchdog] armed
```

### 036_d1_spawn2.log
```
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
[watchdog] armed
step 100/8000 | loss 1.1744 | lr 1.01e-05 | 1.93 it/s
step 200/8000 | loss 1.1658 | lr 2.01e-05 | 1.72 it/s
step 300/8000 | loss 1.0905 | lr 3.00e-05 | 1.86 it/s
step 400/8000 | loss 0.9526 | lr 3.00e-05 | 2.08 it/s
step 500/8000 | loss 1.1182 | lr 3.00e-05 | 1.98 it/s
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

### 080_stackdump_and_solo.log
```
State:	R (running)
Threads:	3
Permission Denied: Try running again with elevated permissions by going 'sudo env "PATH=$PATH" !!'
py-spy dump failed for 3426271
----- pid 3426829 -----
State:	R (running)
Threads:	71
Permission Denied: Try running again with elevated permissions by going 'sudo env "PATH=$PATH" !!'
py-spy dump failed for 3426829

=== killing everything, relaunching D2 ALONE with contention fixes ===
launched D2 solo on GPU3; sleeping 240s to see whether it passes the point where the others died
=== D2 solo after 4 min ===
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
step 100/8000 | loss 3.2712 | lr 1.01e-05 | 0.56 it/s
=== load / gpu ===
 21:04:44 up 112 days,  23:20,  22 users,  load average: 0.04, 0.02, 0.01
0, 16 MiB, 0 %
1, 12 MiB, 0 %
2, 12 MiB, 0 %
3, 2751 MiB, 0 %
```

### 090_stall_hunt.log
```

########## 2. what are the stuck processes waiting on? ##########
--- pid 3432429  state=R (running)  wchan=0  threads=15
--- pid 3432562  state=S (sleeping)  wchan=do_sys_poll  threads=3
--- pid 3432625  state=S (sleeping)  wchan=do_sys_poll  threads=3
(no output above = nothing stuck / already gone)

########## 3. install the stall watchdog ##########
installed: True

########## 4. kill everything, run D2 alone with num_workers=0 ##########
launched D2 with num_workers=0 on GPU3; watching for 5 minutes
########## 5. after 5 minutes ##########
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
[watchdog] armed
step 100/8000 | loss 3.1789 | lr 1.01e-05 | 0.36 it/s
 22:11:17 up 113 days,  27 min,  21 users, 0.04, 0.02, 0.01
0, 16 MiB, 0 %
1, 12 MiB, 0 %
2, 12 MiB, 0 %
3, 3037 MiB, 0 %
```

### 100_io_throughput.log
```
    PID USER     %CPU %MEM     ELAPSED COMMAND
4055697 YuYang   1969  0.0 39-08:52:55 clash-linux-amd
3506994 chenyux+  437  0.1       08:08 pt_main_thread
 840961 wyxuan    9.6  0.0 30-01:47:03 MainThread
   3516 root      3.7  0.0 113-00:29:51 ToDesk_Service
 840349 wyxuan    3.3  0.0 30-01:47:14 sshd
   2244 message+  1.6  0.0 113-00:30:03 dbus-daemon
2013687 weiyich+  1.6  0.0 13-06:07:21 codex
   2380 root      1.5  0.0 113-00:30:03 rustdesk
 841212 wyxuan    1.0  0.0 30-01:47:01 codex
2013249 weiyich+  0.9  0.0 13-06:07:25 node
 840606 wyxuan    0.7  0.0 30-01:47:05 MainThread
--- top IO waiters ---
  15729 root     D    -                    sync
2984980 root     D    -                    kworker/22:1+usb_hub_wq
3083474 root     Ds   -                    mount.exfat
3252434 root     D    -                    sync

########## our run right now ##########
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
[watchdog] armed
step 100/8000 | loss 3.1789 | lr 1.01e-05 | 0.36 it/s
```

### 110_spawn_fix.log
```
########## patch: fork -> spawn for episode dataloader workers ##########
patched to spawn

########## stop the slow workers=0 run, relaunch with spawn ##########
launched D2 (GPU3) and D1 (GPU2) with spawn workers; watching 6 minutes
########## after 6 minutes ##########
--- D2 ---
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
[watchdog] armed
step 100/8000 | loss 3.3284 | lr 1.01e-05 | 0.82 it/s
step 200/8000 | loss 2.7280 | lr 2.01e-05 | 0.86 it/s
--- D1 ---
[model] v1 | 6.25M params total (published backbone 5.29M + ICL module 0.96M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.96M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 98 context tensors keep their initialization
[watchdog] armed
0, 16 MiB, 0 %
1, 12 MiB, 0 %
2, 2443 MiB, 0 %
3, 2483 MiB, 0 %
```

### 120_clean_home.log
```
removed /tmp/v41

########## $HOME after ##########
7.0G	/home/chenyuxiang
0	/home/chenyuxiang/.sudo_as_admin_successful
4.0K	/home/chenyuxiang/.bash_logout
4.0K	/home/chenyuxiang/.condarc
4.0K	/home/chenyuxiang/.profile
4.0K	/home/chenyuxiang/.wget-hsts
4.0K	/home/chenyuxiang/.Xauthority
8.0K	/home/chenyuxiang/.bashrc
12K	/home/chenyuxiang/.conda
20K	/home/chenyuxiang/.ssh
52K	/home/chenyuxiang/.config
72K	/home/chenyuxiang/.bash_history
84K	/home/chenyuxiang/.nv
384K	/home/chenyuxiang/.local
7.0G	/home/chenyuxiang/.cache

NOTE: ~/.ssh/ is left alone on purpose -- ssh only reads keys from there,
      and the whole channel to GitHub depends on it. It is a few KB.

########## confirming our real footprint is all under /data2 ##########
12M	/data2/chenyuxiang/code/myoicl
920M	/data2/chenyuxiang/runs
```

### 130_commit_and_eval_d1.log
```
[A] user7: CER 52.69
[A] mean over users: 54.38
[B] user0: CER 59.19
[B] user1: CER 61.95
[B] user2: CER 45.82
[B] user3: CER 53.32
[B] user4: CER 57.67
[B] user5: CER 49.20
[B] user6: CER 55.23
[B] user7: CER 52.04
[B] mean over users: 54.30
[C] user0: CER 61.94
[C] user1: CER 61.81
[C] user2: CER 45.82
[C] user3: CER 53.45
[C] user4: CER 55.69
[C] user5: CER 49.26
[C] user6: CER 55.51
[C] user7: CER 52.79
[C] mean over users: 54.53
[A] gap closed vs personalization ceiling: 2.3%
[B] gap closed vs personalization ceiling: 2.5%
[C] gap closed vs personalization ceiling: 1.9%
[saved] /data2/chenyuxiang/runs/eval/d1_early_ABC_k256.json
=== DONE: this is the first real mode-C number on the 8 held-out users ===
```

### 140_gates_and_later_evals.log
```
--- waiting for D1 to pass step 3000 ---
=== D1 @ ~step 3000 : 8 official test users, modes A/B/C, K=256 ===
[A] mean over users: 54.57
[B] mean over users: 54.69
[C] mean over users: 54.66
[A] gap closed vs personalization ceiling: 1.9%
[B] gap closed vs personalization ceiling: 1.6%
[C] gap closed vs personalization ceiling: 1.7%
=== /tmp/d1_snap_3000.pt  (step 8000) ===
    CLOSED  cross_post.gate              g=-0.00345  tanh=-0.00345
    CLOSED  cross_pre.gate               g=+0.00061  tanh=+0.00061
    OPEN    film.up.bias                             |w|mean=3.977e-02 max=2.029e-01
    OPEN    film.up.weight                           |w|mean=2.427e-02 max=1.702e-01
  all CLOSED  -> the context pathway never opened. A zero
  some OPEN   -> context does reach the decoder. A zero gain then
--- waiting for D1 to pass step 6000 ---
=== D1 @ ~step 6000 : 8 official test users, modes A/B/C, K=256 ===
=== /tmp/d1_snap_6000.pt  (step 8000) ===
    CLOSED  cross_post.gate              g=-0.00345  tanh=-0.00345
    CLOSED  cross_pre.gate               g=+0.00061  tanh=+0.00061
    OPEN    film.up.bias                             |w|mean=3.977e-02 max=2.029e-01
    OPEN    film.up.weight                           |w|mean=2.427e-02 max=1.702e-01
  all CLOSED  -> the context pathway never opened. A zero
  some OPEN   -> context does reach the decoder. A zero gain then
--- waiting for D1 to pass step 8000 ---
```

### 150_truth_from_ckpt.log
```
########## the log is unreliable -- ask the checkpoint and the process ##########
D1: checkpoint step=2000  written 5.4 min ago
D2: no checkpoint yet

--- processes and how much CPU time they have burned ---
3522548       34:32 00:57:33  166 python -m myoicl.train_qwerty --config myoicl/configs/qwerty_forcectx.yaml --set data.num_workers=4 --set out_dir=/data2/cheny
3579965       24:08 00:21:31 89.2 python -m myoicl.train_qwerty --config myoicl/configs/qwerty_gatefix.yaml --set data.num_workers=4 --set train.save_every=1000

--- true step numbers straight from the live logs on disk ---
bus/results/036_d1_spawn2.log: 5 step lines, last = step 500/8000 | loss 1.1182 | lr 3.00e-05 | 1.98 it/s
bus/results/034_d2_spawn.log: 2 step lines, last = step 200/8000 | loss 2.7280 | lr 2.01e-05 | 0.86 it/s

########## keep future logs OUT of git's reach ##########
runner.sh patched (effective on next runner restart)
```

### 160_eval_curve_by_ckpt.log
```
[B] gap closed vs personalization ceiling: 1.4%
[C] gap closed vs personalization ceiling: 1.5%
=== /tmp/d1_snap.pt  (step 5000) ===
    CLOSED  cross_post.gate              g=+0.00376  tanh=+0.00376
    CLOSED  cross_pre.gate               g=+0.00281  tanh=+0.00281
    OPEN    film.up.bias                             |w|mean=3.866e-02 max=1.895e-01
    OPEN    film.up.weight                           |w|mean=2.401e-02 max=1.547e-01
  all CLOSED  -> the context pathway never opened. A zero
  some OPEN   -> context does reach the decoder. A zero gain then
--- waiting for D1 checkpoint to reach step 8000 (23:20) ---
=== D1 @ step 8000 : 8 official held-out users, modes A/B/C, K=256 ===
[A] mean over users: 54.57
[B] mean over users: 54.69
[C] mean over users: 54.66
[A] gap closed vs personalization ceiling: 1.9%
[B] gap closed vs personalization ceiling: 1.6%
[C] gap closed vs personalization ceiling: 1.7%
=== /tmp/d1_snap.pt  (step 8000) ===
    CLOSED  cross_post.gate              g=-0.00345  tanh=-0.00345
    CLOSED  cross_pre.gate               g=+0.00061  tanh=+0.00061
    OPEN    film.up.bias                             |w|mean=3.977e-02 max=2.029e-01
    OPEN    film.up.weight                           |w|mean=2.427e-02 max=1.702e-01
  all CLOSED  -> the context pathway never opened. A zero
  some OPEN   -> context does reach the decoder. A zero gain then
=== eval curve complete ===
```

### 170_effective_injection.log
```
nothing on its own. The identifiable quantity is the product.

--- myoicl_d1_spawn  step 3000
    cross_pre   tanh(g)=+0.01901  ||W||=  7.1508  ||b||= 1.0978   EFFECTIVE |tanh(g)|*||W|| = 0.13590
    cross_post  tanh(g)=+0.02816  ||W||=  6.0389  ||b||= 0.6200   EFFECTIVE |tanh(g)|*||W|| = 0.17007
    film.up     ||W||=  6.1545
    TOTAL effective cross-attention injection = 0.30598
--- myoicl_d2_spawn  step 2000
    cross_pre   tanh(g)=+0.39751  ||W||=  7.0806  ||b||= 1.0337   EFFECTIVE |tanh(g)|*||W|| = 2.81460
    cross_post  tanh(g)=+0.32762  ||W||=  5.2151  ||b||= 1.1652   EFFECTIVE |tanh(g)|*||W|| = 1.70855
    film.up     ||W||=  6.5064
    TOTAL effective cross-attention injection = 4.52315
--- myoicl_joint  step 20000
    cross_pre   tanh(g)=-0.00012  ||W||= 15.8597  ||b||= 1.4019   EFFECTIVE |tanh(g)|*||W|| = 0.00191
    cross_post  tanh(g)=-0.00142  ||W||= 15.7783  ||b||= 1.3316   EFFECTIVE |tanh(g)|*||W|| = 0.02244
    film.up     ||W||=  2.0835
    TOTAL effective cross-attention injection = 0.02434
--- myoicl_scratch  step 70000
    cross_pre   tanh(g)=-0.00043  ||W||= 22.7253  ||b||= 2.2289   EFFECTIVE |tanh(g)|*||W|| = 0.00984
    cross_post  tanh(g)=-0.00004  ||W||= 14.4087  ||b||= 1.2476   EFFECTIVE |tanh(g)|*||W|| = 0.00060
    film.up     ||W||= 18.3706
    TOTAL effective cross-attention injection = 0.01044

=== patch gate_report.py so every future report prints the product ===
gate_report.py now prints the effective product
```

### 180_pretrain_units.log
```
[stage1'] 8000/12000 | loss 0.0265 | predict-the-mean 0.0370 | skill +28.5%
[stage1'] 8200/12000 | loss 0.0262 | predict-the-mean 0.0359 | skill +26.9%
[stage1'] 8400/12000 | loss 0.0296 | predict-the-mean 0.0400 | skill +26.1%
[stage1'] 8600/12000 | loss 0.0278 | predict-the-mean 0.0381 | skill +27.0%
[stage1'] 8800/12000 | loss 0.0279 | predict-the-mean 0.0383 | skill +27.0%
[stage1'] 9000/12000 | loss 0.0274 | predict-the-mean 0.0379 | skill +27.7%
[stage1'] 9200/12000 | loss 0.0270 | predict-the-mean 0.0388 | skill +30.3%
[stage1'] 9400/12000 | loss 0.0263 | predict-the-mean 0.0362 | skill +27.4%
[stage1'] 9600/12000 | loss 0.0263 | predict-the-mean 0.0371 | skill +29.0%
[stage1'] 9800/12000 | loss 0.0264 | predict-the-mean 0.0377 | skill +29.9%
[stage1'] 10000/12000 | loss 0.0261 | predict-the-mean 0.0367 | skill +28.9%
[stage1'] 10200/12000 | loss 0.0254 | predict-the-mean 0.0365 | skill +30.2%
[stage1'] 10400/12000 | loss 0.0256 | predict-the-mean 0.0361 | skill +29.0%
[stage1'] 10600/12000 | loss 0.0275 | predict-the-mean 0.0385 | skill +28.7%
[stage1'] 10800/12000 | loss 0.0252 | predict-the-mean 0.0365 | skill +30.9%
[stage1'] 11000/12000 | loss 0.0269 | predict-the-mean 0.0387 | skill +30.4%
[stage1'] 11200/12000 | loss 0.0253 | predict-the-mean 0.0359 | skill +29.5%
[stage1'] 11400/12000 | loss 0.0263 | predict-the-mean 0.0362 | skill +27.4%
[stage1'] 11600/12000 | loss 0.0288 | predict-the-mean 0.0393 | skill +26.7%
[stage1'] 11800/12000 | loss 0.0245 | predict-the-mean 0.0345 | skill +28.9%
[stage1'] 12000/12000 | loss 0.0274 | predict-the-mean 0.0382 | skill +28.2%
[saved] /data2/chenyuxiang/runs/units_pretrain.pt
Next: python -m myoicl.loso --holdout 0 --init-from /data2/chenyuxiang/runs/units_pretrain.pt
=== done, checkpoint: ===
-rw-rw-r-- 1 chenyuxiang chenyuxiang 25154965 8月  19 03:36 /data2/chenyuxiang/runs/units_pretrain.pt
```

### 190_init_units_from.log
```
=== add init_units_from: load ONLY the context encoder from a units pretrain ===
init_units_from added

=== smoke: config parses and the key is read ===
621:    units_ckpt = cfg.get("init_units_from")
632:                f"init_units_from={units_ckpt} contains no ctx_encoder.* tensors"
committed
```

### 200_d3_from_pretrained_units.log
```
    cross_pre    |tanh(g)|=0.22478  ||W||=  6.2468  EFFECTIVE=1.40417
    cross_post   |tanh(g)|=0.18466  ||W||=  5.1602  EFFECTIVE=0.95287
--- waiting for D3 checkpoint step >= 5000 (00:00) ---
=== D3 @ step ~5000 : 8 official held-out users, A/B/C, K=256 ===
[A] mean over users: 54.76
[B] mean over users: 55.26
[C] mean over users: 54.95
[A] gap closed vs personalization ceiling: 1.4%
[B] gap closed vs personalization ceiling: 0.3%
[C] gap closed vs personalization ceiling: 1.0%
=== /tmp/d3_snap.pt  (step 5000) ===
    cross_pre    |tanh(g)|=0.00317  ||W||=  8.4206  EFFECTIVE=0.02668
    cross_post   |tanh(g)|=0.00475  ||W||=  7.4897  EFFECTIVE=0.03559
--- waiting for D3 checkpoint step >= 8000 (00:30) ---
=== D3 @ step ~8000 : 8 official held-out users, A/B/C, K=256 ===
[A] mean over users: 54.49
[B] mean over users: 54.77
[C] mean over users: 54.78
[A] gap closed vs personalization ceiling: 2.0%
[B] gap closed vs personalization ceiling: 1.4%
[C] gap closed vs personalization ceiling: 1.4%
=== /tmp/d3_snap.pt  (step 8000) ===
    cross_pre    |tanh(g)|=0.00165  ||W||=  8.4925  EFFECTIVE=0.01399
    cross_post   |tanh(g)|=0.00059  ||W||=  7.6299  EFFECTIVE=0.00450
=== D3 evaluation curve complete ===
```

### 210_encoding_beta_d4.log
```
    cross_pre    |tanh(g)|=0.23011  ||W||=  6.3289  EFFECTIVE=1.45636
    cross_post   |tanh(g)|=0.18240  ||W||=  5.3596  EFFECTIVE=0.97761
--- waiting for D4 checkpoint step >= 5000 (00:51) ---
=== D4 @ step ~5000 : 8 official held-out users, A/B/C, K=256 ===
[A] mean over users: 55.22
[B] mean over users: 55.60
[C] mean over users: 55.59
[A] gap closed vs personalization ceiling: 0.4%
[B] gap closed vs personalization ceiling: -0.5%
[C] gap closed vs personalization ceiling: -0.5%
=== /tmp/d4_snap.pt  (step 5000) ===
    cross_pre    |tanh(g)|=0.00057  ||W||=  8.2775  EFFECTIVE=0.00473
    cross_post   |tanh(g)|=0.00750  ||W||=  7.6652  EFFECTIVE=0.05746
--- waiting for D4 checkpoint step >= 8000 (01:25) ---
=== D4 @ step ~8000 : 8 official held-out users, A/B/C, K=256 ===
[A] mean over users: 55.30
[B] mean over users: 55.27
[C] mean over users: 55.45
[A] gap closed vs personalization ceiling: 0.2%
[B] gap closed vs personalization ceiling: 0.3%
[C] gap closed vs personalization ceiling: -0.1%
=== /tmp/d4_snap.pt  (step 8000) ===
    cross_pre    |tanh(g)|=0.00149  ||W||=  8.4280  EFFECTIVE=0.01255
    cross_post   |tanh(g)|=0.00158  ||W||=  7.6839  EFFECTIVE=0.01210
=== D4 curve complete ===
```

### 220_v3_deploy_smoke_train.log
```
=== deploy v3 code ===
-rw-r--r-- 1 chenyuxiang chenyuxiang 2615 8月  19 01:51 myoicl/configs/qwerty_v3_ctxframe.yaml
-rw-r--r-- 1 chenyuxiang chenyuxiang 5712 8月  19 01:46 myoicl/ctx_frame.py
-rw-r--r-- 1 chenyuxiang chenyuxiang 3695 8月  19 01:51 myoicl/smoke_v3.py

=== v3 smoke test (CPU, tiny model) ===
Traceback (most recent call last):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 91, in <module>
    sys.exit(main())
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 28, in main
    model = build_model(cfg, num_classes=V)
  File "/data2/chenyuxiang/code/myoicl/myoicl/model.py", line 343, in build_model
    return MyoICLModel(
  File "/data2/chenyuxiang/code/myoicl/myoicl/model.py", line 97, in __init__
    assert d_model == num_bands * official_mlp_features[-1], (
AssertionError: d_model (64) must equal num_bands * mlp_features[-1] (128)
SMOKE FAILED -- not launching training. Fix needed.
```

### 230_v3_fix_smoke_and_train.log
```
=== fix the smoke tiny-config (d_model must equal num_bands*mlp[-1]) ===
fixed: True

=== v3 smoke test ===
Traceback (most recent call last):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 91, in <module>
    sys.exit(main())
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 47, in main
    tokens, pooled = model.encode_context(
  File "/data2/chenyuxiang/code/myoicl/myoicl/model.py", line 235, in encode_context
    device=ctx_labeled_feats.device,
AttributeError: 'NoneType' object has no attribute 'device'
SMOKE STILL FAILING -- stopping.
```

### 240_v3_fix2_smoke_train.log
```
=== deploy v3 fix (encode_context v3 branch moved before stats block) ===

=== v3 smoke test ===
[smoke] built 50 support tokens from 6 windows
[smoke] mean |mode C - mode A| = 0.0000e+00
Traceback (most recent call last):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 91, in <module>
    sys.exit(main())
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 61, in main
    assert diff > 1e-6, "context does not change the output (gate stuck closed?)"
AssertionError: context does not change the output (gate stuck closed?)
SMOKE STILL FAILING -- stopping.
```

### 250_v3_smoke_train.log
```
=== deploy corrected smoke (identity-at-init is CORRECT; test open-after-training) ===
=== v3 smoke ===
[smoke] built 50 support tokens from 6 windows
[smoke] init max|mode C - mode A| = 0.00e+00 (want ~0: identity)
[smoke] init grad o_proj=3.382e+01 (want >0: path can open)
[smoke] after 8 steps mean|mode C - mode A| = nan (want >0: context now changes output)
Traceback (most recent call last):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 126, in <module>
    sys.exit(main())
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 100, in main
    assert diff1 > 1e-6, "context still does nothing after training -- path stuck"
AssertionError: context still does nothing after training -- path stuck
SMOKE STILL FAILING -- stopping.
```

### 260_v3_smoke3_train.log
```
[B] mean over users: 54.03
[C] mean over users: 54.57
[A] gap closed vs personalization ceiling: 3.1%
[B] gap closed vs personalization ceiling: 3.1%
[C] gap closed vs personalization ceiling: 1.8%
=== /tmp/v3_snap.pt  (step 2000) ===
    cross_pre    |tanh(g)|=0.45023  ||W||=  6.1459  EFFECTIVE=2.76706
    cross_post   |tanh(g)|=0.42207  ||W||=  4.9640  EFFECTIVE=2.09517
--- waiting for v3 step >= 5000 (02:49) ---
=== v3 @ step ~5000 : 8 official held-out users, A/B/C, K=128 ===
[A] mean over users: 54.18
[B] mean over users: 54.18
[C] mean over users: 55.22
[A] gap closed vs personalization ceiling: 2.7%
[B] gap closed vs personalization ceiling: 2.7%
[C] gap closed vs personalization ceiling: 0.4%
--- waiting for v3 step >= 9000 (03:07) ---
=== v3 @ step ~9000 : 8 official held-out users, A/B/C, K=128 ===
[A] mean over users: 54.12
[B] mean over users: 54.12
=== /tmp/v3_snap.pt  (step 9000) ===
    cross_pre    |tanh(g)|=0.07668  ||W||= 13.7224  EFFECTIVE=1.05230
    cross_post   |tanh(g)|=0.00869  ||W||= 10.9762  EFFECTIVE=0.09533
--- waiting for v3 step >= 12000 (03:33) ---
=== v3 @ step ~12000 : 8 official held-out users, A/B/C, K=128 ===
```

### 270_v3_fullwin_train.log
```
=== deploy full-window fix (support windows must survive TDS 124-frame shrink) ===
=== smoke re-check ===
Traceback (most recent call last):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 123, in <module>
    sys.exit(main())
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 47, in main
    tokens, pooled = model.encode_context(
  File "/data2/chenyuxiang/code/myoicl/myoicl/model.py", line 238, in encode_context
    tokens, pooled = self.ctx_encoder(feats, logp, flens)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/code/myoicl/myoicl/ctx_frame.py", line 113, in forward
    tok = tok[valid]                                      # (M0, d_ctx)
IndexError: The shape of the mask [96] at index 0 does not match the shape of the indexed tensor [192, 32] at index 0
SMOKE FAILED
```

### 280_v3_padfix_train.log
```
[B] mean over users: 54.18
[C] mean over users: 55.22
[A] gap closed vs personalization ceiling: 2.7%
[B] gap closed vs personalization ceiling: 2.7%
[C] gap closed vs personalization ceiling: 0.4%
=== /tmp/v3_snap.pt  (step 5000) ===
    cross_pre    |tanh(g)|=0.09691  ||W||= 11.4994  EFFECTIVE=1.11445
    cross_post   |tanh(g)|=0.01985  ||W||=  9.1071  EFFECTIVE=0.18075
--- waiting for v3 step >= 9000 (03:07) ---
=== v3 @ step ~9000 : 8 official held-out users, A/B/C ===
[A] mean over users: 54.12
[B] mean over users: 54.12
[C] mean over users: 55.88
[A] gap closed vs personalization ceiling: 2.9%
[B] gap closed vs personalization ceiling: 2.9%
[C] gap closed vs personalization ceiling: -1.1%
--- waiting for v3 step >= 12000 (03:33) ---
=== v3 @ step ~12000 : 8 official held-out users, A/B/C ===
[A] mean over users: 54.16
[B] mean over users: 54.16
[C] mean over users: 55.15
[A] gap closed vs personalization ceiling: 2.8%
[B] gap closed vs personalization ceiling: 2.8%
[C] gap closed vs personalization ceiling: 0.5%
=== v3 curve complete ===
```

### 290_v3_cheavy.log
```
=== v3 hedge: context-heavy + longer (GPU1) ===
[model] v1 | 6.01M params total (published backbone 5.29M + ICL module 0.71M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.71M @ lr 2.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 35 context tensors keep their initialization
[watchdog] armed
step 100/20000 | loss 1.4790 | lr 3.79e-06 | 2.89 it/s
step 200/20000 | loss 1.3944 | lr 7.54e-06 | 2.79 it/s
step 300/20000 | loss 1.5722 | lr 1.13e-05 | 2.25 it/s
step 400/20000 | loss 1.4285 | lr 1.50e-05 | 2.48 it/s
step 100/20000 | loss 1.4790 | lr 3.79e-06 | 2.89 it/s
step 200/20000 | loss 1.3944 | lr 7.54e-06 | 2.79 it/s
step 300/20000 | loss 1.5722 | lr 1.13e-05 | 2.25 it/s
step 400/20000 | loss 1.4285 | lr 1.50e-05 | 2.48 it/s
--- wait cheavy step >= 6000 (03:07) ---
=== v3-cheavy @ step ~6000 : 8 users A/B/C ===
[A] mean over users: 53.53
[B] mean over users: 53.53
[C] mean over users: 54.86
    cross_pre    |tanh(g)|=0.00042  ||W||= 26.4393  EFFECTIVE=0.01104
    cross_post   |tanh(g)|=0.00642  ||W||= 20.5838  EFFECTIVE=0.13212
--- wait cheavy step >= 12000 (03:44) ---
```

### 300_v3_frozen.log
```
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.71M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 35 context tensors keep their initialization
[freeze] backbone 5.29M frozen | context modules 0.71M trainable (11.9% of total)
[watchdog] armed
step 100/12000 | loss 1.7924 | lr 3.07e-06 | 2.13 it/s
step 200/12000 | loss 1.6190 | lr 6.30e-06 | 2.71 it/s
step 300/12000 | loss 1.3745 | lr 9.45e-06 | 2.43 it/s
step 100/12000 | loss 1.7924 | lr 3.07e-06 | 2.13 it/s
step 200/12000 | loss 1.6190 | lr 6.30e-06 | 2.71 it/s
step 300/12000 | loss 1.3745 | lr 9.45e-06 | 2.43 it/s
--- wait frozen step >= 6000 (03:07) ---
=== v3-frozen @ step ~6000 : 8 users A/B/C ===
[A] mean over users: 55.40
[B] mean over users: 55.40
[C] mean over users: 55.56
    cross_pre    |tanh(g)|=0.20556  ||W||= 12.7450  EFFECTIVE=2.61982
    cross_post   |tanh(g)|=0.32007  ||W||=  7.5508  EFFECTIVE=2.41680
--- wait frozen step >= 12000 (03:45) ---
=== v3-frozen @ step ~12000 : 8 users A/B/C ===
[A] mean over users: 55.40
[B] mean over users: 55.40
[C] mean over users: 55.51
    cross_pre    |tanh(g)|=0.12490  ||W||= 14.6937  EFFECTIVE=1.83529
    cross_post   |tanh(g)|=0.24797  ||W||=  8.4160  EFFECTIVE=2.08692
=== v3-frozen complete ===
```

### 310_v31_kvsplit.log
```
[smoke] built 50 support tokens from 6 windows
[smoke] init max|mode C - mode A| = 0.00e+00 (want ~0: identity)
[smoke] init grad o_proj=3.354e+01 (want >0: path can open)
[smoke] with o_proj opened, mean|mode C - mode A| = 3.6827e-02 (want >0: context now changes output)
[smoke] grad to frame context encoder now = 1.095e+01
[smoke] full-length support -> 96 tokens (>= 50 masked)
[smoke v3] ALL PASS

===== smoke v3 (kv_split=True) =====
[smoke] built 50 support tokens from 6 windows
[smoke] init max|mode C - mode A| = 0.00e+00 (want ~0: identity)
[smoke] init grad o_proj=4.382e+01 (want >0: path can open)
[smoke] with o_proj opened, mean|mode C - mode A| = 3.0983e-02 (want >0: context now changes output)
[smoke] grad to frame context encoder now = 4.497e+00
Traceback (most recent call last):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 129, in <module>
    rc = main(kv_split=False) or main(kv_split=True)
  File "/data2/chenyuxiang/code/myoicl/myoicl/smoke_v3.py", line 121, in main
    print(f"[smoke] full-length support -> {tok_full.shape[1]} tokens "
AttributeError: 'tuple' object has no attribute 'shape'
SMOKE FAILED -- not launching
```

### 320_v31_relaunch.log
```
=== wait for a free GPU (v3-main on GPU2 to finish) ===
v3-main at step 12000
=== launch v3.1 on GPU2 ===
[model] v1 | 5.94M params total (published backbone 5.29M + ICL module 0.65M) | device=cuda | phase=icl
[data] train sessions=837 val sessions=192
[data] episodic users=86 train + 10 meta-val (held out from module training)
[optim] backbone 5.29M @ lr 3.0e-05 | context 0.65M @ lr 1.0e-03 | 2 params exempt from weight decay
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 31 context tensors keep their initialization
[watchdog] armed
step 100/12000 | loss 1.5150 | lr 3.79e-06 | 2.43 it/s
step 200/12000 | loss 1.4696 | lr 7.54e-06 | 2.81 it/s
step 300/12000 | loss 1.4570 | lr 1.13e-05 | 2.67 it/s
step 400/12000 | loss 1.3018 | lr 1.50e-05 | 3.44 it/s
step 100/12000 | loss 1.5150 | lr 3.79e-06 | 2.43 it/s
step 200/12000 | loss 1.4696 | lr 7.54e-06 | 2.81 it/s
step 300/12000 | loss 1.4570 | lr 1.13e-05 | 2.67 it/s
step 400/12000 | loss 1.3018 | lr 1.50e-05 | 3.44 it/s
--- wait v3.1 step >= 2000 (04:09) ---
=== v3.1 @ step ~2000 : 8 users A/B/C ===
[A] mean over users: 54.12
[B] mean over users: 54.12
[C] mean over users: 54.23
    cross_pre    |tanh(g)|=0.58110  ||W||=  7.3030  EFFECTIVE=4.24381
    cross_post   |tanh(g)|=0.51757  ||W||=  5.7567  EFFECTIVE=2.97949
--- wait v3.1 step >= 5000 (04:22) ---
```

### 330_v32_filmonly.log
```
[watchdog] armed
step 100/12000 | loss 1.6424 | lr 3.79e-06 | 2.62 it/s
step 200/12000 | loss 1.2428 | lr 7.54e-06 | 3.53 it/s
step 300/12000 | loss 1.3811 | lr 1.13e-05 | 3.35 it/s
step 400/12000 | loss 1.3833 | lr 1.50e-05 | 2.74 it/s
step 100/12000 | loss 1.6424 | lr 3.79e-06 | 2.62 it/s
step 200/12000 | loss 1.2428 | lr 7.54e-06 | 3.53 it/s
step 300/12000 | loss 1.3811 | lr 1.13e-05 | 3.35 it/s
step 400/12000 | loss 1.3833 | lr 1.50e-05 | 2.74 it/s
--- wait v3.2 step >= 2000 (04:26) ---
=== v3.2 @ step ~2000 : 8 users A/B/C ===
[A] mean over users: 53.99
[B] mean over users: 53.99
[C] mean over users: 54.52
--- wait v3.2 step >= 6000 (04:38) ---
=== v3.2 @ step ~6000 : 8 users A/B/C ===
[A] mean over users: 54.42
[B] mean over users: 54.42
[C] mean over users: 55.70
--- wait v3.2 step >= 12000 (05:04) ---
=== v3.2 @ step ~12000 : 8 users A/B/C ===
[A] mean over users: 55.77
[B] mean over users: 55.77
[C] mean over users: 57.02
=== v3.2 complete ===
```

### 340_ctxsource_diag.log
```
=== waiting for v3.1 checkpoint step >= 9000 ===
v3.1 checkpoint step 9000
=== v3.1 mode C, --ctx-source cross (8 users, K=12) ===
[A] mean over users: 54.19
[C] mean over users: 55.14
=== v3.1 mode C, --ctx-source prefix (8 users, K=12) ===
```

### 350_samesession_diag.log
```
=== kill the stuck/flawed prefix eval (ctx-source only changes unused ctx_raw for v3) ===
=== deploy correct same-session diagnostic (--support-from-test) ===

############ myoicl_v31_kvsplit (step 12000) ############
--- CROSS-session (labelled support from OTHER sessions; the default all runs used) ---
[A] mean over users: 54.75
[C] mean over users: 55.26
--- SAME-session (labelled support from the DECODED session itself) ---
[A] mean over users: 54.75
[C] mean over users: 55.13

############ myoicl_v32_filmonly (step 12000) ############
--- CROSS-session (labelled support from OTHER sessions; the default all runs used) ---
[A] mean over users: 55.77
[C] mean over users: 57.03
--- SAME-session (labelled support from the DECODED session itself) ---
[A] mean over users: 55.77
[C] mean over users: 57.46

=== READ ===
If SAME-session mode-C < mode-A (gain positive) while CROSS-session hurts,
the universal negative is cross-session electrode staleness, not the method:
in-context calibration works when calibration and use are the same session.
=== same-session diagnostic complete ===
```
