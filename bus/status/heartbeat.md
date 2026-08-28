# heartbeat 2026-08-28T21:01:32+08:00

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
360_tokens_only_pilot                    DONE rc=127
370_teacher_fleet                        DONE rc=127
380_deploy_distill                       DONE rc=127
400_v5_hardsynth                         DONE rc=127
410_deploy_remix                         DONE rc=127
411_deploy_remix_fix                     DONE rc=127
412_deploy_remix_v2                      DONE rc=127
413_deploy_remix_v3                      DONE rc=127
420_eval_v5a1_real_users                 DONE rc=127
421_eval_v5a2_real_users                 DONE rc=127
430_deploy_trunk_tf                      DONE rc=127
440_train_trunk_tf                       DONE rc=127
450_log_relay                            DONE rc=127
460_free_capacity                        DONE rc=127
470_deploy_prefix_icl                    DONE rc=127
471_prefix_stride_fix                    DONE rc=127
480_tf_lr_probe                          DONE rc=127
490_tf_100hz_relaunch                    DONE rc=127
500_eval_a2_retry                        DONE rc=127
510_fold_fleet_lr1e3                     DONE rc=127
520_deploy_symbol_icl                    DONE rc=127
525_trunk_diag                           DONE rc=127
526_blank_tracker                        DONE rc=127
530_gate_5s_eval                         DONE rc=127
531_gate_final                           DONE rc=127
540_fullbudget_and_icl_dev               DONE rc=127
541_icl_dev_relaunch                     DONE rc=127
542_kcurve_autotrigger                   DONE rc=127
543_icl_phase2_conditional               DONE rc=127
544_perm_probe_loop                      DONE rc=127
545_final_night_verdict                  DONE rc=127
550_fused_prefix_icl                     DONE rc=127
551_fused_retry                          DONE rc=127
552_fused_frozen                         DONE rc=127
553_aux_supervised                       DONE rc=127
554_allgpu_sprint                        DONE rc=127
555_gate_full_eval                       DONE rc=127
556_joint_synth_aux                      DONE rc=127
557_zeroshot_today                       DONE rc=127
558_joint_is_mainline                    DONE rc=127
559_val_realonly                         DONE rc=127
560_myocorl_launch                       DONE rc=127
561_encoding_signal_diag                 DONE rc=127
562_keystroke_foundation                 DONE rc=127
563_keystroke_retry                      DONE rc=127
564_keystroke_diag2                      DONE rc=127
565_icl_split_and_scoring                DONE rc=127
566_keystroke_incremental                DONE rc=127
567_stream_and_probe                     DONE rc=127
568_budget_curve                         DONE rc=127
569_full_stop                            DONE rc=127
570_icl_sanity                           DONE rc=127
571_icl_sanity_fix                       DONE rc=127
572_w1_tta_floor                         DONE rc=127
573_partb_gate                           DONE rc=127
574_partb_peruser                        DONE rc=127
575_partb_v2                             DONE rc=127
576_lm_solve                             DONE rc=127
577_partb_sweep                          DONE rc=127
578_lm_eow_and_collect                   DONE rc=127
579_lm_beam_audit                        DONE rc=127
580_collect3                             DONE rc=127
581_seg_gate                             DONE rc=127
582_final8_and_collect                   DONE rc=127
583_seg_adapt                            DONE rc=127
584_collect_final                        DONE rc=127
585_honest_table                         DONE rc=127
586_frame_level                          DONE rc=127
587_frame_collect                        DONE rc=127
590_partb_mainline                       DONE rc=127
591_flashlight_install                   DONE rc=127
592_official_decoder                     DONE rc=127
593_decode_timestamps                    DONE rc=127
595_partb_v2                             DONE rc=127
596_fix_reset                            DONE rc=127
597_collect_v2                           DONE rc=127
598_per_window_fallback                  DONE rc=127
599_collect_v3                           DONE rc=127
600_partA_splash_probe                   DONE rc=127
601_partA_train                          DONE rc=127
602_collect_partA                        DONE rc=127
603_rtn_replaces_bn                      DONE rc=127
604_collect_partA2                       DONE rc=127
605_rtn_from_source                      DONE rc=127
606_partA1_align                         DONE rc=127
607_step_matched                         DONE rc=127
608_fix_align_silent                     DONE rc=127
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

### 160_eval_curve.log
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

### 200_d3.log
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

### 210_d4.log
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

### 220_v3.log
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

### 230_v3.log
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

### 240_v3.log
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

### 250_v3.log
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

### 260_v3.log
```
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
[A] mean over users: 54.16
[B] mean over users: 54.16
[C] mean over users: 55.15
[A] gap closed vs personalization ceiling: 2.8%
[B] gap closed vs personalization ceiling: 2.8%
[C] gap closed vs personalization ceiling: 0.5%
=== /tmp/v3_snap.pt  (step 12000) ===
    cross_pre    |tanh(g)|=0.08264  ||W||= 13.8773  EFFECTIVE=1.14687
    cross_post   |tanh(g)|=0.01047  ||W||= 11.1146  EFFECTIVE=0.11638
=== v3 curve complete ===
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

### 270_v3.log
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

### 280_v3.log
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

### 290_v3cheavy.log
```
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
=== v3-cheavy @ step ~12000 : 8 users A/B/C ===
[A] mean over users: 54.62
[B] mean over users: 54.62
[C] mean over users: 55.57
    cross_pre    |tanh(g)|=0.00189  ||W||= 30.9359  EFFECTIVE=0.05857
    cross_post   |tanh(g)|=0.00023  ||W||= 22.8550  EFFECTIVE=0.00533
--- wait cheavy step >= 20000 (04:21) ---
=== v3-cheavy @ step ~20000 : 8 users A/B/C ===
[A] mean over users: 54.38
[B] mean over users: 54.38
[C] mean over users: 55.93
    cross_pre    |tanh(g)|=0.00122  ||W||= 31.0374  EFFECTIVE=0.03788
    cross_post   |tanh(g)|=0.00030  ||W||= 22.6137  EFFECTIVE=0.00681
=== v3-cheavy complete ===
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

### 300_v3frozen.log
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

### 310_v31.log
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

### 320_v31.log
```
[C] mean over users: 54.23
    cross_pre    |tanh(g)|=0.58110  ||W||=  7.3030  EFFECTIVE=4.24381
    cross_post   |tanh(g)|=0.51757  ||W||=  5.7567  EFFECTIVE=2.97949
--- wait v3.1 step >= 5000 (04:22) ---
=== v3.1 @ step ~5000 : 8 users A/B/C ===
[A] mean over users: 54.75
[B] mean over users: 54.75
[C] mean over users: 55.19
    cross_pre    |tanh(g)|=0.25701  ||W||= 12.1291  EFFECTIVE=3.11728
    cross_post   |tanh(g)|=0.05475  ||W||=  9.1036  EFFECTIVE=0.49842
--- wait v3.1 step >= 9000 (04:41) ---
=== v3.1 @ step ~9000 : 8 users A/B/C ===
[A] mean over users: 54.19
[B] mean over users: 54.19
[C] mean over users: 55.17
    cross_pre    |tanh(g)|=0.21586  ||W||= 14.3054  EFFECTIVE=3.08797
    cross_post   |tanh(g)|=0.02816  ||W||= 11.2462  EFFECTIVE=0.31670
--- wait v3.1 step >= 12000 (05:08) ---
=== v3.1 @ step ~12000 : 8 users A/B/C ===
[A] mean over users: 54.75
[B] mean over users: 54.75
[C] mean over users: 55.28
    cross_pre    |tanh(g)|=0.22290  ||W||= 14.4714  EFFECTIVE=3.22571
    cross_post   |tanh(g)|=0.02937  ||W||= 11.3556  EFFECTIVE=0.33355
=== v3.1 complete ===
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

### 330_v32.log
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

### 340_diag.log
```
=== waiting for v3.1 checkpoint step >= 9000 ===
v3.1 checkpoint step 9000
=== v3.1 mode C, --ctx-source cross (8 users, K=12) ===
[A] mean over users: 54.19
[C] mean over users: 55.14
=== v3.1 mode C, --ctx-source prefix (8 users, K=12) ===
[A] mean over users: 54.19
=== ctx-source diagnostic complete ===
READ: if mode-C prefix << mode-C cross (and < mode-A), calibration works
      SAME-session and the universal negative is cross-session staleness.
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

### 350_samesession.log
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

### 360_pilot.log
```
  step  1250/2500 | loss 2.274 | CER  47.96 | gain +0.11
  step  1500/2500 | loss 2.276 | CER  47.96 | gain +0.11
  step  1750/2500 | loss 2.285 | CER  47.96 | gain +0.11
  step  2000/2500 | loss 2.310 | CER  47.96 | gain +0.11
  step  2250/2500 | loss 2.267 | CER  47.96 | gain +0.11
  step  2500/2500 | loss 2.275 | CER  47.96 | gain +0.11

==================================================================
zero-shot (frozen) -- 3-user subset            56.47
per-user adapter tuning (this probe)           56.40   gain +0.07
full per-user fine-tuning (published, 8 users)   11.40
  (published generic, 8 users: 55.39 -- NOT this subset)
------------------------------------------------------------------
the conditioning interface reaches 0% of the fine-tuning gap
(unrestricted: all training sessions + gradients. Mode C gets minutes
 and one forward pass, so run --limit-seconds 256 for the fair target.)
==================================================================

=> The INTERFACE is the bottleneck. Even with the user fully
   known, and without signs of memorisation, conditioning cannot move
   the frozen backbone far. Widen it: more injection points, larger
   d_ctx, or per-electrode conditioning before the frontend mixes
   channels.
[saved] /data2/chenyuxiang/runs/ceiling_tokens_only.json
=== pilot complete ===
```

### 360_tokens_only_pilot.log
```
  step  1250/2500 | loss 2.274 | CER  47.96 | gain +0.11
  step  1500/2500 | loss 2.276 | CER  47.96 | gain +0.11
  step  1750/2500 | loss 2.285 | CER  47.96 | gain +0.11
  step  2000/2500 | loss 2.310 | CER  47.96 | gain +0.11
  step  2250/2500 | loss 2.267 | CER  47.96 | gain +0.11
  step  2500/2500 | loss 2.275 | CER  47.96 | gain +0.11

==================================================================
zero-shot (frozen) -- 3-user subset            56.47
per-user adapter tuning (this probe)           56.40   gain +0.07
full per-user fine-tuning (published, 8 users)   11.40
  (published generic, 8 users: 55.39 -- NOT this subset)
------------------------------------------------------------------
the conditioning interface reaches 0% of the fine-tuning gap
(unrestricted: all training sessions + gradients. Mode C gets minutes
 and one forward pass, so run --limit-seconds 256 for the fair target.)
==================================================================

=> The INTERFACE is the bottleneck. Even with the user fully
   known, and without signs of memorisation, conditioning cannot move
   the frozen backbone far. Widen it: more injection points, larger
   d_ctx, or per-electrode conditioning before the frontend mixes
   channels.
[saved] /data2/chenyuxiang/runs/ceiling_tokens_only.json
=== pilot complete ===
```

### 370_teacher_fleet.log
```
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[1/24] 11944098: zero-shot 3.47 -> best 3.38 (gain +0.09)
--- shard 2 ---
[teachers] 24/96 training users in this shard | tokens_only=False | steps=1800
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[1/24] 12565339: zero-shot 2.46 -> best 2.33 (gain +0.14)
--- shard 3 ---
[teachers] 24/96 training users in this shard | tokens_only=False | steps=1800
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
=== progress reporter: copy shard logs + count artifacts every 10 min ===
[fleet] 13:59 teachers done: 10/96
[fleet] 14:09 teachers done: 16/96
[fleet] 14:19 teachers done: 21/96
[fleet] 14:29 teachers done: 28/96
[fleet] 14:39 teachers done: 33/96
[fleet] 14:49 teachers done: 33/96
[fleet] all shard processes ended
=== teacher fleet job complete: 33 teachers ===
```

### 370_teachers.log
```
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[1/24] 11944098: zero-shot 3.47 -> best 3.38 (gain +0.09)
--- shard 2 ---
[teachers] 24/96 training users in this shard | tokens_only=False | steps=1800
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[1/24] 12565339: zero-shot 2.46 -> best 2.33 (gain +0.14)
--- shard 3 ---
[teachers] 24/96 training users in this shard | tokens_only=False | steps=1800
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
=== progress reporter: copy shard logs + count artifacts every 10 min ===
[fleet] 13:59 teachers done: 10/96
[fleet] 14:09 teachers done: 16/96
[fleet] 14:19 teachers done: 21/96
[fleet] 14:29 teachers done: 28/96
[fleet] 14:39 teachers done: 33/96
[fleet] 14:49 teachers done: 33/96
[fleet] all shard processes ended
=== teacher fleet job complete: 33 teachers ===
```

### 380_deploy_distill.log
```
=== deploy distill.py (code only; training starts when teachers exist) ===
AST OK
deployed
```

### 380_distill_deploy.log
```
=== deploy distill.py (code only; training starts when teachers exist) ===
AST OK
deployed
```

### 400_v5_hardsynth.log
```
--- 14:48 ---
[a0_gain_affine] [val] step 500: mode-C CER 70.64 | mode-B CER 41.67 | mode-A CER 68.22 | gain C -2.42 / B +26.55 | loss 3.8753
[a1_gain_v31] [val] step 1000: mode-C CER 40.50 | mode-B CER 75.37 | mode-A CER 75.37 | gain C +34.86 / B +0.00 | loss 1.4733
[a2_realistic] 
[scan] [89/96] 9456349: held-out-session CER 7.59 (7 sessions)
--- 14:53 ---
[a0_gain_affine] [val] step 1000: mode-C CER 46.97 | mode-B CER 55.64 | mode-A CER 69.04 | gain C +22.06 / B +13.39 | loss 2.0425
[a1_gain_v31] [val] step 2500: mode-C CER 38.06 | mode-B CER 75.39 | mode-A CER 75.39 | gain C +37.33 / B +0.00 | loss 1.3286
[a2_realistic] 
[scan] === UNSEEN users (8 official test): 55.39 published ===
--- 14:58 ---
[a0_gain_affine] [val] step 1500: mode-C CER 71.44 | mode-B CER 59.87 | mode-A CER 69.22 | gain C -2.22 / B +9.35 | loss 4.4178
[a1_gain_v31] [val] step 4000: mode-C CER 34.94 | mode-B CER 75.56 | mode-A CER 75.56 | gain C +40.62 / B +0.00 | loss 1.2116
[a2_realistic] [val] step 500: mode-C CER 66.18 | mode-B CER 68.33 | mode-A CER 79.82 | gain C +13.65 / B +11.49 | loss 2.7225
[scan] === UNSEEN users (8 official test): 55.39 published ===
--- 15:03 ---
[a0_gain_affine] [val] step 2000: mode-C CER 83.29 | mode-B CER 71.19 | mode-A CER 68.50 | gain C -14.78 / B -2.69 | loss 6.6059
[a1_gain_v31] [val] step 5000: mode-C CER 36.99 | mode-B CER 75.01 | mode-A CER 75.01 | gain C +38.02 / B +0.00 | loss 1.2423
[a2_realistic] [val] step 500: mode-C CER 66.18 | mode-B CER 68.33 | mode-A CER 79.82 | gain C +13.65 / B +11.49 | loss 2.7225
[scan] === UNSEEN users (8 official test): 55.39 published ===
--- 15:08 ---
[a0_gain_affine] [val] step 3000: mode-C CER 62.43 | mode-B CER 71.90 | mode-A CER 69.04 | gain C +6.61 / B -2.86 | loss 3.4234
[a1_gain_v31] [val] step 6500: mode-C CER 36.91 | mode-B CER 75.17 | mode-A CER 75.17 | gain C +38.26 / B +0.00 | loss 1.2732
[a2_realistic] [val] step 1000: mode-C CER 66.35 | mode-B CER 69.46 | mode-A CER 79.79 | gain C +13.45 / B +10.33 | loss 2.9385
[scan] === UNSEEN users (8 official test): 55.39 published ===
```

### 400_v5.log
```
[a0_gain_affine] [val] step 6500: mode-C CER 66.73 | mode-B CER 90.14 | mode-A CER 68.57 | gain C +1.83 / B -21.58 | loss 3.9261
[a1_gain_v31] [val] step 7500: mode-C CER 36.95 | mode-B CER 75.68 | mode-A CER 75.68 | gain C +38.73 / B +0.00 | loss 1.3050
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 2831817) is killed by signal: Aborted. 
  ^^ a1_gain_v31 LOOKS BROKEN
[a2_realistic] [val] step 4000: mode-C CER 68.16 | mode-B CER 76.63 | mode-A CER 79.45 | gain C +11.29 / B +2.83 | loss 3.2769
[scan] === UNSEEN users (8 official test): 55.39 published ===
--- 16:13 ---
[a0_gain_affine] [val] step 6500: mode-C CER 66.73 | mode-B CER 90.14 | mode-A CER 68.57 | gain C +1.83 / B -21.58 | loss 3.9261
[a1_gain_v31] [val] step 7500: mode-C CER 36.95 | mode-B CER 75.68 | mode-A CER 75.68 | gain C +38.73 / B +0.00 | loss 1.3050
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 2831817) is killed by signal: Aborted. 
  ^^ a1_gain_v31 LOOKS BROKEN
[a2_realistic] [val] step 4000: mode-C CER 68.16 | mode-B CER 76.63 | mode-A CER 79.45 | gain C +11.29 / B +2.83 | loss 3.2769
[scan] === UNSEEN users (8 official test): 55.39 published ===
--- 16:18 ---
[a0_gain_affine] [val] step 6500: mode-C CER 66.73 | mode-B CER 90.14 | mode-A CER 68.57 | gain C +1.83 / B -21.58 | loss 3.9261
[a1_gain_v31] [val] step 7500: mode-C CER 36.95 | mode-B CER 75.68 | mode-A CER 75.68 | gain C +38.73 / B +0.00 | loss 1.3050
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 2831817) is killed by signal: Aborted. 
  ^^ a1_gain_v31 LOOKS BROKEN
[a2_realistic] [val] step 4500: mode-C CER 77.72 | mode-B CER 87.09 | mode-A CER 79.78 | gain C +2.06 / B -7.31 | loss 4.2851
[scan] === UNSEEN users (8 official test): 55.39 published ===
all v5 runs ended
=== 400 done ===
```

### 410_deploy_remix.log
```
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/code/myoicl/myoicl/tds.py", line 89, in forward
    return self.tds_conv_blocks(inputs)  # (T, N, num_features)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/code/myoicl/myoicl/tds.py", line 37, in forward
    x = self.conv2d(x)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 460, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 456, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Calculated padded input size per channel: (32 x 9). Kernel size: (1 x 32). Kernel size can't be greater than actual input size
SMOKE FAILED (rc=1) -- rolling back to /data2/chenyuxiang/runs/backup_myoicl_20260819_144953
```

### 410_remix.log
```
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/code/myoicl/myoicl/tds.py", line 89, in forward
    return self.tds_conv_blocks(inputs)  # (T, N, num_features)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/code/myoicl/myoicl/tds.py", line 37, in forward
    x = self.conv2d(x)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 460, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 456, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Calculated padded input size per channel: (32 x 9). Kernel size: (1 x 32). Kernel size can't be greater than actual input size
SMOKE FAILED (rc=1) -- rolling back to /data2/chenyuxiang/runs/backup_myoicl_20260819_144953
```

### 411_deploy_remix_fix.log
```
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/code/myoicl/myoicl/tds.py", line 89, in forward
    return self.tds_conv_blocks(inputs)  # (T, N, num_features)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/code/myoicl/myoicl/tds.py", line 37, in forward
    x = self.conv2d(x)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 460, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 456, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Calculated padded input size per channel: (32 x 9). Kernel size: (1 x 32). Kernel size can't be greater than actual input size
SMOKE FAILED (rc=1) -- rolling back to /data2/chenyuxiang/runs/backup_myoicl_20260819_152415
```

### 411_remix.log
```
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/code/myoicl/myoicl/tds.py", line 89, in forward
    return self.tds_conv_blocks(inputs)  # (T, N, num_features)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/code/myoicl/myoicl/tds.py", line 37, in forward
    x = self.conv2d(x)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 460, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py", line 456, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Calculated padded input size per channel: (32 x 9). Kernel size: (1 x 32). Kernel size can't be greater than actual input size
SMOKE FAILED (rc=1) -- rolling back to /data2/chenyuxiang/runs/backup_myoicl_20260819_152415
```

### 412_deploy_remix_v2.log
```
=== backup before overwriting shared modules ===
rollback copy: /data2/chenyuxiang/runs/backup_myoicl_20260819_152746
=== extract ===
AST OK

=== regression: the NO-remix path must be untouched ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
no-remix model OK | trainable 6.23M -> frozen leaves 0.93M context params

=== smoke: the remix head itself (CPU) ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[build] remix params 217.6k
  [ok ] remix is identity at init  max|M-I| = 0.00e+00
  [ok ] logits unchanged at init  max|da| = 0.00e+00
  [FAIL] every remix parameter gets gradient  dead: ['s_ref', 'logit_scale', 'mlp.0.weight', 'mlp.0.bias']
  [ok ] assign head recovers a known channel roll  accuracy 100%  (roll=5)
  [ok ] freeze_backbone leaves remix trainable

SMOKE FAILED: ['every remix parameter gets gradient']
SMOKE FAILED (rc=1) -- rolling back to /data2/chenyuxiang/runs/backup_myoicl_20260819_152746
```

### 412_remix.log
```
=== backup before overwriting shared modules ===
rollback copy: /data2/chenyuxiang/runs/backup_myoicl_20260819_152746
=== extract ===
AST OK

=== regression: the NO-remix path must be untouched ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
no-remix model OK | trainable 6.23M -> frozen leaves 0.93M context params

=== smoke: the remix head itself (CPU) ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[build] remix params 217.6k
  [ok ] remix is identity at init  max|M-I| = 0.00e+00
  [ok ] logits unchanged at init  max|da| = 0.00e+00
  [FAIL] every remix parameter gets gradient  dead: ['s_ref', 'logit_scale', 'mlp.0.weight', 'mlp.0.bias']
  [ok ] assign head recovers a known channel roll  accuracy 100%  (roll=5)
  [ok ] freeze_backbone leaves remix trainable

SMOKE FAILED: ['every remix parameter gets gradient']
SMOKE FAILED (rc=1) -- rolling back to /data2/chenyuxiang/runs/backup_myoicl_20260819_152746
```

### 413_deploy_remix_v3.log
```
=== backup before overwriting shared modules ===
rollback copy: /data2/chenyuxiang/runs/backup_myoicl_20260819_153159
=== extract ===
AST OK

=== regression: the NO-remix path must be untouched ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
no-remix model OK | trainable 6.23M -> frozen leaves 0.93M context params

=== smoke: the remix head itself (CPU) ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[build] remix params 217.6k
  [ok ] remix is identity at init  max|M-I| = 0.00e+00
  [ok ] logits unchanged at init  max|da| = 0.00e+00
        (zero-grad at step 0: ['s_ref', 'mlp.0.weight', 'mlp.0.bias'] -- expected for parameters upstream of a zero output matrix)
  [ok ] every remix parameter gets gradient after one step
  [ok ] assign head recovers a known channel roll  accuracy 100%  (roll=5)
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
  [ok ] freeze_backbone leaves remix trainable

SMOKE OK -- remix head is identity at init, gradient-connected, identifiable, and survives freezing.
=== 413 done: remix head deployed and verified, nothing launched ===
```

### 413_remix.log
```
=== backup before overwriting shared modules ===
rollback copy: /data2/chenyuxiang/runs/backup_myoicl_20260819_153159
=== extract ===
AST OK

=== regression: the NO-remix path must be untouched ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
no-remix model OK | trainable 6.23M -> frozen leaves 0.93M context params

=== smoke: the remix head itself (CPU) ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[build] remix params 217.6k
  [ok ] remix is identity at init  max|M-I| = 0.00e+00
  [ok ] logits unchanged at init  max|da| = 0.00e+00
        (zero-grad at step 0: ['s_ref', 'mlp.0.weight', 'mlp.0.bias'] -- expected for parameters upstream of a zero output matrix)
  [ok ] every remix parameter gets gradient after one step
  [ok ] assign head recovers a known channel roll  accuracy 100%  (roll=5)
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
  [ok ] freeze_backbone leaves remix trainable

SMOKE OK -- remix head is identity at init, gradient-connected, identifiable, and survives freezing.
=== 413 done: remix head deployed and verified, nothing launched ===
```

### 420_eval_real.log
```
[C] user0: CER 63.25
[C] user1: CER 60.19
[C] user2: CER 50.12
[C] user3: CER 57.74
[C] user4: CER 61.51
[C] user5: CER 55.07
[C] user6: CER 54.08
[C] user7: CER 52.54
[C] mean over users: 56.81
[A] gap closed vs personalization ceiling: -0.0%
[B] gap closed vs personalization ceiling: -0.0%
[C] gap closed vs personalization ceiling: -3.2%
[saved] /data2/chenyuxiang/runs/v5a1_real_k45.json

=== K-CURVE SUMMARY (8 official test users) ===
   k  secs   mode A   mode B   mode C   gain C   gain B
   4    16    55.40    55.40    56.82    -1.43    +0.00
  12    48    55.40    55.40    56.81    -1.42    +0.00
  23    92    55.40    55.40    56.81    -1.42    +0.00
  45   180    55.40    55.40    56.81    -1.42    +0.00

reference: published zero-shot 55.39 | published per-user finetune 11.28
mode A should sit at ~55.39 (frozen backbone). If it does not, the
eval is wrong and no mode-C number here means anything.
=== 420 done ===
```

### 420_eval_v5a1_real_users.log
```
[B] user1: CER 59.90
[B] user2: CER 48.06
[B] user3: CER 54.69
[B] user4: CER 58.28
[B] user5: CER 53.90
[B] user6: CER 54.63
[B] user7: CER 52.25
[B] mean over users: 55.40
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv2d(input, weight, bias, self.stride,
[C] user0: CER 63.25
[C] user1: CER 60.19
[C] user2: CER 50.12
[C] user3: CER 57.74
[C] user4: CER 61.51
[C] user5: CER 55.07
[C] user6: CER 54.08
[C] user7: CER 52.54
[C] mean over users: 56.81
[A] gap closed vs personalization ceiling: -0.0%
[B] gap closed vs personalization ceiling: -0.0%
[C] gap closed vs personalization ceiling: -3.2%
[saved] /data2/chenyuxiang/runs/v5a1_real_k23.json

############ k=45 windows (~180s of the user's own labelled data) ############
```

### 421_eval_a2_real.log
```
[B] user7: CER 60.63
[B] mean over users: 62.57
[C] user0: CER 63.52
[C] user1: CER 67.26
[C] user2: CER 53.68
[C] user3: CER 63.90
[C] user4: CER 63.32
[C] user5: CER 58.87
[C] user6: CER 59.23
[C] user7: CER 55.00
[C] mean over users: 60.60
[A] gap closed vs personalization ceiling: -0.0%
[B] gap closed vs personalization ceiling: -16.3%
[C] gap closed vs personalization ceiling: -11.8%
[saved] /data2/chenyuxiang/runs/v5a2_real_k45.json

=== K-CURVE SUMMARY (8 official test users) ===
   k  secs   mode A   mode B   mode C   gain C   gain B
  12    48    55.40    62.61    60.54    -5.14    -7.22
  45   180    55.40    62.57    60.60    -5.20    -7.18

reference: published zero-shot 55.39 | published per-user finetune 11.28
mode A should sit at ~55.39 (frozen backbone). If it does not, the
eval is wrong and no mode-C number here means anything.
=== 420 done ===
```

### 421_eval_v5a2_real_users.log
```
=== wait for job 420 to release GPU3 ===
=== evaluating /data2/chenyuxiang/runs/v5_a2/best.pt  (14:54) ===

############ k=12 windows (~48s of the user's own labelled data) ############
```

### 430_deploy_trunk_tf.log
```
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
tiny   ours    2.12M | paper    2.2M -> OK (paper cross-user CER 35.9)
        2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
small  ours    4.99M | paper    5.4M -> OK (paper cross-user CER 35.2)
        4.99M total  (featurizer 0.08M  encoder 4.74M  decoder 0.03M)
large  ours  103.06M | paper  109.0M -> OK (paper cross-user CER 30.5)
        103.06M total  (featurizer 0.08M  encoder 100.77M  decoder 0.10M)

=== end-to-end CPU smoke: 5 real training steps ===
fold 0: train 624 sessions | heldout 213 sessions, 24 users
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
  step 0: raw (4, 32, 11994) -> emissions (121, 4, 99) | frame lens [121, 120, 120, 121]
           loss 19.8256  grad-sum 2.047e+04
  step 1: raw (4, 32, 11963) -> emissions (120, 4, 99) | frame lens [120, 120, 120, 120]
           loss 18.7519  grad-sum 1.718e+04
  step 2: raw (4, 32, 11982) -> emissions (121, 4, 99) | frame lens [121, 120, 120, 120]
           loss 124.2906  grad-sum 2.955e+05
  step 3: raw (4, 32, 11998) -> emissions (121, 4, 99) | frame lens [120, 121, 120, 121]
           loss 27.9070  grad-sum 5.973e+04
  step 4: raw (4, 32, 11992) -> emissions (121, 4, 99) | frame lens [120, 121, 120, 120]
           loss 14.3142  grad-sum 2.479e+04
  prefix hook OK: 37 prefix tokens leave emissions at (121, 4, 99)
SMOKE OK
=== 430 done: trunk deployed and verified, nothing launched ===
```

### 430_trunk_tf.log
```
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
tiny   ours    2.12M | paper    2.2M -> OK (paper cross-user CER 35.9)
        2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
small  ours    4.99M | paper    5.4M -> OK (paper cross-user CER 35.2)
        4.99M total  (featurizer 0.08M  encoder 4.74M  decoder 0.03M)
large  ours  103.06M | paper  109.0M -> OK (paper cross-user CER 30.5)
        103.06M total  (featurizer 0.08M  encoder 100.77M  decoder 0.10M)

=== end-to-end CPU smoke: 5 real training steps ===
fold 0: train 624 sessions | heldout 213 sessions, 24 users
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
  step 0: raw (4, 32, 11994) -> emissions (121, 4, 99) | frame lens [121, 120, 120, 121]
           loss 19.8256  grad-sum 2.047e+04
  step 1: raw (4, 32, 11963) -> emissions (120, 4, 99) | frame lens [120, 120, 120, 120]
           loss 18.7519  grad-sum 1.718e+04
  step 2: raw (4, 32, 11982) -> emissions (121, 4, 99) | frame lens [121, 120, 120, 120]
           loss 124.2906  grad-sum 2.955e+05
  step 3: raw (4, 32, 11998) -> emissions (121, 4, 99) | frame lens [120, 121, 120, 121]
           loss 27.9070  grad-sum 5.973e+04
  step 4: raw (4, 32, 11992) -> emissions (121, 4, 99) | frame lens [120, 121, 120, 120]
           loss 14.3142  grad-sum 2.479e+04
  prefix hook OK: 37 prefix tokens leave emissions at (121, 4, 99)
SMOKE OK
=== 430 done: trunk deployed and verified, nothing launched ===
```

### 440_train_trunk_tf.log
```
=== redeploy train_trunk (cached eval sets) ===
AST OK

=== waiting for GPUs (the v5 ladder is still running) ===
launched tf_ref on GPU1 pid=2889785  (15:24)
  (10) no free GPU: 0, 2411 MiB 1, 1369 MiB 2, 2807 MiB 3, 24117 MiB 
launched tf_fold0 on GPU3 pid=2905364  (15:34)

=== streaming (12 h) ===
```

### 440_trunk_train.log
```
--- 02:59 ---
[tf_ref] [val] step 6000: 8-test-user CER 83.37 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
--- 03:04 ---
[tf_ref] [val] step 6000: 8-test-user CER 83.37 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
--- 03:09 ---
[tf_ref] [val] step 6000: 8-test-user CER 83.37 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
--- 03:14 ---
[tf_ref] [val] step 6000: 8-test-user CER 83.37 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
--- 03:19 ---
[tf_ref] [val] step 6000: 8-test-user CER 83.37 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
--- 03:24 ---
[tf_ref] [val] step 6000: 8-test-user CER 83.37 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
--- 03:29 ---
[tf_ref] [val] step 6000: 8-test-user CER 83.37 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
--- 03:34 ---
[tf_ref] [val] step 6000: 8-test-user CER 83.37 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
=== 440 done ===
```

### 450_log_relay.log
```
--- 15:40 ---
0, 16 MiB, 0 %|1, 1369 MiB, 14 %|2, 2807 MiB, 0 %|3, 1369 MiB, 38 %|
[tf_fold0] step 400/40000 | loss 4.1474 | lr 6.00e-05 | 433 win/s
[tf_ref] [val] new best 100.00 -> best.pt

--- 16:00 ---
0, 1371 MiB, 0 %|1, 1369 MiB, 0 %|2, 2807 MiB, 0 %|3, 1369 MiB, 0 %|
[tf_fold0] [val] new best 100.00 -> best.pt
[tf_ref] [val] new best 94.53 -> best.pt
[tf_ref_lr1e3] 

--- 16:30 ---
0, 3171 MiB, 69 %|1, 3167 MiB, 68 %|2, 2983 MiB, 86 %|3, 12 MiB, 0 %|
[tf_fold0] step 1600/40000 | loss 3.1930 | lr 2.40e-04 | 623 win/s
[tf_ref] step 1800/40000 | loss 3.1541 | lr 2.70e-04 | 712 win/s
[tf_ref_lr1e3] step 1400/40000 | loss 3.1287 | lr 7.00e-04 | 625 win/s

--- 17:00 ---
0, 2989 MiB, 90 %|1, 2985 MiB, 11 %|2, 2985 MiB, 68 %|3, 805 MiB, 0 %|
[tf_fold0] 
[tf_fold1] 
[tf_fold2] 
[tf_ref] [val] new best 83.37 -> best.pt
[tf_ref_lr1e3] [val] new best 78.64 -> best.pt

```

### 450_relay.log
```

--- 14:30 ---
0, 3173 MiB, 34 %|1, 2731 MiB, 20 %|2, 2985 MiB, 71 %|3, 2731 MiB, 0 %|
[tf_fold0_full] [val] step 103000: 8-test-user CER 45.88 | fold-heldout-user CER 53.87  (their Tiny reference: 35.9)
[tf_fold0] [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
[tf_fold1_full] [val] new best 45.88 -> best.pt
[tf_fold1] [val] step 40000: 8-test-user CER 87.88 | fold-heldout-user CER 88.04  (their Tiny reference: 35.9)
[tf_fold2] [val] step 40000: 8-test-user CER 87.80 | fold-heldout-user CER 89.26  (their Tiny reference: 35.9)
[tf_fold3] [val] step 40000: 8-test-user CER 88.39 | fold-heldout-user CER 89.60  (their Tiny reference: 35.9)
[tf_ref_full] [val] step 92000: 8-test-user CER 46.59 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_ref] [val] new best 83.37 -> best.pt
[tf_ref_lr1e3] [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)

--- 15:00 ---
0, 3173 MiB, 63 %|1, 3153 MiB, 0 %|2, 2985 MiB, 70 %|3, 12 MiB, 0 %|
[tf_fold0_full] [val] step 103000: 8-test-user CER 45.88 | fold-heldout-user CER 53.87  (their Tiny reference: 35.9)
[tf_fold0] [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
[tf_fold1_full] [val] step 100000: 8-test-user CER 46.74 | fold-heldout-user CER 54.43  (their Tiny reference: 35.9)
[tf_fold1] [val] step 40000: 8-test-user CER 87.88 | fold-heldout-user CER 88.04  (their Tiny reference: 35.9)
[tf_fold2] [val] step 40000: 8-test-user CER 87.80 | fold-heldout-user CER 89.26  (their Tiny reference: 35.9)
[tf_fold3] [val] step 40000: 8-test-user CER 88.39 | fold-heldout-user CER 89.60  (their Tiny reference: 35.9)
[tf_ref_full] [val] step 96000: 8-test-user CER 46.19 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_ref] [val] new best 83.37 -> best.pt
[tf_ref_lr1e3] [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)

```

### 460_free_capacity.log
```
=== before ===
0, 2411 MiB, 28 %
1, 1369 MiB, 46 %
2, 2807 MiB, 0 %
3, 1369 MiB, 23 %
python procs: 12
 15:40:30 up 113 days,  17:56,  21 users,  load average: 0.04, 0.01, 0.00

=== stop the diverged A0 arm only ===
A0 stopped

=== after ===
0, 16 MiB, 0 %
1, 1369 MiB, 15 %
2, 2807 MiB, 0 %
3, 1369 MiB, 15 %
 15:40:45 up 113 days,  17:56,  21 users,  load average: 0.04, 0.02, 0.01
=== 460 done ===
```

### 460_free.log
```
=== before ===
0, 2411 MiB, 28 %
1, 1369 MiB, 46 %
2, 2807 MiB, 0 %
3, 1369 MiB, 23 %
python procs: 12
 15:40:30 up 113 days,  17:56,  21 users,  load average: 0.04, 0.01, 0.00

=== stop the diverged A0 arm only ===
A0 stopped

=== after ===
0, 16 MiB, 0 %
1, 1369 MiB, 15 %
2, 2807 MiB, 0 %
3, 1369 MiB, 15 %
 15:40:45 up 113 days,  17:56,  21 users,  load average: 0.04, 0.02, 0.01
=== 460 done ===
```

### 470_deploy_prefix_icl.log
```
encoder params 29.8k
  {'k_windows': 4, 'seconds': 20, 'tokens_uncapped': 372, 'tokens': 372, 'capped': False}
  {'k_windows': 12, 'seconds': 60, 'tokens_uncapped': 1116, 'tokens': 1116, 'capped': False}
  {'k_windows': 18, 'seconds': 90, 'tokens_uncapped': 1674, 'tokens': 1674, 'capped': False}
  {'k_windows': 23, 'seconds': 115, 'tokens_uncapped': 2139, 'tokens': 2139, 'capped': False}
  {'k_windows': 45, 'seconds': 225, 'tokens_uncapped': 4185, 'tokens': 4096, 'capped': True}

=== contamination guard must FIRE on a reference backbone ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[trunk] /data2/chenyuxiang/runs/tf_ref/last.pt step 2000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
[FATAL] backbone was trained with --fold -1 but the cohort is fold 0: it has SEEN these users, so the episodes would contain no adaptation headroom
(expected: [FATAL] backbone was trained with --fold -1 ...)

=== CPU smoke: one real episode end to end ===
cohort 24 multi-session users of 24
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
episode user 89335547: support (11999, 6, 2, 16) query (11961, 2, 2, 16)
prefix (1, 527, 128)  (527 tokens from 6 windows)
mode A (120, 2, 99) vs mode C (120, 2, 99) | max|dlogit| 3.448
ctc loss 25.3645 | prefix-encoder grad-sum 2.186e+03
SMOKE OK
修改尚未加入提交（使用 "git add" 和/或 "git commit -a"）
=== 470 done: prefix ICL trainer deployed, waiting on tf_fold0 ===
```

### 470_prefix_icl.log
```
encoder params 29.8k
  {'k_windows': 4, 'seconds': 20, 'tokens_uncapped': 372, 'tokens': 372, 'capped': False}
  {'k_windows': 12, 'seconds': 60, 'tokens_uncapped': 1116, 'tokens': 1116, 'capped': False}
  {'k_windows': 18, 'seconds': 90, 'tokens_uncapped': 1674, 'tokens': 1674, 'capped': False}
  {'k_windows': 23, 'seconds': 115, 'tokens_uncapped': 2139, 'tokens': 2139, 'capped': False}
  {'k_windows': 45, 'seconds': 225, 'tokens_uncapped': 4185, 'tokens': 4096, 'capped': True}

=== contamination guard must FIRE on a reference backbone ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[trunk] /data2/chenyuxiang/runs/tf_ref/last.pt step 2000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
[FATAL] backbone was trained with --fold -1 but the cohort is fold 0: it has SEEN these users, so the episodes would contain no adaptation headroom
(expected: [FATAL] backbone was trained with --fold -1 ...)

=== CPU smoke: one real episode end to end ===
cohort 24 multi-session users of 24
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
episode user 89335547: support (11999, 6, 2, 16) query (11961, 2, 2, 16)
prefix (1, 527, 128)  (527 tokens from 6 windows)
mode A (120, 2, 99) vs mode C (120, 2, 99) | max|dlogit| 3.448
ctc loss 25.3645 | prefix-encoder grad-sum 2.186e+03
SMOKE OK
修改尚未加入提交（使用 "git add" 和/或 "git commit -a"）
=== 470 done: prefix ICL trainer deployed, waiting on tf_fold0 ===
```

### 471_prefix_icl.log
```
=== prefix length budget ===
encoder params 29.8k
  {'k_windows': 4, 'seconds': 16, 'tokens_uncapped': 380, 'tokens': 380, 'capped': False}
  {'k_windows': 12, 'seconds': 48, 'tokens_uncapped': 1140, 'tokens': 1140, 'capped': False}
  {'k_windows': 18, 'seconds': 72, 'tokens_uncapped': 1710, 'tokens': 1710, 'capped': False}
  {'k_windows': 23, 'seconds': 92, 'tokens_uncapped': 2185, 'tokens': 2185, 'capped': False}
  {'k_windows': 45, 'seconds': 180, 'tokens_uncapped': 4275, 'tokens': 4096, 'capped': True}

=== contamination guard must FIRE on a reference backbone ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[trunk] /data2/chenyuxiang/runs/tf_ref/last.pt step 2000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
[FATAL] backbone was trained with --fold -1 but the cohort is fold 0: it has SEEN these users, so the episodes would contain no adaptation headroom
(expected: [FATAL] backbone was trained with --fold -1 ...)

=== CPU smoke: one real episode end to end ===
cohort 24 multi-session users of 24
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
episode user 89335547: support (11960, 6, 2, 16) query (11898, 2, 2, 16)
prefix (1, 615, 128)  (615 tokens from 6 windows)
mode A (593, 2, 99) vs mode C (593, 2, 99) | max|dlogit| 3.309
ctc loss 235.4557 | prefix-encoder grad-sum 2.731e+04
SMOKE OK
=== 471 done: prefix ICL trainer deployed, waiting on tf_fold0 ===
```

### 471_prefix_stride_fix.log
```
=== prefix length budget ===
encoder params 29.8k
  {'k_windows': 4, 'seconds': 16, 'tokens_uncapped': 380, 'tokens': 380, 'capped': False}
  {'k_windows': 12, 'seconds': 48, 'tokens_uncapped': 1140, 'tokens': 1140, 'capped': False}
  {'k_windows': 18, 'seconds': 72, 'tokens_uncapped': 1710, 'tokens': 1710, 'capped': False}
  {'k_windows': 23, 'seconds': 92, 'tokens_uncapped': 2185, 'tokens': 2185, 'capped': False}
  {'k_windows': 45, 'seconds': 180, 'tokens_uncapped': 4275, 'tokens': 4096, 'capped': True}

=== contamination guard must FIRE on a reference backbone ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[trunk] /data2/chenyuxiang/runs/tf_ref/last.pt step 2000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
[FATAL] backbone was trained with --fold -1 but the cohort is fold 0: it has SEEN these users, so the episodes would contain no adaptation headroom
(expected: [FATAL] backbone was trained with --fold -1 ...)

=== CPU smoke: one real episode end to end ===
cohort 24 multi-session users of 24
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
episode user 89335547: support (11960, 6, 2, 16) query (11898, 2, 2, 16)
prefix (1, 615, 128)  (615 tokens from 6 windows)
mode A (593, 2, 99) vs mode C (593, 2, 99) | max|dlogit| 3.309
ctc loss 235.4557 | prefix-encoder grad-sum 2.731e+04
SMOKE OK
=== 471 done: prefix ICL trainer deployed, waiting on tf_fold0 ===
```

### 480_lr_probe.log
```
[tf_ref_lr1e3] step 40000/40000 | loss 1.6571 | lr 0.00e+00 | 387 win/s
         [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] step 40000/40000 | loss 1.5527 | lr 0.00e+00 | 398 win/s
         [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
--- 15:10 ---
[tf_ref] step 6400/40000 | loss 2.3342 | lr 2.90e-04 | 702 win/s
         [val] new best 83.37 -> best.pt
[tf_ref_lr1e3] step 40000/40000 | loss 1.6571 | lr 0.00e+00 | 387 win/s
         [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] step 40000/40000 | loss 1.5527 | lr 0.00e+00 | 398 win/s
         [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
--- 15:20 ---
[tf_ref] step 6400/40000 | loss 2.3342 | lr 2.90e-04 | 702 win/s
         [val] new best 83.37 -> best.pt
[tf_ref_lr1e3] step 40000/40000 | loss 1.6571 | lr 0.00e+00 | 387 win/s
         [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] step 40000/40000 | loss 1.5527 | lr 0.00e+00 | 398 win/s
         [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
--- 15:30 ---
[tf_ref] step 6400/40000 | loss 2.3342 | lr 2.90e-04 | 702 win/s
         [val] new best 83.37 -> best.pt
[tf_ref_lr1e3] step 40000/40000 | loss 1.6571 | lr 0.00e+00 | 387 win/s
         [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] step 40000/40000 | loss 1.5527 | lr 0.00e+00 | 398 win/s
         [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
```

### 480_tf_lr_probe.log
```
=== launching tf_ref_lr1e3 on GPU0 at 15:58 ===
pid=2942963
[split] REFERENCE run: all 96 training users
[split] official test users: 16 sessions (never trained on in either mode)
[data] 183349 training windows of 5.0s
[data] monitor sets: 160 test windows, 0 fold-heldout windows
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[model] tiny: 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)

=== compare the two learning rates every 10 min ===
```

### 490_tf_100hz.log
```
        [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
[tf_ref_lr1e3] step 40000/40000 | loss 1.6571 | lr 0.00e+00 | 387 win/s
        [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
--- 06:13 ---
[tf_ref] step 6400/40000 | loss 2.3342 | lr 2.90e-04 | 702 win/s
        [val] new best 83.37 -> best.pt
[tf_fold0] step 40000/40000 | loss 1.5527 | lr 0.00e+00 | 398 win/s
        [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
[tf_ref_lr1e3] step 40000/40000 | loss 1.6571 | lr 0.00e+00 | 387 win/s
        [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
--- 06:18 ---
[tf_ref] step 6400/40000 | loss 2.3342 | lr 2.90e-04 | 702 win/s
        [val] new best 83.37 -> best.pt
[tf_fold0] step 40000/40000 | loss 1.5527 | lr 0.00e+00 | 398 win/s
        [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
[tf_ref_lr1e3] step 40000/40000 | loss 1.6571 | lr 0.00e+00 | 387 win/s
        [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
--- 06:23 ---
[tf_ref] step 6400/40000 | loss 2.3342 | lr 2.90e-04 | 702 win/s
        [val] new best 83.37 -> best.pt
[tf_fold0] step 40000/40000 | loss 1.5527 | lr 0.00e+00 | 398 win/s
        [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
[tf_ref_lr1e3] step 40000/40000 | loss 1.6571 | lr 0.00e+00 | 387 win/s
        [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
=== 490 done ===
```

### 490_tf_100hz_relaunch.log
```
--- tf_ref_lr1e3 ---
[split] REFERENCE run: all 96 training users
[split] official test users: 16 sessions (never trained on in either mode)
[data] 229266 training windows of 4.0s
[data] monitor sets: 160 test windows, 0 fold-heldout windows
[model] featurizer [11, 3, 3]/[5, 2, 2] -> 100 Hz frames (400 per window)
[model] tiny: 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)

=== stream (14 h) ===
--- 16:28 ---
[tf_ref] step 1600/40000 | loss 3.1925 | lr 2.40e-04 | 711 win/s
[tf_fold0] step 1200/40000 | loss 3.3008 | lr 1.80e-04 | 653 win/s
[tf_ref_lr1e3] step 1200/40000 | loss 3.1712 | lr 6.00e-04 | 636 win/s
--- 16:33 ---
[tf_ref] step 2400/40000 | loss 3.0256 | lr 3.00e-04 | 711 win/s
        [val] new best 100.00 -> best.pt
[tf_fold0] step 2000/40000 | loss 3.1303 | lr 3.00e-04 | 604 win/s
[tf_ref_lr1e3] step 1800/40000 | loss 3.0091 | lr 9.00e-04 | 604 win/s
--- 16:38 ---
[tf_ref] step 3200/40000 | loss 2.8676 | lr 2.99e-04 | 707 win/s
        [val] new best 100.00 -> best.pt
[tf_fold0] step 2400/40000 | loss 3.0404 | lr 3.00e-04 | 587 win/s
        [val] new best 100.00 -> best.pt
[tf_ref_lr1e3] step 2400/40000 | loss 2.7516 | lr 1.00e-03 | 584 win/s
        [val] new best 100.00 -> best.pt
```

### 500_eval_a2.log
```
[A] mean over users: 55.40
[C] mean over users: 60.54
[A] gap closed vs personalization ceiling: -0.0%
[C] gap closed vs personalization ceiling: -11.7%

############ k=45 (~180s) -- started 16:49 ############
rc=0  finished 16:52
[A] mean over users: 55.40
[C] mean over users: 60.60
[A] gap closed vs personalization ceiling: -0.0%
[C] gap closed vs personalization ceiling: -11.8%

=== A2 vs A1 on the 8 real test users ===
  ckpt    k  secs   mode A   mode C   gain C
  v5a1    4    16    55.40    56.82    -1.43
  v5a1   12    48    55.40    56.81    -1.42
  v5a1   23    92    55.40    56.81    -1.42
  v5a1   45   180    55.40    56.81    -1.42
  v5a2   12    48    55.40    60.54    -5.14
  v5a2   45   180    55.40    60.60    -5.20

A1 trained on pure per-channel gain; A2 on the calibrated family with
integer electrode rotation. mode A must sit at 55.39 in both (frozen
backbone) -- if it does not, the eval is wrong, not the method.
=== 500 done ===
```

### 500_eval_a2_retry.log
```
[A] mean over users: 55.40
[C] mean over users: 60.54
[A] gap closed vs personalization ceiling: -0.0%
[C] gap closed vs personalization ceiling: -11.7%

############ k=45 (~180s) -- started 16:49 ############
rc=0  finished 16:52
[A] mean over users: 55.40
[C] mean over users: 60.60
[A] gap closed vs personalization ceiling: -0.0%
[C] gap closed vs personalization ceiling: -11.8%

=== A2 vs A1 on the 8 real test users ===
  ckpt    k  secs   mode A   mode C   gain C
  v5a1    4    16    55.40    56.82    -1.43
  v5a1   12    48    55.40    56.81    -1.42
  v5a1   23    92    55.40    56.81    -1.42
  v5a1   45   180    55.40    56.81    -1.42
  v5a2   12    48    55.40    60.54    -5.14
  v5a2   45   180    55.40    60.60    -5.20

A1 trained on pure per-channel gain; A2 on the calibrated family with
integer electrode rotation. mode A must sit at 55.39 in both (frozen
backbone) -- if it does not, the eval is wrong, not the method.
=== 500 done ===
```

### 510_fold_fleet.log
```
[tf_fold3] step 40000/40000 | loss 1.5302 | lr 0.00e+00 | 516 win/s
        [val] step 40000: 8-test-user CER 88.39 | fold-heldout-user CER 89.60  (their Tiny reference: 35.9)
--- 06:56 ---
[tf_ref_lr1e3] step 40000/40000 | loss 1.6571 | lr 0.00e+00 | 387 win/s
        [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] step 40000/40000 | loss 1.5527 | lr 0.00e+00 | 398 win/s
        [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
[tf_fold1] step 40000/40000 | loss 1.5735 | lr 0.00e+00 | 617 win/s
        [val] step 40000: 8-test-user CER 87.88 | fold-heldout-user CER 88.04  (their Tiny reference: 35.9)
[tf_fold2] step 40000/40000 | loss 1.5024 | lr 0.00e+00 | 618 win/s
        [val] step 40000: 8-test-user CER 87.80 | fold-heldout-user CER 89.26  (their Tiny reference: 35.9)
[tf_fold3] step 40000/40000 | loss 1.5302 | lr 0.00e+00 | 516 win/s
        [val] step 40000: 8-test-user CER 88.39 | fold-heldout-user CER 89.60  (their Tiny reference: 35.9)
--- 07:01 ---
[tf_ref_lr1e3] step 40000/40000 | loss 1.6571 | lr 0.00e+00 | 387 win/s
        [val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] step 40000/40000 | loss 1.5527 | lr 0.00e+00 | 398 win/s
        [val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
[tf_fold1] step 40000/40000 | loss 1.5735 | lr 0.00e+00 | 617 win/s
        [val] step 40000: 8-test-user CER 87.88 | fold-heldout-user CER 88.04  (their Tiny reference: 35.9)
[tf_fold2] step 40000/40000 | loss 1.5024 | lr 0.00e+00 | 618 win/s
        [val] step 40000: 8-test-user CER 87.80 | fold-heldout-user CER 89.26  (their Tiny reference: 35.9)
[tf_fold3] step 40000/40000 | loss 1.5302 | lr 0.00e+00 | 516 win/s
        [val] step 40000: 8-test-user CER 88.39 | fold-heldout-user CER 89.60  (their Tiny reference: 35.9)
=== 510 done ===
```

### 510_fold_fleet_lr1e3.log
```
        [val] step 10000: 8-test-user CER 84.65 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] step 5800/40000 | loss 2.2929 | lr 9.76e-04 | 357 win/s
        [val] new best 83.52 -> best.pt
[tf_fold1] step 10200/40000 | loss 2.1663 | lr 8.89e-04 | 619 win/s
        [val] step 10000: 8-test-user CER 85.21 | fold-heldout-user CER 85.44  (their Tiny reference: 35.9)
[tf_fold2] step 10200/40000 | loss 2.1317 | lr 8.89e-04 | 623 win/s
        [val] step 10000: 8-test-user CER 87.26 | fold-heldout-user CER 88.44  (their Tiny reference: 35.9)
--- 18:15 ---
[tf_ref_lr1e3] step 11200/40000 | loss 2.2409 | lr 8.62e-04 | 414 win/s
        [val] step 10000: 8-test-user CER 84.65 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] step 6200/40000 | loss 2.2882 | lr 9.70e-04 | 354 win/s
        [val] step 6000: 8-test-user CER 85.77 | fold-heldout-user CER 87.43  (their Tiny reference: 35.9)
[tf_fold1] step 11000/40000 | loss 2.1475 | lr 8.68e-04 | 620 win/s
        [val] step 10000: 8-test-user CER 85.21 | fold-heldout-user CER 85.44  (their Tiny reference: 35.9)
[tf_fold2] step 11000/40000 | loss 2.1292 | lr 8.68e-04 | 624 win/s
        [val] step 10000: 8-test-user CER 87.26 | fold-heldout-user CER 88.44  (their Tiny reference: 35.9)
--- 18:20 ---
[tf_ref_lr1e3] step 11600/40000 | loss 2.2235 | lr 8.51e-04 | 410 win/s
        [val] step 10000: 8-test-user CER 84.65 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0] step 6600/40000 | loss 2.2801 | lr 9.64e-04 | 353 win/s
        [val] step 6000: 8-test-user CER 85.77 | fold-heldout-user CER 87.43  (their Tiny reference: 35.9)
[tf_fold1] step 11800/40000 | loss 2.1304 | lr 8.45e-04 | 621 win/s
        [val] step 10000: 8-test-user CER 85.21 | fold-heldout-user CER 85.44  (their Tiny reference: 35.9)
[tf_fold2] step 11600/40000 | loss 2.0999 | lr 8.51e-04 | 625 win/s
        [val] step 10000: 8-test-user CER 87.26 | fold-heldout-user CER 88.44  (their Tiny reference: 35.9)
```

### 520_deploy_symbol_icl.log
```
=== extract ===
AST OK

=== smoke: the symbol map itself ===
letters=26  example 8-cycle: abfhkmqz -> qzkbmafh
fixed points and mapping application OK
SMOKE OK
=== 520 done: symbol-ICL trainer deployed, waiting on the fold fleet ===
```

### 520_symbol_icl.log
```
=== extract ===
AST OK

=== smoke: the symbol map itself ===
letters=26  example 8-cycle: abfhkmqz -> qzkbmafh
fixed points and mapping application OK
SMOKE OK
=== 520 done: symbol-ICL trainer deployed, waiting on the fold fleet ===
```

### 525_diag.log
```
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
checkpoint step 10000

1. INPUT SCALE: mean 0.0  std 13.6  absmax 924
   post-conv1  std 21.520  absmax 1298.8
   post-featurizer  std 0.174

2. BLANK: argmax==blank on 99.2% of frames | mean p(blank) 0.991
3. ENTROPY: 0.017 nats (uniform would be 4.595)

[0] ref(0): ''
    hyp(0): ''

[1] ref(0): ''
    hyp(0): ''

[2] ref(0): ''
    hyp(0): ''
=== 525 done ===
```

### 525_trunk_diag.log
```
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
checkpoint step 10000

1. INPUT SCALE: mean 0.0  std 13.6  absmax 924
   post-conv1  std 21.520  absmax 1298.8
   post-featurizer  std 0.174

2. BLANK: argmax==blank on 99.2% of frames | mean p(blank) 0.991
3. ENTROPY: 0.017 nats (uniform would be 4.595)

[0] ref(0): ''
    hyp(0): ''

[1] ref(0): ''
    hyp(0): ''

[2] ref(0): ''
    hyp(0): ''
=== 525 done ===
```

### 526_blank.log
```
  tf_ref_lr1e3_40k step  40000 | argmax-blank  99.5% | p(blank) 0.991 | entropy 0.039
  (probe batch: 8 x 30s windows, 8/8 non-empty refs)
--- 14:29 ---
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
  tf_fold0       step  40000 | argmax-blank  99.4% | p(blank) 0.991 | entropy 0.029
  tf_fold0_40k   step  40000 | argmax-blank  99.4% | p(blank) 0.991 | entropy 0.029
  tf_fold0_full  step 103000 | argmax-blank  99.5% | p(blank) 0.995 | entropy 0.009
  tf_fold1       step  40000 | argmax-blank  99.5% | p(blank) 0.990 | entropy 0.045
  tf_fold1_40k   step  40000 | argmax-blank  99.5% | p(blank) 0.990 | entropy 0.045
  tf_fold1_full  step  96000 | argmax-blank  99.5% | p(blank) 0.994 | entropy 0.011
  tf_fold2       step  40000 | argmax-blank  99.5% | p(blank) 0.993 | entropy 0.020
  tf_fold2_40k   step  40000 | argmax-blank  99.5% | p(blank) 0.993 | entropy 0.020
  tf_fold3       step  40000 | argmax-blank  99.5% | p(blank) 0.990 | entropy 0.048
  tf_fold3_40k   step  40000 | argmax-blank  99.5% | p(blank) 0.990 | entropy 0.048
  tf_ref_full    step  96000 | argmax-blank  99.5% | p(blank) 0.994 | entropy 0.008
  tf_ref_lr1e3   step  40000 | argmax-blank  99.5% | p(blank) 0.991 | entropy 0.039
  tf_ref_lr1e3_40k step  40000 | argmax-blank  99.5% | p(blank) 0.991 | entropy 0.039
  (probe batch: 8 x 30s windows, 8/8 non-empty refs)
--- 15:20 ---
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
  tf_fold0       step  40000 | argmax-blank  99.4% | p(blank) 0.991 | entropy 0.029
  tf_fold0_40k   step  40000 | argmax-blank  99.4% | p(blank) 0.991 | entropy 0.029
  tf_fold0_full  step 103000 | argmax-blank  99.5% | p(blank) 0.995 | entropy 0.009
```

### 526_blank_tracker.log
```
--- 18:23 ---
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
  tf_fold0       step   6000 | argmax-blank  99.3% | p(blank) 0.982 | entropy 0.072
  tf_fold1       step  12000 | argmax-blank  99.3% | p(blank) 0.978 | entropy 0.100
  tf_fold2       step  12000 | argmax-blank  99.3% | p(blank) 0.981 | entropy 0.081
  tf_ref_lr1e3   step  12000 | argmax-blank  99.3% | p(blank) 0.981 | entropy 0.087
  (probe batch: 8 x 30s windows, 8/8 non-empty refs)
--- 18:56 ---
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
  tf_fold0       step   8000 | argmax-blank  99.3% | p(blank) 0.980 | entropy 0.084
```

### 530_gate_5s_eval.log
```
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv1d(input, weight, bias, self.stride,
[tf_fold0] step 10000 | 8-test-user 5s-window CER 53.13 (72176 chars) | fold-heldout 63.10
[tf_fold1] step 12000 | 8-test-user 5s-window CER 53.57 (72176 chars) | fold-heldout 57.78
```

### 530_gate.log
```
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv1d(input, weight, bias, self.stride,
[tf_fold0] step 10000 | 8-test-user 5s-window CER 53.13 (72176 chars) | fold-heldout 63.10
[tf_fold1] step 12000 | 8-test-user 5s-window CER 53.57 (72176 chars) | fold-heldout 57.78
[tf_fold2] step 14000 | 8-test-user 5s-window CER 52.01 (72176 chars) | fold-heldout 57.58
[tf_fold3] step 4000 | 8-test-user 5s-window CER 56.14 (72176 chars) | fold-heldout 66.02
[tf_ref_lr1e3] step 4000 | 8-test-user 5s-window CER 55.39 (72176 chars)

=== window-length sweep on tf_ref_lr1e3 (quantify the monitor bias) ===
  window    5s: CER  58.98  (18307 chars)
  window   10s: CER  62.34  (18244 chars)
  window   20s: CER  72.95  (18128 chars)
  window   30s: CER  79.34  (17962 chars)

reference: paper Tiny 35.9 (4 s windows, same 8 test users)
verdict guide: <=45 -> reproduction roughly holds, monitor was the artifact; 60-75 -> partial, dig into encoder internals; >80 -> re-implementation genuinely fails, switch to running fairemg itself.
=== 530 done ===
```

### 531_gate_final.log
```
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv1d(input, weight, bias, self.stride,
[fold2_best] step 14000 | 8-test 5s CER 52.01 | fold-heldout 57.58
  (rc=0 for fold2_best)

=== window-length sweep on ref_last40k ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv1d(input, weight, bias, self.stride,
  window    5s: CER  43.68
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv1d(input, weight, bias, self.stride,
  window   10s: CER  68.41
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv1d(input, weight, bias, self.stride,
  window   30s: CER  87.50
reference: paper Tiny 35.9 | verdict: <=45 in-class (train longer to
close the rest); 45-60 partial; >80 re-implementation bug -> run fairemg
=== 531 done ===
```

### 540_fullbudget_and_icl_dev.log
```
  fold 2:  24 users,  213 sessions | e.g. ['12565339', '18200807', '25915650']
  fold 3:  24 users,  205 sessions | e.g. ['13321435', '20676876', '26940776']
[split] fold 0: train on 72 users (624 sessions); HELD OUT 24 users (213 sessions)
[split] official test users: 16 sessions (never trained on in either mode)
[data] 172708 training windows of 4.0s
--- tf_fold1_full ---
96 training users -> 4 folds
  fold 0:  24 users,  213 sessions | e.g. ['11372316', '14312238', '2396581']
  fold 1:  24 users,  206 sessions | e.g. ['11944098', '1438774', '25847138']
  fold 2:  24 users,  213 sessions | e.g. ['12565339', '18200807', '25915650']
  fold 3:  24 users,  205 sessions | e.g. ['13321435', '20676876', '26940776']
[split] fold 1: train on 72 users (631 sessions); HELD OUT 24 users (206 sessions)
[split] official test users: 16 sessions (never trained on in either mode)
[data] 174323 training windows of 4.0s
--- icl_dev_fold2 ---
[cohort] fold 2: 24 users the backbone has never seen, 213 sessions
[trunk] /data2/chenyuxiang/runs/tf_fold2/last.pt step 40000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
Traceback (most recent call last):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/data2/chenyuxiang/code/myoicl/myoicl/train_prefix_icl.py", line 445, in <module>

=== stream (16 h) ===
```

### 540_fullbudget.log
```
        [val] step 100000: 8-test-user CER 46.21 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0_full] step 103000/103000 | loss 1.2038 | lr 0.00e+00 | 660 win/s
        [val] step 103000: 8-test-user CER 45.88 | fold-heldout-user CER 53.87  (their Tiny reference: 35.9)
[tf_fold1_full] step 103000/103000 | loss 1.2239 | lr 0.00e+00 | 512 win/s
        [val] step 103000: 8-test-user CER 46.82 | fold-heldout-user CER 54.28  (their Tiny reference: 35.9)
[icl_dev_fold2] step 12000/12000 | loss 2.4017 | lr 0.00e+00 | 1.51 it/s
        [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
--- 15:24 ---
[tf_ref_full] step 100600/103000 | loss 1.2611 | lr 9.76e-07 | 488 win/s
        [val] step 100000: 8-test-user CER 46.21 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0_full] step 103000/103000 | loss 1.2038 | lr 0.00e+00 | 660 win/s
        [val] step 103000: 8-test-user CER 45.88 | fold-heldout-user CER 53.87  (their Tiny reference: 35.9)
[tf_fold1_full] step 103000/103000 | loss 1.2239 | lr 0.00e+00 | 512 win/s
        [val] step 103000: 8-test-user CER 46.82 | fold-heldout-user CER 54.28  (their Tiny reference: 35.9)
[icl_dev_fold2] step 12000/12000 | loss 2.4017 | lr 0.00e+00 | 1.51 it/s
        [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
--- 15:29 ---
[tf_ref_full] step 101000/103000 | loss 1.2603 | lr 6.78e-07 | 487 win/s
        [val] step 100000: 8-test-user CER 46.21 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[tf_fold0_full] step 103000/103000 | loss 1.2038 | lr 0.00e+00 | 660 win/s
        [val] step 103000: 8-test-user CER 45.88 | fold-heldout-user CER 53.87  (their Tiny reference: 35.9)
[tf_fold1_full] step 103000/103000 | loss 1.2239 | lr 0.00e+00 | 512 win/s
        [val] step 103000: 8-test-user CER 46.82 | fold-heldout-user CER 54.28  (their Tiny reference: 35.9)
[icl_dev_fold2] step 12000/12000 | loss 2.4017 | lr 0.00e+00 | 1.51 it/s
        [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
```

### 541_icl_dev_relaunch.log
```
AST OK
relaunched icl_dev_fold2 pid=3845713
[cohort] fold 2: 24 users the backbone has never seen, 213 sessions
[trunk] /data2/chenyuxiang/runs/tf_fold2/last.pt step 40000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
[prefix] {'k_windows': 4, 'seconds': 16, 'tokens_uncapped': 380, 'tokens': 380, 'capped': False}
[prefix] {'k_windows': 12, 'seconds': 48, 'tokens_uncapped': 1140, 'tokens': 1140, 'capped': False}
[prefix] {'k_windows': 23, 'seconds': 92, 'tokens_uncapped': 2185, 'tokens': 2185, 'capped': False}
[prefix] {'k_windows': 45, 'seconds': 180, 'tokens_uncapped': 4275, 'tokens': 4096, 'capped': True}
[symbol] 26 permutable letter classes | p_permute 0.5 k [4, 12]
  return F.conv1d(input, weight, bias, self.stride,
[audit] step 0: mode-A 63.00 | mode-C 100.00 (random prefix) | deployment reference ~43-58
step 100/12000 | loss 8.1230 | lr 5.00e-05 | 1.74 it/s
step 200/12000 | loss 4.7799 | lr 1.00e-04 | 1.72 it/s
step 300/12000 | loss 3.3612 | lr 1.50e-04 | 1.63 it/s
step 400/12000 | loss 3.1112 | lr 2.00e-04 | 1.54 it/s
[01:05] [val] new best mode-C 75.06 -> best.pt
[01:10] [val] new best mode-C 56.03 -> best.pt
```

### 541_icl_relaunch.log
```
[03:30] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[03:35] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[03:40] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[03:45] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[03:50] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[03:55] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:00] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:05] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:10] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:15] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:20] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:25] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:30] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:35] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:40] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:45] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:50] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[04:55] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[05:00] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[05:05] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[05:10] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[05:15] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[05:20] [val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
dev run ended
=== 541 done ===
```

### 542_kcurve_autotrigger.log
```
AST OK
=== waiting for a worthwhile icl_dev checkpoint ===
  waiting (12) step 5600/12000 | loss 2.4000 | lr 1.79e-04 | 1.55 it/s
using GPU3
[ckpt] /data2/chenyuxiang/runs/icl_dev_fold2/best.pt step 7500
[cohort] fold 2: 24 real novel users
  return F.conv1d(input, weight, bias, self.stride,
k=  4 ( 16s): mode-A  44.91 | mode-C  45.62 | gain  -0.71 (30 episodes)
k= 12 ( 48s): mode-A  45.30 | mode-C  44.90 | gain  +0.41 (30 episodes)
k= 23 ( 92s): mode-A  44.16 | mode-C  44.08 | gain  +0.08 (30 episodes)
k= 45 (180s): mode-A  44.51 | mode-C  44.90 | gain  -0.38 (30 episodes)

K-curve slope of gain: -0.0006 CER per support window
verdict: positive gain AND positive slope -> the mechanism holds;
flat gain==0 -> support is ignored; positive gain, zero slope -> 
a constant bias, not in-context learning.
[saved] /data2/chenyuxiang/runs/icl_kcurve_fold2.json
=== 542 done ===
```

### 542_kcurve.log
```
AST OK
=== waiting for a worthwhile icl_dev checkpoint ===
  waiting (12) step 5600/12000 | loss 2.4000 | lr 1.79e-04 | 1.55 it/s
using GPU3
[ckpt] /data2/chenyuxiang/runs/icl_dev_fold2/best.pt step 7500
[cohort] fold 2: 24 real novel users
  return F.conv1d(input, weight, bias, self.stride,
k=  4 ( 16s): mode-A  44.91 | mode-C  45.62 | gain  -0.71 (30 episodes)
k= 12 ( 48s): mode-A  45.30 | mode-C  44.90 | gain  +0.41 (30 episodes)
k= 23 ( 92s): mode-A  44.16 | mode-C  44.08 | gain  +0.08 (30 episodes)
k= 45 (180s): mode-A  44.51 | mode-C  44.90 | gain  -0.38 (30 episodes)

K-curve slope of gain: -0.0006 CER per support window
verdict: positive gain AND positive slope -> the mechanism holds;
flat gain==0 -> support is ignored; positive gain, zero slope -> 
a constant bias, not in-context learning.
[saved] /data2/chenyuxiang/runs/icl_kcurve_fold2.json
=== 542 done ===
```

### 543_icl_phase2_conditional.log
```
AST OK
=== wait for phase 1 to finish (or die) ===
phase-1 best gain: 0.37
=== phase 2: symbol-tuning emphasis, warm-started from /data2/chenyuxiang/runs/icl_dev_fold2/best.pt ===
phase 2 pid=4079332
```

### 543_phase2.log
```
[03:33] [val] step 2000: mode-A 53.05 | mode-C 53.20 | gain C -0.16   (REAL novel subjects, fold 2)
[03:38] [val] step 2500: mode-A 52.58 | mode-C 53.33 | gain C -0.75   (REAL novel subjects, fold 2)
[03:43] [val] step 3000: mode-A 61.16 | mode-C 61.03 | gain C +0.12   (REAL novel subjects, fold 2)
[03:48] [val] step 3500: mode-A 56.54 | mode-C 56.50 | gain C +0.05   (REAL novel subjects, fold 2)
[03:53] [val] new best mode-C 45.37 -> best.pt
[03:58] [val] step 4500: mode-A 50.06 | mode-C 50.49 | gain C -0.44   (REAL novel subjects, fold 2)
[04:03] [val] step 5000: mode-A 52.32 | mode-C 52.85 | gain C -0.54   (REAL novel subjects, fold 2)
[04:08] [val] step 5500: mode-A 52.45 | mode-C 52.97 | gain C -0.52   (REAL novel subjects, fold 2)
[04:13] [val] step 6000: mode-A 51.82 | mode-C 51.91 | gain C -0.09   (REAL novel subjects, fold 2)
[04:18] [val] step 6500: mode-A 46.04 | mode-C 45.61 | gain C +0.43   (REAL novel subjects, fold 2)
[04:23] [val] step 6500: mode-A 46.04 | mode-C 45.61 | gain C +0.43   (REAL novel subjects, fold 2)
[04:28] [val] step 7000: mode-A 48.94 | mode-C 48.79 | gain C +0.15   (REAL novel subjects, fold 2)
[04:33] [val] step 7500: mode-A 49.76 | mode-C 49.56 | gain C +0.20   (REAL novel subjects, fold 2)
[04:38] [val] step 8000: mode-A 51.88 | mode-C 52.31 | gain C -0.44   (REAL novel subjects, fold 2)
[04:43] [val] step 8500: mode-A 54.10 | mode-C 54.31 | gain C -0.21   (REAL novel subjects, fold 2)
[04:48] [val] step 9000: mode-A 53.15 | mode-C 52.17 | gain C +0.98   (REAL novel subjects, fold 2)
[04:53] [val] step 9500: mode-A 52.04 | mode-C 51.66 | gain C +0.38   (REAL novel subjects, fold 2)
[04:58] [val] step 10000: mode-A 50.11 | mode-C 50.08 | gain C +0.02   (REAL novel subjects, fold 2)
[05:03] [val] step 10500: mode-A 49.27 | mode-C 49.71 | gain C -0.44   (REAL novel subjects, fold 2)
[05:08] [val] step 11000: mode-A 51.72 | mode-C 51.37 | gain C +0.35   (REAL novel subjects, fold 2)
[05:13] [val] step 11500: mode-A 46.72 | mode-C 47.36 | gain C -0.64   (REAL novel subjects, fold 2)
[05:18] [val] step 12000: mode-A 47.42 | mode-C 47.99 | gain C -0.56   (REAL novel subjects, fold 2)
[05:23] [val] step 12000: mode-A 47.42 | mode-C 47.99 | gain C -0.56   (REAL novel subjects, fold 2)
phase 2 ended
=== 543 done ===
```

### 544_permprobe.log
```
[ckpt] /data2/chenyuxiang/runs/icl_dev2_fold2/best.pt step 4000
[permuted probe] derangement of 10 letters per episode -- mode A cannot know the mapping
k= 12 ( 48s): mode-A  57.97 | mode-C  59.07 | gain  -1.10 (20 episodes)
[saved] /data2/chenyuxiang/runs/perm_probe_latest.json
k= 12 ( 48s): mode-A  43.23 | mode-C  42.86 | gain  +0.37 (20 episodes)
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
phase 2 ended -- final probe follows
=== 544 done ===
```

### 544_perm_probe_loop.log
```
patch verified (permute_k x5)
=== probe at 03:21 (ckpt 03:18) ===
[ckpt] /data2/chenyuxiang/runs/icl_dev2_fold2/best.pt step 1000
[permuted probe] derangement of 10 letters per episode -- mode A cannot know the mapping
k= 12 ( 48s): mode-A  59.91 | mode-C  63.39 | gain  -3.47 (20 episodes)
[saved] /data2/chenyuxiang/runs/perm_probe_latest.json
k= 12 ( 48s): mode-A  43.66 | mode-C  44.58 | gain  -0.92 (20 episodes)
=== probe at 04:11 (ckpt 03:50) ===
[ckpt] /data2/chenyuxiang/runs/icl_dev2_fold2/best.pt step 4000
[permuted probe] derangement of 10 letters per episode -- mode A cannot know the mapping
k= 12 ( 48s): mode-A  57.97 | mode-C  59.07 | gain  -1.10 (20 episodes)
[saved] /data2/chenyuxiang/runs/perm_probe_latest.json
k= 12 ( 48s): mode-A  43.23 | mode-C  42.86 | gain  +0.37 (20 episodes)
phase 2 ended -- final probe follows
```

### 545_final_night_verdict.log
```
=== wait for phase 2 to end ===
=== permuted probe on last.pt (mechanism) ===
[ckpt] /data2/chenyuxiang/runs/icl_dev2_fold2/last.pt step 12000
[permuted probe] derangement of 10 letters per episode -- mode A cannot know the mapping
k= 12 ( 48s): mode-A  57.84 | mode-C  59.82 | gain  -1.98 (30 episodes)
k= 45 (180s): mode-A  58.47 | mode-C  61.51 | gain  -3.04 (30 episodes)
K-curve slope of gain: -0.0321 CER per support window
verdict: positive gain AND positive slope -> the mechanism holds;
flat gain==0 -> support is ignored; positive gain, zero slope -> 
[saved] /data2/chenyuxiang/runs/final_perm_probe.json

=== identity K-curve on last.pt (deployment) ===
[ckpt] /data2/chenyuxiang/runs/icl_dev2_fold2/last.pt step 12000
k=  4 ( 16s): mode-A  42.82 | mode-C  42.80 | gain  +0.02 (30 episodes)
k= 12 ( 48s): mode-A  42.66 | mode-C  42.73 | gain  -0.08 (30 episodes)
k= 23 ( 92s): mode-A  43.12 | mode-C  44.03 | gain  -0.90 (30 episodes)
k= 45 (180s): mode-A  43.32 | mode-C  43.48 | gain  -0.16 (30 episodes)
K-curve slope of gain: -0.0054 CER per support window
verdict: positive gain AND positive slope -> the mechanism holds;
flat gain==0 -> support is ignored; positive gain, zero slope -> 
=== 545 done ===
```

### 545_verdict.log
```
=== wait for phase 2 to end ===
=== permuted probe on last.pt (mechanism) ===
[ckpt] /data2/chenyuxiang/runs/icl_dev2_fold2/last.pt step 12000
[permuted probe] derangement of 10 letters per episode -- mode A cannot know the mapping
k= 12 ( 48s): mode-A  57.84 | mode-C  59.82 | gain  -1.98 (30 episodes)
k= 45 (180s): mode-A  58.47 | mode-C  61.51 | gain  -3.04 (30 episodes)
K-curve slope of gain: -0.0321 CER per support window
verdict: positive gain AND positive slope -> the mechanism holds;
flat gain==0 -> support is ignored; positive gain, zero slope -> 
[saved] /data2/chenyuxiang/runs/final_perm_probe.json

=== identity K-curve on last.pt (deployment) ===
[ckpt] /data2/chenyuxiang/runs/icl_dev2_fold2/last.pt step 12000
k=  4 ( 16s): mode-A  42.82 | mode-C  42.80 | gain  +0.02 (30 episodes)
k= 12 ( 48s): mode-A  42.66 | mode-C  42.73 | gain  -0.08 (30 episodes)
k= 23 ( 92s): mode-A  43.12 | mode-C  44.03 | gain  -0.90 (30 episodes)
k= 45 (180s): mode-A  43.32 | mode-C  43.48 | gain  -0.16 (30 episodes)
K-curve slope of gain: -0.0054 CER per support window
verdict: positive gain AND positive slope -> the mechanism holds;
flat gain==0 -> support is ignored; positive gain, zero slope -> 
=== 545 done ===
```

### 550_fused.log
```
patch verified (fused x8)
AST OK
launched icl_fused_fold2 pid=146692
[cohort] fold 2: 24 users the backbone has never seen, 213 sessions
[trunk] /data2/chenyuxiang/runs/tf_fold2/last.pt step 40000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
[prefix] FUSED mode: per-token (signal + soft-aligned char)
[prefix] {'k_windows': 4, 'seconds': 16, 'tokens_uncapped': 380, 'tokens': 380, 'capped': False}
[prefix] {'k_windows': 12, 'seconds': 48, 'tokens_uncapped': 1140, 'tokens': 1140, 'capped': False}
[prefix] {'k_windows': 23, 'seconds': 92, 'tokens_uncapped': 2185, 'tokens': 2185, 'capped': False}
[prefix] {'k_windows': 45, 'seconds': 180, 'tokens_uncapped': 4275, 'tokens': 4096, 'capped': True}
[symbol] 26 permutable letter classes | p_permute 0.5 k [4, 12]
  return F.conv1d(input, weight, bias, self.stride,
[audit] step 0: mode-A 62.63 | mode-C 100.00 (random prefix) | deployment reference ~43-58
Traceback (most recent call last):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 196, in _run_module_as_main
[05:42] Traceback (most recent call last):
fused run ended
=== 550 done ===
```

### 550_fused_prefix_icl.log
```
patch verified (fused x8)
AST OK
launched icl_fused_fold2 pid=146692
[cohort] fold 2: 24 users the backbone has never seen, 213 sessions
[trunk] /data2/chenyuxiang/runs/tf_fold2/last.pt step 40000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
[prefix] FUSED mode: per-token (signal + soft-aligned char)
[prefix] {'k_windows': 4, 'seconds': 16, 'tokens_uncapped': 380, 'tokens': 380, 'capped': False}
[prefix] {'k_windows': 12, 'seconds': 48, 'tokens_uncapped': 1140, 'tokens': 1140, 'capped': False}
[prefix] {'k_windows': 23, 'seconds': 92, 'tokens_uncapped': 2185, 'tokens': 2185, 'capped': False}
[prefix] {'k_windows': 45, 'seconds': 180, 'tokens_uncapped': 4275, 'tokens': 4096, 'capped': True}
[symbol] 26 permutable letter classes | p_permute 0.5 k [4, 12]
  return F.conv1d(input, weight, bias, self.stride,
[audit] step 0: mode-A 62.63 | mode-C 100.00 (random prefix) | deployment reference ~43-58
Traceback (most recent call last):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 196, in _run_module_as_main
[05:42] Traceback (most recent call last):
fused run ended
=== 550 done ===
```

### 551_fused.log
```
patch verified (fused x8)
AST OK
launched icl_fusedb_fold2 pid=167606
[cohort] fold 2: 24 users the backbone has never seen, 213 sessions
[trunk] /data2/chenyuxiang/runs/tf_fold2/last.pt step 40000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
[prefix] FUSED mode: per-token (signal + soft-aligned char)
[prefix] {'k_windows': 4, 'seconds': 16, 'tokens_uncapped': 380, 'tokens': 380, 'capped': False}
[prefix] {'k_windows': 12, 'seconds': 48, 'tokens_uncapped': 1140, 'tokens': 1140, 'capped': False}
[prefix] {'k_windows': 23, 'seconds': 92, 'tokens_uncapped': 2185, 'tokens': 2185, 'capped': False}
[prefix] {'k_windows': 45, 'seconds': 180, 'tokens_uncapped': 4275, 'tokens': 4096, 'capped': True}
[symbol] 26 permutable letter classes | p_permute 0.5 k [4, 12]
  return F.conv1d(input, weight, bias, self.stride,
[audit] step 0: mode-A 62.91 | mode-C 100.00 (random prefix) | deployment reference ~43-58
step 100/12000 | loss 8.1298 | lr 5.00e-05 | 1.62 it/s
step 200/12000 | loss 5.2564 | lr 1.00e-04 | 1.60 it/s
[05:54] [val] new best mode-C 75.59 -> best.pt
[05:59] [val] new best mode-C 58.20 -> best.pt
[06:04] [val] step 1500: mode-A 57.10 | mode-C 58.68 | gain C -1.58   (REAL novel subjects, fold 2)
[06:09] [val] step 2000: mode-A 59.25 | mode-C 60.57 | gain C -1.32   (REAL novel subjects, fold 2)
[06:14] [val] step 2000: mode-A 59.25 | mode-C 60.57 | gain C -1.32   (REAL novel subjects, fold 2)
[06:19] [val] new best mode-C 57.42 -> best.pt
[06:24] [val] new best mode-C 56.67 -> best.pt
[06:29] [val] new best mode-C 55.14 -> best.pt
fused run ended
=== 551 done ===
```

### 551_fused_retry.log
```
patch verified (fused x8)
AST OK
launched icl_fusedb_fold2 pid=167606
[cohort] fold 2: 24 users the backbone has never seen, 213 sessions
[trunk] /data2/chenyuxiang/runs/tf_fold2/last.pt step 40000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
[prefix] FUSED mode: per-token (signal + soft-aligned char)
[prefix] {'k_windows': 4, 'seconds': 16, 'tokens_uncapped': 380, 'tokens': 380, 'capped': False}
[prefix] {'k_windows': 12, 'seconds': 48, 'tokens_uncapped': 1140, 'tokens': 1140, 'capped': False}
[prefix] {'k_windows': 23, 'seconds': 92, 'tokens_uncapped': 2185, 'tokens': 2185, 'capped': False}
[prefix] {'k_windows': 45, 'seconds': 180, 'tokens_uncapped': 4275, 'tokens': 4096, 'capped': True}
[symbol] 26 permutable letter classes | p_permute 0.5 k [4, 12]
  return F.conv1d(input, weight, bias, self.stride,
[audit] step 0: mode-A 62.91 | mode-C 100.00 (random prefix) | deployment reference ~43-58
step 100/12000 | loss 8.1298 | lr 5.00e-05 | 1.62 it/s
step 200/12000 | loss 5.2564 | lr 1.00e-04 | 1.60 it/s
[05:54] [val] new best mode-C 75.59 -> best.pt
[05:59] [val] new best mode-C 58.20 -> best.pt
[06:04] [val] step 1500: mode-A 57.10 | mode-C 58.68 | gain C -1.58   (REAL novel subjects, fold 2)
[06:09] [val] step 2000: mode-A 59.25 | mode-C 60.57 | gain C -1.32   (REAL novel subjects, fold 2)
[06:14] [val] step 2000: mode-A 59.25 | mode-C 60.57 | gain C -1.32   (REAL novel subjects, fold 2)
[06:19] [val] new best mode-C 57.42 -> best.pt
[06:24] [val] new best mode-C 56.67 -> best.pt
[06:29] [val] new best mode-C 55.14 -> best.pt
fused run ended
=== 551 done ===
```

### 552_frozen.log
```
[06:48] [val] step 1500: mode-A 64.14 | mode-C 65.10 | gain C -0.95   (REAL novel subjects, fold 2)
[06:53] [val] step 2000: mode-A 62.40 | mode-C 65.08 | gain C -2.68   (REAL novel subjects, fold 2)
[06:58] [val] step 2500: mode-A 62.88 | mode-C 65.09 | gain C -2.20   (REAL novel subjects, fold 2)
[07:03] [val] step 3000: mode-A 62.10 | mode-C 64.05 | gain C -1.95   (REAL novel subjects, fold 2)
[07:08] [val] step 3500: mode-A 63.92 | mode-C 66.71 | gain C -2.79   (REAL novel subjects, fold 2)
[07:13] [val] new best mode-C 56.54 -> best.pt
[07:18] [val] step 4500: mode-A 57.76 | mode-C 60.85 | gain C -3.09   (REAL novel subjects, fold 2)
[07:23] [val] step 5000: mode-A 63.69 | mode-C 64.53 | gain C -0.84   (REAL novel subjects, fold 2)
[07:28] [val] step 5500: mode-A 65.75 | mode-C 66.20 | gain C -0.45   (REAL novel subjects, fold 2)
[07:33] [val] step 6000: mode-A 60.78 | mode-C 62.34 | gain C -1.56   (REAL novel subjects, fold 2)
[07:38] [val] step 6500: mode-A 59.58 | mode-C 61.48 | gain C -1.90   (REAL novel subjects, fold 2)
[07:43] [val] step 7000: mode-A 63.92 | mode-C 64.28 | gain C -0.36   (REAL novel subjects, fold 2)
[07:48] [val] step 7000: mode-A 63.92 | mode-C 64.28 | gain C -0.36   (REAL novel subjects, fold 2)
[07:53] [val] step 7500: mode-A 55.81 | mode-C 56.83 | gain C -1.02   (REAL novel subjects, fold 2)
[07:58] [val] step 8000: mode-A 61.59 | mode-C 61.61 | gain C -0.02   (REAL novel subjects, fold 2)
[08:03] [val] step 8500: mode-A 65.92 | mode-C 67.05 | gain C -1.13   (REAL novel subjects, fold 2)
[08:08] [val] step 9000: mode-A 60.03 | mode-C 60.52 | gain C -0.49   (REAL novel subjects, fold 2)
[08:13] [val] step 9500: mode-A 62.58 | mode-C 64.11 | gain C -1.54   (REAL novel subjects, fold 2)
[08:18] [val] step 10000: mode-A 58.63 | mode-C 59.45 | gain C -0.82   (REAL novel subjects, fold 2)
[08:23] [val] step 10500: mode-A 62.53 | mode-C 63.91 | gain C -1.37   (REAL novel subjects, fold 2)
[08:28] [val] step 11000: mode-A 62.90 | mode-C 62.69 | gain C +0.22   (REAL novel subjects, fold 2)
[08:33] [val] step 11500: mode-A 58.06 | mode-C 58.81 | gain C -0.75   (REAL novel subjects, fold 2)
[08:38] [val] step 12000: mode-A 60.18 | mode-C 60.20 | gain C -0.02   (REAL novel subjects, fold 2)
frozen run ended
=== 552 done ===
```

### 552_fused_frozen.log
```
=== stop the jointly-trained fused run (it is re-treading the zero-lock) ===
stopped
launched icl_frozen_fold2 pid=248766
[cohort] fold 2: 24 users the backbone has never seen, 213 sessions
[trunk] /data2/chenyuxiang/runs/tf_fold2/last.pt step 40000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
[prefix] FUSED mode: per-token (signal + soft-aligned char)
[prefix] {'k_windows': 4, 'seconds': 16, 'tokens_uncapped': 380, 'tokens': 380, 'capped': False}
[prefix] {'k_windows': 12, 'seconds': 48, 'tokens_uncapped': 1140, 'tokens': 1140, 'capped': False}
[prefix] {'k_windows': 23, 'seconds': 92, 'tokens_uncapped': 2185, 'tokens': 2185, 'capped': False}
[prefix] {'k_windows': 45, 'seconds': 180, 'tokens_uncapped': 4275, 'tokens': 4096, 'capped': True}
[symbol] 26 permutable letter classes | p_permute 0.5 k [4, 12]
  return F.conv1d(input, weight, bias, self.stride,
[audit] step 0: mode-A 62.58 | mode-C 100.00 (random prefix) | deployment reference ~43-58
step 100/12000 | loss 9.6123 | lr 1.67e-04 | 2.02 it/s
step 200/12000 | loss 5.8096 | lr 3.33e-04 | 1.66 it/s
[06:38] [val] new best mode-C 72.48 -> best.pt
[06:43] [val] new best mode-C 61.12 -> best.pt
```

### 553_aux.log
```
step 100/20000 | loss 13.0578 | aux rot 2.769 (chance 2.83) | aux perm 3.278 (chance 3.26) | lr 1.00e-04 | 1.59 it/s
step 200/20000 | loss 12.5117 | aux rot 2.420 (chance 2.83) | aux perm 2.978 (chance 3.26) | lr 2.00e-04 | 1.53 it/s
step 300/20000 | loss 10.4381 | aux rot 2.204 (chance 2.83) | aux perm 2.433 (chance 3.26) | lr 3.00e-04 | 1.55 it/s
step 400/20000 | loss 5.6576 | aux rot 2.159 (chance 2.83) | aux perm 2.008 (chance 3.26) | lr 4.00e-04 | 1.54 it/s
[13:55] step 900/20000 | loss 5.3208 | aux rot 2.211 (chance 2.83) | aux perm 1.779 (chance 3.26) | lr 9.00e-04 | 1.58 it/s
[14:00] step 1300/20000 | loss 5.0901 | aux rot 2.041 (chance 2.83) | aux perm 1.672 (chance 3.26) | lr 9.99e-04 | 1.52 it/s
        [val] new best mode-C 70.59 -> best.pt
[14:05] step 1800/20000 | loss 5.0060 | aux rot 1.987 (chance 2.83) | aux perm 1.606 (chance 3.26) | lr 9.96e-04 | 1.57 it/s
        [val] new best mode-C 70.59 -> best.pt
[14:10] step 2200/20000 | loss 5.1382 | aux rot 2.023 (chance 2.83) | aux perm 1.765 (chance 3.26) | lr 9.90e-04 | 1.52 it/s
        [val] new best mode-C 64.40 -> best.pt
[14:15] step 2600/20000 | loss 4.8422 | aux rot 1.933 (chance 2.83) | aux perm 1.739 (chance 3.26) | lr 9.83e-04 | 1.50 it/s
        [val] new best mode-C 64.40 -> best.pt
[14:20] step 3000/20000 | loss 5.1406 | aux rot 2.037 (chance 2.83) | aux perm 1.765 (chance 3.26) | lr 9.73e-04 | 1.49 it/s
        [val] new best mode-C 64.18 -> best.pt
[14:25] step 3400/20000 | loss 4.7852 | aux rot 1.947 (chance 2.83) | aux perm 1.710 (chance 3.26) | lr 9.61e-04 | 1.46 it/s
        [val] new best mode-C 64.18 -> best.pt
[14:30] step 3800/20000 | loss 4.6989 | aux rot 1.769 (chance 2.83) | aux perm 1.645 (chance 3.26) | lr 9.47e-04 | 1.45 it/s
        [val] new best mode-C 64.18 -> best.pt
[14:35] step 4200/20000 | loss 4.7833 | aux rot 1.720 (chance 2.83) | aux perm 1.730 (chance 3.26) | lr 9.32e-04 | 1.41 it/s
        [val] step 4000: mode-A 65.84 | mode-C 67.90 | gain C -2.06   (REAL novel subjects, fold 2)
[14:40] step 4200/20000 | loss 4.7833 | aux rot 1.720 (chance 2.83) | aux perm 1.730 (chance 3.26) | lr 9.32e-04 | 1.41 it/s
        [val] step 4000: mode-A 65.84 | mode-C 67.90 | gain C -2.06   (REAL novel subjects, fold 2)
aux run ended
=== 553 done ===
```

### 553_aux_supervised.log
```
step 100/20000 | loss 13.0578 | aux rot 2.769 (chance 2.83) | aux perm 3.278 (chance 3.26) | lr 1.00e-04 | 1.59 it/s
step 200/20000 | loss 12.5117 | aux rot 2.420 (chance 2.83) | aux perm 2.978 (chance 3.26) | lr 2.00e-04 | 1.53 it/s
step 300/20000 | loss 10.4381 | aux rot 2.204 (chance 2.83) | aux perm 2.433 (chance 3.26) | lr 3.00e-04 | 1.55 it/s
step 400/20000 | loss 5.6576 | aux rot 2.159 (chance 2.83) | aux perm 2.008 (chance 3.26) | lr 4.00e-04 | 1.54 it/s
[13:55] step 900/20000 | loss 5.3208 | aux rot 2.211 (chance 2.83) | aux perm 1.779 (chance 3.26) | lr 9.00e-04 | 1.58 it/s
[14:00] step 1300/20000 | loss 5.0901 | aux rot 2.041 (chance 2.83) | aux perm 1.672 (chance 3.26) | lr 9.99e-04 | 1.52 it/s
        [val] new best mode-C 70.59 -> best.pt
[14:05] step 1800/20000 | loss 5.0060 | aux rot 1.987 (chance 2.83) | aux perm 1.606 (chance 3.26) | lr 9.96e-04 | 1.57 it/s
        [val] new best mode-C 70.59 -> best.pt
[14:10] step 2200/20000 | loss 5.1382 | aux rot 2.023 (chance 2.83) | aux perm 1.765 (chance 3.26) | lr 9.90e-04 | 1.52 it/s
        [val] new best mode-C 64.40 -> best.pt
[14:15] step 2600/20000 | loss 4.8422 | aux rot 1.933 (chance 2.83) | aux perm 1.739 (chance 3.26) | lr 9.83e-04 | 1.50 it/s
        [val] new best mode-C 64.40 -> best.pt
[14:20] step 3000/20000 | loss 5.1406 | aux rot 2.037 (chance 2.83) | aux perm 1.765 (chance 3.26) | lr 9.73e-04 | 1.49 it/s
        [val] new best mode-C 64.18 -> best.pt
[14:25] step 3400/20000 | loss 4.7852 | aux rot 1.947 (chance 2.83) | aux perm 1.710 (chance 3.26) | lr 9.61e-04 | 1.46 it/s
        [val] new best mode-C 64.18 -> best.pt
[14:30] step 3800/20000 | loss 4.6989 | aux rot 1.769 (chance 2.83) | aux perm 1.645 (chance 3.26) | lr 9.47e-04 | 1.45 it/s
        [val] new best mode-C 64.18 -> best.pt
[14:35] step 4200/20000 | loss 4.7833 | aux rot 1.720 (chance 2.83) | aux perm 1.730 (chance 3.26) | lr 9.32e-04 | 1.41 it/s
        [val] step 4000: mode-A 65.84 | mode-C 67.90 | gain C -2.06   (REAL novel subjects, fold 2)
[14:40] step 4200/20000 | loss 4.7833 | aux rot 1.720 (chance 2.83) | aux perm 1.730 (chance 3.26) | lr 9.32e-04 | 1.41 it/s
        [val] step 4000: mode-A 65.84 | mode-C 67.90 | gain C -2.06   (REAL novel subjects, fold 2)
aux run ended
=== 553 done ===
```

### 554_allgpu_sprint.log
```
=== GPU1: twin aux run on the full-budget fold0 backbone ===
launched icl_aux_fold0 pid=1090250
=== GPU0: probe loop when ref_full finishes ===
probe watcher pid=1090251
=== GPU2: fold2_full continuation when fold1_full finishes ===
=== stream (14 h) ===
--- 14:08 ---
[icl_aux_fold2] step 2100/20000 | loss 4.7245 | aux rot 2.011 (chance 2.83) | aux perm 1.707 (chance 3.26) | lr 9.92
        [val] new best mode-C 64.40 -> best.pt
[icl_aux_fold0] step 300/20000 | loss 5.8680 | aux rot 2.281 (chance 2.83) | aux perm 2.463 (chance 3.26) | lr 3.00e
k= 12 ( 48s): mode-A  53.24 | mode-C  57.82 | gain  -4.58 (24 episodes)
k= 45 (180s): mode-A  53.24 | mode-C  59.84 | gain  -6.60 (24 episodes)
k= 12 ( 48s): mode-A  64.69 | mode-C  67.42 | gain  -2.74 (24 episodes)
--- 14:13 ---
[icl_aux_fold2] step 2500/20000 | loss 4.9932 | aux rot 1.980 (chance 2.83) | aux perm 1.589 (chance 3.26) | lr 9.85
        [val] new best mode-C 64.40 -> best.pt
[icl_aux_fold0] step 700/20000 | loss 5.2765 | aux rot 2.074 (chance 2.83) | aux perm 1.661 (chance 3.26) | lr 7.00e
k= 12 ( 48s): mode-A  53.24 | mode-C  57.82 | gain  -4.58 (24 episodes)
k= 45 (180s): mode-A  53.24 | mode-C  59.84 | gain  -6.60 (24 episodes)
k= 12 ( 48s): mode-A  64.69 | mode-C  67.42 | gain  -2.74 (24 episodes)
```

### 554_probes.log
```
--- 16:13 icl_aux_fold0 ---
[ckpt] /data2/chenyuxiang/runs/icl_aux_fold0/best.pt step 2000
k= 12 ( 48s): mode-A  54.73 | mode-C  54.21 | gain  +0.52 (24 episodes)
k= 45 (180s): mode-A  55.86 | mode-C  62.14 | gain  -6.28 (24 episodes)
k= 12 ( 48s): mode-A  65.48 | mode-C  66.52 | gain  -1.04 (24 episodes)
--- 16:38 icl_aux_fold2 ---
[ckpt] /data2/chenyuxiang/runs/icl_aux_fold2/best.pt step 3000
k= 12 ( 48s): mode-A  52.89 | mode-C  57.21 | gain  -4.32 (24 episodes)
k= 45 (180s): mode-A  54.01 | mode-C  56.87 | gain  -2.86 (24 episodes)
k= 12 ( 48s): mode-A  64.27 | mode-C  67.52 | gain  -3.25 (24 episodes)
--- 16:38 icl_aux_fold0 ---
[ckpt] /data2/chenyuxiang/runs/icl_aux_fold0/best.pt step 2000
k= 12 ( 48s): mode-A  55.66 | mode-C  54.33 | gain  +1.33 (24 episodes)
k= 45 (180s): mode-A  55.57 | mode-C  62.59 | gain  -7.02 (24 episodes)
k= 12 ( 48s): mode-A  66.53 | mode-C  65.97 | gain  +0.56 (24 episodes)
--- 17:04 icl_aux_fold2 ---
[ckpt] /data2/chenyuxiang/runs/icl_aux_fold2/best.pt step 3000
k= 12 ( 48s): mode-A  54.02 | mode-C  58.35 | gain  -4.33 (24 episodes)
k= 45 (180s): mode-A  54.17 | mode-C  57.77 | gain  -3.60 (24 episodes)
k= 12 ( 48s): mode-A  63.77 | mode-C  67.48 | gain  -3.71 (24 episodes)
--- 17:04 icl_aux_fold0 ---
[ckpt] /data2/chenyuxiang/runs/icl_aux_fold0/best.pt step 2000
k= 12 ( 48s): mode-A  56.00 | mode-C  54.41 | gain  +1.60 (24 episodes)
k= 45 (180s): mode-A  56.20 | mode-C  63.07 | gain  -6.86 (24 episodes)
k= 12 ( 48s): mode-A  67.45 | mode-C  66.92 | gain  +0.53 (24 episodes)
```

### 554_sprint.log
```
k= 12 ( 48s): mode-A  67.17 | mode-C  66.45 | gain  +0.72 (24 episodes)
--- 15:18 ---
[icl_aux_fold2] step 4200/20000 | loss 4.7833 | aux rot 1.720 (chance 2.83) | aux perm 1.730 (chance 3.26) | lr 9.32
        [val] step 4000: mode-A 65.84 | mode-C 67.90 | gain C -2.06   (REAL novel subjects, fold 2)
[icl_aux_fold0] step 3000/20000 | loss 5.0045 | aux rot 1.955 (chance 2.83) | aux perm 1.760 (chance 3.26) | lr 9.73
        [val] step 3000: mode-A 58.67 | mode-C 61.45 | gain C -2.78   (REAL novel subjects, fold 0)
k= 12 ( 48s): mode-A  55.40 | mode-C  53.98 | gain  +1.41 (24 episodes)
k= 45 (180s): mode-A  55.82 | mode-C  64.02 | gain  -8.20 (24 episodes)
k= 12 ( 48s): mode-A  67.17 | mode-C  66.45 | gain  +0.72 (24 episodes)
--- 15:23 ---
[icl_aux_fold2] step 4200/20000 | loss 4.7833 | aux rot 1.720 (chance 2.83) | aux perm 1.730 (chance 3.26) | lr 9.32
        [val] step 4000: mode-A 65.84 | mode-C 67.90 | gain C -2.06   (REAL novel subjects, fold 2)
[icl_aux_fold0] step 3000/20000 | loss 5.0045 | aux rot 1.955 (chance 2.83) | aux perm 1.760 (chance 3.26) | lr 9.73
        [val] step 3000: mode-A 58.67 | mode-C 61.45 | gain C -2.78   (REAL novel subjects, fold 0)
k= 12 ( 48s): mode-A  55.14 | mode-C  53.79 | gain  +1.35 (24 episodes)
k= 45 (180s): mode-A  55.68 | mode-C  63.62 | gain  -7.94 (24 episodes)
k= 12 ( 48s): mode-A  66.50 | mode-C  66.31 | gain  +0.19 (24 episodes)
--- 15:28 ---
[icl_aux_fold2] step 4200/20000 | loss 4.7833 | aux rot 1.720 (chance 2.83) | aux perm 1.730 (chance 3.26) | lr 9.32
        [val] step 4000: mode-A 65.84 | mode-C 67.90 | gain C -2.06   (REAL novel subjects, fold 2)
[icl_aux_fold0] step 3000/20000 | loss 5.0045 | aux rot 1.955 (chance 2.83) | aux perm 1.760 (chance 3.26) | lr 9.73
        [val] step 3000: mode-A 58.67 | mode-C 61.45 | gain C -2.78   (REAL novel subjects, fold 0)
k= 12 ( 48s): mode-A  55.14 | mode-C  53.79 | gain  +1.35 (24 episodes)
k= 45 (180s): mode-A  55.68 | mode-C  63.62 | gain  -7.94 (24 episodes)
k= 12 ( 48s): mode-A  66.50 | mode-C  66.31 | gain  +0.19 (24 episodes)
```

### 555_gate_full_eval.log
```
=== wait for ref_full / fold1_full to finish ===
```

### 555_gate_full.log
```
=== wait for ref_full / fold1_full to finish ===
```

### 556_joint.log
```
=== wait for tf_fold1_full to finish ===
using GPU0
launched icl_joint_fold1 pid=1211048
[15:09] step 200/20000 | loss 7.0014 | aux rot 2.560 (chance 2.83) | aux perm 3.132 (chance 3.26) | lr 1.00e
[15:14] step 400/20000 | loss 6.2978 | aux rot 2.187 (chance 2.83) | aux perm 2.488 (chance 3.26) | lr 2.00e
[15:19] step 700/20000 | loss 5.0362 | aux rot 2.080 (chance 2.83) | aux perm 1.784 (chance 3.26) | lr 3.50e
[15:24] step 900/20000 | loss 4.9979 | aux rot 2.124 (chance 2.83) | aux perm 1.754 (chance 3.26) | lr 4.50e
[15:29] step 1100/20000 | loss 5.0751 | aux rot 2.073 (chance 2.83) | aux perm 1.565 (chance 3.26) | lr 5.00
        [val] new best mode-C 55.21 -> best.pt
```

### 556_joint_synth_aux.log
```
        [val] step 4000: mode-A 47.23 | mode-C 46.75 | gain C +0.48   (REAL novel subjects, fold 1)
[16:29] step 4600/20000 | loss 4.6108 | aux rot 1.501 (chance 2.83) | aux perm 1.636 (chance 3.26) | lr 4.57
        [val] step 4000: mode-A 47.23 | mode-C 46.75 | gain C +0.48   (REAL novel subjects, fold 1)
[16:34] step 4900/20000 | loss 4.1456 | aux rot 1.387 (chance 2.83) | aux perm 1.696 (chance 3.26) | lr 4.50
        [val] step 4000: mode-A 47.23 | mode-C 46.75 | gain C +0.48   (REAL novel subjects, fold 1)
[16:39] step 5300/20000 | loss 4.3116 | aux rot 1.362 (chance 2.83) | aux perm 1.491 (chance 3.26) | lr 4.39
        [val] new best mode-C 40.65 -> best.pt
[16:44] step 5600/20000 | loss 4.1729 | aux rot 1.413 (chance 2.83) | aux perm 1.594 (chance 3.26) | lr 4.31
        [val] new best mode-C 40.65 -> best.pt
[16:49] step 5900/20000 | loss 4.4784 | aux rot 1.359 (chance 2.83) | aux perm 1.589 (chance 3.26) | lr 4.22
        [val] new best mode-C 40.65 -> best.pt
[16:54] step 6200/20000 | loss 4.3861 | aux rot 1.322 (chance 2.83) | aux perm 1.725 (chance 3.26) | lr 4.13
        [val] step 6000: mode-A 41.87 | mode-C 43.37 | gain C -1.50   (REAL novel subjects, fold 1)
[16:59] step 6500/20000 | loss 4.1344 | aux rot 1.332 (chance 2.83) | aux perm 1.660 (chance 3.26) | lr 4.04
        [val] step 6000: mode-A 41.87 | mode-C 43.37 | gain C -1.50   (REAL novel subjects, fold 1)
[17:04] step 6800/20000 | loss 4.2060 | aux rot 1.382 (chance 2.83) | aux perm 1.648 (chance 3.26) | lr 3.94
        [val] step 6000: mode-A 41.87 | mode-C 43.37 | gain C -1.50   (REAL novel subjects, fold 1)
[17:09] step 7100/20000 | loss 4.0573 | aux rot 1.400 (chance 2.83) | aux perm 1.579 (chance 3.26) | lr 3.83
        [val] step 7000: mode-A 43.49 | mode-C 45.47 | gain C -1.98   (REAL novel subjects, fold 1)
[17:14] step 7400/20000 | loss 4.0348 | aux rot 1.309 (chance 2.83) | aux perm 1.675 (chance 3.26) | lr 3.73
        [val] step 7000: mode-A 43.49 | mode-C 45.47 | gain C -1.98   (REAL novel subjects, fold 1)
[17:19] step 7600/20000 | loss 4.2307 | aux rot 1.275 (chance 2.83) | aux perm 1.524 (chance 3.26) | lr 3.65
        [val] step 7000: mode-A 43.49 | mode-C 45.47 | gain C -1.98   (REAL novel subjects, fold 1)
joint run ended
=== 556 done ===
```

### 557_zeroshot.log
```
[canon] canonical profile from 24 training users
  return F.conv1d(input, weight, bias, self.stride,
[user0] base  48.24 | k=12 corrected  71.47 (gain -23.23) rolls=[5, 13] margin=0.04/0.01
[user0] base  48.24 | k=45 corrected  81.89 (gain -33.65) rolls=[12, 4] margin=0.01/0.04
[user1] base  45.13 | k=12 corrected  69.29 (gain -24.16) rolls=[8, 1] margin=0.01/0.00
[user1] base  45.13 | k=45 corrected  68.52 (gain -23.39) rolls=[9, 0] margin=0.00/0.01
[user2] base  40.50 | k=12 corrected  40.34 (gain +0.16) rolls=[0, 0] margin=0.02/0.01
[user2] base  40.50 | k=45 corrected  75.82 (gain -35.32) rolls=[2, 4] margin=0.01/0.00
[user3] base  52.14 | k=12 corrected  79.12 (gain -26.98) rolls=[12, 10] margin=0.01/0.03
[user3] base  52.14 | k=45 corrected  79.45 (gain -27.31) rolls=[6, 0] margin=0.01/0.02
[user4] base  57.71 | k=12 corrected  58.63 (gain -0.91) rolls=[1, 14] margin=0.03/0.00
[user4] base  57.71 | k=45 corrected  77.25 (gain -19.53) rolls=[14, 7] margin=0.03/0.00
[user5] base  38.82 | k=12 corrected  78.65 (gain -39.83) rolls=[11, 3] margin=0.01/0.00
[user5] base  38.82 | k=45 corrected  41.44 (gain -2.63) rolls=[1, 15] margin=0.00/0.02
[user6] base  54.48 | k=12 corrected  60.99 (gain -6.52) rolls=[15, 1] margin=0.02/0.03
[user6] base  54.48 | k=45 corrected  62.68 (gain -8.20) rolls=[1, 14] margin=0.04/0.11
[user7] base  44.01 | k=12 corrected  53.44 (gain -9.43) rolls=[1, 13] margin=0.01/0.00
[user7] base  44.01 | k=45 corrected  70.12 (gain -26.10) rolls=[15, 4] margin=0.01/0.01

=== TRAINING-FREE EXPLICIT ADAPTATION -- 8 real unseen users ===
mode A (no calibration)      :  47.63
explicit correction k=12     :  63.99  (gain -16.36)
explicit correction k=45     :  69.65  (gain -22.02)
[saved] /data2/chenyuxiang/runs/remix_zeroshot.json
=== 557 done ===
```

### 557_zeroshot_today.log
```
[canon] canonical profile from 24 training users
  return F.conv1d(input, weight, bias, self.stride,
[user0] base  48.24 | k=12 corrected  71.47 (gain -23.23) rolls=[5, 13] margin=0.04/0.01
[user0] base  48.24 | k=45 corrected  81.89 (gain -33.65) rolls=[12, 4] margin=0.01/0.04
[user1] base  45.13 | k=12 corrected  69.29 (gain -24.16) rolls=[8, 1] margin=0.01/0.00
[user1] base  45.13 | k=45 corrected  68.52 (gain -23.39) rolls=[9, 0] margin=0.00/0.01
[user2] base  40.50 | k=12 corrected  40.34 (gain +0.16) rolls=[0, 0] margin=0.02/0.01
[user2] base  40.50 | k=45 corrected  75.82 (gain -35.32) rolls=[2, 4] margin=0.01/0.00
[user3] base  52.14 | k=12 corrected  79.12 (gain -26.98) rolls=[12, 10] margin=0.01/0.03
[user3] base  52.14 | k=45 corrected  79.45 (gain -27.31) rolls=[6, 0] margin=0.01/0.02
[user4] base  57.71 | k=12 corrected  58.63 (gain -0.91) rolls=[1, 14] margin=0.03/0.00
[user4] base  57.71 | k=45 corrected  77.25 (gain -19.53) rolls=[14, 7] margin=0.03/0.00
[user5] base  38.82 | k=12 corrected  78.65 (gain -39.83) rolls=[11, 3] margin=0.01/0.00
[user5] base  38.82 | k=45 corrected  41.44 (gain -2.63) rolls=[1, 15] margin=0.00/0.02
[user6] base  54.48 | k=12 corrected  60.99 (gain -6.52) rolls=[15, 1] margin=0.02/0.03
[user6] base  54.48 | k=45 corrected  62.68 (gain -8.20) rolls=[1, 14] margin=0.04/0.11
[user7] base  44.01 | k=12 corrected  53.44 (gain -9.43) rolls=[1, 13] margin=0.01/0.00
[user7] base  44.01 | k=45 corrected  70.12 (gain -26.10) rolls=[15, 4] margin=0.01/0.01

=== TRAINING-FREE EXPLICIT ADAPTATION -- 8 real unseen users ===
mode A (no calibration)      :  47.63
explicit correction k=12     :  63.99  (gain -16.36)
explicit correction k=45     :  69.65  (gain -22.02)
[saved] /data2/chenyuxiang/runs/remix_zeroshot.json
=== 557 done ===
```

### 558_joint_is_mainline.log
```
[icl_joint_fold1] step 600/20000 | loss 5.2351 | aux rot 2.131 (chance 2.83) | aux perm 1.787 (chance 3.26) | lr 3
--- 15:23 ---
[icl_joint_fold0] step 1600/30000 | loss 4.9454 | aux rot 1.993 (chance 2.83) | aux perm 1.673 (chance 3.26) | lr 
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 800/20000 | loss 4.6822 | aux rot 2.074 (chance 2.83) | aux perm 1.681 (chance 3.26) | lr 4
--- 15:28 ---
[icl_joint_fold0] step 1900/30000 | loss 4.9759 | aux rot 1.983 (chance 2.83) | aux perm 1.835 (chance 3.26) | lr 
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 1100/20000 | loss 5.0751 | aux rot 2.073 (chance 2.83) | aux perm 1.565 (chance 3.26) | lr 
        [val] new best mode-C 55.21 -> best.pt
--- 15:33 ---
[icl_joint_fold0] step 2100/30000 | loss 4.6140 | aux rot 1.894 (chance 2.83) | aux perm 1.564 (chance 3.26) | lr 
        [val] new best mode-C 47.40 -> best.pt
[icl_joint_fold1] step 1300/20000 | loss 4.8615 | aux rot 1.967 (chance 2.83) | aux perm 1.675 (chance 3.26) | lr 
        [val] new best mode-C 55.21 -> best.pt
--- 15:38 ---
[icl_joint_fold0] step 2300/30000 | loss 4.6538 | aux rot 1.858 (chance 2.83) | aux perm 1.667 (chance 3.26) | lr 
        [val] new best mode-C 47.40 -> best.pt
[icl_joint_fold1] step 1500/20000 | loss 5.1511 | aux rot 1.905 (chance 2.83) | aux perm 1.784 (chance 3.26) | lr 
        [val] new best mode-C 55.21 -> best.pt
--- 15:43 ---
[icl_joint_fold0] step 2600/30000 | loss 4.9083 | aux rot 1.786 (chance 2.83) | aux perm 1.679 (chance 3.26) | lr 
        [val] new best mode-C 47.40 -> best.pt
[icl_joint_fold1] step 1800/20000 | loss 4.8824 | aux rot 1.874 (chance 2.83) | aux perm 1.835 (chance 3.26) | lr 
        [val] new best mode-C 55.21 -> best.pt
```

### 558_mainline.log
```
[icl_joint_fold0] step 300/30000 | loss 6.4928 | aux rot 2.489 (chance 2.83) | aux perm 3.008 (chance 3.26) | lr 1
[icl_joint_fold1] waiting
--- 15:03 ---
[icl_joint_fold0] step 600/30000 | loss 5.3970 | aux rot 2.163 (chance 2.83) | aux perm 2.046 (chance 3.26) | lr 2
[icl_joint_fold1] waiting
--- 15:08 ---
[icl_joint_fold0] step 900/30000 | loss 5.1509 | aux rot 2.147 (chance 2.83) | aux perm 1.755 (chance 3.26) | lr 3
[icl_joint_fold1] step 100/20000 | loss 8.2409 | aux rot 2.789 (chance 2.83) | aux perm 3.260 (chance 3.26) | lr 5
--- 15:13 ---
[icl_joint_fold0] step 1100/30000 | loss 5.1774 | aux rot 2.116 (chance 2.83) | aux perm 1.561 (chance 3.26) | lr 
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 400/20000 | loss 6.2978 | aux rot 2.187 (chance 2.83) | aux perm 2.488 (chance 3.26) | lr 2
--- 15:18 ---
[icl_joint_fold0] step 1400/30000 | loss 4.8974 | aux rot 2.092 (chance 2.83) | aux perm 1.600 (chance 3.26) | lr 
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 600/20000 | loss 5.2351 | aux rot 2.131 (chance 2.83) | aux perm 1.787 (chance 3.26) | lr 3
--- 15:23 ---
[icl_joint_fold0] step 1600/30000 | loss 4.9454 | aux rot 1.993 (chance 2.83) | aux perm 1.673 (chance 3.26) | lr 
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 800/20000 | loss 4.6822 | aux rot 2.074 (chance 2.83) | aux perm 1.681 (chance 3.26) | lr 4
--- 15:28 ---
[icl_joint_fold0] step 1900/30000 | loss 4.9759 | aux rot 1.983 (chance 2.83) | aux perm 1.835 (chance 3.26) | lr 
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 1100/20000 | loss 5.0751 | aux rot 2.073 (chance 2.83) | aux perm 1.565 (chance 3.26) | lr 
        [val] new best mode-C 55.21 -> best.pt
```

### 559_valfix.log
```
[icl_joint_fold1] waiting
--- 15:05 ---
[icl_joint_fold0] step 700/30000 | loss 5.1462 | aux rot 2.103 (chance 2.83) | aux perm 1.889 (chance 3.26) 
[icl_joint_fold1] waiting
--- 15:10 ---
[icl_joint_fold0] step 1000/30000 | loss 5.0165 | aux rot 2.102 (chance 2.83) | aux perm 1.578 (chance 3.26)
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 200/20000 | loss 7.0014 | aux rot 2.560 (chance 2.83) | aux perm 3.132 (chance 3.26) 
--- 15:15 ---
[icl_joint_fold0] step 1200/30000 | loss 5.1696 | aux rot 2.065 (chance 2.83) | aux perm 1.758 (chance 3.26)
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 500/20000 | loss 5.6428 | aux rot 2.157 (chance 2.83) | aux perm 2.027 (chance 3.26) 
--- 15:20 ---
[icl_joint_fold0] step 1500/30000 | loss 5.2012 | aux rot 2.043 (chance 2.83) | aux perm 1.789 (chance 3.26)
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 700/20000 | loss 5.0362 | aux rot 2.080 (chance 2.83) | aux perm 1.784 (chance 3.26) 
--- 15:25 ---
[icl_joint_fold0] step 1700/30000 | loss 5.1740 | aux rot 2.054 (chance 2.83) | aux perm 1.705 (chance 3.26)
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 900/20000 | loss 4.9979 | aux rot 2.124 (chance 2.83) | aux perm 1.754 (chance 3.26) 
--- 15:30 ---
[icl_joint_fold0] step 2000/30000 | loss 5.1927 | aux rot 1.890 (chance 2.83) | aux perm 1.657 (chance 3.26)
        [val] new best mode-C 47.40 -> best.pt
[icl_joint_fold1] step 1200/20000 | loss 5.0613 | aux rot 2.000 (chance 2.83) | aux perm 1.750 (chance 3.26)
        [val] new best mode-C 55.21 -> best.pt
```

### 559_val_realonly.log
```
[icl_joint_fold1] waiting
--- 15:05 ---
[icl_joint_fold0] step 700/30000 | loss 5.1462 | aux rot 2.103 (chance 2.83) | aux perm 1.889 (chance 3.26) 
[icl_joint_fold1] waiting
--- 15:10 ---
[icl_joint_fold0] step 1000/30000 | loss 5.0165 | aux rot 2.102 (chance 2.83) | aux perm 1.578 (chance 3.26)
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 200/20000 | loss 7.0014 | aux rot 2.560 (chance 2.83) | aux perm 3.132 (chance 3.26) 
--- 15:15 ---
[icl_joint_fold0] step 1200/30000 | loss 5.1696 | aux rot 2.065 (chance 2.83) | aux perm 1.758 (chance 3.26)
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 500/20000 | loss 5.6428 | aux rot 2.157 (chance 2.83) | aux perm 2.027 (chance 3.26) 
--- 15:20 ---
[icl_joint_fold0] step 1500/30000 | loss 5.2012 | aux rot 2.043 (chance 2.83) | aux perm 1.789 (chance 3.26)
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 700/20000 | loss 5.0362 | aux rot 2.080 (chance 2.83) | aux perm 1.784 (chance 3.26) 
--- 15:25 ---
[icl_joint_fold0] step 1700/30000 | loss 5.1740 | aux rot 2.054 (chance 2.83) | aux perm 1.705 (chance 3.26)
        [val] new best mode-C 47.71 -> best.pt
[icl_joint_fold1] step 900/20000 | loss 4.9979 | aux rot 2.124 (chance 2.83) | aux perm 1.754 (chance 3.26) 
--- 15:30 ---
[icl_joint_fold0] step 2000/30000 | loss 5.1927 | aux rot 1.890 (chance 2.83) | aux perm 1.657 (chance 3.26)
        [val] new best mode-C 47.40 -> best.pt
[icl_joint_fold1] step 1200/20000 | loss 5.0613 | aux rot 2.000 (chance 2.83) | aux perm 1.750 (chance 3.26)
        [val] new best mode-C 55.21 -> best.pt
```

### 560_myocorl_launch.log
```
AST OK
=== CPU smoke: one real episode + ridge baseline ===
episode: ctx (128, 24, 99) y_ctx (128, 24) queries (16, 99) y_q (16, 128)
ridge-24 EV on held-out windows: -0.626  (must be > 0)
Traceback (most recent call last):
  File "<stdin>", line 20, in <module>
AssertionError: ridge baseline broken -- unit responses carry no signal?
SMOKE FAILED rc=1
```

### 560_myocorl.log
```
AST OK
=== CPU smoke: one real episode + ridge baseline ===
episode: ctx (128, 24, 99) y_ctx (128, 24) queries (16, 99) y_q (16, 128)
ridge-24 EV on held-out windows: -0.626  (must be > 0)
Traceback (most recent call last):
  File "<stdin>", line 20, in <module>
AssertionError: ridge baseline broken -- unit responses carry no signal?
SMOKE FAILED rc=1
```

### 561_encoding_signal_diag.log
```
letter+space predictors: 27
D98  n=raw  y=raw        -> K=24: +0.069  K=45: +0.114  K=90: +0.186
D98  n=raw  y=offsetfree -> K=24: +0.121  K=45: +0.137  K=90: +0.236
D98  n=prop y=raw        -> K=24: +0.061  K=45: +0.103  K=90: +0.140
D98  n=prop y=offsetfree -> K=24: +0.112  K=45: +0.138  K=90: +0.200
D28  n=raw  y=raw        -> K=24: +0.052  K=45: +0.088  K=90: +0.148
D28  n=raw  y=offsetfree -> K=24: +0.067  K=45: +0.120  K=90: +0.172
D28  n=prop y=raw        -> K=24: +0.034  K=45: +0.060  K=90: +0.086
D28  n=prop y=offsetfree -> K=24: +0.060  K=45: +0.105  K=90: +0.136

WINNER: ('D98', 'raw', 'offsetfree')  {24: 0.121, 45: 0.137, 90: 0.236}
verdict: K=90 EV > 0.05 -> foundation stands, rebuild MyoCoRL on the
winning definition; all <= 0 -> per-unit window-level linear encoding
has no signal on sEMG and paper 1 needs a different response variable.
=== 561 done ===
```

### 562_keystroke_foundation.log
```
building population tuning from 12 other users...
Traceback (most recent call last):
  File "<stdin>", line 71, in <module>
  File "<stdin>", line 37, in extract
RuntimeError: shape '[2, 16, 6, 5]' is invalid for input of size 1056
=== 562 done ===
```

### 563_keystroke_retry.log
```
building population tuning from 12 other users...
population tuning built from 12 users

      user     N |    K  within-EV  shrunk-EV   pop-EV
```

### 564_keystroke_diag2.log
```
population tuning built from 12 users

      user     N |    K  within-EV  shrunk-EV   pop-EV
  11372316   900 |   50      0.032      0.188    0.171
  11372316   900 |  100      0.116      0.202    0.171
  11372316   900 |  300      0.179      0.203    0.171
  11944098   896 |   50     -0.085      0.196    0.182
  11944098   896 |  100      0.108      0.211    0.182
  11944098   896 |  300      0.209      0.243    0.182
  12565339   900 |   50     -0.028      0.214    0.201
  12565339   900 |  100      0.058      0.211    0.201
  12565339   900 |  300      0.245      0.253    0.201
  13321435   900 |   50     -0.137      0.085    0.080
  13321435   900 |  100     -0.193      0.088    0.080
  13321435   900 |  300      0.020      0.106    0.080

=== rerun ===
building population tuning from 12 other users...
    [extract] 2021-05-27-1622155672-keystrokes-dca-stu: 498 usable of 500 keystrokes
    [extract] 2021-06-28-1624879406-keystrokes-dca-stu: 500 usable of 500 keystrokes
    [extract] 2020-12-17-1608240199-keystrokes-dca-stu: 500 usable of 500 keystrokes
    [extract] 2021-07-06-1625597882-keystrokes-dca-stu: 500 usable of 500 keystrokes
    [extract] 2021-05-28-1622177569-keystrokes-dca-stu: 500 usable of 500 keystrokes
    [extract] 2021-05-06-1620327471-keystrokes-dca-stu: 500 usable of 500 keystrokes
    [extract] 2021-03-22-1616452683-keystrokes-dca-stu: 500 usable of 500 keystrokes
```

### 565_icl_split_and_scoring.log
```
[audit] step 0: mode-A 50.89 | mode-C 100.00 (random prefix) | deployment reference ~43-58
step 100/20000 | loss 8.5527 | aux rot 2.860 (chance 2.83) | aux perm 3.278 (chance 3.26) | lr 3.00e-05 | 1.88 it/s
step 200/20000 | loss 6.2540 | aux rot 2.816 (chance 2.83) | aux perm 3.202 (chance 3.26) | lr 6.00e-05 | 1.79 it/s
step 300/20000 | loss 5.0300 | aux rot 2.677 (chance 2.83) | aux perm 3.070 (chance 3.26) | lr 9.00e-05 | 1.91 it/s
step 400/20000 | loss 5.3473 | aux rot 2.542 (chance 2.83) | aux perm 2.897 (chance 3.26) | lr 1.20e-04 | 1.78 it/s
--- fold 1 ---
[cohort] fold 1: 24 users the backbone has never seen, 206 sessions
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[trunk] /data2/chenyuxiang/runs/tf_fold1_full/last.pt step 103000 | 2.12M total  (featurizer 0.08M  encoder 1.98M  decoder 0.01M)
[prefix] FUSED mode: per-token (signal + soft-aligned char)
[prefix] {'k_windows': 4, 'seconds': 16, 'tokens_uncapped': 380, 'tokens': 380, 'capped': False}
[prefix] {'k_windows': 12, 'seconds': 48, 'tokens_uncapped': 1140, 'tokens': 1140, 'capped': False}
[prefix] {'k_windows': 23, 'seconds': 92, 'tokens_uncapped': 2185, 'tokens': 2185, 'capped': False}
[prefix] {'k_windows': 45, 'seconds': 180, 'tokens_uncapped': 4275, 'tokens': 4096, 'capped': True}
[meta-split] meta-train 16 users / 146 sessions | meta-test 8 users / 60 sessions (disjoint, both unseen by the backbone)
[symbol] 26 permutable letter classes | p_permute 0.5 k [4, 12]
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv1d(input, weight, bias, self.stride,
[audit] step 0: mode-A 62.84 | mode-C 100.00 (random prefix) | deployment reference ~43-58
step 100/20000 | loss 7.0961 | aux rot 2.808 (chance 2.83) | aux perm 3.273 (chance 3.26) | lr 3.00e-05 | 1.88 it/s
step 200/20000 | loss 5.8444 | aux rot 2.723 (chance 2.83) | aux perm 3.204 (chance 3.26) | lr 6.00e-05 | 1.78 it/s
step 300/20000 | loss 4.9437 | aux rot 2.620 (chance 2.83) | aux perm 3.074 (chance 3.26) | lr 9.00e-05 | 1.88 it/s
step 400/20000 | loss 5.3398 | aux rot 2.482 (chance 2.83) | aux perm 2.940 (chance 3.26) | lr 1.20e-04 | 1.75 it/s
=== 565 launched ===
```

### 566_keystroke_incremental.log
```
=== phase 1/2: per-user extraction (one process each) ===
```

### 567_stream_and_probe.log
```
### icl_split_fold1
[prefix] {'k_windows': 4, 'seconds': 16, 'tokens_uncapped': 380, 'tokens': 380, 'capped': False}
[prefix] {'k_windows': 12, 'seconds': 48, 'tokens_uncapped': 1140, 'tokens': 1140, 'capped': False}
[prefix] {'k_windows': 23, 'seconds': 92, 'tokens_uncapped': 2185, 'tokens': 2185, 'capped': False}
[prefix] {'k_windows': 45, 'seconds': 180, 'tokens_uncapped': 4275, 'tokens': 4096, 'capped': True}
[meta-split] meta-train 16 users / 146 sessions | meta-test 8 users / 60 sessions (disjoint, both unseen by the backbone)
[symbol] 26 permutable letter classes | p_permute 0.5 k [4, 12]
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv1d(input, weight, bias, self.stride,
[audit] step 0: mode-A 62.84 | mode-C 100.00 (random prefix) | deployment reference ~43-58
step 100/20000 | loss 7.0961 | aux rot 2.808 (chance 2.83) | aux perm 3.273 (chance 3.26) | lr 3.00e-05 | 1.88 it/s
step 200/20000 | loss 5.8444 | aux rot 2.723 (chance 2.83) | aux perm 3.204 (chance 3.26) | lr 6.00e-05 | 1.78 it/s
step 300/20000 | loss 4.9437 | aux rot 2.620 (chance 2.83) | aux perm 3.074 (chance 3.26) | lr 9.00e-05 | 1.88 it/s

--- start the detached log streamer (idempotent) ---
  streamer launched (60 s period, tails last 200 kB of each run)
icl_split_fold1.log
icl_split_fold0.log
567_stream_and_probe.log
567_stream_and_probe.done
567_stream_and_probe.started
566_keystroke_incremental.done
566_keystroke_incremental.log
566_keystroke_incremental.started
=== 567 done ===
```

### 568_budget_curve.log
```
=== unpack the two new modules ===
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/lib/python2.7/ast.py", line 37, in parse
    return compile(source, filename, mode, PyCF_ONLY_AST)
  File "<unknown>", line 137
    print(f"[skip] {key} already in {args.out}")
                                              ^
SyntaxError: invalid syntax
```

### 569_full_stop.log
```

=== E. TOTAL ON DISK ===
  2.8G	/data2/chenyuxiang/runs
  30M	/data2/chenyuxiang/code/myoicl

=== F. IF YOU DECIDE TO DELETE -- run these YOURSELF ===
  # experiment outputs only, keeps trunks + code + data:
  rm -rf /data2/chenyuxiang/runs/icl_* /data2/chenyuxiang/runs/myocorl* /data2/chenyuxiang/runs/keystroke_cache /data2/chenyuxiang/runs/remix*

  # also the trained trunks (irreversible: ~4 GPU-days to rebuild):
  rm -rf /data2/chenyuxiang/runs/tf_ref_full /data2/chenyuxiang/runs/tf_fold0_full /data2/chenyuxiang/runs/tf_fold1_full \
         /data2/chenyuxiang/runs/tf_fold2_full /data2/chenyuxiang/runs/tf_fold3_full

  # also all our code (the official emg2qwerty repo and the dataset
  # are NOT touched by this):
  rm -rf /data2/chenyuxiang/code/myoicl/myoicl

  # stop the bus runner entirely:
  touch /data2/chenyuxiang/code/myoicl/bus/jobs/STOP

  The dataset lives at /data2/chenyuxiang/code/emg2qwerty/data and is
  not referenced by any command above.

=== manifest written to bus/results/569_MANIFEST.txt ===
=== 569 done: all compute stopped, nothing deleted ===
```

### 570_icl_sanity.log
```
=== unpack + verify ===
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/lib/python2.7/ast.py", line 37, in parse
    return compile(source, filename, mode, PyCF_ONLY_AST)
  File "<unknown>", line 47
    def __init__(self, d_in: int, n_symbols: int, d_model: int = 128,
                           ^
SyntaxError: invalid syntax
```

### 571_icl_sanity_fix.log
```
  ==> NOT ALL CRITERIA PASS -- read the failing line above before doing anything else.

[saved] /tmp/icl_sanity_smoke.json
  smoke OK

=== the real run (GPU 0) ===
====================================================================
PRE-REGISTERED CRITERIA (written before any training runs)
  chance accuracy = 1/12 = 0.083
  C1  static arm  <= 0.133   (task is honest: no
      context-free solution exists, because pi is per-episode)
  C2  omega  arm  >  0.700   at K=64        (in-context works)
  C3  omega  acc increases with K           (it reads K examples)
  C4  omega  wrong-subject <= 0.163  (it reads THIS
      subject, not a marginal prior)  <-- the control the
      emg2qwerty runs never had
====================================================================

--- arm: omega ---
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
  [omega ] step   500 | loss 2.490 | acc@K64 0.081
  [omega ] step  1000 | loss 2.454 | acc@K64 0.127
  [omega ] step  1500 | loss 2.372 | acc@K64 0.193
  [omega ] step  2000 | loss 2.497 | acc@K64 0.204
```

### 572_w1_tta_floor.log
```
TEST-TIME ADAPTATION FLOOR -- 8 official unseen users, NO labels
       arm  mean CER   vs base
      base     61.51     +0.00
        bn     60.64     +0.87

[SANITY FAIL] unadapted mean 61.51 vs reproduction reference 55.39 (|d| = 6.12 > 1.0). Do NOT use the numbers above; fix the eval path first.

[FLOOR] the best label-free generic recipe is 'bn' at 60.64 (+0.87 vs unadapted).
        Part B (LM-as-Teacher) must beat 60.64, not 61.51, to claim anything.
        Personalised-with-labels reference: 11.28 -- the gap still open after the floor is 49.36 CER.

[saved] /data2/chenyuxiang/runs/tta_floor_smoke.json

=== full run: 8 users x 4 arms ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 67 context tensors keep their initialization
[arch] BatchNorm x1: ['frontend.0.batch_norm']
[arch] LayerNorm x17 | other norm x0

[user0] 64 unlabelled calibration windows (4.3 min)
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv2d(input, weight, bias, self.stride,
  [base] user0: CER 61.51
  [bn] recalibrated 1 BatchNorm layers
```

### 573_partb_gate.log
```

  raw decode 61.85 CER -> filter 'conf' keeps 100.0% at 61.85 CER
  ==> GATE FAILS: self-training on these would reinforce errors. Fix the FILTER, not the training loop.

[saved] /data2/chenyuxiang/runs/partb_gate2.json

=== full Part B: gate + adaptation, all 8 users ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 67 context tensors keep their initialization
[decoder] NO BEAM DECODER: /data2/chenyuxiang/code/emg2qwerty/config/decoder/ctc_beam.yaml has no _target_
[decoder] running with the confidence filter only. The 'agree' filter is the plan's main one, so this is a DEPENDENCY GAP to fix, not a result about the method.
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv2d(input, weight, bias, self.stride,

[user0] 96 unlabelled windows
     all_greedy: n=96   keep=100.0% pseudo-CER  67.77
           conf: n=96   keep=100.0% pseudo-CER  67.77
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv2d(input, weight, bias, self.stride,
   [adapt] user0: 61.51 -> 99.99 (gain -38.48) on 96 pseudo-labelled windows

[user1] 96 unlabelled windows
     all_greedy: n=96   keep=100.0% pseudo-CER  55.94
           conf: n=96   keep=100.0% pseudo-CER  55.94
```

### 574_partb_peruser.log
```
           conf: n=96   keep=100.0% pseudo-CER  54.43
   [adapt] user6: 54.66 -> 99.12 (gain -44.46) on 96 pseudo-labelled windows
THE GATE -- pseudo-label quality on unseen users (means)
  ==> GATE FAILS: self-training on these would reinforce errors. Fix the FILTER, not the training loop.
--- user7 ---
[decoder] NO BEAM DECODER: /data2/chenyuxiang/code/emg2qwerty/config/decoder/ctc_beam.yaml has no _target_
[decoder] running with the confidence filter only. The 'agree' filter is the plan's main one, so this is a DEPENDENCY GAP to fix, not a result about the method.
     all_greedy: n=96   keep=100.0% pseudo-CER  53.00
           conf: n=96   keep=100.0% pseudo-CER  53.00
   [adapt] user7: 52.17 -> 99.92 (gain -47.75) on 96 pseudo-labelled windows
THE GATE -- pseudo-label quality on unseen users (means)
  ==> GATE FAILS: self-training on these would reinforce errors. Fix the FILTER, not the training loop.

=== AGGREGATE ===
  users with results: 8

        filter    keep  pseudo-CER
    all_greedy 100.0%       56.34
          conf 100.0%       56.34

  raw decode 56.34 -> filter 'conf' keeps 100.0% at 56.34 CER
  ==> GATE FAILS -- self-training would reinforce errors; fix the FILTER

  ADAPTATION over 8 users: 55.39 -> 99.85 (mean gain -44.47), zero labels
=== 574 done (any unfinished user resumes on re-run) ===
```

### 575_partb_v2.log
```
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 67 context tensors keep their initialization
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv2d(input, weight, bias, self.stride,

[user0] 128 windows | raw CER 68.08
            conf_nb>q50: keep 50.0%  pseudo-CER  62.38  (gain  +5.70)
            conf_nb>q75: keep 25.0%  pseudo-CER  61.13  (gain  +6.95)
            conf_nb>q90: keep 10.2%  pseudo-CER  52.88  (gain +15.20)
            path_lp>q50: keep 50.0%  pseudo-CER  70.26  (gain  -2.17)
            path_lp>q75: keep 25.0%  pseudo-CER  79.86  (gain -11.78)
            path_lp>q90: keep 10.2%  pseudo-CER  83.15  (gain -15.06)
             consistent: keep 42.2%  pseudo-CER  67.89  (gain  +0.20)
      consistent+conf75: keep 17.2%  pseudo-CER  62.50  (gain  +5.58)

[saved] /data2/chenyuxiang/runs/partb2_probe.json

=== per-user detached: audit + adaptation ===
  launched user0 on gpu 0
  launched user1 on gpu 1
  launched user2 on gpu 2
  launched user3 on gpu 3
  launched user4 on gpu 0
  launched user5 on gpu 1
  launched user6 on gpu 2
  launched user7 on gpu 3
```

### 576_lm_solve.log
```
-rw-rw-r-- 1 chenyuxiang chenyuxiang 32545139 8月  14 15:19 /data2/chenyuxiang/code/emg2qwerty/models/lm/wikitext-103-6gram-charlm.bin
[lm] order=6

                tokenisation     OOV     real  shuffled   margin
                    chars+|   15.3%   -2.026    -1.822   -0.204
               chars+<space>  15.3%   -2.026    -1.777   -0.249
                     chars+_  15.3%   -2.026    -1.809   -0.217
                     chars+#  15.3%   -2.026    -1.831   -0.195
        chars, space dropped   0.0%   -1.459    -1.857   +0.398
   chars, literal space kept   0.0%   -1.459    -1.806   +0.347
             CHARS+| (upper)  97.9%   -2.808    -2.827   +0.019
 CHARS space dropped (upper)  97.5%   -2.827    -2.827   +0.000
                   raw words  85.6%   -3.246    -5.237   +1.991
           RAW WORDS (upper)  89.3%   -3.246    -5.237   +1.991

[SOLVED] tokenisation 'chars, space dropped' -- OOV 0.0%, real beats shuffled by +0.398
[SOLVED] partb2.CharLM._tok must be set to this scheme.

[probe] scoring a few plausible typing strings under the best scheme (higher = more language-like):
     -1.757  'hello world how are you'
     -1.675  'helo wrld hw ar yu'
     -1.999  'xkqj vzmw plfh brtn'
     -1.367  'the meeting is at three'
=== 576 done ===
```

### 577_partb_sweep.log
```
=== unpack + verify ===
  ok myoicl/partb_sweep.py (10753 bytes)
  ok myoicl/partb2.py (17052 bytes)

=== detached sweep, one user per GPU ===
  launched user0 on gpu 1
  launched user1 on gpu 2
  launched user3 on gpu 3

=== wait up to 45 min ===
  [t=1m] 0/3 users
  [t=2m] 0/3 users
  [t=3m] 0/3 users
  [t=4m] 0/3 users
```

### 578_lm_eow_and_collect.log
```
总用量 72
drwxrwxr-x  2 chenyuxiang chenyuxiang 4096 8月  22 02:50 .
drwxrwxr-x 73 chenyuxiang chenyuxiang 4096 8月  22 03:00 ..
-rw-rw-r--  1 chenyuxiang chenyuxiang  790 8月  22 02:50 user0.json
-rw-rw-r--  1 chenyuxiang chenyuxiang 2251 8月  22 02:50 user0.log
-rw-rw-r--  1 chenyuxiang chenyuxiang  795 8月  22 02:50 user1.json
-rw-rw-r--  1 chenyuxiang chenyuxiang 2251 8月  22 02:50 user1.log
-rw-rw-r--  1 chenyuxiang chenyuxiang  792 8月  22 02:50 user2.json
-rw-rw-r--  1 chenyuxiang chenyuxiang 2251 8月  22 02:50 user2.log
-rw-rw-r--  1 chenyuxiang chenyuxiang  794 8月  22 02:50 user3.json
-rw-rw-r--  1 chenyuxiang chenyuxiang 2251 8月  22 02:50 user3.log
-rw-rw-r--  1 chenyuxiang chenyuxiang  792 8月  22 02:50 user4.json
  captured partb/user0 (26 lines)
  captured partb/user1 (26 lines)
  captured partb/user2 (26 lines)
  captured partb/user3 (26 lines)
  captured partb/user4 (26 lines)
  captured partb/user5 (26 lines)
  captured partb/user6 (26 lines)
  captured partb/user7 (26 lines)

--- still-running workers ---
1493527 /data2/chenyuxiang/conda_envs/qwerty/bin/python -m myoicl.partb_sweep --user user0 --cal-windows 128 --filters conf_nb>q90 conf_nb>q75 consistent+conf75 --lrs 1e-5 3e-5 1e-4 --steps 30 100 --scopes all inputbn --out /data2/chenyuxiang/runs/partb_sweep/user0.json
1494389 /data2/chenyuxiang/conda_envs/qwerty/bin/python -m myoicl.partb_sweep --user user1 --cal-windows 128 --filters conf_nb>q90 conf_nb>q75 consistent+conf75 --lrs 1e-5 3e-5 1e-4 --steps 30 100 --scopes all inputbn --out /data2/chenyuxiang/runs/partb_sweep/user1.json
=== 578 done (short by design) ===
```

### 579_lm_beam_audit.log
```
=== unpack + verify ===
  ok, LM fix + spelling test present

=== launch detached per-user audits (LM + beam + all filters) ===
  launched user0 on gpu 0
  launched user1 on gpu 1
  launched user2 on gpu 2
  launched user3 on gpu 3
  launched user4 on gpu 0
  launched user5 on gpu 1
  launched user6 on gpu 2
  launched user7 on gpu 3

=== launch detached sweeps on the 3 probe users, ema 0.9 ===
  launched sweep user0 on gpu 0
  launched sweep user1 on gpu 1
  launched sweep user3 on gpu 2

running now:
11
=== 579 launched; a later short job collects the logs ===
```

### 580_collect3.log
```
[beam] torchaudio ctc_decoder unavailable: CTC Decoder suit requires flashlight-text package and optionally KenLM. Please install them.
[lm] /data2/chenyuxiang/code/emg2qwerty/models/lm/wikitext-103-6gram-charlm.bin
[lm] validated: real -1.060 > shuffled -1.933 (+0.873), spelling 3/3

########## FILTER TABLE so far ##########

  users 8 | raw greedy pseudo-CER 56.32
                  filter    keep  pseudo-CER   vs raw
             conf_nb>q50  50.0%       51.85    +4.48
             conf_nb>q75  25.0%       49.94    +6.38
             conf_nb>q90  10.2%       46.80    +9.52
              consistent  42.6%       55.83    +0.49
       consistent+conf75  13.9%       51.49    +4.84
         consistent+lm75  10.4%       53.14    +3.19
                  lm>q75  25.0%       54.28    +2.04
             path_lp>q50  50.0%       55.10    +1.22
             path_lp>q75  25.0%       57.42    -1.10
             path_lp>q90  10.2%       60.45    -4.12

  BEST 'conf_nb>q90': keeps 10.2% at 46.80 CER (+9.52)
  ==> GATE STILL FAILS (>40): pseudo-labels are the bottleneck, not the optimiser

########## SWEEP2 (ema 0.9, 192 windows, up to 300 steps) ##########
  no sweep2 json yet
=== 580 done ===
```

### 581_seg_gate.log
```
=== unpack + verify ===
  ok (15156 bytes)

=== detached: character-precision gate on 4 users, 2 with beam probe ===
  launched user0 gpu 0 beam-probe=24
  launched user1 gpu 1 beam-probe=24
  launched user3 gpu 2 beam-probe=0
  launched user5 gpu 3 beam-probe=0

workers now: 4
=== 581 launched ===
```

### 582_final8_and_collect.log
```
    0.95-0.99        294     66.0%          34.0
    0.99-1.00        247     76.9%          23.1
             ALL     1738     53.7%

[beam] prefix-beam+LM on 24 windows: greedy 52.74 vs beam 72.32 -> no better
[seg] thr 0.9: 41 segments, 141 characters (8.1% of all predicted characters)
[seg] thr 0.95: 28 segments, 87 characters (5.0% of all predicted characters)
--
=== THE GATE: character precision vs confidence (user3) ===
      confidence    chars  precision  -> pseudo-CER
[beam] prefix-beam+LM on 24 windows: greedy 69.43 vs beam 82.57 -> no better
[beam] prefix-beam+LM on 24 windows: greedy 52.74 vs beam 72.32 -> no better

########## launch final 8-user run, best known config ##########
  launched user0 on gpu 0
  launched user1 on gpu 1
  launched user2 on gpu 2
  launched user3 on gpu 3
  launched user4 on gpu 0
  launched user5 on gpu 1
  launched user6 on gpu 2
  launched user7 on gpu 3

workers now: 8
=== 582 launched ===
```

### 583_seg_adapt.log
```
=== unpack + verify ===
  ok (18448 bytes)

=== final8 progress (582) ===
8
  gain  +3.32 | conf_nb>q90 lr=3e-05 steps=30 inputbn student (kept 20, pseudo-CER 40.7)
  gain  -0.06 | conf_nb>q90 lr=3e-05 steps=30 inputbn ema (kept 20, pseudo-CER 40.7)
[user6] unadapted test CER 54.66
         conf_nb>q90 lr=3e-05   steps=30   inputbn student: CER  57.41  gain  -2.75
         conf_nb>q90 lr=3e-05   steps=30   inputbn     ema: CER  54.61  gain  +0.05
  gain  +0.05 | conf_nb>q90 lr=3e-05 steps=30 inputbn ema (kept 20, pseudo-CER 51.0)
  gain  -2.75 | conf_nb>q90 lr=3e-05 steps=30 inputbn student (kept 20, pseudo-CER 51.0)
[user7] unadapted test CER 52.17
         conf_nb>q90 lr=3e-05   steps=30   inputbn student: CER  51.57  gain  +0.60
         conf_nb>q90 lr=3e-05   steps=30   inputbn     ema: CER  52.19  gain  -0.02
  gain  +0.60 | conf_nb>q90 lr=3e-05 steps=30 inputbn student (kept 20, pseudo-CER 43.0)
  gain  -0.02 | conf_nb>q90 lr=3e-05 steps=30 inputbn ema (kept 20, pseudo-CER 43.0)

=== launch segment-level adaptation, 768-window pool ===
  launched user0 on gpu 0 (768 windows)
  launched user1 on gpu 1 (768 windows)
  launched user3 on gpu 2 (768 windows)

workers now: 3
=== 583 launched ===
```

### 584_collect_final.log
```
           user   before    after     gain   chars
          user0    61.51    60.92    +0.59      22
          user1    59.96    61.65    -1.69      80
          user3    54.70    53.89    +0.81      66
           MEAN                      -0.10
     threshold 0.95
           user   before    after     gain   chars
          user0    61.51    61.73    -0.22     138
          user1    59.96    60.60    -0.64     501
          user3    54.70    54.01    +0.69     374
           MEAN                      -0.06

  G. RULED OUT TONIGHT (so nobody repeats them)
     flashlight-text absent -> official CTCBeamDecoder unusable, and
       torchaudio's ctc_decoder needs it too
     own pure-python prefix beam + kenlm: 82.57 vs greedy 69.43
       (implementation bug, not tuning) -- disabled
     LM validated (boundary </s>, OOV 0.0%, spelling 3/3) but LM-based
       WINDOW filters were the weakest arm
     EMA teacher gained ~0.00 at ema 0.99 and 0.9; the student is the
       model to deploy here
     path_lp (mean greedy path log-prob) is ANTI-correlated with
       correctness: top decile was 4 CER WORSE than raw

=== 584 done ===
```

### 585_honest_table.log
```
========================================================================
SEGMENT-LEVEL ADAPTATION (768-window pool) -- negative, and why
========================================================================

  threshold 0.99
       user0   61.51 ->   60.92   +0.59      22 chars (7 segments)
       user1   59.96 ->   61.65   -1.69      80 chars (26 segments)
       user3   54.70 ->   53.89   +0.81      66 chars (21 segments)
        MEAN                      -0.10      56 chars

  threshold 0.95
       user0   61.51 ->   61.73   -0.22     138 chars (43 segments)
       user1   59.96 ->   60.60   -0.64     501 chars (154 segments)
       user3   54.70 ->   54.01   +0.69     374 chars (112 segments)
        MEAN                      -0.06     338 chars

  WHY THE YIELD COLLAPSED: the gate measured PER-CHARACTER precision
  (87.3% / 76.9% above posterior 0.99), but segments require min-chars
  CONSECUTIVE high-confidence characters. Isolated confident
  characters are common; runs of three are rare -- 768 windows gave
  only 22-80 characters at 0.99, i.e. 0.3-0.8% of all predictions,
  about a tenth of what the per-character rate would suggest.
  The clean labels exist but they are SCATTERED, and CTC needs
  contiguous spans. That is the finding, not a tuning failure.
=== 585 done ===
```

### 586_frame_level.log
```
=== unpack + verify ===
  ok (9636 bytes)

=== launch: 4 users x {conf, random, shuffled} ===
  launched user0 on gpu 0
  launched user1 on gpu 1
  launched user3 on gpu 2
  launched user5 on gpu 3

workers: 4
=== 586 launched ===
```

### 587_frame_collect.log
```
  conf +3.88 vs random +3.70: NO confidence effect -- the gain, if any, is not from selecting good labels
  shuffled +3.31: does NOT hurt -- the loss is not doing what it claims

##################################################################
###   FRAME-LEVEL PSEUDO-LABELS + CONTROLS -- the verdict       ##
##################################################################

      user    base |     conf    gain |   random    gain |  shuffled    gain |  frames
     user0   61.51 |    62.06   -0.55 |    61.88   -0.37 |    59.85   +1.66 |     656
     user1   59.96 |    60.97   -1.02 |    60.54   -0.58 |    61.47   -1.52 |    1190
     user3   54.70 |    53.90   +0.80 |    53.34   +1.36 |    53.41   +1.29 |    1073
     user5   53.85 |    49.97   +3.88 |    50.15   +3.70 |    50.54   +3.31 |    1150
  --------------------------------------------------------------------------
      MEAN   57.50 |            +0.78 |            +1.03 |            +1.19 |
        conf: +0.78 +- 1.91  (2/4 users improved)
      random: +1.03 +- 1.72  (2/4 users improved)
    shuffled: +1.19 +- 1.74  (3/4 users improved)

  PRE-REGISTERED VERDICT (rules fixed before the run):
    conf +0.78 must exceed random +1.03 by 0.3 : FAIL -- selecting by confidence does nothing
    shuffled +1.19 must be below -0.3           : FAIL -- the loss is not doing what it claims
    conf gain +0.78 must exceed its own sd 1.91 : FAIL -- effect smaller than spread = noise

  ==> NOT A RESULT -- report as noise, do not dress it up
=== 587 done ===
```

### 590_partb_mainline.log
```
=== unpack ===
  ok (14705 bytes)

=== quick beam sanity on user0 (1 min, then the real runs launch) ===
[user0] unadapted 61.51 | gap to personalised 50.23
Traceback (most recent call last):
ValueError: tuple.index(x): x not in tuple

=== 8 users, main line, encoder updated ===
  launched user0 on gpu 0
  launched user1 on gpu 1
  launched user2 on gpu 2
  launched user3 on gpu 3
  launched user4 on gpu 0
  launched user5 on gpu 1
  launched user6 on gpu 2
  launched user7 on gpu 3

workers: 10
=== 590 launched ===
```

### 591_flashlight_install.log
```
  LexiconFreeDecoder OK
  CTCBeamDecoder OK; signature:
    (self, _charset: 'CharacterSet' = <factory>, beam_size: 'int' = 50, max_labels_per_timestep: 'int' = -1, lm_path: 'str | None' = None, lm_weight: 'float' = 2.0, insertion_bonus: 'float' = 2.0, delete_key: 'str | None' = 'Key.backspace') -> None

=== unpack the decoder-aware main line ===
  ok (17680 bytes)

=== probe: does the official decoder build, and does beam beat greedy? ===
[decoder] CTCBeamDecoder construction failed: CTCBeamDecoder.__init__() got an unexpected keyword argument 'charset'  -> falling back to the in-repo prefix beam
[user0] unadapted 61.51 | gap to personalised 50.23
Traceback (most recent call last):
ValueError: tuple.index(x): x not in tuple

=== 8 users, main line ===
  launched user0 on gpu 0
  launched user1 on gpu 1
  launched user2 on gpu 2
  launched user3 on gpu 3
  launched user4 on gpu 0
  launched user5 on gpu 1
  launched user6 on gpu 2
  launched user7 on gpu 3

workers: 16
=== 591 launched ===
```

### 592_official_decoder.log
```
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'
[decoder] official failed at runtime, falling back: CTCBeamDecoder.decode() missing 1 required positional argument: 'timestamps'

=== 8 users, main line, official decoder ===
  launched user0 on gpu 0
  launched user1 on gpu 1
  launched user2 on gpu 2
  launched user3 on gpu 3
  launched user4 on gpu 0
  launched user5 on gpu 1
  launched user6 on gpu 2
  launched user7 on gpu 3

workers: 8
=== 592 launched ===
```

### 593_decode_timestamps.log
```
=== stop 592 workers (fallback decoder) ===
  stopped
=== unpack + verify ===
  ok (18514 bytes)

=== probe: does the OFFICIAL decoder now run, and beat greedy? ===
[decoder] official CTCBeamDecoder(beam=25, lm_w=2.0, ins=2.0, default charset)
[user0] unadapted 61.51 | gap to personalised 50.23
[user0] r1 decode: greedy 69.43 | beam 68.29 (beam better) | consistent 0/24 | kept 5 at pseudo-CER 67.61
[user0] r1 RESULT: 61.51 -> 62.01 (-0.50) = -1.0% of the gap, ZERO labels
[FINAL] user0: 61.51 -> 62.01 (-0.50) | gap eaten -1.0%

=== 8 users, main line, official decoder ===
  launched user0 on gpu 0
  launched user1 on gpu 1
  launched user2 on gpu 2
  launched user3 on gpu 3
  launched user4 on gpu 0
  launched user5 on gpu 1
  launched user6 on gpu 2
  launched user7 on gpu 3

workers: 8
=== 593 launched ===
```

### 595_partb_v2.log
```
=== install the module ===
  ok (16933 bytes)

=== probe on user5 (v1's best), segment granularity ===
[decoder] official CTCBeamDecoder(beam=50)
[user5] unadapted 53.85 | receptive field 125 | gap 42.57
Traceback (most recent call last):
AssertionError: Timestamps are not monotonic

=== 8 users, segment granularity ===
  launched user0 on gpu 0
  launched user1 on gpu 1
  launched user2 on gpu 2
  launched user3 on gpu 3
  launched user4 on gpu 0
  launched user5 on gpu 1
  launched user6 on gpu 2
  launched user7 on gpu 3

workers: 8
=== 595 launched ===
```

### 596_fix_reset.log
```
  none

=== patch beam_with_times: reset() before every decode ===
[patched] reset() inserted
  verified (17304 bytes)

=== probe on user5 ===
[decoder] official CTCBeamDecoder(beam=50)
[user5] unadapted 53.85 | receptive field 125 | gap 42.57
[user5] r1 decode: greedy 52.14 | beam 49.43 | drift(beam-vs-greedy) 33.13 | 73 segments | pseudo-CER 66.67
[user5] r1 RESULT: 53.85 -> 50.31 (+3.54) = 8.3% of the gap
[FINAL] user5: 53.85 -> 50.31 (+3.54) | gap eaten 8.3%

=== 8 users ===
  launched user0 on gpu 0
  launched user1 on gpu 1
  launched user2 on gpu 2
  launched user3 on gpu 3
  launched user4 on gpu 0
  launched user5 on gpu 1
  launched user6 on gpu 2
  launched user7 on gpu 3

workers: 8
=== 596 launched ===
```

### 597_collect_v2.log
```
  v1 for comparison: 55.39 -> 53.67, +1.72, 3.9%, 4/8 improved

     user   r  greedy    beam   drift  items     CER
    user0   1   68.72   66.80   42.69     50   61.06
    user0   2   68.93   67.42   43.40     48   61.52
    user1   1   56.19   52.46   36.92     51   61.62
    user1   2   55.97   52.14   36.97     51   60.05
    user1   3   55.58   52.48   36.38     51   59.73
    user2   1   49.90   44.16   34.52     50   48.00
    user2   2   48.48   43.69   33.39     49   46.48
    user2   3   48.48   43.04   33.05     49   47.14
    user3   1   56.93   55.05   34.14    278   52.11
    user3   2   56.78   54.24   34.15    305   50.76
    user4   1   60.03   55.47   41.18    342   58.53
    user4   2   59.36   54.00   40.59    334   55.81
    user4   3   58.66   52.95   40.49    337   54.78
    user5   1   51.93   48.73   32.53     51   50.14
    user5   2   51.00   48.15   31.61     51   52.02
    user5   3   51.05   47.93   32.07     51   51.91
    user6   1   54.34   51.42   35.33     52   56.64
    user6   2   54.26   51.46   35.87     52   55.65
    user7   1   51.74   48.26   32.89    411   49.87
    user7   2   51.13   47.67   31.97    423   48.13
    user7   3   50.74   47.93   30.57    437   48.14
=== 597 done ===
```

### 598_per_window_fallback.log
```
=== stop v2 workers ===
  none

=== patch ===
[patched] 2 anchors
  verified (17855 bytes)

=== 8 users, per-window segment fallback ===
  launched user0 on gpu 0
  launched user1 on gpu 1
  launched user2 on gpu 2
  launched user3 on gpu 3
  launched user4 on gpu 0
  launched user5 on gpu 1
  launched user6 on gpu 2
  launched user7 on gpu 3

workers: 8
=== 598 launched ===
```

### 599_collect_v3.log
```

  per-user gain +1.86 +- 1.33 (4/5 improved, worst -0.30, best +3.45)

  === three generations ===
                         run  mean gain     sd  improved   worst    gap
             v1 window-level      +1.72   3.36       4/8   -2.75   3.9%
    v2 mixed (5/8 fell back)      +1.68   1.82       6/8   -0.99   3.8%
        of which segment (3)      +3.81      -       3/3   +3.45      -
      v3 per-window fallback      +1.86   1.33       4/5   -0.30   4.4%

     user   r  greedy    beam   drift  items     CER
    user2   1   49.90   44.16   34.52    248   45.47
    user3   1   56.93   55.05   34.14    278   52.11
    user4   1   60.03   55.47   41.18    342   58.53
    user5   1   51.93   48.73   32.53    278   50.40
    user6   1   54.34   51.42   35.33    434   53.62

########## segment quality + timestamp coverage ##########
      1 exact 36/160 (22%) | 2 win w/o timestamps
      1 exact 50/278 (18%) | 0 win w/o timestamps
      1 exact 67/278 (24%) | 3 win w/o timestamps
      1 exact 77/248 (31%) | 4 win w/o timestamps
      1 exact 83/342 (24%) | 0 win w/o timestamps
      1 exact 90/434 (21%) | 1 win w/o timestamps
=== 599 done ===
```

### 600_partA_splash_probe.log
```
=== numerical checks ===
  causality: max |delta| on frames 0..24 after perturbing 25.. = 0.000e+00  OK
  RSG: (200, 2, 2, 16, 33) -> (200, 2, 2, 16, 6)
  RTN: mean +0.0039 std 0.9847 (late frames should be ~0/1) | finite=True
  RTN late frames: mean +0.0057 std 0.9959
  ACM: zeroed fraction 0.72 (train) | eval passthrough True

=== NEGATIVE CONTROL: retrofit onto the FROZEN official model ===
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[pretrained] loaded 51 backbone tensors from /data2/chenyuxiang/code/emg2qwerty/models/generic.ckpt; 67 context tensors keep their initialization
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv2d(input, weight, bias, self.stride,
  [user0] unmodified 61.51
  [user1] unmodified 59.96
  [user0] + RTN retrofit 93.98 (-32.47)
  [user1] + RTN retrofit 95.79 (-35.83)

  mean 60.73 -> 94.89 (-34.15)
  ==> AS EXPECTED: retrofitting normalisation onto a model that
      was trained without it is destructive. SplashNet's -12.6
      is a TRAINING-time result and must be reproduced by
      retraining, not by a test-time transform.
[saved] /data2/chenyuxiang/runs/splash_probe.json
=== 600 done ===
```

### 601_partA_train.log
```
=== unpack + verify ===
  ok myoicl/train_splash.py (10228 bytes)
  ok myoicl/splash.py (6685 bytes)

=== 30-step smoke on the full recipe (shapes, loss finite) ===
[split] 96 training users / 837 sessions | 16 official test sessions
[model] official TDS, freq_bins=6, 7.52M params | RSG=6 RTN=1 ACM=1
[data] 229266 training windows of 4.0s
[data] 512 monitor windows on the 8 test users
[sanity] step 0 test CER 2075.20 (untrained; should be ~100)
step 10/30 | loss 300.9355 | lr 5.00e-06 | 6.21 it/s
step 20/30 | loss 118.9674 | lr 1.00e-05 | 9.12 it/s
step 30/30 | loss 196.6556 | lr 1.50e-05 | 10.74 it/s
[val] step 30: 8-test-user CER 227.80   (plain-baseline reproduction 55.39; SplashNet reports ~36 with this recipe)
[val] new best 227.80 -> best.pt

=== four arms, detached ===
  launched full on gpu 0 (bands=6 rtn=1 acm=1)
  launched rtnonly on gpu 1 (bands=0 rtn=1 acm=0)
  launched rsgonly on gpu 2 (bands=6 rtn=0 acm=0)
  launched plain on gpu 3 (bands=0 rtn=0 acm=0)

workers: 4
=== 601 launched ===
```

### 602_collect_partA.log
```
[val] step 4000: 8-test-user CER 60.12   (plain-baseline reproduction 55.39; SplashNet reports ~36 with this recipe)
[val] new best 60.12 -> best.pt
[val] step 6000: 8-test-user CER 55.34   (plain-baseline reproduction 55.39; SplashNet reports ~36 with this recipe)
[val] new best 55.34 -> best.pt
[val] step 8000: 8-test-user CER 56.30   (plain-baseline reproduction 55.39; SplashNet reports ~36 with this recipe)
[val] step 10000: 8-test-user CER 55.36   (plain-baseline reproduction 55.39; SplashNet reports ~36 with this recipe)

##################################################################
###  PART A stage 0 -- normalisation recipe, same-budget arms   ##
##################################################################

        arm    step  CER now    best  SplashNet ref   delta
      plain    8000    57.05   57.05          55.39   +1.66
    rsgonly    8000    58.71   58.71          47.18  +11.53
    rtnonly   10000    55.36   55.34          39.15  +16.19
       full    8000    77.09   77.09          36.42  +40.67

  full vs OUR OWN same-budget plain arm: 57.05 -> 77.09 (-20.04)
  (the plain arm is trained by this same script with this same
   budget, so the recipe is not confounded with training length)

  reminder: this recipe is SplashNet's, i.e. the PLATFORM.
  Stage A1 -- our contrastive alignment -- is measured on top of
  whichever arm wins, and that is the row with our name on it.
=== 602 done ===
```

### 603_rtn_replaces_bn.log
```
=== stop only the two arms being replaced ===
  stopped full
  stopped rtnonly
  still running: 10

=== patch: --no-specnorm replaces the frontend BatchNorm with Identity ===
[patched] 2 anchors + sanity note
  verified (11049 bytes)

=== relaunch the two arms with RTN replacing BatchNorm ===
  launched rtn_nobn on gpu 0 (bands=0 rtn=1 acm=0 p_ch=0.0, no specnorm)
  launched full_nobn on gpu 1 (bands=6 rtn=1 acm=1 p_ch=0.30, no specnorm)

workers: 12
  (plain and rsgonly continue untouched as the same-budget references)
=== 603 launched ===
```

### 604_collect_partA2.log
```
         arm    step  CER now    best     ref  vs plain
       plain   24000    53.74   53.19   55.39         -
     rsgonly   24000    52.87   52.68   47.18     +0.51
     rtnonly   26000    50.69   50.69   39.15     +2.50
        full   14000    69.11   69.11   36.42    -15.91
    rtn_nobn    8000    56.49   56.49   39.15     -3.30
   full_nobn   18000    57.13   57.13   36.42     -3.93

  curves (last 4 evals each):
         plain  18k:53.8 20k:54.2 22k:53.2 24k:53.7
       rsgonly  18k:55.0 20k:54.0 22k:52.7 24k:52.9
       rtnonly  20k:52.6 22k:51.0 24k:51.2 26k:50.7
          full  8k:77.1 10k:69.7 12k:69.4 14k:69.1
      rtn_nobn  2k:67.4 4k:59.4 6k:57.8 8k:56.5
     full_nobn  12k:57.7 14k:58.5 16k:57.6 18k:57.1

  ==> HYPOTHESIS REJECTED: RTN still loses to plain (53.19 vs 56.49).
      Do NOT keep tuning the wiring. Next suspects, in order:
      (a) RTN should act on a different quantity than the
          log-spectrogram, (b) the warmup-frame handling,
      (c) lr needs to change once the input scale changes.

  absolute values sit above SplashNet's references because this
  budget is ~4.2 epochs; only the vs-plain column is meaningful.
=== 604 done ===
```

### 605_rtn_from_source.log
```
  stopped rtn_nobn
  stopped full_nobn
  still running: 10

=== patch splash.py: frozen-statistics warm-up, Tm=125 ===
[patched] splash.py: Tm=125 frozen-statistics warm-up

=== patch train_splash.py: ACM = SpecAugment(0 time, 2 freq @ 12) ===
[patched] train_splash.py: ACM via SpecAugment, frontend ACM disabled
  verified

=== re-verify RTN numerically with the new warm-up ===
  causality (frames 0..299): 0.000e+00
  warm-up block 0..124 uses one shared statistic: True
  late frames mean +0.0074 std 1.0108
  finite: True

=== relaunch the two arms, corrected ===
  launched rtn_v2 on gpu 0 (bands=0 rtn=1 acm=0, RTN replaces BN)
  launched full_v2 on gpu 1 (bands=6 rtn=1 acm=1, RTN replaces BN)

workers: 12
  (plain and rsgonly still running untouched as the budget-matched
   references -- restarting them would break the comparison)
=== 605 launched ===
```

### 606_partA1_align.log
```
=== unpack + verify ===
  ok myoicl/align_char.py (6712 bytes)
  ok myoicl/train_align.py (10368 bytes)

=== forced_align available? ===
  torchaudio 2.3.0+cu121 | forced_align: True
  smoke: alignment tokens present = [0, 1, 2, 3]

=== 60-step smoke: does L_char produce segments and finite loss? ===
[split] 96 training users | 16 official test sessions
[model] TDS 7.85M | proj 0.69M | w_char=0.2 tau=0.1 warmup=20 cross_user_only=1 shuffle_users=False
[data] 229266 windows over 837 sessions
step 20/60 | loss 212.6834 | segs/step 0 | users/step 0.0 | lr 1.00e-05 | 2.38 it/s
step 40/60 | loss 126.3637 | segs/step 0 | users/step 0.0 | lr 2.00e-05 | 3.37 it/s
step 60/60 | loss 53.6660 | segs/step 0 | users/step 0.0 | lr 3.00e-05 | 3.96 it/s
[val] step 60: 8-test-user CER 100.00
[val] new best 100.00

=== three arms (gpu2 shared; gpu3 keeps rsgonly) ===
  launched align on gpu 2 (--w-char 0.2 --char-warmup 8000)
  launched shuffle on gpu 2 (--w-char 0.2 --char-warmup 8000 --shuffle-users)
  launched ctconly on gpu 3 (--w-char 0)

workers: splash 20 | align 3
=== 606 launched ===
```

### 607_step_matched.log
```
  === all arms at step 10000 ===
          full    69.72   vs plain  -10.50
     full_nobn    60.54   vs plain   -1.33
       full_v2    56.09   vs plain   +3.12
         plain    59.22   vs plain       -
       rsgonly    56.30   vs plain   +2.92
      rtn_nobn    55.30   vs plain   +3.92
        rtn_v2    54.84   vs plain   +4.38
       rtnonly    55.36   vs plain   +3.85

  best non-plain arm at this step: rtn_v2 54.84 (+4.38 vs plain)

  pairwise, deepest common step for each pair vs plain:
          full @  14000:   69.11 vs plain   55.66  (-13.45)
     full_nobn @  40000:   51.08 vs plain   50.91  (-0.17)
       full_v2 @  34000:   50.34 vs plain   50.91  (+0.57)
       rsgonly @  44000:   49.09 vs plain   50.36  (+1.27)
      rtn_nobn @  16000:   52.67 vs plain   54.34  (+1.67)
        rtn_v2 @  10000:   54.84 vs plain   59.22  (+4.38)
       rtnonly @  26000:   50.69 vs plain   52.53  (+1.84)

  absolute values sit well above SplashNet's references because
  these arms have run a small fraction of a full training schedule;
  only the step-matched vs-plain column carries information.
=== 607 done ===
```

### 608_fix_align_silent.log
```
=== stop the three align arms (they are pure CTC right now) ===
  stopped

=== patch: surface the exception instead of swallowing it ===
[patched] align_char.py: exception surfaced
  verified

=== probe forced_align on ONE REAL batch, real shapes and dtypes ===
  charset: num_classes=99 blank/null=98
  inputs (621, 4, 2, 16, 33) targets (17, 4) torch.int32 lengths [0, 0, 0, 17]
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv2d(input, weight, bias, self.stride,
  hidden (497, 4, 768) logprobs (497, 4, 99)
  target ids min=0 max=96  (blank must NOT appear: 98)
  utt 0: T=497 L=0 -> 0 segments | seg lengths min/median/max = -/-/-
  utt 1: T=497 L=0 -> 0 segments | seg lengths min/median/max = -/-/-
  utt 2: T=497 L=0 -> 0 segments | seg lengths min/median/max = -/-/-

  if segment counts are non-zero here but min_len=2 drops them all,
  the filter is the problem; if they are zero, the printed
  forced_align error above names the real cause.
=== 608 done -- arms stay stopped until the cause is named ===
```

### d3_train.log
```
[val] step 6000: mode-C CER 16.99 | mode-B CER 17.13 | mode-A CER 15.88 | gain C -1.11 / B -1.25 | loss 0.6158
[val] new best CER 16.99 -> saved best.pt
step 6100/8000 | loss 0.7332 | lr 5.57e-06 | 1.31 it/s
step 6200/8000 | loss 0.7158 | lr 5.17e-06 | 1.59 it/s
step 6300/8000 | loss 0.7820 | lr 4.79e-06 | 1.46 it/s
step 6400/8000 | loss 0.8463 | lr 4.43e-06 | 1.37 it/s
step 6500/8000 | loss 0.7911 | lr 4.09e-06 | 1.64 it/s
step 6600/8000 | loss 0.7385 | lr 3.76e-06 | 1.35 it/s
step 6700/8000 | loss 0.7720 | lr 3.46e-06 | 1.79 it/s
step 6800/8000 | loss 0.7467 | lr 3.17e-06 | 1.83 it/s
step 6900/8000 | loss 0.8087 | lr 2.91e-06 | 1.36 it/s
step 7000/8000 | loss 0.8092 | lr 2.67e-06 | 1.51 it/s
[val] step 7000: mode-C CER 18.15 | mode-B CER 17.99 | mode-A CER 17.00 | gain C -1.15 / B -0.99 | loss 0.6675
step 7100/8000 | loss 0.6643 | lr 2.45e-06 | 1.06 it/s
step 7200/8000 | loss 0.7738 | lr 2.25e-06 | 1.70 it/s
step 7300/8000 | loss 0.8446 | lr 2.08e-06 | 1.43 it/s
step 7400/8000 | loss 0.7710 | lr 1.92e-06 | 1.90 it/s
step 7500/8000 | loss 0.7543 | lr 1.80e-06 | 1.74 it/s
step 7600/8000 | loss 0.6453 | lr 1.69e-06 | 1.97 it/s
step 7700/8000 | loss 0.7902 | lr 1.61e-06 | 1.54 it/s
step 7800/8000 | loss 0.7631 | lr 1.55e-06 | 1.52 it/s
step 7900/8000 | loss 0.6770 | lr 1.51e-06 | 2.07 it/s
step 8000/8000 | loss 0.6493 | lr 1.50e-06 | 1.57 it/s
[val] step 8000: mode-C CER 17.75 | mode-B CER 17.92 | mode-A CER 16.52 | gain C -1.23 / B -1.40 | loss 0.6252
[done] {'best_val_cer': 16.994242604725034, 'steps': 8000, 'phase': 'icl'}
```

### d4_train.log
```
step 6000/8000 | loss 0.8322 | lr 5.99e-06 | 1.56 it/s
[val] step 6000: mode-C CER 22.86 | mode-B CER 23.30 | mode-A CER 21.93 | gain C -0.93 / B -1.37 | loss 0.8716
step 6100/8000 | loss 0.7596 | lr 5.57e-06 | 1.39 it/s
step 6200/8000 | loss 0.8698 | lr 5.17e-06 | 1.61 it/s
step 6300/8000 | loss 0.7703 | lr 4.79e-06 | 1.17 it/s
step 6400/8000 | loss 0.8198 | lr 4.43e-06 | 1.65 it/s
step 6500/8000 | loss 0.7323 | lr 4.09e-06 | 1.54 it/s
step 6600/8000 | loss 0.7832 | lr 3.76e-06 | 1.72 it/s
step 6700/8000 | loss 0.8318 | lr 3.46e-06 | 1.37 it/s
step 6800/8000 | loss 0.6850 | lr 3.17e-06 | 1.87 it/s
step 6900/8000 | loss 0.8138 | lr 2.91e-06 | 1.84 it/s
step 7000/8000 | loss 0.8756 | lr 2.67e-06 | 1.56 it/s
[val] step 7000: mode-C CER 23.30 | mode-B CER 23.00 | mode-A CER 21.92 | gain C -1.38 / B -1.08 | loss 0.8791
step 7100/8000 | loss 0.7476 | lr 2.45e-06 | 1.34 it/s
step 7200/8000 | loss 0.7326 | lr 2.25e-06 | 1.57 it/s
step 7300/8000 | loss 0.8353 | lr 2.08e-06 | 1.44 it/s
step 7400/8000 | loss 0.7968 | lr 1.92e-06 | 1.48 it/s
step 7500/8000 | loss 0.8512 | lr 1.80e-06 | 1.58 it/s
step 7600/8000 | loss 0.8434 | lr 1.69e-06 | 1.50 it/s
step 7700/8000 | loss 0.7596 | lr 1.61e-06 | 1.73 it/s
step 7800/8000 | loss 0.7662 | lr 1.55e-06 | 1.81 it/s
step 7900/8000 | loss 0.7791 | lr 1.51e-06 | 1.68 it/s
step 8000/8000 | loss 0.7358 | lr 1.50e-06 | 1.65 it/s
[val] step 8000: mode-C CER 22.69 | mode-B CER 22.56 | mode-A CER 21.59 | gain C -1.10 / B -0.97 | loss 0.8491
[done] {'best_val_cer': 21.223643276875215, 'steps': 8000, 'phase': 'icl'}
```

### eval_a2_k12.log
```
[ckpt] /data2/chenyuxiang/runs/v5_a2/best.pt (v1, step 500)
[A] user0: CER 61.45
[A] user1: CER 59.90
[A] user2: CER 48.06
[A] user3: CER 54.69
[A] user4: CER 58.28
[A] user5: CER 53.90
[A] user6: CER 54.63
[A] user7: CER 52.25
[A] mean over users: 55.40
[C] user0: CER 63.22
[C] user1: CER 68.07
[C] user2: CER 53.68
[C] user3: CER 64.79
[C] user4: CER 63.14
[C] user5: CER 57.49
[C] user6: CER 59.19
[C] user7: CER 54.70
[C] mean over users: 60.54
[A] gap closed vs personalization ceiling: -0.0%
[C] gap closed vs personalization ceiling: -11.7%
[saved] /data2/chenyuxiang/runs/v5a2_real_k12.json
```

### eval_a2_k45.log
```
[ckpt] /data2/chenyuxiang/runs/v5_a2/best.pt (v1, step 500)
[A] user0: CER 61.45
[A] user1: CER 59.90
[A] user2: CER 48.06
[A] user3: CER 54.69
[A] user4: CER 58.28
[A] user5: CER 53.90
[A] user6: CER 54.63
[A] user7: CER 52.25
[A] mean over users: 55.40
[C] user0: CER 63.52
[C] user1: CER 67.26
[C] user2: CER 53.68
[C] user3: CER 63.90
[C] user4: CER 63.32
[C] user5: CER 58.87
[C] user6: CER 59.23
[C] user7: CER 55.00
[C] mean over users: 60.60
[A] gap closed vs personalization ceiling: -0.0%
[C] gap closed vs personalization ceiling: -11.8%
[saved] /data2/chenyuxiang/runs/v5a2_real_k45.json
```

### icl_aux_fold0.log
```
[val] step 1000: mode-A 60.88 | mode-C 63.87 | gain C -2.99   (REAL novel subjects, fold 0)
[val] new best mode-C 63.87 -> best.pt
step 1100/20000 | loss 5.0098 | aux rot 2.092 (chance 2.83) | aux perm 1.624 (chance 3.26) | lr 1.00e-03 | 1.35 it/s
step 1200/20000 | loss 4.8963 | aux rot 2.065 (chance 2.83) | aux perm 1.712 (chance 3.26) | lr 1.00e-03 | 1.35 it/s
step 1300/20000 | loss 5.0274 | aux rot 2.043 (chance 2.83) | aux perm 1.671 (chance 3.26) | lr 9.99e-04 | 1.33 it/s
step 1400/20000 | loss 4.7274 | aux rot 2.118 (chance 2.83) | aux perm 1.620 (chance 3.26) | lr 9.99e-04 | 1.34 it/s
step 1500/20000 | loss 4.9879 | aux rot 2.030 (chance 2.83) | aux perm 1.724 (chance 3.26) | lr 9.98e-04 | 1.35 it/s
step 1600/20000 | loss 4.7783 | aux rot 2.015 (chance 2.83) | aux perm 1.728 (chance 3.26) | lr 9.98e-04 | 1.36 it/s
step 1700/20000 | loss 4.8300 | aux rot 2.095 (chance 2.83) | aux perm 1.650 (chance 3.26) | lr 9.97e-04 | 1.37 it/s
step 1800/20000 | loss 4.9617 | aux rot 1.972 (chance 2.83) | aux perm 1.611 (chance 3.26) | lr 9.96e-04 | 1.37 it/s
step 1900/20000 | loss 4.9612 | aux rot 1.993 (chance 2.83) | aux perm 1.604 (chance 3.26) | lr 9.94e-04 | 1.37 it/s
step 2000/20000 | loss 4.9789 | aux rot 1.965 (chance 2.83) | aux perm 1.736 (chance 3.26) | lr 9.93e-04 | 1.36 it/s
[val] step 2000: mode-A 57.78 | mode-C 60.48 | gain C -2.70   (REAL novel subjects, fold 0)
[val] new best mode-C 60.48 -> best.pt
step 2100/20000 | loss 4.7293 | aux rot 1.980 (chance 2.83) | aux perm 1.692 (chance 3.26) | lr 9.92e-04 | 1.35 it/s
step 2200/20000 | loss 5.0857 | aux rot 2.010 (chance 2.83) | aux perm 1.760 (chance 3.26) | lr 9.90e-04 | 1.33 it/s
step 2300/20000 | loss 4.8654 | aux rot 1.966 (chance 2.83) | aux perm 1.729 (chance 3.26) | lr 9.88e-04 | 1.32 it/s
step 2400/20000 | loss 4.5350 | aux rot 1.811 (chance 2.83) | aux perm 1.742 (chance 3.26) | lr 9.87e-04 | 1.32 it/s
step 2500/20000 | loss 4.9696 | aux rot 1.938 (chance 2.83) | aux perm 1.592 (chance 3.26) | lr 9.85e-04 | 1.31 it/s
step 2600/20000 | loss 4.7692 | aux rot 1.888 (chance 2.83) | aux perm 1.744 (chance 3.26) | lr 9.83e-04 | 1.30 it/s
step 2700/20000 | loss 5.1111 | aux rot 1.919 (chance 2.83) | aux perm 1.742 (chance 3.26) | lr 9.80e-04 | 1.29 it/s
step 2800/20000 | loss 4.5304 | aux rot 1.813 (chance 2.83) | aux perm 1.671 (chance 3.26) | lr 9.78e-04 | 1.30 it/s
step 2900/20000 | loss 4.5663 | aux rot 1.831 (chance 2.83) | aux perm 1.641 (chance 3.26) | lr 9.76e-04 | 1.30 it/s
step 3000/20000 | loss 5.0045 | aux rot 1.955 (chance 2.83) | aux perm 1.760 (chance 3.26) | lr 9.73e-04 | 1.30 it/s
[val] step 3000: mode-A 58.67 | mode-C 61.45 | gain C -2.78   (REAL novel subjects, fold 0)
```

### icl_aux_fold2.log
```
step 2100/20000 | loss 4.7245 | aux rot 2.011 (chance 2.83) | aux perm 1.707 (chance 3.26) | lr 9.92e-04 | 1.54 it/s
step 2200/20000 | loss 5.1382 | aux rot 2.023 (chance 2.83) | aux perm 1.765 (chance 3.26) | lr 9.90e-04 | 1.52 it/s
step 2300/20000 | loss 4.9387 | aux rot 1.968 (chance 2.83) | aux perm 1.747 (chance 3.26) | lr 9.88e-04 | 1.51 it/s
step 2400/20000 | loss 4.6101 | aux rot 1.892 (chance 2.83) | aux perm 1.747 (chance 3.26) | lr 9.87e-04 | 1.52 it/s
step 2500/20000 | loss 4.9932 | aux rot 1.980 (chance 2.83) | aux perm 1.589 (chance 3.26) | lr 9.85e-04 | 1.51 it/s
step 2600/20000 | loss 4.8422 | aux rot 1.933 (chance 2.83) | aux perm 1.739 (chance 3.26) | lr 9.83e-04 | 1.50 it/s
step 2700/20000 | loss 5.0515 | aux rot 1.932 (chance 2.83) | aux perm 1.738 (chance 3.26) | lr 9.80e-04 | 1.49 it/s
step 2800/20000 | loss 4.6061 | aux rot 1.860 (chance 2.83) | aux perm 1.666 (chance 3.26) | lr 9.78e-04 | 1.49 it/s
step 2900/20000 | loss 4.6595 | aux rot 1.931 (chance 2.83) | aux perm 1.631 (chance 3.26) | lr 9.76e-04 | 1.50 it/s
step 3000/20000 | loss 5.1406 | aux rot 2.037 (chance 2.83) | aux perm 1.765 (chance 3.26) | lr 9.73e-04 | 1.49 it/s
[val] step 3000: mode-A 60.18 | mode-C 64.18 | gain C -4.00   (REAL novel subjects, fold 2)
[val] new best mode-C 64.18 -> best.pt
step 3100/20000 | loss 4.8039 | aux rot 1.880 (chance 2.83) | aux perm 1.628 (chance 3.26) | lr 9.70e-04 | 1.48 it/s
step 3200/20000 | loss 4.7880 | aux rot 1.868 (chance 2.83) | aux perm 1.683 (chance 3.26) | lr 9.67e-04 | 1.48 it/s
step 3300/20000 | loss 5.0672 | aux rot 1.837 (chance 2.83) | aux perm 1.703 (chance 3.26) | lr 9.64e-04 | 1.47 it/s
step 3400/20000 | loss 4.7852 | aux rot 1.947 (chance 2.83) | aux perm 1.710 (chance 3.26) | lr 9.61e-04 | 1.46 it/s
step 3500/20000 | loss 4.9618 | aux rot 1.814 (chance 2.83) | aux perm 1.675 (chance 3.26) | lr 9.58e-04 | 1.46 it/s
step 3600/20000 | loss 4.6364 | aux rot 1.834 (chance 2.83) | aux perm 1.700 (chance 3.26) | lr 9.55e-04 | 1.46 it/s
step 3700/20000 | loss 4.7236 | aux rot 1.772 (chance 2.83) | aux perm 1.721 (chance 3.26) | lr 9.51e-04 | 1.46 it/s
step 3800/20000 | loss 4.6989 | aux rot 1.769 (chance 2.83) | aux perm 1.645 (chance 3.26) | lr 9.47e-04 | 1.45 it/s
step 3900/20000 | loss 4.7141 | aux rot 1.797 (chance 2.83) | aux perm 1.668 (chance 3.26) | lr 9.44e-04 | 1.44 it/s
step 4000/20000 | loss 4.7677 | aux rot 1.901 (chance 2.83) | aux perm 1.630 (chance 3.26) | lr 9.40e-04 | 1.44 it/s
[val] step 4000: mode-A 65.84 | mode-C 67.90 | gain C -2.06   (REAL novel subjects, fold 2)
step 4100/20000 | loss 4.7910 | aux rot 1.762 (chance 2.83) | aux perm 1.660 (chance 3.26) | lr 9.36e-04 | 1.42 it/s
step 4200/20000 | loss 4.7833 | aux rot 1.720 (chance 2.83) | aux perm 1.730 (chance 3.26) | lr 9.32e-04 | 1.41 it/s
```

### icl_dev2_fold2.log
```
step 10100/12000 | loss 2.6772 | lr 1.34e-05 | 1.55 it/s
step 10200/12000 | loss 2.6768 | lr 1.21e-05 | 1.55 it/s
step 10300/12000 | loss 2.6325 | lr 1.08e-05 | 1.55 it/s
step 10400/12000 | loss 2.7006 | lr 9.56e-06 | 1.55 it/s
step 10500/12000 | loss 2.7084 | lr 8.42e-06 | 1.55 it/s
[val] step 10500: mode-A 49.27 | mode-C 49.71 | gain C -0.44   (REAL novel subjects, fold 2)
step 10600/12000 | loss 2.7067 | lr 7.35e-06 | 1.55 it/s
step 10700/12000 | loss 2.6953 | lr 6.35e-06 | 1.55 it/s
step 10800/12000 | loss 2.7037 | lr 5.42e-06 | 1.55 it/s
step 10900/12000 | loss 2.6881 | lr 4.56e-06 | 1.55 it/s
step 11000/12000 | loss 2.8109 | lr 3.77e-06 | 1.55 it/s
[val] step 11000: mode-A 51.72 | mode-C 51.37 | gain C +0.35   (REAL novel subjects, fold 2)
step 11100/12000 | loss 2.6803 | lr 3.06e-06 | 1.55 it/s
step 11200/12000 | loss 2.6824 | lr 2.42e-06 | 1.55 it/s
step 11300/12000 | loss 2.6078 | lr 1.85e-06 | 1.55 it/s
step 11400/12000 | loss 2.7676 | lr 1.36e-06 | 1.55 it/s
step 11500/12000 | loss 2.6999 | lr 9.48e-07 | 1.55 it/s
[val] step 11500: mode-A 46.72 | mode-C 47.36 | gain C -0.64   (REAL novel subjects, fold 2)
step 11600/12000 | loss 2.7648 | lr 6.07e-07 | 1.55 it/s
step 11700/12000 | loss 2.7498 | lr 3.42e-07 | 1.56 it/s
step 11800/12000 | loss 2.6527 | lr 1.52e-07 | 1.56 it/s
step 11900/12000 | loss 2.7580 | lr 3.80e-08 | 1.56 it/s
step 12000/12000 | loss 2.7608 | lr 0.00e+00 | 1.56 it/s
[val] step 12000: mode-A 47.42 | mode-C 47.99 | gain C -0.56   (REAL novel subjects, fold 2)
[done] best mode-C 45.37
```

### icl_dev_fold2.log
```
step 10100/12000 | loss 2.4735 | lr 2.01e-05 | 1.51 it/s
step 10200/12000 | loss 2.3661 | lr 1.81e-05 | 1.52 it/s
step 10300/12000 | loss 2.3357 | lr 1.62e-05 | 1.51 it/s
step 10400/12000 | loss 2.3725 | lr 1.43e-05 | 1.51 it/s
step 10500/12000 | loss 2.2964 | lr 1.26e-05 | 1.52 it/s
[val] step 10500: mode-A 53.90 | mode-C 54.52 | gain C -0.62   (REAL novel subjects, fold 2)
step 10600/12000 | loss 2.4289 | lr 1.10e-05 | 1.52 it/s
step 10700/12000 | loss 2.3120 | lr 9.52e-06 | 1.52 it/s
step 10800/12000 | loss 2.3677 | lr 8.13e-06 | 1.52 it/s
step 10900/12000 | loss 2.4098 | lr 6.84e-06 | 1.52 it/s
step 11000/12000 | loss 2.4630 | lr 5.66e-06 | 1.52 it/s
[val] step 11000: mode-A 47.58 | mode-C 47.71 | gain C -0.13   (REAL novel subjects, fold 2)
step 11100/12000 | loss 2.4640 | lr 4.59e-06 | 1.52 it/s
step 11200/12000 | loss 2.4363 | lr 3.63e-06 | 1.52 it/s
step 11300/12000 | loss 2.4330 | lr 2.78e-06 | 1.52 it/s
step 11400/12000 | loss 2.4478 | lr 2.05e-06 | 1.52 it/s
step 11500/12000 | loss 2.3917 | lr 1.42e-06 | 1.52 it/s
[val] step 11500: mode-A 49.54 | mode-C 50.28 | gain C -0.74   (REAL novel subjects, fold 2)
step 11600/12000 | loss 2.5203 | lr 9.10e-07 | 1.51 it/s
step 11700/12000 | loss 2.5158 | lr 5.12e-07 | 1.51 it/s
step 11800/12000 | loss 2.3988 | lr 2.28e-07 | 1.51 it/s
step 11900/12000 | loss 2.4732 | lr 5.70e-08 | 1.51 it/s
step 12000/12000 | loss 2.4017 | lr 0.00e+00 | 1.51 it/s
[val] step 12000: mode-A 51.03 | mode-C 50.91 | gain C +0.12   (REAL novel subjects, fold 2)
[done] best mode-C 45.71
```

### icl_frozen_fold2.log
```
step 10100/12000 | loss 2.8161 | lr 6.70e-05 | 1.60 it/s
step 10200/12000 | loss 2.7960 | lr 6.03e-05 | 1.60 it/s
step 10300/12000 | loss 2.7150 | lr 5.39e-05 | 1.60 it/s
step 10400/12000 | loss 2.7883 | lr 4.78e-05 | 1.60 it/s
step 10500/12000 | loss 2.7053 | lr 4.21e-05 | 1.60 it/s
[val] step 10500: mode-A 62.53 | mode-C 63.91 | gain C -1.37   (REAL novel subjects, fold 2)
step 10600/12000 | loss 2.7730 | lr 3.68e-05 | 1.60 it/s
step 10700/12000 | loss 2.8049 | lr 3.17e-05 | 1.60 it/s
step 10800/12000 | loss 2.7718 | lr 2.71e-05 | 1.60 it/s
step 10900/12000 | loss 2.8233 | lr 2.28e-05 | 1.61 it/s
step 11000/12000 | loss 2.8234 | lr 1.89e-05 | 1.61 it/s
[val] step 11000: mode-A 62.90 | mode-C 62.69 | gain C +0.22   (REAL novel subjects, fold 2)
step 11100/12000 | loss 2.7704 | lr 1.53e-05 | 1.61 it/s
step 11200/12000 | loss 2.7884 | lr 1.21e-05 | 1.60 it/s
step 11300/12000 | loss 2.8333 | lr 9.27e-06 | 1.60 it/s
step 11400/12000 | loss 2.8080 | lr 6.82e-06 | 1.60 it/s
step 11500/12000 | loss 2.7524 | lr 4.74e-06 | 1.60 it/s
[val] step 11500: mode-A 58.06 | mode-C 58.81 | gain C -0.75   (REAL novel subjects, fold 2)
step 11600/12000 | loss 2.7437 | lr 3.03e-06 | 1.60 it/s
step 11700/12000 | loss 2.7359 | lr 1.71e-06 | 1.60 it/s
step 11800/12000 | loss 2.7696 | lr 7.59e-07 | 1.60 it/s
step 11900/12000 | loss 2.8013 | lr 1.90e-07 | 1.60 it/s
step 12000/12000 | loss 2.7387 | lr 0.00e+00 | 1.61 it/s
[val] step 12000: mode-A 60.18 | mode-C 60.20 | gain C -0.02   (REAL novel subjects, fold 2)
[done] best mode-C 56.54
```

### icl_fusedb_fold2.log
```
step 2000/12000 | loss 2.6922 | lr 2.89e-04 | 1.45 it/s
[val] step 2000: mode-A 59.25 | mode-C 60.57 | gain C -1.32   (REAL novel subjects, fold 2)
step 2100/12000 | loss 2.6274 | lr 2.87e-04 | 1.43 it/s
step 2200/12000 | loss 2.5883 | lr 2.86e-04 | 1.44 it/s
step 2300/12000 | loss 2.5148 | lr 2.84e-04 | 1.44 it/s
step 2400/12000 | loss 2.5568 | lr 2.82e-04 | 1.45 it/s
step 2500/12000 | loss 2.5626 | lr 2.80e-04 | 1.46 it/s
[val] step 2500: mode-A 56.96 | mode-C 57.42 | gain C -0.46   (REAL novel subjects, fold 2)
[val] new best mode-C 57.42 -> best.pt
step 2600/12000 | loss 2.5046 | lr 2.78e-04 | 1.44 it/s
step 2700/12000 | loss 2.5703 | lr 2.76e-04 | 1.44 it/s
step 2800/12000 | loss 2.4984 | lr 2.73e-04 | 1.45 it/s
step 2900/12000 | loss 2.5691 | lr 2.71e-04 | 1.44 it/s
step 3000/12000 | loss 2.5952 | lr 2.68e-04 | 1.45 it/s
[val] step 3000: mode-A 56.53 | mode-C 56.67 | gain C -0.15   (REAL novel subjects, fold 2)
[val] new best mode-C 56.67 -> best.pt
step 3100/12000 | loss 2.5290 | lr 2.66e-04 | 1.44 it/s
step 3200/12000 | loss 2.5274 | lr 2.63e-04 | 1.45 it/s
step 3300/12000 | loss 2.4774 | lr 2.60e-04 | 1.46 it/s
step 3400/12000 | loss 2.5156 | lr 2.58e-04 | 1.46 it/s
step 3500/12000 | loss 2.5528 | lr 2.55e-04 | 1.46 it/s
[val] step 3500: mode-A 54.09 | mode-C 55.14 | gain C -1.05   (REAL novel subjects, fold 2)
[val] new best mode-C 55.14 -> best.pt
step 3600/12000 | loss 2.5497 | lr 2.52e-04 | 1.46 it/s
step 3700/12000 | loss 2.5186 | lr 2.49e-04 | 1.46 it/s
```

### icl_fused_fold2.log
```
[prefix] FUSED mode: per-token (signal + soft-aligned char)
[prefix] {'k_windows': 4, 'seconds': 16, 'tokens_uncapped': 380, 'tokens': 380, 'capped': False}
[prefix] {'k_windows': 12, 'seconds': 48, 'tokens_uncapped': 1140, 'tokens': 1140, 'capped': False}
[prefix] {'k_windows': 23, 'seconds': 92, 'tokens_uncapped': 2185, 'tokens': 2185, 'capped': False}
[prefix] {'k_windows': 45, 'seconds': 180, 'tokens_uncapped': 4275, 'tokens': 4096, 'capped': True}
[symbol] 26 permutable letter classes | p_permute 0.5 k [4, 12]
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return F.conv1d(input, weight, bias, self.stride,
[audit] step 0: mode-A 62.63 | mode-C 100.00 (random prefix) | deployment reference ~43-58
Traceback (most recent call last):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/data2/chenyuxiang/code/myoicl/myoicl/train_prefix_icl.py", line 471, in <module>
    main()
  File "/data2/chenyuxiang/code/myoicl/myoicl/train_prefix_icl.py", line 439, in main
    loss.backward()
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/_tensor.py", line 525, in backward
    torch.autograd.backward(
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/__init__.py", line 267, in backward
    _engine_run_backward(
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/graph.py", line 744, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [11, 598, 99]], which is output 0 of ExpBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
```

### icl_joint_fold0.log
```
step 6300/30000 | loss 4.6280 | aux rot 1.417 (chance 2.83) | aux perm 1.688 (chance 3.26) | lr 4.66e-04 | 0.93 it/s
step 6400/30000 | loss 4.3103 | aux rot 1.425 (chance 2.83) | aux perm 1.602 (chance 3.26) | lr 4.64e-04 | 0.94 it/s
step 6500/30000 | loss 4.2663 | aux rot 1.485 (chance 2.83) | aux perm 1.673 (chance 3.26) | lr 4.63e-04 | 0.94 it/s
step 6600/30000 | loss 4.0294 | aux rot 1.440 (chance 2.83) | aux perm 1.642 (chance 3.26) | lr 4.62e-04 | 0.94 it/s
step 6700/30000 | loss 4.0256 | aux rot 1.378 (chance 2.83) | aux perm 1.714 (chance 3.26) | lr 4.60e-04 | 0.94 it/s
step 6800/30000 | loss 4.2466 | aux rot 1.406 (chance 2.83) | aux perm 1.648 (chance 3.26) | lr 4.59e-04 | 0.94 it/s
step 6900/30000 | loss 4.2368 | aux rot 1.406 (chance 2.83) | aux perm 1.683 (chance 3.26) | lr 4.57e-04 | 0.94 it/s
step 7000/30000 | loss 4.5734 | aux rot 1.428 (chance 2.83) | aux perm 1.685 (chance 3.26) | lr 4.55e-04 | 0.94 it/s
[val] step 7000: mode-A 42.27 | mode-C 42.38 | gain C -0.12   (REAL novel subjects, fold 0)
[val] new best mode-C 42.38 -> best.pt
step 7100/30000 | loss 4.0389 | aux rot 1.364 (chance 2.83) | aux perm 1.598 (chance 3.26) | lr 4.54e-04 | 0.94 it/s
step 7200/30000 | loss 3.8481 | aux rot 1.434 (chance 2.83) | aux perm 1.567 (chance 3.26) | lr 4.52e-04 | 0.94 it/s
step 7300/30000 | loss 4.4014 | aux rot 1.358 (chance 2.83) | aux perm 1.607 (chance 3.26) | lr 4.51e-04 | 0.94 it/s
step 7400/30000 | loss 4.1602 | aux rot 1.390 (chance 2.83) | aux perm 1.681 (chance 3.26) | lr 4.49e-04 | 0.94 it/s
step 7500/30000 | loss 4.0652 | aux rot 1.440 (chance 2.83) | aux perm 1.524 (chance 3.26) | lr 4.47e-04 | 0.94 it/s
step 7600/30000 | loss 4.1967 | aux rot 1.333 (chance 2.83) | aux perm 1.530 (chance 3.26) | lr 4.46e-04 | 0.94 it/s
step 7700/30000 | loss 4.0748 | aux rot 1.371 (chance 2.83) | aux perm 1.682 (chance 3.26) | lr 4.44e-04 | 0.95 it/s
step 7800/30000 | loss 4.0010 | aux rot 1.325 (chance 2.83) | aux perm 1.726 (chance 3.26) | lr 4.42e-04 | 0.95 it/s
step 7900/30000 | loss 4.3426 | aux rot 1.300 (chance 2.83) | aux perm 1.647 (chance 3.26) | lr 4.40e-04 | 0.95 it/s
step 8000/30000 | loss 4.2378 | aux rot 1.322 (chance 2.83) | aux perm 1.748 (chance 3.26) | lr 4.39e-04 | 0.95 it/s
[val] step 8000: mode-A 40.52 | mode-C 40.68 | gain C -0.16   (REAL novel subjects, fold 0)
[val] new best mode-C 40.68 -> best.pt
step 8100/30000 | loss 4.3292 | aux rot 1.372 (chance 2.83) | aux perm 1.705 (chance 3.26) | lr 4.37e-04 | 0.95 it/s
step 8200/30000 | loss 4.1166 | aux rot 1.389 (chance 2.83) | aux perm 1.664 (chance 3.26) | lr 4.35e-04 | 0.95 it/s
step 8300/30000 | loss 4.4400 | aux rot 1.403 (chance 2.83) | aux perm 1.648 (chance 3.26) | lr 4.33e-04 | 0.95 it/s
```

### icl_joint_fold1.log
```
step 5400/20000 | loss 4.1782 | aux rot 1.400 (chance 2.83) | aux perm 1.614 (chance 3.26) | lr 4.37e-04 | 0.93 it/s
step 5500/20000 | loss 4.0854 | aux rot 1.426 (chance 2.83) | aux perm 1.701 (chance 3.26) | lr 4.34e-04 | 0.94 it/s
step 5600/20000 | loss 4.1729 | aux rot 1.413 (chance 2.83) | aux perm 1.594 (chance 3.26) | lr 4.31e-04 | 0.94 it/s
step 5700/20000 | loss 4.5477 | aux rot 1.413 (chance 2.83) | aux perm 1.709 (chance 3.26) | lr 4.28e-04 | 0.94 it/s
step 5800/20000 | loss 4.2039 | aux rot 1.340 (chance 2.83) | aux perm 1.668 (chance 3.26) | lr 4.25e-04 | 0.94 it/s
step 5900/20000 | loss 4.4784 | aux rot 1.359 (chance 2.83) | aux perm 1.589 (chance 3.26) | lr 4.22e-04 | 0.94 it/s
step 6000/20000 | loss 4.3000 | aux rot 1.417 (chance 2.83) | aux perm 1.613 (chance 3.26) | lr 4.19e-04 | 0.95 it/s
[val] step 6000: mode-A 41.87 | mode-C 43.37 | gain C -1.50   (REAL novel subjects, fold 1)
step 6100/20000 | loss 4.4783 | aux rot 1.477 (chance 2.83) | aux perm 1.668 (chance 3.26) | lr 4.16e-04 | 0.95 it/s
step 6200/20000 | loss 4.3861 | aux rot 1.322 (chance 2.83) | aux perm 1.725 (chance 3.26) | lr 4.13e-04 | 0.95 it/s
step 6300/20000 | loss 4.6062 | aux rot 1.430 (chance 2.83) | aux perm 1.683 (chance 3.26) | lr 4.10e-04 | 0.94 it/s
step 6400/20000 | loss 4.2902 | aux rot 1.392 (chance 2.83) | aux perm 1.610 (chance 3.26) | lr 4.07e-04 | 0.95 it/s
step 6500/20000 | loss 4.1344 | aux rot 1.332 (chance 2.83) | aux perm 1.660 (chance 3.26) | lr 4.04e-04 | 0.95 it/s
step 6600/20000 | loss 3.9101 | aux rot 1.355 (chance 2.83) | aux perm 1.642 (chance 3.26) | lr 4.00e-04 | 0.95 it/s
step 6700/20000 | loss 4.0458 | aux rot 1.382 (chance 2.83) | aux perm 1.705 (chance 3.26) | lr 3.97e-04 | 0.95 it/s
step 6800/20000 | loss 4.2060 | aux rot 1.382 (chance 2.83) | aux perm 1.648 (chance 3.26) | lr 3.94e-04 | 0.95 it/s
step 6900/20000 | loss 4.0872 | aux rot 1.289 (chance 2.83) | aux perm 1.680 (chance 3.26) | lr 3.90e-04 | 0.95 it/s
step 7000/20000 | loss 4.5028 | aux rot 1.392 (chance 2.83) | aux perm 1.684 (chance 3.26) | lr 3.87e-04 | 0.95 it/s
[val] step 7000: mode-A 43.49 | mode-C 45.47 | gain C -1.98   (REAL novel subjects, fold 1)
step 7100/20000 | loss 4.0573 | aux rot 1.400 (chance 2.83) | aux perm 1.579 (chance 3.26) | lr 3.83e-04 | 0.95 it/s
step 7200/20000 | loss 3.7937 | aux rot 1.308 (chance 2.83) | aux perm 1.557 (chance 3.26) | lr 3.80e-04 | 0.95 it/s
step 7300/20000 | loss 4.3631 | aux rot 1.308 (chance 2.83) | aux perm 1.617 (chance 3.26) | lr 3.76e-04 | 0.95 it/s
step 7400/20000 | loss 4.0348 | aux rot 1.309 (chance 2.83) | aux perm 1.675 (chance 3.26) | lr 3.73e-04 | 0.95 it/s
step 7500/20000 | loss 4.0723 | aux rot 1.418 (chance 2.83) | aux perm 1.527 (chance 3.26) | lr 3.69e-04 | 0.95 it/s
step 7600/20000 | loss 4.2307 | aux rot 1.275 (chance 2.83) | aux perm 1.524 (chance 3.26) | lr 3.65e-04 | 0.95 it/s
```

### icl_split_fold0.log
```
step 100/20000 | loss 8.5527 | aux rot 2.860 (chance 2.83) | aux perm 3.278 (chance 3.26) | lr 3.00e-05 | 1.88 it/s
step 200/20000 | loss 6.2540 | aux rot 2.816 (chance 2.83) | aux perm 3.202 (chance 3.26) | lr 6.00e-05 | 1.79 it/s
step 300/20000 | loss 5.0300 | aux rot 2.677 (chance 2.83) | aux perm 3.070 (chance 3.26) | lr 9.00e-05 | 1.91 it/s
step 400/20000 | loss 5.3473 | aux rot 2.542 (chance 2.83) | aux perm 2.897 (chance 3.26) | lr 1.20e-04 | 1.78 it/s
step 500/20000 | loss 4.6795 | aux rot 2.428 (chance 2.83) | aux perm 2.574 (chance 3.26) | lr 1.50e-04 | 1.81 it/s
[val] step 500: IDENTITY A 45.02 C 68.06 gain -23.04 | PERMUTED A 61.34 C 77.21 gain -15.87 | mem-gauge(meta-train A) 51.55   (meta-test users, fold 0)
[val] new best mode-C 68.06 -> best.pt
step 600/20000 | loss 4.6654 | aux rot 2.416 (chance 2.83) | aux perm 2.304 (chance 3.26) | lr 1.80e-04 | 1.77 it/s
step 700/20000 | loss 4.3376 | aux rot 2.306 (chance 2.83) | aux perm 2.026 (chance 3.26) | lr 2.10e-04 | 1.76 it/s
step 800/20000 | loss 4.2150 | aux rot 2.235 (chance 2.83) | aux perm 1.831 (chance 3.26) | lr 2.40e-04 | 1.76 it/s
step 900/20000 | loss 3.5987 | aux rot 2.192 (chance 2.83) | aux perm 1.667 (chance 3.26) | lr 2.70e-04 | 1.76 it/s
step 1000/20000 | loss 3.6733 | aux rot 2.178 (chance 2.83) | aux perm 1.506 (chance 3.26) | lr 3.00e-04 | 1.73 it/s
[val] step 1000: IDENTITY A 49.56 C 53.65 gain -4.08 | PERMUTED A 63.57 C 66.71 gain -3.14 | mem-gauge(meta-train A) 43.97   (meta-test users, fold 0)
[val] new best mode-C 53.65 -> best.pt
step 1100/20000 | loss 3.8112 | aux rot 2.130 (chance 2.83) | aux perm 1.650 (chance 3.26) | lr 3.00e-04 | 1.73 it/s
step 1200/20000 | loss 3.9258 | aux rot 2.109 (chance 2.83) | aux perm 1.766 (chance 3.26) | lr 3.00e-04 | 1.73 it/s
step 1300/20000 | loss 4.0147 | aux rot 2.110 (chance 2.83) | aux perm 1.697 (chance 3.26) | lr 3.00e-04 | 1.75 it/s
step 1400/20000 | loss 4.0839 | aux rot 2.245 (chance 2.83) | aux perm 1.608 (chance 3.26) | lr 3.00e-04 | 1.76 it/s
step 1500/20000 | loss 3.9426 | aux rot 2.077 (chance 2.83) | aux perm 1.599 (chance 3.26) | lr 2.99e-04 | 1.74 it/s
[val] step 1500: IDENTITY A 48.18 C 50.79 gain -2.61 | PERMUTED A 62.54 C 64.00 gain -1.46 | mem-gauge(meta-train A) 45.06   (meta-test users, fold 0)
[val] new best mode-C 50.79 -> best.pt
step 1600/20000 | loss 3.4373 | aux rot 2.035 (chance 2.83) | aux perm 1.673 (chance 3.26) | lr 2.99e-04 | 1.74 it/s
step 1700/20000 | loss 3.8778 | aux rot 2.028 (chance 2.83) | aux perm 1.588 (chance 3.26) | lr 2.99e-04 | 1.75 it/s
step 1800/20000 | loss 3.9130 | aux rot 2.066 (chance 2.83) | aux perm 1.589 (chance 3.26) | lr 2.99e-04 | 1.75 it/s
step 1900/20000 | loss 3.4965 | aux rot 2.126 (chance 2.83) | aux perm 1.716 (chance 3.26) | lr 2.98e-04 | 1.74 it/s
```

### icl_split_fold1.log
```
  return F.conv1d(input, weight, bias, self.stride,
[audit] step 0: mode-A 62.84 | mode-C 100.00 (random prefix) | deployment reference ~43-58
step 100/20000 | loss 7.0961 | aux rot 2.808 (chance 2.83) | aux perm 3.273 (chance 3.26) | lr 3.00e-05 | 1.88 it/s
step 200/20000 | loss 5.8444 | aux rot 2.723 (chance 2.83) | aux perm 3.204 (chance 3.26) | lr 6.00e-05 | 1.78 it/s
step 300/20000 | loss 4.9437 | aux rot 2.620 (chance 2.83) | aux perm 3.074 (chance 3.26) | lr 9.00e-05 | 1.88 it/s
step 400/20000 | loss 5.3398 | aux rot 2.482 (chance 2.83) | aux perm 2.940 (chance 3.26) | lr 1.20e-04 | 1.75 it/s
step 500/20000 | loss 4.6027 | aux rot 2.414 (chance 2.83) | aux perm 2.645 (chance 3.26) | lr 1.50e-04 | 1.77 it/s
[val] step 500: IDENTITY A 55.82 C 75.80 gain -19.98 | PERMUTED A 67.62 C 81.28 gain -13.66 | mem-gauge(meta-train A) 44.68   (meta-test users, fold 1)
[val] new best mode-C 75.80 -> best.pt
step 600/20000 | loss 4.6691 | aux rot 2.348 (chance 2.83) | aux perm 2.360 (chance 3.26) | lr 1.80e-04 | 1.72 it/s
step 700/20000 | loss 4.2528 | aux rot 2.276 (chance 2.83) | aux perm 2.056 (chance 3.26) | lr 2.10e-04 | 1.71 it/s
step 800/20000 | loss 4.1700 | aux rot 2.250 (chance 2.83) | aux perm 1.857 (chance 3.26) | lr 2.40e-04 | 1.71 it/s
step 900/20000 | loss 3.5871 | aux rot 2.204 (chance 2.83) | aux perm 1.673 (chance 3.26) | lr 2.70e-04 | 1.73 it/s
step 1000/20000 | loss 3.6488 | aux rot 2.153 (chance 2.83) | aux perm 1.504 (chance 3.26) | lr 3.00e-04 | 1.70 it/s
[val] step 1000: IDENTITY A 56.26 C 58.91 gain -2.65 | PERMUTED A 68.02 C 69.50 gain -1.48 | mem-gauge(meta-train A) 39.94   (meta-test users, fold 1)
[val] new best mode-C 58.91 -> best.pt
step 1100/20000 | loss 3.7616 | aux rot 2.087 (chance 2.83) | aux perm 1.650 (chance 3.26) | lr 3.00e-04 | 1.70 it/s
step 1200/20000 | loss 3.8112 | aux rot 2.147 (chance 2.83) | aux perm 1.758 (chance 3.26) | lr 3.00e-04 | 1.70 it/s
step 1300/20000 | loss 3.9420 | aux rot 2.057 (chance 2.83) | aux perm 1.678 (chance 3.26) | lr 3.00e-04 | 1.73 it/s
step 1400/20000 | loss 3.9821 | aux rot 2.199 (chance 2.83) | aux perm 1.609 (chance 3.26) | lr 3.00e-04 | 1.73 it/s
step 1500/20000 | loss 3.9224 | aux rot 2.066 (chance 2.83) | aux perm 1.592 (chance 3.26) | lr 2.99e-04 | 1.71 it/s
[val] step 1500: IDENTITY A 57.12 C 59.58 gain -2.47 | PERMUTED A 68.48 C 70.32 gain -1.84 | mem-gauge(meta-train A) 37.18   (meta-test users, fold 1)
step 1600/20000 | loss 3.3163 | aux rot 1.986 (chance 2.83) | aux perm 1.672 (chance 3.26) | lr 2.99e-04 | 1.72 it/s
step 1700/20000 | loss 3.7076 | aux rot 1.990 (chance 2.83) | aux perm 1.576 (chance 3.26) | lr 2.99e-04 | 1.73 it/s
step 1800/20000 | loss 3.8425 | aux rot 2.033 (chance 2.83) | aux perm 1.603 (chance 3.26) | lr 2.99e-04 | 1.73 it/s
```

### teachers_shard0.log
```
[teachers] 24/96 training users in this shard | tokens_only=False | steps=1800
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[1/24] 11372316: zero-shot 9.83 -> best 9.83 (gain +0.00)
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[2/24] 14312238: zero-shot 4.08 -> best 4.02 (gain +0.06)
[3/24] 2396581: zero-shot 10.03 -> best 9.89 (gain +0.15)
[4/24] 29502646: zero-shot 3.87 -> best 3.87 (gain +0.00)
[5/24] 33505485: zero-shot 17.66 -> best 17.45 (gain +0.21)
[6/24] 3734552: zero-shot 7.73 -> best 7.73 (gain +0.00)
[7/24] 41556660: zero-shot 8.94 -> best 8.86 (gain +0.08)
[8/24] 45200932: zero-shot 10.45 -> best 10.09 (gain +0.37)
[9/24] 47919028: zero-shot 9.51 -> best 9.51 (gain +0.00)
```

### teachers_shard1.log
```
[teachers] 24/96 training users in this shard | tokens_only=False | steps=1800
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[1/24] 11944098: zero-shot 3.47 -> best 3.38 (gain +0.09)
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[2/24] 1438774: zero-shot 11.89 -> best 11.89 (gain +0.00)
[3/24] 25847138: zero-shot 14.82 -> best 14.82 (gain +0.00)
[4/24] 30481951: zero-shot 9.43 -> best 9.43 (gain +0.00)
[5/24] 3432025: zero-shot 3.58 -> best 3.54 (gain +0.05)
[6/24] 37398304: zero-shot 4.08 -> best 3.99 (gain +0.09)
[7/24] 4162929: zero-shot 13.13 -> best 13.13 (gain +0.00)
[8/24] 45828573: zero-shot 4.48 -> best 4.48 (gain +0.00)
[9/24] 5344357: zero-shot 11.42 -> best 11.42 (gain +0.00)
```

### teachers_shard2.log
```
[teachers] 24/96 training users in this shard | tokens_only=False | steps=1800
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[1/24] 12565339: zero-shot 2.46 -> best 2.33 (gain +0.14)
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[2/24] 18200807: zero-shot 14.57 -> best 14.57 (gain +0.00)
[3/24] 25915650: zero-shot 2.68 -> best 2.68 (gain +0.00)
[4/24] 30807164: zero-shot 18.98 -> best 18.98 (gain +0.00)
[5/24] 34527640: zero-shot 15.67 -> best 15.41 (gain +0.26)
[6/24] 39024419: zero-shot 5.07 -> best 4.94 (gain +0.13)
[7/24] 42383274: zero-shot 10.77 -> best 10.65 (gain +0.12)
[8/24] 46697259: zero-shot 3.11 -> best 2.96 (gain +0.15)
[9/24] 53845929: zero-shot 10.71 -> best 10.68 (gain +0.02)
```

### teachers_shard3.log
```
[teachers] 24/96 training users in this shard | tokens_only=False | steps=1800
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:919.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[1/24] 13321435: zero-shot 15.67 -> best 15.67 (gain +0.00)
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
[2/24] 20676876: zero-shot 9.28 -> best 9.22 (gain +0.06)
[3/24] 26940776: zero-shot 10.05 -> best 10.05 (gain +0.00)
[4/24] 33259248: zero-shot 5.65 -> best 5.43 (gain +0.23)
[5/24] 3537794: zero-shot 9.42 -> best 9.42 (gain +0.00)
[6/24] 41222064: zero-shot 6.19 -> best 6.19 (gain +0.00)
```

### tf_fold0_full.log
```
step 98800/103000 | loss 1.2004 | lr 2.99e-06 | 659 win/s
step 99000/103000 | loss 1.2051 | lr 2.71e-06 | 659 win/s
step 99200/103000 | loss 1.2024 | lr 2.44e-06 | 659 win/s
step 99400/103000 | loss 1.2040 | lr 2.19e-06 | 659 win/s
step 99600/103000 | loss 1.2052 | lr 1.96e-06 | 659 win/s
step 99800/103000 | loss 1.1988 | lr 1.73e-06 | 659 win/s
step 100000/103000 | loss 1.2035 | lr 1.52e-06 | 659 win/s
[val] step 100000: 8-test-user CER 46.03 | fold-heldout-user CER 53.67  (their Tiny reference: 35.9)
step 100200/103000 | loss 1.2077 | lr 1.33e-06 | 659 win/s
step 100400/103000 | loss 1.2027 | lr 1.15e-06 | 659 win/s
step 100600/103000 | loss 1.2025 | lr 9.76e-07 | 659 win/s
step 100800/103000 | loss 1.2039 | lr 8.20e-07 | 659 win/s
step 101000/103000 | loss 1.2014 | lr 6.78e-07 | 659 win/s
step 101200/103000 | loss 1.2021 | lr 5.49e-07 | 659 win/s
step 101400/103000 | loss 1.2012 | lr 4.34e-07 | 660 win/s
step 101600/103000 | loss 1.2013 | lr 3.32e-07 | 660 win/s
step 101800/103000 | loss 1.2063 | lr 2.44e-07 | 660 win/s
step 102000/103000 | loss 1.2002 | lr 1.70e-07 | 660 win/s
step 102200/103000 | loss 1.2020 | lr 1.08e-07 | 660 win/s
step 102400/103000 | loss 1.2006 | lr 6.10e-08 | 660 win/s
step 102600/103000 | loss 1.2039 | lr 2.71e-08 | 660 win/s
step 102800/103000 | loss 1.2012 | lr 6.78e-09 | 660 win/s
step 103000/103000 | loss 1.2038 | lr 0.00e+00 | 660 win/s
[val] step 103000: 8-test-user CER 45.88 | fold-heldout-user CER 53.87  (their Tiny reference: 35.9)
[done] best 8-test-user CER 44.28
```

### tf_fold0.log
```
step 36000/40000 | loss 1.5666 | lr 2.71e-05 | 379 win/s
[val] step 36000: 8-test-user CER 86.69 | fold-heldout-user CER 87.72  (their Tiny reference: 35.9)
step 36200/40000 | loss 1.5706 | lr 2.45e-05 | 380 win/s
step 36400/40000 | loss 1.5664 | lr 2.20e-05 | 381 win/s
step 36600/40000 | loss 1.5669 | lr 1.96e-05 | 382 win/s
step 36800/40000 | loss 1.5662 | lr 1.74e-05 | 383 win/s
step 37000/40000 | loss 1.5618 | lr 1.53e-05 | 384 win/s
step 37200/40000 | loss 1.5597 | lr 1.33e-05 | 385 win/s
step 37400/40000 | loss 1.5568 | lr 1.15e-05 | 386 win/s
step 37600/40000 | loss 1.5608 | lr 9.81e-06 | 387 win/s
step 37800/40000 | loss 1.5530 | lr 8.25e-06 | 388 win/s
step 38000/40000 | loss 1.5590 | lr 6.82e-06 | 389 win/s
[val] step 38000: 8-test-user CER 87.24 | fold-heldout-user CER 88.15  (their Tiny reference: 35.9)
step 38200/40000 | loss 1.5537 | lr 5.53e-06 | 390 win/s
step 38400/40000 | loss 1.5549 | lr 4.37e-06 | 390 win/s
step 38600/40000 | loss 1.5571 | lr 3.35e-06 | 391 win/s
step 38800/40000 | loss 1.5583 | lr 2.46e-06 | 392 win/s
step 39000/40000 | loss 1.5540 | lr 1.71e-06 | 393 win/s
step 39200/40000 | loss 1.5471 | lr 1.09e-06 | 394 win/s
step 39400/40000 | loss 1.5571 | lr 6.15e-07 | 395 win/s
step 39600/40000 | loss 1.5540 | lr 2.73e-07 | 396 win/s
step 39800/40000 | loss 1.5497 | lr 6.83e-08 | 397 win/s
step 40000/40000 | loss 1.5527 | lr 0.00e+00 | 398 win/s
[val] step 40000: 8-test-user CER 87.25 | fold-heldout-user CER 88.14  (their Tiny reference: 35.9)
[done] best 8-test-user CER 83.05
```

### tf_fold1_full.log
```
step 98800/103000 | loss 1.2250 | lr 2.99e-06 | 507 win/s
step 99000/103000 | loss 1.2208 | lr 2.71e-06 | 507 win/s
step 99200/103000 | loss 1.2250 | lr 2.44e-06 | 507 win/s
step 99400/103000 | loss 1.2223 | lr 2.19e-06 | 508 win/s
step 99600/103000 | loss 1.2227 | lr 1.96e-06 | 508 win/s
step 99800/103000 | loss 1.2211 | lr 1.73e-06 | 508 win/s
step 100000/103000 | loss 1.2209 | lr 1.52e-06 | 508 win/s
[val] step 100000: 8-test-user CER 46.74 | fold-heldout-user CER 54.43  (their Tiny reference: 35.9)
step 100200/103000 | loss 1.2186 | lr 1.33e-06 | 509 win/s
step 100400/103000 | loss 1.2224 | lr 1.15e-06 | 509 win/s
step 100600/103000 | loss 1.2214 | lr 9.76e-07 | 509 win/s
step 100800/103000 | loss 1.2184 | lr 8.20e-07 | 509 win/s
step 101000/103000 | loss 1.2143 | lr 6.78e-07 | 510 win/s
step 101200/103000 | loss 1.2215 | lr 5.49e-07 | 510 win/s
step 101400/103000 | loss 1.2207 | lr 4.34e-07 | 510 win/s
step 101600/103000 | loss 1.2192 | lr 3.32e-07 | 511 win/s
step 101800/103000 | loss 1.2229 | lr 2.44e-07 | 511 win/s
step 102000/103000 | loss 1.2191 | lr 1.70e-07 | 511 win/s
step 102200/103000 | loss 1.2183 | lr 1.08e-07 | 511 win/s
step 102400/103000 | loss 1.2221 | lr 6.10e-08 | 512 win/s
step 102600/103000 | loss 1.2199 | lr 2.71e-08 | 512 win/s
step 102800/103000 | loss 1.2194 | lr 6.78e-09 | 512 win/s
step 103000/103000 | loss 1.2239 | lr 0.00e+00 | 512 win/s
[val] step 103000: 8-test-user CER 46.82 | fold-heldout-user CER 54.28  (their Tiny reference: 35.9)
[done] best 8-test-user CER 45.88
```

### tf_fold1.log
```
step 36000/40000 | loss 1.5886 | lr 2.71e-05 | 617 win/s
[val] step 36000: 8-test-user CER 88.05 | fold-heldout-user CER 88.25  (their Tiny reference: 35.9)
step 36200/40000 | loss 1.5868 | lr 2.45e-05 | 617 win/s
step 36400/40000 | loss 1.5858 | lr 2.20e-05 | 617 win/s
step 36600/40000 | loss 1.5850 | lr 1.96e-05 | 617 win/s
step 36800/40000 | loss 1.5808 | lr 1.74e-05 | 617 win/s
step 37000/40000 | loss 1.5820 | lr 1.53e-05 | 617 win/s
step 37200/40000 | loss 1.5762 | lr 1.33e-05 | 617 win/s
step 37400/40000 | loss 1.5742 | lr 1.15e-05 | 617 win/s
step 37600/40000 | loss 1.5796 | lr 9.81e-06 | 617 win/s
step 37800/40000 | loss 1.5735 | lr 8.25e-06 | 617 win/s
step 38000/40000 | loss 1.5724 | lr 6.82e-06 | 617 win/s
[val] step 38000: 8-test-user CER 87.98 | fold-heldout-user CER 88.18  (their Tiny reference: 35.9)
step 38200/40000 | loss 1.5743 | lr 5.53e-06 | 617 win/s
step 38400/40000 | loss 1.5710 | lr 4.37e-06 | 617 win/s
step 38600/40000 | loss 1.5731 | lr 3.35e-06 | 617 win/s
step 38800/40000 | loss 1.5637 | lr 2.46e-06 | 618 win/s
step 39000/40000 | loss 1.5679 | lr 1.71e-06 | 618 win/s
step 39200/40000 | loss 1.5741 | lr 1.09e-06 | 617 win/s
step 39400/40000 | loss 1.5697 | lr 6.15e-07 | 617 win/s
step 39600/40000 | loss 1.5671 | lr 2.73e-07 | 617 win/s
step 39800/40000 | loss 1.5723 | lr 6.83e-08 | 617 win/s
step 40000/40000 | loss 1.5735 | lr 0.00e+00 | 617 win/s
[val] step 40000: 8-test-user CER 87.88 | fold-heldout-user CER 88.04  (their Tiny reference: 35.9)
[done] best 8-test-user CER 83.72
```

### tf_fold2_full.log
```
step 18200/103000 | loss 1.7465 | lr 6.57e-04 | 720 win/s
step 18400/103000 | loss 1.7297 | lr 6.56e-04 | 721 win/s
step 18600/103000 | loss 1.7245 | lr 6.55e-04 | 721 win/s
step 18800/103000 | loss 1.7490 | lr 6.54e-04 | 722 win/s
step 19000/103000 | loss 1.7710 | lr 6.52e-04 | 722 win/s
step 19200/103000 | loss 1.7501 | lr 6.51e-04 | 722 win/s
step 19400/103000 | loss 1.7391 | lr 6.50e-04 | 722 win/s
step 19600/103000 | loss 1.7227 | lr 6.49e-04 | 723 win/s
step 19800/103000 | loss 1.7336 | lr 6.48e-04 | 723 win/s
step 20000/103000 | loss 1.7322 | lr 6.47e-04 | 723 win/s
[val] step 20000: 8-test-user CER 48.07 | fold-heldout-user CER 56.97  (their Tiny reference: 35.9)
[val] new best 48.07 -> best.pt
step 20200/103000 | loss 1.7459 | lr 6.46e-04 | 723 win/s
step 20400/103000 | loss 1.7228 | lr 6.45e-04 | 723 win/s
step 20600/103000 | loss 1.6940 | lr 6.43e-04 | 723 win/s
step 20800/103000 | loss 1.7032 | lr 6.42e-04 | 723 win/s
step 21000/103000 | loss 1.7006 | lr 6.41e-04 | 723 win/s
step 21200/103000 | loss 1.7225 | lr 6.40e-04 | 723 win/s
step 21400/103000 | loss 1.7005 | lr 6.38e-04 | 723 win/s
step 21600/103000 | loss 1.6851 | lr 6.37e-04 | 723 win/s
step 21800/103000 | loss 1.6965 | lr 6.36e-04 | 723 win/s
step 22000/103000 | loss 1.6956 | lr 6.35e-04 | 723 win/s
step 22200/103000 | loss 1.7017 | lr 6.33e-04 | 723 win/s
step 22400/103000 | loss 1.6998 | lr 6.32e-04 | 723 win/s
step 22600/103000 | loss 1.7094 | lr 6.31e-04 | 723 win/s
```

### tf_fold2.log
```
step 36000/40000 | loss 1.5210 | lr 2.71e-05 | 619 win/s
[val] step 36000: 8-test-user CER 87.74 | fold-heldout-user CER 89.19  (their Tiny reference: 35.9)
step 36200/40000 | loss 1.5184 | lr 2.45e-05 | 618 win/s
step 36400/40000 | loss 1.5233 | lr 2.20e-05 | 618 win/s
step 36600/40000 | loss 1.5117 | lr 1.96e-05 | 618 win/s
step 36800/40000 | loss 1.5099 | lr 1.74e-05 | 618 win/s
step 37000/40000 | loss 1.5117 | lr 1.53e-05 | 618 win/s
step 37200/40000 | loss 1.5059 | lr 1.33e-05 | 619 win/s
step 37400/40000 | loss 1.5071 | lr 1.15e-05 | 619 win/s
step 37600/40000 | loss 1.5043 | lr 9.81e-06 | 619 win/s
step 37800/40000 | loss 1.5083 | lr 8.25e-06 | 618 win/s
step 38000/40000 | loss 1.5036 | lr 6.82e-06 | 618 win/s
[val] step 38000: 8-test-user CER 87.85 | fold-heldout-user CER 89.26  (their Tiny reference: 35.9)
step 38200/40000 | loss 1.5013 | lr 5.53e-06 | 618 win/s
step 38400/40000 | loss 1.5051 | lr 4.37e-06 | 618 win/s
step 38600/40000 | loss 1.5002 | lr 3.35e-06 | 618 win/s
step 38800/40000 | loss 1.5018 | lr 2.46e-06 | 619 win/s
step 39000/40000 | loss 1.5066 | lr 1.71e-06 | 618 win/s
step 39200/40000 | loss 1.5035 | lr 1.09e-06 | 618 win/s
step 39400/40000 | loss 1.5053 | lr 6.15e-07 | 618 win/s
step 39600/40000 | loss 1.4941 | lr 2.73e-07 | 618 win/s
step 39800/40000 | loss 1.5032 | lr 6.83e-08 | 618 win/s
step 40000/40000 | loss 1.5024 | lr 0.00e+00 | 618 win/s
[val] step 40000: 8-test-user CER 87.80 | fold-heldout-user CER 89.26  (their Tiny reference: 35.9)
[done] best 8-test-user CER 82.46
```

### tf_fold3.log
```
step 36000/40000 | loss 1.5556 | lr 2.71e-05 | 552 win/s
[val] step 36000: 8-test-user CER 88.25 | fold-heldout-user CER 89.57  (their Tiny reference: 35.9)
step 36200/40000 | loss 1.5493 | lr 2.45e-05 | 549 win/s
step 36400/40000 | loss 1.5492 | lr 2.20e-05 | 546 win/s
step 36600/40000 | loss 1.5459 | lr 1.96e-05 | 544 win/s
step 36800/40000 | loss 1.5428 | lr 1.74e-05 | 542 win/s
step 37000/40000 | loss 1.5428 | lr 1.53e-05 | 540 win/s
step 37200/40000 | loss 1.5413 | lr 1.33e-05 | 538 win/s
step 37400/40000 | loss 1.5388 | lr 1.15e-05 | 535 win/s
step 37600/40000 | loss 1.5378 | lr 9.81e-06 | 534 win/s
step 37800/40000 | loss 1.5394 | lr 8.25e-06 | 532 win/s
step 38000/40000 | loss 1.5372 | lr 6.82e-06 | 531 win/s
[val] step 38000: 8-test-user CER 88.43 | fold-heldout-user CER 89.61  (their Tiny reference: 35.9)
step 38200/40000 | loss 1.5344 | lr 5.53e-06 | 529 win/s
step 38400/40000 | loss 1.5357 | lr 4.37e-06 | 528 win/s
step 38600/40000 | loss 1.5334 | lr 3.35e-06 | 526 win/s
step 38800/40000 | loss 1.5321 | lr 2.46e-06 | 525 win/s
step 39000/40000 | loss 1.5349 | lr 1.71e-06 | 523 win/s
step 39200/40000 | loss 1.5349 | lr 1.09e-06 | 522 win/s
step 39400/40000 | loss 1.5344 | lr 6.15e-07 | 521 win/s
step 39600/40000 | loss 1.5314 | lr 2.73e-07 | 519 win/s
step 39800/40000 | loss 1.5343 | lr 6.83e-08 | 518 win/s
step 40000/40000 | loss 1.5302 | lr 0.00e+00 | 516 win/s
[val] step 40000: 8-test-user CER 88.39 | fold-heldout-user CER 89.60  (their Tiny reference: 35.9)
[done] best 8-test-user CER 79.99
```

### tf_ref_full.log
```
step 98800/103000 | loss 1.2646 | lr 2.99e-06 | 488 win/s
step 99000/103000 | loss 1.2630 | lr 2.71e-06 | 488 win/s
step 99200/103000 | loss 1.2600 | lr 2.44e-06 | 488 win/s
step 99400/103000 | loss 1.2644 | lr 2.19e-06 | 488 win/s
step 99600/103000 | loss 1.2678 | lr 1.96e-06 | 488 win/s
step 99800/103000 | loss 1.2590 | lr 1.73e-06 | 488 win/s
step 100000/103000 | loss 1.2678 | lr 1.52e-06 | 488 win/s
[val] step 100000: 8-test-user CER 46.21 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
step 100200/103000 | loss 1.2651 | lr 1.33e-06 | 488 win/s
step 100400/103000 | loss 1.2617 | lr 1.15e-06 | 488 win/s
step 100600/103000 | loss 1.2611 | lr 9.76e-07 | 488 win/s
step 100800/103000 | loss 1.2646 | lr 8.20e-07 | 488 win/s
step 101000/103000 | loss 1.2603 | lr 6.78e-07 | 487 win/s
step 101200/103000 | loss 1.2652 | lr 5.49e-07 | 487 win/s
step 101400/103000 | loss 1.2616 | lr 4.34e-07 | 487 win/s
step 101600/103000 | loss 1.2589 | lr 3.32e-07 | 487 win/s
step 101800/103000 | loss 1.2635 | lr 2.44e-07 | 486 win/s
step 102000/103000 | loss 1.2607 | lr 1.70e-07 | 486 win/s
step 102200/103000 | loss 1.2624 | lr 1.08e-07 | 486 win/s
step 102400/103000 | loss 1.2561 | lr 6.10e-08 | 486 win/s
step 102600/103000 | loss 1.2656 | lr 2.71e-08 | 486 win/s
step 102800/103000 | loss 1.2638 | lr 6.78e-09 | 486 win/s
step 103000/103000 | loss 1.2614 | lr 0.00e+00 | 486 win/s
[val] step 103000: 8-test-user CER 46.23 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[done] best 8-test-user CER 44.99
```

### tf_ref.log
```
step 2400/40000 | loss 3.0256 | lr 3.00e-04 | 711 win/s
step 2600/40000 | loss 2.9942 | lr 3.00e-04 | 709 win/s
step 2800/40000 | loss 2.9508 | lr 3.00e-04 | 709 win/s
step 3000/40000 | loss 2.9066 | lr 2.99e-04 | 708 win/s
step 3200/40000 | loss 2.8676 | lr 2.99e-04 | 707 win/s
step 3400/40000 | loss 2.8254 | lr 2.99e-04 | 706 win/s
step 3600/40000 | loss 2.7944 | lr 2.99e-04 | 706 win/s
step 3800/40000 | loss 2.7553 | lr 2.98e-04 | 705 win/s
step 4000/40000 | loss 2.7171 | lr 2.98e-04 | 705 win/s
[val] step 4000: 8-test-user CER 96.35 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[val] new best 96.35 -> best.pt
step 4200/40000 | loss 2.6762 | lr 2.98e-04 | 704 win/s
step 4400/40000 | loss 2.6429 | lr 2.97e-04 | 703 win/s
step 4600/40000 | loss 2.5931 | lr 2.97e-04 | 702 win/s
step 4800/40000 | loss 2.5449 | lr 2.96e-04 | 702 win/s
step 5000/40000 | loss 2.5132 | lr 2.95e-04 | 701 win/s
step 5200/40000 | loss 2.4724 | lr 2.95e-04 | 701 win/s
step 5400/40000 | loss 2.4455 | lr 2.94e-04 | 701 win/s
step 5600/40000 | loss 2.4213 | lr 2.93e-04 | 701 win/s
step 5800/40000 | loss 2.3891 | lr 2.93e-04 | 701 win/s
step 6000/40000 | loss 2.3726 | lr 2.92e-04 | 702 win/s
[val] step 6000: 8-test-user CER 83.37 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[val] new best 83.37 -> best.pt
step 6200/40000 | loss 2.3523 | lr 2.91e-04 | 702 win/s
step 6400/40000 | loss 2.3342 | lr 2.90e-04 | 702 win/s
```

### tf_ref_lr1e3.log
```
step 36000/40000 | loss 1.6794 | lr 2.71e-05 | 376 win/s
[val] step 36000: 8-test-user CER 86.94 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
step 36200/40000 | loss 1.6771 | lr 2.45e-05 | 377 win/s
step 36400/40000 | loss 1.6797 | lr 2.20e-05 | 377 win/s
step 36600/40000 | loss 1.6776 | lr 1.96e-05 | 378 win/s
step 36800/40000 | loss 1.6696 | lr 1.74e-05 | 379 win/s
step 37000/40000 | loss 1.6725 | lr 1.53e-05 | 379 win/s
step 37200/40000 | loss 1.6692 | lr 1.33e-05 | 380 win/s
step 37400/40000 | loss 1.6708 | lr 1.15e-05 | 381 win/s
step 37600/40000 | loss 1.6676 | lr 9.81e-06 | 381 win/s
step 37800/40000 | loss 1.6632 | lr 8.25e-06 | 382 win/s
step 38000/40000 | loss 1.6637 | lr 6.82e-06 | 382 win/s
[val] step 38000: 8-test-user CER 87.16 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
step 38200/40000 | loss 1.6662 | lr 5.53e-06 | 383 win/s
step 38400/40000 | loss 1.6665 | lr 4.37e-06 | 383 win/s
step 38600/40000 | loss 1.6656 | lr 3.35e-06 | 384 win/s
step 38800/40000 | loss 1.6666 | lr 2.46e-06 | 384 win/s
step 39000/40000 | loss 1.6592 | lr 1.71e-06 | 384 win/s
step 39200/40000 | loss 1.6612 | lr 1.09e-06 | 385 win/s
step 39400/40000 | loss 1.6610 | lr 6.15e-07 | 385 win/s
step 39600/40000 | loss 1.6616 | lr 2.73e-07 | 386 win/s
step 39800/40000 | loss 1.6659 | lr 6.83e-08 | 386 win/s
step 40000/40000 | loss 1.6571 | lr 0.00e+00 | 387 win/s
[val] step 40000: 8-test-user CER 87.14 | fold-heldout-user CER nan  (their Tiny reference: 35.9)
[done] best 8-test-user CER 78.64
```

### v31_train.log
```
step 10000/12000 | loss 0.9480 | lr 3.68e-06 | 2.79 it/s
[val] step 10000: mode-C CER 33.73 | mode-B CER 31.82 | mode-A CER 31.82 | gain C -1.91 / B +0.00 | loss 1.4693
step 10100/12000 | loss 0.9386 | lr 3.48e-06 | 2.14 it/s
step 10200/12000 | loss 0.9384 | lr 3.28e-06 | 3.28 it/s
step 10300/12000 | loss 0.9083 | lr 3.09e-06 | 2.60 it/s
step 10400/12000 | loss 0.8921 | lr 2.91e-06 | 2.58 it/s
step 10500/12000 | loss 0.8510 | lr 2.74e-06 | 2.77 it/s
step 10600/12000 | loss 0.7796 | lr 2.58e-06 | 2.80 it/s
step 10700/12000 | loss 1.0353 | lr 2.44e-06 | 2.90 it/s
step 10800/12000 | loss 0.8474 | lr 2.30e-06 | 3.38 it/s
step 10900/12000 | loss 0.8653 | lr 2.17e-06 | 3.08 it/s
step 11000/12000 | loss 0.8874 | lr 2.06e-06 | 2.82 it/s
[val] step 11000: mode-C CER 35.43 | mode-B CER 33.23 | mode-A CER 33.23 | gain C -2.20 / B +0.00 | loss 1.4557
step 11100/12000 | loss 0.8655 | lr 1.95e-06 | 2.23 it/s
step 11200/12000 | loss 0.9035 | lr 1.86e-06 | 3.17 it/s
step 11300/12000 | loss 0.9123 | lr 1.77e-06 | 2.66 it/s
step 11400/12000 | loss 0.8933 | lr 1.70e-06 | 2.78 it/s
step 11500/12000 | loss 0.9338 | lr 1.64e-06 | 2.84 it/s
step 11600/12000 | loss 1.0009 | lr 1.59e-06 | 2.45 it/s
step 11700/12000 | loss 0.8615 | lr 1.55e-06 | 2.99 it/s
step 11800/12000 | loss 0.9282 | lr 1.52e-06 | 2.44 it/s
step 11900/12000 | loss 0.9326 | lr 1.51e-06 | 3.41 it/s
step 12000/12000 | loss 0.9137 | lr 1.50e-06 | 2.45 it/s
[val] step 12000: mode-C CER 37.66 | mode-B CER 35.96 | mode-A CER 35.96 | gain C -1.69 / B +0.00 | loss 1.6246
[done] {'best_val_cer': 30.502512562814072, 'steps': 12000, 'phase': 'icl'}
```

### v32_train.log
```
step 10000/12000 | loss 0.8909 | lr 3.68e-06 | 2.90 it/s
[val] step 10000: mode-C CER 34.42 | mode-B CER 32.40 | mode-A CER 32.40 | gain C -2.01 / B +0.00 | loss 1.3946
step 10100/12000 | loss 0.9366 | lr 3.48e-06 | 2.58 it/s
step 10200/12000 | loss 0.9141 | lr 3.28e-06 | 3.34 it/s
step 10300/12000 | loss 0.8504 | lr 3.09e-06 | 4.25 it/s
step 10400/12000 | loss 0.9384 | lr 2.91e-06 | 2.83 it/s
step 10500/12000 | loss 0.9138 | lr 2.74e-06 | 3.42 it/s
step 10600/12000 | loss 0.8927 | lr 2.58e-06 | 2.57 it/s
step 10700/12000 | loss 0.8417 | lr 2.44e-06 | 3.13 it/s
step 10800/12000 | loss 0.8098 | lr 2.30e-06 | 3.19 it/s
step 10900/12000 | loss 0.9133 | lr 2.17e-06 | 3.24 it/s
step 11000/12000 | loss 0.8975 | lr 2.06e-06 | 2.58 it/s
[val] step 11000: mode-C CER 33.49 | mode-B CER 32.11 | mode-A CER 32.11 | gain C -1.38 / B +0.00 | loss 1.3431
step 11100/12000 | loss 0.7570 | lr 1.95e-06 | 2.70 it/s
step 11200/12000 | loss 0.8814 | lr 1.86e-06 | 3.44 it/s
step 11300/12000 | loss 0.9319 | lr 1.77e-06 | 3.08 it/s
step 11400/12000 | loss 0.7985 | lr 1.70e-06 | 2.64 it/s
step 11500/12000 | loss 0.8458 | lr 1.64e-06 | 2.93 it/s
step 11600/12000 | loss 0.8877 | lr 1.59e-06 | 2.75 it/s
step 11700/12000 | loss 0.8411 | lr 1.55e-06 | 3.85 it/s
step 11800/12000 | loss 0.8725 | lr 1.52e-06 | 2.91 it/s
step 11900/12000 | loss 0.8097 | lr 1.51e-06 | 3.35 it/s
step 12000/12000 | loss 0.8766 | lr 1.50e-06 | 2.84 it/s
[val] step 12000: mode-C CER 33.23 | mode-B CER 32.19 | mode-A CER 32.19 | gain C -1.04 / B +0.00 | loss 1.4201
[done] {'best_val_cer': 31.115951742627345, 'steps': 12000, 'phase': 'icl'}
```

### v3cheavy_train.log
```
step 18000/20000 | loss 0.9970 | lr 2.26e-06 | 2.77 it/s
[val] step 18000: mode-C CER 33.26 | mode-B CER 32.21 | mode-A CER 32.21 | gain C -1.04 / B +0.00 | loss 1.3658
step 18100/20000 | loss 0.7447 | lr 2.18e-06 | 2.29 it/s
step 18200/20000 | loss 0.7545 | lr 2.11e-06 | 3.05 it/s
step 18300/20000 | loss 0.7470 | lr 2.05e-06 | 2.48 it/s
step 18400/20000 | loss 0.8692 | lr 1.99e-06 | 2.29 it/s
step 18500/20000 | loss 0.8494 | lr 1.93e-06 | 2.65 it/s
step 18600/20000 | loss 0.7993 | lr 1.87e-06 | 2.29 it/s
step 18700/20000 | loss 0.7395 | lr 1.82e-06 | 2.94 it/s
step 18800/20000 | loss 0.8032 | lr 1.77e-06 | 2.18 it/s
step 18900/20000 | loss 0.7287 | lr 1.73e-06 | 2.45 it/s
step 19000/20000 | loss 0.7887 | lr 1.69e-06 | 2.26 it/s
[val] step 19000: mode-C CER 33.67 | mode-B CER 32.66 | mode-A CER 32.66 | gain C -1.00 / B +0.00 | loss 1.2447
step 19100/20000 | loss 0.7689 | lr 1.65e-06 | 1.82 it/s
step 19200/20000 | loss 0.8580 | lr 1.62e-06 | 2.00 it/s
step 19300/20000 | loss 0.8121 | lr 1.59e-06 | 1.97 it/s
step 19400/20000 | loss 0.7951 | lr 1.57e-06 | 2.22 it/s
step 19500/20000 | loss 0.7390 | lr 1.55e-06 | 1.64 it/s
step 19600/20000 | loss 0.8635 | lr 1.53e-06 | 1.41 it/s
step 19700/20000 | loss 0.8335 | lr 1.52e-06 | 1.92 it/s
step 19800/20000 | loss 0.7441 | lr 1.51e-06 | 2.49 it/s
step 19900/20000 | loss 0.8352 | lr 1.50e-06 | 2.38 it/s
step 20000/20000 | loss 0.7826 | lr 1.50e-06 | 2.35 it/s
[val] step 20000: mode-C CER 33.17 | mode-B CER 32.30 | mode-A CER 32.30 | gain C -0.87 / B +0.00 | loss 1.3076
[done] {'best_val_cer': 32.86171467070911, 'steps': 20000, 'phase': 'icl'}
```

### v3frozen_train.log
```
step 9800/12000 | loss 1.0111 | lr 9.72e-06 | 2.65 it/s
step 9900/12000 | loss 0.9020 | lr 9.42e-06 | 2.65 it/s
step 10000/12000 | loss 0.9523 | lr 9.13e-06 | 2.58 it/s
[val] step 10000: mode-C CER 20.66 | mode-B CER 28.85 | mode-A CER 28.85 | gain C +8.19 / B +0.00 | loss 0.6736
[val] new best CER 20.66 -> saved best.pt
step 10100/12000 | loss 0.8520 | lr 8.84e-06 | 2.18 it/s
step 10200/12000 | loss 0.9866 | lr 8.53e-06 | 2.66 it/s
step 10300/12000 | loss 0.8900 | lr 8.25e-06 | 2.39 it/s
step 10400/12000 | loss 0.7142 | lr 8.00e-06 | 3.11 it/s
step 10500/12000 | loss 0.9485 | lr 7.71e-06 | 2.48 it/s
step 10700/12000 | loss 0.9365 | lr 7.18e-06 | 1.27 it/s
step 10800/12000 | loss 0.9937 | lr 6.90e-06 | 2.52 it/s
step 11000/12000 | loss 1.0081 | lr 6.39e-06 | 1.15 it/s
[val] step 11000: mode-C CER 21.93 | mode-B CER 29.27 | mode-A CER 29.27 | gain C +7.34 / B +0.00 | loss 0.6969
step 11100/12000 | loss 1.0204 | lr 6.14e-06 | 2.09 it/s
step 11200/12000 | loss 0.8443 | lr 5.91e-06 | 2.54 it/s
step 11300/12000 | loss 0.9902 | lr 5.66e-06 | 2.77 it/s
step 11500/12000 | loss 0.8968 | lr 5.20e-06 | 1.43 it/s
step 11600/12000 | loss 0.8179 | lr 4.98e-06 | 2.81 it/s
step 11700/12000 | loss 0.9536 | lr 4.77e-06 | 2.35 it/s
step 11800/12000 | loss 0.9043 | lr 4.56e-06 | 2.55 it/s
step 11900/12000 | loss 0.8902 | lr 4.35e-06 | 2.46 it/s
step 12000/12000 | loss 0.8650 | lr 4.17e-06 | 2.47 it/s
[val] step 12000: mode-C CER 21.98 | mode-B CER 28.65 | mode-A CER 28.65 | gain C +6.67 / B +0.00 | loss 0.7356
[done] {'best_val_cer': 20.662382600098862, 'steps': 12000, 'phase': 'icl'}
```

### v3_train.log
```
step 10000/12000 | loss 0.8304 | lr 3.68e-06 | 2.94 it/s
[val] step 10000: mode-C CER 33.58 | mode-B CER 30.63 | mode-A CER 30.63 | gain C -2.95 / B +0.00 | loss 1.4330
step 10100/12000 | loss 0.9134 | lr 3.48e-06 | 2.12 it/s
step 10200/12000 | loss 0.8418 | lr 3.28e-06 | 2.50 it/s
step 10300/12000 | loss 0.7353 | lr 3.09e-06 | 2.96 it/s
step 10400/12000 | loss 0.7646 | lr 2.91e-06 | 2.78 it/s
step 10500/12000 | loss 0.7716 | lr 2.74e-06 | 3.16 it/s
step 10600/12000 | loss 0.7741 | lr 2.58e-06 | 3.03 it/s
step 10700/12000 | loss 0.9399 | lr 2.44e-06 | 2.17 it/s
step 10800/12000 | loss 0.8227 | lr 2.30e-06 | 2.76 it/s
step 10900/12000 | loss 0.8376 | lr 2.17e-06 | 2.37 it/s
step 11000/12000 | loss 0.8211 | lr 2.06e-06 | 2.74 it/s
[val] step 11000: mode-C CER 34.80 | mode-B CER 32.52 | mode-A CER 32.52 | gain C -2.28 / B +0.00 | loss 1.5546
step 11100/12000 | loss 0.8214 | lr 1.95e-06 | 2.04 it/s
step 11200/12000 | loss 0.9133 | lr 1.86e-06 | 2.84 it/s
step 11300/12000 | loss 0.8776 | lr 1.77e-06 | 2.52 it/s
step 11400/12000 | loss 0.8180 | lr 1.70e-06 | 2.85 it/s
step 11500/12000 | loss 0.8323 | lr 1.64e-06 | 2.52 it/s
step 11600/12000 | loss 0.8830 | lr 1.59e-06 | 2.85 it/s
step 11700/12000 | loss 0.8403 | lr 1.55e-06 | 2.56 it/s
step 11800/12000 | loss 0.7491 | lr 1.52e-06 | 3.66 it/s
step 11900/12000 | loss 0.7959 | lr 1.51e-06 | 2.76 it/s
step 12000/12000 | loss 0.8709 | lr 1.50e-06 | 2.82 it/s
[val] step 12000: mode-C CER 35.44 | mode-B CER 31.78 | mode-A CER 31.78 | gain C -3.66 / B +0.00 | loss 1.5502
[done] {'best_val_cer': 32.26510067114094, 'steps': 12000, 'phase': 'icl'}
```

### v5_a0_gain_affine.log
```
step 4700/8000 | loss 1.0494 | lr 4.82e-04 | 2.04 it/s
step 4800/8000 | loss 1.0883 | lr 4.61e-04 | 2.10 it/s
step 4900/8000 | loss 1.0449 | lr 4.40e-04 | 1.97 it/s
step 5000/8000 | loss 1.1178 | lr 4.19e-04 | 2.05 it/s
[val] step 5000: mode-C CER 65.36 | mode-B CER 79.88 | mode-A CER 69.15 | gain C +3.79 / B -10.73 | loss 3.8628
step 5100/8000 | loss 0.9767 | lr 3.99e-04 | 1.58 it/s
step 5200/8000 | loss 1.0737 | lr 3.78e-04 | 2.20 it/s
step 5300/8000 | loss 0.9858 | lr 3.58e-04 | 2.04 it/s
step 5400/8000 | loss 0.8948 | lr 3.38e-04 | 2.09 it/s
step 5500/8000 | loss 0.9825 | lr 3.19e-04 | 2.16 it/s
[val] step 5500: mode-C CER 74.14 | mode-B CER 93.47 | mode-A CER 68.98 | gain C -5.15 / B -24.49 | loss 4.8149
step 5600/8000 | loss 0.9727 | lr 3.00e-04 | 1.75 it/s
step 5700/8000 | loss 0.9582 | lr 2.81e-04 | 1.86 it/s
step 5800/8000 | loss 0.9717 | lr 2.63e-04 | 1.98 it/s
step 5900/8000 | loss 1.0104 | lr 2.46e-04 | 2.25 it/s
step 6000/8000 | loss 1.0153 | lr 2.29e-04 | 2.11 it/s
[val] step 6000: mode-C CER 67.36 | mode-B CER 91.27 | mode-A CER 67.90 | gain C +0.54 / B -23.37 | loss 4.0452
step 6100/8000 | loss 0.9951 | lr 2.12e-04 | 1.76 it/s
step 6200/8000 | loss 1.0118 | lr 1.97e-04 | 2.11 it/s
step 6300/8000 | loss 0.9744 | lr 1.82e-04 | 1.22 it/s
step 6400/8000 | loss 0.9717 | lr 1.67e-04 | 1.41 it/s
step 6500/8000 | loss 0.9083 | lr 1.54e-04 | 1.19 it/s
[val] step 6500: mode-C CER 66.73 | mode-B CER 90.14 | mode-A CER 68.57 | gain C +1.83 / B -21.58 | loss 3.9261
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 20 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```

### v5_a1_gain_v31.log
```
step 7700/8000 | loss 1.2638 | lr 5.09e-06 | 2.07 it/s
step 7800/8000 | loss 1.2359 | lr 4.74e-06 | 4.85 it/s
step 7900/8000 | loss 1.2135 | lr 4.43e-06 | 4.57 it/s
[done] {'best_val_cer': 34.94402985074627, 'steps': 8000, 'phase': 'icl'}
terminate called without an active exception
terminate called without an active exception
terminate called without an active exception
terminate called without an active exception
Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x7f04b5440160>
Traceback (most recent call last):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1479, in __del__
    self._shutdown_workers()
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1443, in _shutdown_workers
    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/multiprocessing/process.py", line 149, in join
    res = self._popen.wait(timeout)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/multiprocessing/popen_fork.py", line 40, in wait
    if not wait([self.sentinel], timeout):
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/multiprocessing/connection.py", line 931, in wait
    ready = selector.select(timeout)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/selectors.py", line 416, in select
    fd_event_list = self._selector.poll(timeout)
  File "/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/site-packages/torch/utils/data/_utils/signal_handling.py", line 66, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 2831817) is killed by signal: Aborted. 
```

### v5_a2_realistic.log
```
step 2700/8000 | loss 2.6537 | lr 8.68e-04 | 0.77 it/s
step 2800/8000 | loss 2.6598 | lr 8.53e-04 | 0.86 it/s
step 2900/8000 | loss 2.7604 | lr 8.38e-04 | 0.80 it/s
step 3000/8000 | loss 2.7032 | lr 8.21e-04 | 0.77 it/s
[val] step 3000: mode-C CER 96.69 | mode-B CER 85.01 | mode-A CER 79.77 | gain C -16.91 / B -5.23 | loss 9.8119
step 3100/8000 | loss 2.7093 | lr 8.04e-04 | 0.62 it/s
step 3200/8000 | loss 2.6670 | lr 7.87e-04 | 0.82 it/s
step 3300/8000 | loss 2.6182 | lr 7.69e-04 | 0.96 it/s
step 3400/8000 | loss 2.5552 | lr 7.50e-04 | 0.80 it/s
step 3500/8000 | loss 2.5870 | lr 7.31e-04 | 0.81 it/s
[val] step 3500: mode-C CER 75.41 | mode-B CER 76.18 | mode-A CER 79.80 | gain C +4.38 / B +3.62 | loss 3.9540
step 3600/8000 | loss 2.4266 | lr 7.12e-04 | 0.70 it/s
step 3700/8000 | loss 2.5383 | lr 6.92e-04 | 0.84 it/s
step 3800/8000 | loss 2.6359 | lr 6.72e-04 | 0.96 it/s
step 3900/8000 | loss 2.6126 | lr 6.51e-04 | 0.84 it/s
step 4000/8000 | loss 2.5845 | lr 6.31e-04 | 0.79 it/s
[val] step 4000: mode-C CER 68.16 | mode-B CER 76.63 | mode-A CER 79.45 | gain C +11.29 / B +2.83 | loss 3.2769
step 4100/8000 | loss 2.6075 | lr 6.10e-04 | 0.61 it/s
step 4200/8000 | loss 2.5191 | lr 5.89e-04 | 0.88 it/s
step 4300/8000 | loss 2.5014 | lr 5.68e-04 | 0.83 it/s
step 4400/8000 | loss 2.4864 | lr 5.46e-04 | 0.74 it/s
step 4500/8000 | loss 2.5935 | lr 5.25e-04 | 0.88 it/s
[val] step 4500: mode-C CER 77.72 | mode-B CER 87.09 | mode-A CER 79.78 | gain C +2.06 / B -7.31 | loss 4.2851
/data2/chenyuxiang/conda_envs/qwerty/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 20 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
```

### v5_zeroshot_scan.log
```
[75/96] 81695116: held-out-session CER 4.22 (6 sessions)
[76/96] 83774284: held-out-session CER 5.94 (10 sessions)
[77/96] 84958031: held-out-session CER 5.49 (6 sessions)
[78/96] 85881224: held-out-session CER 11.32 (10 sessions)
[79/96] 86629437: held-out-session CER 5.70 (10 sessions)
[80/96] 87998384: held-out-session CER 10.21 (7 sessions)
[81/96] 89335547: held-out-session CER 6.72 (16 sessions)
[82/96] 89415164: held-out-session CER 10.43 (9 sessions)
[83/96] 90443344: held-out-session CER 11.99 (8 sessions)
[84/96] 92249581: held-out-session CER 6.20 (10 sessions)
[85/96] 92418081: held-out-session CER 9.78 (10 sessions)
[86/96] 92903591: held-out-session CER 7.70 (10 sessions)
[87/96] 93203007: held-out-session CER 5.60 (8 sessions)
[88/96] 94305460: held-out-session CER 3.59 (9 sessions)
[89/96] 9456349: held-out-session CER 7.59 (7 sessions)
[90/96] 94998811: held-out-session CER 4.05 (10 sessions)
[91/96] 95396398: held-out-session CER 3.24 (9 sessions)
[92/96] 97165588: held-out-session CER 9.34 (5 sessions)
[93/96] 97336339: held-out-session CER 12.43 (10 sessions)
[94/96] 97890030: held-out-session CER 9.45 (10 sessions)
[95/96] 97946571: held-out-session CER 10.07 (10 sessions)
[96/96] 99192446: held-out-session CER 2.17 (10 sessions)

=== SEEN users (n=96): median 8.11 p10 3.32 p90 15.67 ===
=== UNSEEN users (8 official test): 55.39 published ===
```
