# heartbeat 2026-08-18T20:26:07+08:00

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
=== gate report (pre-v4.1 checkpoints) ===
```

### 020_d1_gatefix.log
```
```

### 030_d2_forcectx.log
```
```
