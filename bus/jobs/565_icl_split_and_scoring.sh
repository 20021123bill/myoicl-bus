set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export PYTHONUNBUFFERED=1

# =============================================================================
# 565: the two real bugs behind "gain = 0", found by reading the trainer.
#
# BUG 1 -- META-TRAIN AND META-VAL ARE THE SAME 24 USERS.
#   tr_ep = UserEpisodes(held_pairs, ...)      # the fold's 24 novel users
#   va_ep = UserEpisodes(held_pairs, ...)      # THE SAME 24 USERS
#   The trunk is trainable (correctly -- BrainCoDec freezes nothing), so every
#   step it fits the very users the gain is measured on.  Evidence, fold 0:
#   mode-A 53.72 (step 0) -> 43.53 -> 45.36 -> 43.13 -> 43.14.  The trunk ate
#   10.6 CER of adaptation headroom in 6k steps.  Context cannot earn what the
#   weights have already taken.  Fix: disjoint meta-train (16) / meta-test (8)
#   users inside the fold cohort.  Both stay novel to the BACKBONE, so the
#   contamination guard still binds; meta-test additionally stays novel to the
#   ICL run itself.
#
# BUG 2 -- WE NEVER SCORE THE TASK WE TRAIN ON.
#   Training draws p_permute = 0.5 (symbol map) and p_synth (transform): the
#   only episodes with unmemorisable headroom.  The aux heads prove the encoder
#   reads them (perm 1.6 vs chance 3.26).  But validate() pins allow_permute =
#   False, so the end-to-end CER number is measured ONLY on the identity task
#   -- exactly the task where memorising the user beats reading the context.
#   Fix: score BOTH.  The permuted-task gain (A' vs C') is the honest ICL
#   readout: with a novel gesture->symbol mapping, mode A is near-chance and
#   only 3 minutes of labelled support can recover it.
#
# Also adds a memorisation gauge: mode-A on the META-TRAIN users, printed next
# to mode-A on the META-TEST users every validation.  If the first collapses
# while the second holds, bug 1's mechanism is visible in one line.
# =============================================================================

python - <<'PY'
import re, sys
p = "myoicl/train_prefix_icl.py"
src = open(p).read()
orig = src

# ---- patch 1: disjoint meta-train / meta-test users -------------------------
A1 = """    tr_ep = UserEpisodes(held_pairs, seed=args.seed)
    va_ep = UserEpisodes(held_pairs, seed=args.seed + 1000)
"""
B1 = '''    # META-SPLIT (2026-08-20). The cohort's users are novel to the BACKBONE,
    # but the ICL run trains the trunk too, so users used for meta-training
    # stop being novel to THIS run within a few thousand steps. Measuring gain
    # on them reports the headroom the trunk already consumed. Deterministic
    # 2:1 split: every third user is meta-test and is never drawn for a
    # gradient step.
    _cohort = sorted(held_users)
    meta_te = [u for i, u in enumerate(_cohort) if i % 3 == 2]
    meta_tr = [u for i, u in enumerate(_cohort) if i % 3 != 2]
    assert not (set(meta_tr) & set(meta_te))
    tr_pairs = [(u, p) for (u, p) in held_pairs if u in set(meta_tr)]
    te_pairs = [(u, p) for (u, p) in held_pairs if u in set(meta_te)]
    print(f"[meta-split] meta-train {len(meta_tr)} users / {len(tr_pairs)} "
          f"sessions | meta-test {len(meta_te)} users / {len(te_pairs)} "
          f"sessions (disjoint, both unseen by the backbone)")
    tr_ep = UserEpisodes(tr_pairs, seed=args.seed)
    va_ep = UserEpisodes(te_pairs, seed=args.seed + 1000)
    mem_ep = UserEpisodes(tr_pairs, seed=args.seed + 2000)
'''

# ---- patch 2: validate() scores identity AND permuted, plus memorisation ----
A2 = '''            u, sb, qb, _, _, _ = draw(va_ep, allow_permute=False,
                                      allow_synth=False)
            for mode in ("A", "C"):
                _, em, in_len, _ = run_episode(sb, qb, mode, train=False)
                preds = greedy_ctc_decode(em.float(), in_len.cpu(),
                                          blank=cs.null_class)
                tg, tl = qb["targets"].numpy(), qb["target_lengths"].numpy()
                for n, p in enumerate(preds):
                    accs[mode].update(
                        LabelData.from_labels(p).text,
                        LabelData.from_labels(tg[: tl[n], n]).text)
        trunk.train(); enc.train()
        return accs["A"].cer, accs["C"].cer
'''
B2 = '''            u, sb, qb, _, _, _ = draw(va_ep, allow_permute=False,
                                      allow_synth=False)
            def _score(sbx, qbx, key):
                _, em, in_len, _ = run_episode(sbx, qbx, key[0], train=False)
                preds = greedy_ctc_decode(em.float(), in_len.cpu(),
                                          blank=cs.null_class)
                tg, tl = qbx["targets"].numpy(), qbx["target_lengths"].numpy()
                for n, p in enumerate(preds):
                    accs[key].update(
                        LabelData.from_labels(p).text,
                        LabelData.from_labels(tg[: tl[n], n]).text)
            for mode in ("A", "C"):
                _score(sb, qb, mode)
            # THE TASK WE ACTUALLY TRAIN ON, scored end to end for the first
            # time. A fixed rng makes the permutation set identical at every
            # validation, so the curve is comparable across steps and does not
            # perturb the training draw sequence.
            if LETTERS:
                vrng = np.random.default_rng(12345)
                m = sample_symbol_map(vrng, LETTERS, args.permute_k[1],
                                      cs.num_classes)
                sbp = _apply_symbol_map(dict(sb), m)
                qbp = _apply_symbol_map(dict(qb), m)
                _score(sbp, qbp, "Ap")
                _score(sbp, qbp, "Cp")
        # MEMORISATION GAUGE: zero-context difficulty on the users this run
        # trains on. Falling here while mode-A on meta-test holds = the trunk
        # is absorbing subjects rather than the encoder learning to read them.
        for _ in range(max(4, args.val_episodes // 3)):
            u, sb, qb, _, _, _ = draw(mem_ep, allow_permute=False,
                                      allow_synth=False)
            _, em, in_len, _ = run_episode(sb, qb, "A", train=False)
            preds = greedy_ctc_decode(em.float(), in_len.cpu(),
                                      blank=cs.null_class)
            tg, tl = qb["targets"].numpy(), qb["target_lengths"].numpy()
            for n, p in enumerate(preds):
                accs["mem"].update(
                    LabelData.from_labels(p).text,
                    LabelData.from_labels(tg[: tl[n], n]).text)
        trunk.train(); enc.train()
        return (accs["A"].cer, accs["C"].cer, accs["Ap"].cer,
                accs["Cp"].cer, accs["mem"].cer)
'''

A3 = '''        accs = {"A": CERAccumulator(), "C": CERAccumulator()}'''
B3 = '''        accs = {k: CERAccumulator()
                for k in ("A", "C", "Ap", "Cp", "mem")}'''

A4 = "    a0, c0 = validate()"
B4 = "    a0, c0, ap0, cp0, mem0 = validate()"

A5 = '''            a, c = validate()
            print(f"[val] step {step+1}: mode-A {a:.2f} | mode-C {c:.2f} | "
                  f"gain C {a - c:+.2f}   (REAL novel subjects, fold "
                  f"{args.fold})", flush=True)
            hist.append({"step": step + 1, "A": a, "C": c, "gain": a - c})'''
B5 = '''            a, c, ap, cp, memA = validate()
            print(f"[val] step {step+1}: IDENTITY A {a:.2f} C {c:.2f} "
                  f"gain {a - c:+.2f} | PERMUTED A {ap:.2f} C {cp:.2f} "
                  f"gain {ap - cp:+.2f} | mem-gauge(meta-train A) "
                  f"{memA:.2f}   (meta-test users, fold {args.fold})",
                  flush=True)
            hist.append({"step": step + 1, "A": a, "C": c, "gain": a - c,
                         "Ap": ap, "Cp": cp, "gain_perm": ap - cp,
                         "memA": memA})'''

A6 = '''            json.dump({"args": vars(args), "cohort": held_users,'''
B6 = '''            json.dump({"args": vars(args), "cohort": held_users,
                       "meta_train_users": meta_tr,
                       "meta_test_users": meta_te,'''

for i, (a, b) in enumerate([(A1, B1), (A2, B2), (A3, B3), (A4, B4),
                            (A5, B5), (A6, B6)], start=1):
    n = src.count(a)
    if n != 1:
        sys.exit(f"[FATAL] patch {i}: anchor found {n} times, expected 1. "
                 f"Refusing to write a half-patched trainer.")
    src = src.replace(a, b)

# audit print references c0 only; keep it valid
src = src.replace(
    '    print(f"[audit] step 0: mode-A {a0:.2f} | mode-C {c0:.2f} '
    '(random prefix)"',
    '    print(f"[audit] step 0: mode-A {a0:.2f} | mode-C {c0:.2f} '
    '(random prefix)"')

assert src != orig
open(p, "w").write(src)
print("[patched] myoicl/train_prefix_icl.py -- 6 anchors replaced")
PY

echo "=== grep verification (unverified patch = no patch) ==="
grep -n "meta-split\|mem_ep\|gain_perm\|PERMUTED A" myoicl/train_prefix_icl.py \
  | head -10
python -c "import ast;ast.parse(open('myoicl/train_prefix_icl.py').read());print('[syntax] OK')"

echo
echo "=== relaunch: fold 0 and fold 1, correct split + both tasks scored ==="
R=/data2/chenyuxiang/runs
for F in 0 1; do
  GPU=$F
  OUT=$R/icl_split_fold$F
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=$GPU nohup python -m myoicl.train_prefix_icl \
    --backbone $R/tf_fold${F}_full/last.pt --fold $F \
    --out-dir "$OUT" --fused-prefix \
    --p-synth 0.5 --p-permute 0.5 --p-modeA 0.25 \
    --k-support 6 18 --max-steps 20000 --val-every 500 --val-episodes 24 \
    --lr 3e-4 --trunk-lr-mult 0.1 --w-aux 1.0 \
    > "$OUT/train.log" 2>&1 &
  echo "launched fold $F on GPU $GPU -> $OUT/train.log"
  sleep 5
done
sleep 240
echo "=== first 240 s of each ==="
for F in 0 1; do
  echo "--- fold $F ---"
  tail -n 20 $R/icl_split_fold$F/train.log
done
echo "=== 565 launched ==="
