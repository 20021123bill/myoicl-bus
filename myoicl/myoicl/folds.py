# Copyright (c) 2026 MyoICL authors. MIT License.
"""User-level fold holdout for meta-training.

WHY (2026-08-19). Measured today: the released backbone's CER on a held-out
SESSION of a user it was TRAINED on is a median 8.11 (n=96, p10 3.32, p90
15.67), versus 55.39 on the 8 unseen test users. Per-user gradient adaptation
on those training users gains +0.00. So an episode drawn from a user the
backbone has seen contains NO adaptation signal, and that is why every v1-v3.2
context architecture converged to ignoring its context.

Training our own backbone does not fix this by itself -- any backbone that saw
user u has memorised u. The fix is to make sure that the backbone used to
score an episode has NOT seen that episode's user:

    fold f:  backbone_f is trained on the users NOT in fold f
             the users IN fold f are genuinely novel to backbone_f
             -> episodes from fold f carry real, ~55-CER-sized headroom

With F folds every one of the 96 training users becomes a valid novel-subject
meta-training task exactly once, at the cost of F backbone trainings. The 8
official test users are never in any fold and are never trained on by any
backbone, so the final number stays clean under every fold.

The split is deterministic (sorted user ids, round-robin) so that every job,
rerun and paper table refers to the same partition without carrying a file
around.
"""
from __future__ import annotations


def user_folds(repo_root: str, n_folds: int = 4, data_root=None):
    """-> (folds, sessions_by_user).

    folds[i] is the sorted list of user ids held out from backbone_i.
    Round-robin over sorted ids keeps the folds balanced in count; sessions
    per user vary (2..16) so fold sizes in HOURS differ slightly, which is
    reported by fold_report() rather than balanced away -- balancing on hours
    would make the split depend on the data version.
    """
    from .qwerty_data import group_by_user, load_user_sessions

    s = load_user_sessions(repo_root, "generic", data_root)
    by_user = {u: p for u, p in sorted(group_by_user(s["train"]).items())}
    ids = list(by_user)
    folds = [sorted(ids[i::n_folds]) for i in range(n_folds)]
    return folds, by_user


def split_for_fold(repo_root: str, fold: int, n_folds: int = 4, data_root=None):
    """-> (train_pairs, heldout_pairs, heldout_users).

    train_pairs   (user, path) for every user NOT in this fold -- what
                  backbone_fold is trained on.
    heldout_pairs (user, path) for the users IN this fold -- novel subjects
                  for backbone_fold, i.e. the meta-training/validation tasks
                  that actually contain adaptation headroom.
    """
    folds, by_user = user_folds(repo_root, n_folds, data_root)
    held = set(folds[fold]) if 0 <= fold < n_folds else set()
    tr, ho = [], []
    for u, paths in by_user.items():
        (ho if u in held else tr).extend((u, p) for p in paths)
    return tr, ho, sorted(held)


def fold_report(repo_root: str, n_folds: int = 4, data_root=None) -> str:
    folds, by_user = user_folds(repo_root, n_folds, data_root)
    lines = [f"{len(by_user)} training users -> {n_folds} folds"]
    for i, f in enumerate(folds):
        n_sess = sum(len(by_user[u]) for u in f)
        lines.append(f"  fold {i}: {len(f):3d} users, {n_sess:4d} sessions "
                     f"| e.g. {f[:3]}")
    return "\n".join(lines)
