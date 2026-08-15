#!/usr/bin/env python
"""Learning-curve experiment — does train-set SIZE limit validation F1?

Question (raised 2026-08-14): everything so far was trained on the full
gemma_v2 train split (1985 rows / 800 groups). No run ever varied train-set
size, so the size->performance relationship is unknown. This script measures
it directly: train the FINAL LOCKED LoRA config (lr=3e-4, r=8, alpha=16,
dropout=0.0 — the Phase 2 / multiseed winner "Config A") on nested,
group-aware, label-stratified subsets of train.jsonl and evaluate each on
the same untouched 259-row validation split.

Run plan (7 points total, 5 new trainings + 2 read-only references):
    seed 42: 100, 200, 400, 600 groups  -> trained here
    seed 42: 800 groups (full train)    -> REFERENCE, reused from
        results/gemma_lora_v2_sweep_phase2/lr3e-4_r8_alpha16_dropout0.0_seed42/
        (identical config + data + seed; retraining would be a bit-for-bit
        redo of a finished run — same reuse policy as Phase 1/2 references)
    seed 23: 200 groups                 -> trained here (seed-noise check)
    seed 23: 800 groups (full train)    -> REFERENCE, reused from
        results/gemma_lora_v2_multiseed/config_a_lr3e-4_r8_alpha16_dropout0.0_seed23/

Subset construction (deterministic, nested, stratified):
    Groups (original_idx) are label-pure (verified: 0/800 mixed) and exactly
    400 harmful / 400 unharmful. The two pools are shuffled once with
    Random(SUBSET_SEED); a k-group subset takes the first k//2 groups of each
    pool, so subsets are nested (100 ⊂ 200 ⊂ 400 ⊂ 600 ⊂ 800) and every
    subset keeps the full-data ~60/40 row-label balance. Subset membership
    depends ONLY on SUBSET_SEED — the seed-23 run at 200 groups trains on
    exactly the same rows as the seed-42 run at 200 groups.

Everything except the train subset is identical to the Phase 2 recipe: the
training/eval machinery is imported from sweep_lora_v2_phase2.py (never
reimplemented), same tokenizer/prompt/truncation/early-stopping/generation.
The held-out split is never opened (self-audit in --dry-run).

Usage:
    ~/ccpp_env/bin/python learning_curve_v2.py --dry-run
    ~/ccpp_env/bin/python learning_curve_v2.py

Safe to re-run: completed runs are skipped, failed runs retried up to
--max-attempts times, a run found "running" at startup (previous process
died mid-training) is archived and restarted from the base model — Trainer
state is never resumed mid-training (same policy as Phase 1/2/multiseed).
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Required env-var block — MUST run before importing torch/transformers
# (uid 1562 has no /etc/passwd entry; see sweep_lora_v2_phase2.py header).
# ---------------------------------------------------------------------------
os.environ.setdefault("USER", "mls01")
os.environ.setdefault("LOGNAME", "mls01")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/mls01/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/home/mls01/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/home/mls01/.cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_lora_v2_phase2 as p2  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed paths and constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("/home/mls01/scripts/model/results/learning_curve_v2")
LOCK_PATH = RESULTS_DIR / "learning_curve.pid"
SUMMARY_CSV = RESULTS_DIR / "learning_curve_summary.csv"
STATE_JSON = RESULTS_DIR / "learning_curve_state.json"
REPORT_MD = RESULTS_DIR / "REPORT.md"

# Read-only reference runs (identical locked config on the FULL 800-group
# train split). Never written to by this script.
REFERENCES = [
    {
        "n_groups": 800, "seed": 42, "source": "reference_phase2",
        "run_summary": Path("/home/mls01/scripts/model/results/gemma_lora_v2_sweep_phase2/"
                            "lr3e-4_r8_alpha16_dropout0.0_seed42/run_summary.json"),
    },
    {
        "n_groups": 800, "seed": 23, "source": "reference_multiseed",
        "run_summary": Path("/home/mls01/scripts/model/results/gemma_lora_v2_multiseed/"
                            "config_a_lr3e-4_r8_alpha16_dropout0.0_seed23/run_summary.json"),
    },
]

# Locked config (final winner of Phase 2 + multiseed: "Config A").
LOCKED = {"learning_rate": 3e-4, "rank": 8, "alpha": 16, "dropout": 0.0}

# New training runs: (n_groups, seed), in run order (smallest first — if the
# curve is obviously flat early, later runs can be cancelled sooner).
NEW_RUNS = [(100, 42), (200, 42), (400, 42), (600, 42), (200, 23)]

SUBSET_SEED = 42          # subset membership RNG — independent of training seed
GROUP_LABELS = ["harmful", "unharmful"]
DEFAULT_MAX_ATTEMPTS = 2

# Built via concatenation so the held-out filename never appears literally
# in this file's source (self-audit in do_dry_run). Never opened.
_HELD_OUT_SPLIT_NAME = "test" + "." + "jsonl"


def now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def run_id(n_groups, seed):
    return f"groups{n_groups}_seed{seed}"


# ---------------------------------------------------------------------------
# Subset construction — nested, group-aware, label-stratified
# ---------------------------------------------------------------------------
def build_nested_subsets(train_df):
    """Return {n_groups: sub-dataframe} for every size needed by NEW_RUNS.

    Groups are label-pure (asserted), so stratification is exact at group
    level: shuffle each label pool once with Random(SUBSET_SEED), then take
    nested prefixes (k//2 groups from each pool for a k-group subset).
    """
    group_label = train_df.groupby("original_idx")["final_label"].first()
    assert group_label.nunique() == 2 and not (
        train_df.groupby("original_idx")["final_label"].nunique() > 1
    ).any(), "groups must be label-pure for stratified nested subsets"
    pools = {label: sorted(group_label[group_label == label].index) for label in GROUP_LABELS}
    rng = random.Random(SUBSET_SEED)
    for label in GROUP_LABELS:
        rng.shuffle(pools[label])

    needed = sorted({n for n, _ in NEW_RUNS} | {800})
    subsets = {}
    for k in needed:
        half = k // 2
        chosen = set(pools["harmful"][:half]) | set(pools["unharmful"][:half])
        subsets[k] = train_df[train_df["original_idx"].isin(chosen)].reset_index(drop=True)
    # Nestedness check: each smaller subset's groups are contained in the next.
    for small, big in zip(needed, needed[1:]):
        assert set(subsets[small]["original_idx"]) <= set(subsets[big]["original_idx"])
    return subsets


# ---------------------------------------------------------------------------
# Reference rows (read-only)
# ---------------------------------------------------------------------------
def load_reference_rows():
    rows = []
    for ref in REFERENCES:
        summary = json.loads(ref["run_summary"].read_text())
        m = summary["best_metrics"]
        rows.append({
            "run_id": run_id(ref["n_groups"], ref["seed"]),
            "n_groups": ref["n_groups"], "n_rows": 1985, "seed": ref["seed"],
            "status": "completed", "best_epoch": summary["best_epoch"],
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "invalid_rate": m["invalid_rate"],
            "duration_seconds": summary.get("total_training_duration_seconds"),
            "source": ref["source"], "run_path": str(ref["run_summary"].parent),
        })
    return rows


# ---------------------------------------------------------------------------
# Summary / state / report
# ---------------------------------------------------------------------------
def collect_trained_rows():
    rows = []
    for n_groups, seed in NEW_RUNS:
        run_dir = RESULTS_DIR / run_id(n_groups, seed)
        status = p2.read_status(run_dir)
        summary_path = run_dir / "run_summary.json"
        row = {"run_id": run_id(n_groups, seed), "n_groups": n_groups, "seed": seed,
               "status": status.get("status", "pending"), "source": "trained_here",
               "run_path": str(run_dir)}
        meta_path = run_dir / "subset_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            row["n_rows"] = meta["n_rows"]
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            m = s["best_metrics"]
            row.update({"best_epoch": s["best_epoch"], "precision": m["precision"],
                        "recall": m["recall"], "f1": m["f1"], "invalid_rate": m["invalid_rate"],
                        "duration_seconds": s.get("total_training_duration_seconds")})
        rows.append(row)
    return rows


def write_summary():
    rows = collect_trained_rows() + load_reference_rows()
    df = pd.DataFrame(rows).sort_values(["seed", "n_groups"]).reset_index(drop=True)
    df.to_csv(SUMMARY_CSV, index=False)
    return df


def write_state(incomplete, started_at):
    STATE_JSON.write_text(json.dumps({
        "started_at": started_at, "updated_at": now_iso(), "incomplete": incomplete,
        "locked_config": LOCKED, "subset_seed": SUBSET_SEED,
        "new_runs": [list(r) for r in NEW_RUNS],
        "references": [{k: str(v) for k, v in ref.items()} for ref in REFERENCES],
    }, indent=2))


def write_report(df):
    """Auto-generated narrative. Regenerated after every run, so it always
    reflects the current state (partial curves included)."""
    done = df[df["status"] == "completed"].copy()
    lines = []
    lines.append("# Learning curve v2 — does train-set size limit validation F1?\n")
    lines.append(f"_Auto-generated by `learning_curve_v2.py`, last update: {now_iso()}_\n")
    lines.append("## Question\n")
    lines.append(
        "Every prior training run used the full gemma_v2 train split (1985 rows / 800 groups), "
        "so the dataset-size → validation-performance relationship was unmeasurable from existing "
        "results. This experiment trains the **final locked LoRA config** "
        "(`lr=3e-4, r=8, alpha=16, dropout=0.0` — the Phase 2 / multiseed winner) on nested, "
        "group-aware, label-stratified subsets of `train.jsonl` and evaluates each on the same "
        "untouched 259-row validation split. The held-out test split was never touched.\n")
    lines.append("## Method\n")
    lines.append(
        f"- Subsets: nested (100 ⊂ 200 ⊂ 400 ⊂ 600 ⊂ 800 groups), stratified 50/50 at group level "
        f"(groups are label-pure), membership fixed by `SUBSET_SEED={SUBSET_SEED}` — identical "
        f"subsets across training seeds.\n"
        f"- Training/eval pipeline: imported unchanged from `sweep_lora_v2_phase2.py` "
        f"(same tokenizer, `PROMPT_1`, head-tail truncation, F1 early stopping, greedy generation).\n"
        f"- The two 800-group points are **read-only references** of already-completed runs with "
        f"identical config+data+seed (Phase 2 seed 42, multiseed seed 23) — not retrained.\n"
        f"- Seed 23 at 200 groups (plus the existing seed-23 800-group reference) gives a "
        f"seed-noise reading at two curve points.\n")
    lines.append("## Results (validation, 259 rows)\n")
    lines.append("| n_groups | n_rows | seed | best_epoch | precision | recall | F1 | invalid | source |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in df.itertuples(index=False):
        if r.status != "completed":
            lines.append(f"| {r.n_groups} | {getattr(r, 'n_rows', '')} | {r.seed} | — | — | — | — | — | {r.status} |")
            continue
        lines.append(f"| {r.n_groups} | {int(r.n_rows)} | {r.seed} | {int(r.best_epoch)} "
                     f"| {r.precision:.4f} | {r.recall:.4f} | **{r.f1:.4f}** "
                     f"| {r.invalid_rate:.3f} | {r.source} |")
    lines.append("")

    c42 = done[done["seed"] == 42].sort_values("n_groups")
    if len(c42) >= 2:
        import math
        lines.append("## Analysis (seed-42 curve)\n")
        pts = [(r.n_rows, r.f1) for r in c42.itertuples(index=False)]
        for (n0, f0), (n1, f1_) in zip(pts, pts[1:]):
            lines.append(f"- {n0} → {n1} rows (×{n1/n0:.2f}): ΔF1 = **{f1_ - f0:+.4f}**")
        # Log-linear fit F1 = a + b*log2(n_rows) over all completed seed-42 points.
        xs = [math.log2(n) for n, _ in pts]
        ys = [f for _, f in pts]
        xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
        b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum((x - xbar) ** 2 for x in xs)
        a = ybar - b * xbar
        lines.append(f"\nLog-linear fit `F1 ≈ {a:.4f} + {b:.4f}·log2(n_rows)` over {len(pts)} points "
                     f"→ **{b:+.4f} F1 per doubling of data**.")
        for mult in (2, 4):
            n_now = pts[-1][0]
            extrap = a + b * math.log2(n_now * mult)
            lines.append(f"- Naive extrapolation to ×{mult} data ({n_now * mult} rows): F1 ≈ {extrap:.4f} "
                         f"(caution: power-law extrapolation from {len(pts)} noisy points)")
        lines.append("")
        seed23 = done[done["seed"] == 23].set_index("n_groups")["f1"]
        seed42 = c42.set_index("n_groups")["f1"]
        common = sorted(set(seed23.index) & set(seed42.index))
        if common:
            lines.append("## Seed noise check\n")
            for k in common:
                lines.append(f"- {k} groups: seed42 F1 {seed42[k]:.4f} vs seed23 F1 {seed23[k]:.4f} "
                             f"(Δ {seed42[k] - seed23[k]:+.4f})")
            lines.append("\nMultiseed context (full data): Config A seed std was ±0.0032 F1 — "
                         "deltas within ~±0.005 are indistinguishable from seed noise.\n")
        if len(c42) >= 5:
            last_delta = pts[-1][1] - pts[-2][1]
            lines.append("## Verdict\n")
            if b > 0.005 and last_delta > 0.003:
                lines.append(
                    f"**The curve is still rising at full data** ({b:+.4f} F1/doubling, last segment "
                    f"{last_delta:+.4f}). Scaling this dataset further (more of the same distribution) "
                    f"would likely still help.\n")
            else:
                lines.append(
                    f"**The curve is essentially flat near full data** ({b:+.4f} F1/doubling, last "
                    f"segment {last_delta:+.4f} — within seed noise). More of the *same* data is "
                    f"unlikely to help much; the known blind spots (no benign-prompt+harmful-response "
                    f"rows, obfuscation/framing weak spots, multi-turn) are coverage problems that "
                    f"require *different* data, not *more* data.\n")
            lines.append("\nCaveats: validation is only 259 rows (1 FN ≈ 0.6pp recall), so differences "
                         "under ~0.01 F1 are weak evidence; val→test ranking inversions have occurred "
                         "before at this val size.\n")
    REPORT_MD.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def do_dry_run(train_df):
    print("[DRY-RUN] Learning curve v2 — plan i provere, bez modela/treninga.\n")
    # Self-audit: the held-out split filename must not appear in this source.
    src = Path(__file__).read_text()
    assert _HELD_OUT_SPLIT_NAME not in src, "held-out split filename found in source!"
    print(f"[OK] Self-audit: '{_HELD_OUT_SPLIT_NAME}' se ne pojavljuje u izvornom kodu.")

    for ref in REFERENCES:
        assert ref["run_summary"].exists(), f"reference run_summary missing: {ref['run_summary']}"
    print(f"[OK] Oba reference run_summary.json postoje ({len(REFERENCES)}).")

    subsets = build_nested_subsets(train_df)
    print("\nSubset composition (nested, stratified):")
    for k in sorted(subsets):
        sub = subsets[k]
        vc = sub["final_label"].value_counts().to_dict()
        tag = " (reference only — not retrained)" if k == 800 else ""
        print(f"  {k:>4} groups -> {len(sub):>5} rows | harmful {vc.get('harmful', 0)} / "
              f"unharmful {vc.get('unharmful', 0)}{tag}")
    print("\nNew training runs (in order):")
    for n_groups, seed in NEW_RUNS:
        print(f"  {run_id(n_groups, seed)}  <-  {LOCKED}")
    print("\n[DRY-RUN] Sve provere prošle.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                       help="Prikaži plan i provere bez učitavanja modela ili treninga.")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    args = parser.parse_args()

    train_df = pd.read_json(p2.TRAIN_PATH, lines=True)
    val_df = pd.read_json(p2.VAL_PATH, lines=True)
    assert len(train_df) == 1985 and train_df["original_idx"].nunique() == 800
    assert len(val_df) == 259 and val_df["original_idx"].nunique() == 100
    assert not (set(train_df["original_idx"]) & set(val_df["original_idx"]))

    if args.dry_run:
        do_dry_run(train_df)
        return

    # Point the imported module's lock at OUR results dir (its lock helpers
    # read module globals at call time).
    p2.SWEEP_DIR = RESULTS_DIR
    p2.LOCK_PATH = LOCK_PATH
    p2.acquire_lock()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    print(f"[START] Learning curve v2 pokrenut {started_at} (PID {os.getpid()}). "
          f"GPU slobodno pre starta provereno ručno.")

    subsets = build_nested_subsets(train_df)
    tokenizer, _eot, build_example, parse_label = p2.build_tokenizer_and_helpers()
    val_examples, _ = p2.build_split(val_df, "validation", build_example)
    val_prefixes = p2.build_val_prefixes(val_df, build_example)
    evaluate_checkpoint = p2.make_evaluate_checkpoint(tokenizer, parse_label)
    SFTDataset = p2.make_sft_dataset_cls()
    collate_fn = p2.make_collate_fn(tokenizer)

    incomplete = False
    for n_groups, seed in NEW_RUNS:
        rid = run_id(n_groups, seed)
        run_dir = RESULTS_DIR / rid
        status_data = p2.read_status(run_dir)
        status, attempt = status_data.get("status", "pending"), status_data.get("attempt", 0)

        if status == "completed":
            print(f"[SKIP] {rid} već completed.")
            continue
        if status == "running":
            print(f"[INTERRUPTED] {rid} je bio 'running' — prethodni proces prekinut.")
            p2.archive_stale_run(run_dir, attempt + 1)
        if status == "failed" and attempt >= args.max_attempts:
            print(f"[SKIP] {rid} već failed {attempt}x (max-attempts={args.max_attempts}).")
            incomplete = True
            continue

        sub = subsets[n_groups]
        cfg = {"config_id": rid, "seed": seed, **LOCKED}
        next_attempt = attempt + 1
        print(f"\n[RUN] {rid} (pokušaj {next_attempt}/{args.max_attempts}) — "
              f"{len(sub)} rows / {n_groups} groups")
        t0 = time.time()
        train_examples, _ = p2.build_split(sub, f"train_subset_{n_groups}", build_example)
        result = p2.run_one_config(cfg, run_dir, next_attempt, tokenizer, train_examples,
                                   val_examples, val_df, val_prefixes, evaluate_checkpoint,
                                   SFTDataset, collate_fn)
        print(f"[{result['status'].upper()}] {rid} ({result['duration']:.0f}s)")

        if result["status"] == "completed":
            # Patch run_config.json: correct phase label + subset provenance.
            rc_path = run_dir / "run_config.json"
            rc = json.loads(rc_path.read_text())
            rc["phase"] = "learning_curve_v2"
            rc["subset"] = {"n_groups": n_groups, "n_rows": len(sub), "subset_seed": SUBSET_SEED,
                            "nested": True, "stratified_by_group_label": True}
            rc_path.write_text(json.dumps(rc, indent=2, ensure_ascii=False))
            (run_dir / "subset_meta.json").write_text(json.dumps({
                "n_groups": n_groups, "n_rows": len(sub), "subset_seed": SUBSET_SEED,
                "group_ids": sorted(sub["original_idx"].unique().tolist()),
            }, indent=2))
        else:
            incomplete = True

        summary_df = write_summary()
        write_report(summary_df)
        write_state(incomplete, started_at)

    summary_df = write_summary()
    write_report(summary_df)
    write_state(incomplete, started_at)
    print("\n" + "=" * 90)
    print(f"LEARNING CURVE {'NEPOTPUN — nastaviće se pri sledećem pokretanju' if incomplete else 'KOMPLETAN'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
