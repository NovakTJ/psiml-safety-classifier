"""Build the v2 (no-refusal) labeled dataset + group-stratified splits.

final_label rule (v2, DIFFERENT from v1): harmful iff prompt_harm_label == "harmful"
OR response_harm_label == "harmful". response_refusal_label is NOT used to decide
final_label — it is carried through as metadata only.

Reads /home/mls01/data/complete_dataset.jsonl (read-only, never modified) and writes:
  /home/mls01/data/gemma_v2_no_refusal/complete_dataset_v2_with_splits.jsonl
  /home/mls01/data/gemma_v2_no_refusal/train.jsonl
  /home/mls01/data/gemma_v2_no_refusal/validation.jsonl
  /home/mls01/data/gemma_v2_no_refusal/test.jsonl
"""
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_PATH = Path("/home/mls01/data/complete_dataset.jsonl")
OUT_DIR = Path("/home/mls01/data/gemma_v2_no_refusal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPLETE_PATH = OUT_DIR / "complete_dataset_v2_with_splits.jsonl"
SPLIT_PATHS = {
    "train": OUT_DIR / "train.jsonl",
    "validation": OUT_DIR / "validation.jsonl",
    "test": OUT_DIR / "test.jsonl",
}

FINAL_COLUMNS = [
    "row_id", "original_idx", "prompt", "response",
    "prompt_harm_label", "response_harm_label", "response_refusal_label",
    "final_label", "language", "adversarial", "augmentation_type",
    "encoding_type", "subcategory", "response_truncated", "split",
]

RANDOM_STATE = 42


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    source_md5_before = md5(SOURCE_PATH)
    print(f"Original dataset MD5 (pre obrade): {source_md5_before}")

    df = pd.read_json(SOURCE_PATH, lines=True)
    print(f"Original dataset učitan: {len(df)} redova, {df.shape[1]} kolona.")

    df["response"] = df["response"].fillna("")

    # --- v2 pravilo: response_refusal_label je SAMO metadata, ne utiče na final_label ---
    df["final_label"] = "unharmful"
    harmful_mask = (
        (df["prompt_harm_label"] == "harmful")
        | (df["response_harm_label"] == "harmful")
    )
    df.loc[harmful_mask, "final_label"] = "harmful"

    counts = df["final_label"].value_counts()
    pct = df["final_label"].value_counts(normalize=True) * 100
    print("\nv2 final_label distribucija (bez refusal-a u pravilu):")
    for label in counts.index:
        print(f"  {label}: {counts[label]} redova ({pct[label]:.2f}%)")

    # --- Validacija pre splita ---
    assert "row_id" in df.columns, "Kolona 'row_id' ne postoji."
    assert df["row_id"].is_unique, "Postoje duplikati u 'row_id'."
    assert df["original_idx"].notna().all(), "Postoje prazne vrednosti u 'original_idx'."

    label_counts_per_group = df.groupby("original_idx")["final_label"].nunique()
    inconsistent_groups = label_counts_per_group[label_counts_per_group > 1]
    if len(inconsistent_groups) > 0:
        raise ValueError(
            f"Nekonzistentne v2 final_label vrednosti unutar original_idx grupa: "
            f"{list(inconsistent_groups.index)}"
        )
    print("\nValidacija OK: row_id jedinstven, original_idx popunjen, "
          "v2 final_label konzistentan po original_idx grupama.")

    # --- Group-stratified 80/10/10 split po original_idx, isti random_state kao v1 ---
    groups = (
        df[["original_idx", "final_label"]]
        .drop_duplicates("original_idx")
        .reset_index(drop=True)
    )
    train_ids, temp_ids = train_test_split(
        groups["original_idx"], test_size=0.2,
        stratify=groups["final_label"], random_state=RANDOM_STATE,
    )
    temp_labels = groups.set_index("original_idx").loc[temp_ids, "final_label"]
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, stratify=temp_labels, random_state=RANDOM_STATE,
    )

    split_map = {}
    split_map.update({i: "train" for i in train_ids})
    split_map.update({i: "validation" for i in val_ids})
    split_map.update({i: "test" for i in test_ids})
    df["split"] = df["original_idx"].map(split_map)

    assert df["split"].notna().all(), "Neki redovi nisu dobili split."

    print(f"\nSplit grupa (original_idx): train={len(train_ids)} "
          f"validation={len(val_ids)} test={len(test_ids)} "
          f"(ukupno {len(train_ids) + len(val_ids) + len(test_ids)})")

    for name in ["train", "validation", "test"]:
        n_rows = (df["split"] == name).sum()
        n_groups = df.loc[df["split"] == name, "original_idx"].nunique()
        print(f"  {name}: {n_rows} redova, {n_groups} grupa")

    # --- Integritetske provere pre čuvanja ---
    train_ids_set, val_ids_set, test_ids_set = set(train_ids), set(val_ids), set(test_ids)
    assert not (train_ids_set & val_ids_set), "Preklapanje original_idx između train i validation."
    assert not (train_ids_set & test_ids_set), "Preklapanje original_idx između train i test."
    assert not (val_ids_set & test_ids_set), "Preklapanje original_idx između validation i test."
    assert (train_ids_set | val_ids_set | test_ids_set) == set(df["original_idx"]), \
        "Svaki original_idx mora pripadati tačno jednom splitu."

    final_df = df[FINAL_COLUMNS].copy()
    assert len(final_df) == len(df), "Broj redova se promenio pri selekciji kolona."
    print(f"\nFinalni DataFrame (v2, sa splitom): {final_df.shape}")

    # --- Čuvanje: po jedan JSON objekat po redu, Unicode bez ASCII escaping-a, bez indeksa ---
    final_df.to_json(COMPLETE_PATH, orient="records", lines=True, force_ascii=False)
    print(f"\nSačuvano: {COMPLETE_PATH} ({len(final_df)} redova)")

    split_dfs_written = {}
    for name, path in SPLIT_PATHS.items():
        split_df = final_df[final_df["split"] == name].reset_index(drop=True)
        split_df.to_json(path, orient="records", lines=True, force_ascii=False)
        split_dfs_written[name] = split_df
        print(f"Sačuvano: {path} ({len(split_df)} redova)")

    # --- Potvrda da originalni dataset NIJE promenjen ---
    source_md5_after = md5(SOURCE_PATH)
    assert source_md5_before == source_md5_after, \
        "Originalni complete_dataset.jsonl je promenjen — ovo NE SME da se desi!"
    print(f"\nOriginal dataset MD5 (posle obrade): {source_md5_after} — nepromenjen, potvrđeno.")

    # =========================================================================
    # Ponovno učitavanje sva 4 fajla i potvrda integriteta.
    # =========================================================================
    print("\n" + "=" * 90)
    print("PONOVNO UČITAVANJE I VALIDACIJA")
    print("=" * 90)

    reloaded_complete = pd.read_json(COMPLETE_PATH, lines=True)
    reloaded_splits = {name: pd.read_json(path, lines=True) for name, path in SPLIT_PATHS.items()}

    # 1) Kompletan fajl ima 2.471 red.
    assert len(reloaded_complete) == 2471, \
        f"Kompletan fajl ima {len(reloaded_complete)} redova, očekivano 2471."
    print(f"[OK] Kompletan fajl ima {len(reloaded_complete)} redova (očekivano 2471).")

    # 2) train + validation + test = 2.471.
    total_split_rows = sum(len(d) for d in reloaded_splits.values())
    assert total_split_rows == 2471, \
        f"train+validation+test = {total_split_rows}, očekivano 2471."
    print(f"[OK] train({len(reloaded_splits['train'])}) + "
          f"validation({len(reloaded_splits['validation'])}) + "
          f"test({len(reloaded_splits['test'])}) = {total_split_rows} (očekivano 2471).")

    # 3) Svaki row_id postoji tačno jednom (u kompletnom fajlu).
    assert reloaded_complete["row_id"].is_unique, "row_id nije jedinstven u kompletnom fajlu."
    print("[OK] Svaki row_id postoji tačno jednom u kompletnom fajlu.")

    # 3b) row_id-jevi iz split fajlova se poklapaju tačno (bez duplikata, bez gubitaka) sa kompletnim.
    all_split_row_ids = pd.concat([d["row_id"] for d in reloaded_splits.values()])
    assert all_split_row_ids.is_unique, "Postoje duplikati row_id između/unutar split fajlova."
    assert set(all_split_row_ids) == set(reloaded_complete["row_id"]), \
        "row_id skup iz split fajlova se ne poklapa sa kompletnim fajlom."
    print("[OK] row_id skup iz train+validation+test se tačno poklapa sa kompletnim fajlom "
          "(bez duplikata, bez gubitaka).")

    # 4) Svaki original_idx pripada tačno jednom splitu.
    idx_to_splits = reloaded_complete.groupby("original_idx")["split"].nunique()
    assert (idx_to_splits == 1).all(), \
        f"original_idx grupe koje pripadaju više od jednog splita: " \
        f"{list(idx_to_splits[idx_to_splits > 1].index)}"
    print("[OK] Svaki original_idx pripada tačno jednom splitu.")

    # 5) Nema leakage-a između splitova (na nivou original_idx, iz reload-ovanih split fajlova).
    reloaded_id_sets = {name: set(d["original_idx"]) for name, d in reloaded_splits.items()}
    assert not (reloaded_id_sets["train"] & reloaded_id_sets["validation"]), \
        "Leakage: train i validation dele original_idx."
    assert not (reloaded_id_sets["train"] & reloaded_id_sets["test"]), \
        "Leakage: train i test dele original_idx."
    assert not (reloaded_id_sets["validation"] & reloaded_id_sets["test"]), \
        "Leakage: validation i test dele original_idx."
    print("[OK] Nema leakage-a original_idx između train/validation/test.")

    # 6) Labele i splitovi posle ponovnog učitavanja identični su DataFrame-ovima u memoriji.
    mem_sorted = final_df.sort_values("row_id").reset_index(drop=True)
    reloaded_sorted = reloaded_complete[FINAL_COLUMNS].sort_values("row_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(mem_sorted, reloaded_sorted, check_dtype=False)
    print("[OK] Ponovo učitan kompletan fajl je identičan in-memory DataFrame-u "
          "(sve kolone, uključujući final_label i split).")

    for name, mem_split_df in split_dfs_written.items():
        mem_split_sorted = mem_split_df.sort_values("row_id").reset_index(drop=True)
        reloaded_split_sorted = (
            reloaded_splits[name][FINAL_COLUMNS].sort_values("row_id").reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(mem_split_sorted, reloaded_split_sorted, check_dtype=False)
    print("[OK] Ponovo učitani train/validation/test fajlovi identični su in-memory split DataFrame-ovima.")

    # 7) final_label svuda odgovara novom (no-refusal) pravilu.
    expected_label = pd.Series("unharmful", index=reloaded_complete.index)
    recomputed_mask = (
        (reloaded_complete["prompt_harm_label"] == "harmful")
        | (reloaded_complete["response_harm_label"] == "harmful")
    )
    expected_label.loc[recomputed_mask] = "harmful"
    assert (reloaded_complete["final_label"] == expected_label).all(), \
        "final_label u ponovo učitanom fajlu ne odgovara v2 (no-refusal) pravilu."
    print("[OK] final_label u ponovo učitanom fajlu svuda odgovara v2 (no-refusal) pravilu "
          "(response_refusal_label nije korišćen u izračunavanju).")

    # 8) Originalni dataset nije promenjen (dupla provera posle reload-a).
    source_md5_final = md5(SOURCE_PATH)
    assert source_md5_final == source_md5_before, \
        "Originalni complete_dataset.jsonl je promenjen tokom izvršavanja skripte!"
    print(f"[OK] Originalni {SOURCE_PATH} je nepromenjen (MD5 {source_md5_final}).")

    print("\n" + "=" * 90)
    print("SVE PROVERE PROŠLE — v2 (no-refusal) dataset je konzistentan source of truth.")
    print("=" * 90)

    print("\nSačuvani fajlovi (putanja — broj redova):")
    print(f"  {COMPLETE_PATH}  —  {len(reloaded_complete)} redova")
    for name, path in SPLIT_PATHS.items():
        print(f"  {path}  —  {len(reloaded_splits[name])} redova")


if __name__ == "__main__":
    main()
