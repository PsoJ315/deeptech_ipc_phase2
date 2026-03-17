from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


GENERIC_PARENT_EXCLUDE = {
    "조성물", "장치", "시스템", "방법", "프로그램", "디바이스", "기기",
    "부재", "기판", "회로", "구조", "모듈", "물질", "재료", "부품",
    "표면", "용액", "기재", "유닛", "소자", "부", "층", "부분", "장비",
}

SOFT_PARENT_ALLOWED = {
    "반도체", "배터리", "전지", "전극", "전해질", "촉매", "합금",
    "이미지", "영상", "신호", "데이터", "디지털", "센서",
    "리튬", "그래핀", "카본", "탄소", "실리콘", "규소",
}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def choose_parent_label(phrase: str) -> str:
    phrase = normalize_spaces(phrase)
    tokens = phrase.split()

    if not tokens:
        return phrase

    if len(tokens) == 1:
        return phrase

    first = tokens[0]
    first_two = " ".join(tokens[:2]) if len(tokens) >= 2 else first

    if first_two in {
        "디지털 트윈", "이미지 분석", "영상 분석", "신호 처리",
        "데이터 분석", "자연어 처리", "머신 러닝", "딥 러닝",
        "반도체 장치", "컴퓨터 프로그램", "사용자 단말",
        "리튬 이온", "금속 이온", "나트륨 이온",
    }:
        return first_two

    if first in SOFT_PARENT_ALLOWED:
        return first

    if first in GENERIC_PARENT_EXCLUDE:
        return phrase

    return first


def resolve_input_file(
    input_dir: str | Path,
    output_dir: str | Path,
    filename: str,
) -> Path:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    project_root = input_dir.parent if input_dir.name == "data" else input_dir
    cwd = Path.cwd()

    candidates = [
        input_dir / filename,
        output_dir / filename,
        project_root / "data" / filename,
        project_root / filename,
        cwd / "data" / filename,
        cwd / filename,
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    recursive_hits = list(project_root.rglob(filename))
    if recursive_hits:
        return recursive_hits[0].resolve()

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Required input file not found: {filename}\n"
        f"Searched paths:\n{searched}\n"
        f"Also searched recursively under: {project_root}"
    )


def build_concept_table(canonical_df: pd.DataFrame) -> pd.DataFrame:
    df = canonical_df.copy()

    if "phrase_canonical" not in df.columns:
        raise ValueError("Input dataframe must contain 'phrase_canonical' column.")

    df = df.drop_duplicates(subset=["phrase_canonical"]).copy()
    df["concept_label"] = df["phrase_canonical"].map(choose_parent_label)

    if "representative_phrase" not in df.columns:
        df["representative_phrase"] = df["phrase_canonical"]

    agg_dict = {
        "phrase_canonical": lambda s: sorted(set(s)),
        "representative_phrase": lambda s: sorted(set(s)),
    }

    if "df" in df.columns:
        agg_dict["df"] = "max"
    if "tf" in df.columns:
        agg_dict["tf"] = "sum"
    if "cleaned_score" in df.columns:
        agg_dict["cleaned_score"] = "max"
    if "canonical_score_proxy" in df.columns:
        agg_dict["canonical_score_proxy"] = "max"

    concept_df = (
        df.groupby("concept_label", as_index=False)
        .agg(agg_dict)
        .rename(columns={
            "phrase_canonical": "member_phrases",
            "representative_phrase": "member_representatives",
            "df": "concept_df_proxy",
            "tf": "concept_tf_sum",
            "cleaned_score": "concept_best_phrase_score",
            "canonical_score_proxy": "concept_score_proxy",
        })
    )

    concept_df["concept_size"] = concept_df["member_phrases"].map(len)

    sort_cols = [
        c for c in [
            "concept_df_proxy",
            "concept_score_proxy",
            "concept_best_phrase_score",
            "concept_tf_sum",
            "concept_size",
        ]
        if c in concept_df.columns
    ]

    if sort_cols:
        concept_df = concept_df.sort_values(
            by=sort_cols,
            ascending=[False] * len(sort_cols)
        ).reset_index(drop=True)

    concept_df["concept_id"] = [f"C{idx:04d}" for idx in range(1, len(concept_df) + 1)]
    concept_df["member_phrases_str"] = concept_df["member_phrases"].map(lambda x: " || ".join(x))
    concept_df["member_representatives_str"] = concept_df["member_representatives"].map(lambda x: " || ".join(x))

    return concept_df


def build_doc_concept_map(
    doc_phrase_df: pd.DataFrame,
    concept_df: pd.DataFrame,
) -> pd.DataFrame:
    phrase_to_concept: dict[str, tuple[str, str]] = {}
    for _, row in concept_df.iterrows():
        concept_id = row["concept_id"]
        concept_label = row["concept_label"]
        members = row["member_phrases"]
        for phrase in members:
            phrase_to_concept[normalize_spaces(phrase)] = (concept_id, concept_label)

    df = doc_phrase_df.copy()

    phrase_col = None
    for cand in ["phrase", "phrase_canonical", "matched_phrase"]:
        if cand in df.columns:
            phrase_col = cand
            break
    if phrase_col is None:
        raise ValueError("doc_phrase_map must contain one of: phrase, phrase_canonical, matched_phrase")

    doc_col = None
    for cand in ["doc_id", "document_id", "id"]:
        if cand in df.columns:
            doc_col = cand
            break
    if doc_col is None:
        raise ValueError("doc_phrase_map must contain one of: doc_id, document_id, id")

    df["phrase_norm"] = df[phrase_col].astype(str).map(normalize_spaces)
    df["concept_tuple"] = df["phrase_norm"].map(phrase_to_concept)

    mapped = df.loc[df["concept_tuple"].notna()].copy()
    mapped["concept_id"] = mapped["concept_tuple"].map(lambda x: x[0])
    mapped["concept_label"] = mapped["concept_tuple"].map(lambda x: x[1])

    out = mapped[[doc_col, phrase_col, "phrase_norm", "concept_id", "concept_label"]].drop_duplicates().reset_index(drop=True)
    out = out.rename(columns={doc_col: "doc_id", phrase_col: "phrase_original_in_doc"})
    return out


def run_step3_build_concepts(
    input_dir: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_map_path = output_dir / "phrase_canonical_map.csv"
    concept_families_path = output_dir / "concept_families.csv"
    doc_concept_map_path = output_dir / "doc_concept_map.csv"

    if not canonical_map_path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {canonical_map_path}\n"
            "Run STEP 2 first."
        )

    doc_phrase_map_path = resolve_input_file(
        input_dir=input_dir,
        output_dir=output_dir,
        filename="doc_phrase_map.csv",
    )

    print(f"[STEP 3] canonical_map_path resolved to: {canonical_map_path.resolve()}")
    print(f"[STEP 3] doc_phrase_map_path resolved to: {doc_phrase_map_path}")

    canonical_df = pd.read_csv(canonical_map_path)
    concept_df = build_concept_table(canonical_df)

    concept_df_for_save = concept_df.drop(columns=["member_phrases", "member_representatives"]).copy()
    concept_df_for_save.to_csv(concept_families_path, index=False, encoding="utf-8-sig")

    doc_phrase_df = pd.read_csv(doc_phrase_map_path)
    doc_concept_map_df = build_doc_concept_map(doc_phrase_df, concept_df)
    doc_concept_map_df.to_csv(doc_concept_map_path, index=False, encoding="utf-8-sig")

    print("=== STEP 3 COMPLETE ===")
    print(f"concept families : {len(concept_df):,}")
    print(f"doc-concept rows : {len(doc_concept_map_df):,}")
    print()

    preview_cols = [
        c for c in [
            "concept_id",
            "concept_label",
            "concept_size",
            "concept_df_proxy",
            "concept_tf_sum",
            "concept_best_phrase_score",
            "concept_score_proxy",
            "member_phrases_str",
        ]
        if c in concept_df_for_save.columns
    ]
    print("[Top 30 concept families]")
    print(concept_df_for_save[preview_cols].head(30).to_string(index=False))

    return concept_df_for_save, doc_concept_map_df