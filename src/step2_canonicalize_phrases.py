from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


SPACE_NORMALIZE_PAT = re.compile(r"\s+")
MULTI_HYPHEN_PAT = re.compile(r"[-‐-‒–—―]+")

# 완전 동일 의미에 가까운 축약/표기 흔들림
EXACT_COMPACT_EQUIVALENTS = {
    "리튬이온": "리튬 이온",
    "나트륨이온": "나트륨 이온",
    "금속이온": "금속 이온",
    "표시장치": "표시 장치",
    "반도체장치": "반도체 장치",
    "컴퓨터프로그램": "컴퓨터 프로그램",
    "사용자단말": "사용자 단말",
    "영상분석": "영상 분석",
    "이미지분석": "이미지 분석",
    "신호처리": "신호 처리",
    "데이터분석": "데이터 분석",
    "자연어처리": "자연어 처리",
    "디지털트윈": "디지털 트윈",
    "머신러닝": "머신 러닝",
    "딥러닝": "딥 러닝",
}

REGEX_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b3\s*d\b", re.I), "3D"),
    (re.compile(r"\b2\s*d\b", re.I), "2D"),
    (re.compile(r"\brf\b", re.I), "RF"),
    (re.compile(r"\bai\b", re.I), "AI"),
    (re.compile(r"\bar\b", re.I), "AR"),
    (re.compile(r"\bvr\b", re.I), "VR"),
    (re.compile(r"\biot\b", re.I), "IoT"),
    (re.compile(r"\bict\b", re.I), "ICT"),
    (re.compile(r"\bmems\b", re.I), "MEMS"),
    (re.compile(r"리튬\s*-\s*이온"), "리튬 이온"),
    (re.compile(r"나트륨\s*-\s*이온"), "나트륨 이온"),
    (re.compile(r"금속\s*-\s*이온"), "금속 이온"),
    (re.compile(r"리튬\s*이온"), "리튬 이온"),
    (re.compile(r"나트륨\s*이온"), "나트륨 이온"),
    (re.compile(r"금속\s*이온"), "금속 이온"),
    (re.compile(r"디지털\s*트윈"), "디지털 트윈"),
    (re.compile(r"이미지\s*분석"), "이미지 분석"),
    (re.compile(r"영상\s*분석"), "영상 분석"),
    (re.compile(r"신호\s*처리"), "신호 처리"),
    (re.compile(r"데이터\s*분석"), "데이터 분석"),
    (re.compile(r"자연어\s*처리"), "자연어 처리"),
    (re.compile(r"머신\s*러닝"), "머신 러닝"),
    (re.compile(r"딥\s*러닝"), "딥 러닝"),
]

# 대표 표기 통합
CANONICAL_EQUIVALENTS = {
    "디스플레이 장치": "표시 장치",
    "표시장치": "표시 장치",
    "컴퓨터프로그램": "컴퓨터 프로그램",
    "반도체장치": "반도체 장치",
    "사용자단말": "사용자 단말",
    "리튬이온 전지": "리튬 이온 전지",
    "리튬이온 배터리": "리튬 이온 배터리",
    "금속이온 배터리": "금속 이온 배터리",
    "이미지분석": "이미지 분석",
    "영상분석": "영상 분석",
    "신호처리": "신호 처리",
    "데이터분석": "데이터 분석",
    "디지털트윈": "디지털 트윈",
}


def normalize_spaces(text: str) -> str:
    return SPACE_NORMALIZE_PAT.sub(" ", str(text)).strip()


def normalize_hyphens(text: str) -> str:
    return MULTI_HYPHEN_PAT.sub("-", text)


def normalize_basic(phrase: str) -> str:
    p = str(phrase).strip()
    p = normalize_hyphens(p)
    p = p.replace("/", " / ")
    p = p.replace("(", " ( ").replace(")", " ) ")
    p = normalize_spaces(p)
    return p


def apply_regex_replacements(phrase: str) -> str:
    out = phrase
    for pat, repl in REGEX_REPLACEMENTS:
        out = pat.sub(repl, out)
    return normalize_spaces(out)


def canonicalize_phrase(phrase: str) -> str:
    p = normalize_basic(phrase)
    p = apply_regex_replacements(p)

    compact = p.replace(" ", "")
    if compact in EXACT_COMPACT_EQUIVALENTS:
        p = EXACT_COMPACT_EQUIVALENTS[compact]

    compact = p.replace(" ", "")
    if compact in CANONICAL_EQUIVALENTS:
        p = CANONICAL_EQUIVALENTS[compact]
    elif p in CANONICAL_EQUIVALENTS:
        p = CANONICAL_EQUIVALENTS[p]

    p = normalize_spaces(p)
    return p


def build_phrase_canonical_map(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    df = cleaned_df.copy()

    if "phrase" not in df.columns:
        raise ValueError("Input dataframe must contain 'phrase' column.")

    df["phrase_original"] = df["phrase"].astype(str)
    df["phrase_canonical"] = df["phrase_original"].map(canonicalize_phrase)

    sort_cols = [c for c in ["cleaned_score", "df", "tf", "ngram", "char_len"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    representative_rows = (
        df.groupby("phrase_canonical", as_index=False)
        .first()
        .rename(columns={"phrase_original": "representative_phrase"})
    )

    rep_map = representative_rows[["phrase_canonical", "representative_phrase"]]

    out = df.merge(rep_map, on="phrase_canonical", how="left")

    family_size = (
        out.groupby("phrase_canonical")["phrase_original"]
        .nunique()
        .reset_index(name="canonical_family_size")
    )
    out = out.merge(family_size, on="phrase_canonical", how="left")

    # canonical 기준 집계 보조 컬럼
    if "df" in out.columns:
        canonical_doc_support = (
            out.groupby("phrase_canonical")["df"]
            .max()
            .reset_index(name="canonical_df_proxy")
        )
        out = out.merge(canonical_doc_support, on="phrase_canonical", how="left")

    if "cleaned_score" in out.columns:
        canonical_score_proxy = (
            out.groupby("phrase_canonical")["cleaned_score"]
            .max()
            .reset_index(name="canonical_score_proxy")
        )
        out = out.merge(canonical_score_proxy, on="phrase_canonical", how="left")

    return out


def run_step2_canonicalize_phrases(
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / "candidate_phrases_cleaned.csv"
    canonical_map_path = output_dir / "phrase_canonical_map.csv"

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {cleaned_path}\n"
            "Run STEP 1 first to create candidate_phrases_cleaned.csv"
        )

    cleaned_df = pd.read_csv(cleaned_path)
    canonical_df = build_phrase_canonical_map(cleaned_df)
    canonical_df.to_csv(canonical_map_path, index=False, encoding="utf-8-sig")

    print("=== STEP 2 COMPLETE ===")
    print(f"input rows   : {len(cleaned_df):,}")
    print(f"output rows  : {len(canonical_df):,}")
    print()

    preview_cols = [
        c for c in [
            "phrase_original",
            "phrase_canonical",
            "representative_phrase",
            "canonical_family_size",
            "df",
            "tf",
            "cleaned_score",
            "canonical_df_proxy",
            "canonical_score_proxy",
        ]
        if c in canonical_df.columns
    ]
    print("[Top 30 canonical map]")
    print(canonical_df[preview_cols].head(30).to_string(index=False))

    return canonical_df