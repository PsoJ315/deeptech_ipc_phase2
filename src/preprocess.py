from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import print_df_overview


TITLE_CANDIDATES = [
    "title", "발명의명칭", "명칭", "invention_title", "patent_title"
]
ABSTRACT_CANDIDATES = [
    "abstract", "요약", "초록", "summary", "patent_abstract"
]
CLAIMS_CANDIDATES = [
    "claims", "청구항", "claim", "대표청구항", "전체청구항"
]
DOC_ID_CANDIDATES = [
    "doc_id", "id", "순번", "번호", "출원번호", "publication_number", "application_number"
]

@dataclass
class ColumnMapping:
    doc_id: str | None
    title: str | None
    abstract: str | None
    claims: str | None


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def basic_text_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = normalize_whitespace(text)

    # 흔한 분절 복원 일부
    text = re.sub(r"(?<=\d)\s+(?=[A-Za-z])", "", text)   # 3 d -> 3d
    text = re.sub(r"(?<=[A-Za-z])\s+(?=\d)", "", text)   # A 1 -> A1
    text = re.sub(r"(?<=\d)\s+(?=[가-힣])", "", text)     # 2 차 -> 2차
    text = re.sub(r"(?<=[가-힣])\s+(?=\d)", "", text)     # 제 1 -> 제1

    return text.strip()


def lower_strip_columns(df: pd.DataFrame) -> dict[str, str]:
    return {str(col).strip().lower(): col for col in df.columns}


def find_first_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = lower_strip_columns(df)
    for cand in candidates:
        key = cand.strip().lower()
        if key in lowered:
            return lowered[key]

    # 부분 매칭 보조
    for lowered_name, original_name in lowered.items():
        for cand in candidates:
            if cand.strip().lower() in lowered_name:
                return original_name
    return None


def infer_column_mapping(df: pd.DataFrame) -> ColumnMapping:
    return ColumnMapping(
        doc_id=find_first_matching_column(df, DOC_ID_CANDIDATES),
        title=find_first_matching_column(df, TITLE_CANDIDATES),
        abstract=find_first_matching_column(df, ABSTRACT_CANDIDATES),
        claims=find_first_matching_column(df, CLAIMS_CANDIDATES),
    )


def standardize_patent_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, ColumnMapping]:
    mapping = infer_column_mapping(df)

    if mapping.title is None and mapping.abstract is None:
        raise ValueError(
            "Could not infer title/abstract columns. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame()

    if mapping.doc_id is not None:
        out["doc_id"] = df[mapping.doc_id].astype(str).fillna("")
    else:
        out["doc_id"] = [f"doc_{i}" for i in range(len(df))]

    out["title"] = (
        df[mapping.title].fillna("").astype(str).map(basic_text_clean)
        if mapping.title is not None
        else ""
    )
    out["abstract"] = (
        df[mapping.abstract].fillna("").astype(str).map(basic_text_clean)
        if mapping.abstract is not None
        else ""
    )
    out["claims"] = (
        df[mapping.claims].fillna("").astype(str).map(basic_text_clean)
        if mapping.claims is not None
        else ""
    )

    out["title_abstract"] = (
        out["title"].fillna("").str.strip() + " " + out["abstract"].fillna("").str.strip()
    ).str.strip()

    out["has_title"] = out["title"].str.len() > 0
    out["has_abstract"] = out["abstract"].str.len() > 0
    out["has_claims"] = out["claims"].str.len() > 0

    # 완전 빈 문서 제거
    out = out[(out["title"].str.len() > 0) | (out["abstract"].str.len() > 0)].copy()

    # 중복 제거: doc_id 중복 있으면 title_abstract 기준 보조
    out = out.drop_duplicates(subset=["doc_id"], keep="first")
    out = out.drop_duplicates(subset=["title_abstract"], keep="first")

    return out.reset_index(drop=True), mapping


def summarize_preprocessed_df(df: pd.DataFrame) -> None:
    print_df_overview(df, "preprocessed")
    print("[preprocessed] non-empty title:", int(df["has_title"].sum()))
    print("[preprocessed] non-empty abstract:", int(df["has_abstract"].sum()))
    print("[preprocessed] non-empty claims:", int(df["has_claims"].sum()))
    print("[preprocessed] sample title_abstract:")
    for i, row in df.head(3).iterrows():
        print(f"  - {row['title_abstract'][:180]}")