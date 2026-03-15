from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SupportScoringResult:
    scored_candidates: pd.DataFrame
    support_details: pd.DataFrame


def _safe_text(x: object) -> str:
    if x is None:
        return ""
    if pd.isna(x):
        return ""
    return str(x)


def _contains_phrase(text: str, phrase: str) -> bool:
    text = _safe_text(text)
    phrase = _safe_text(phrase)
    if not text or not phrase:
        return False
    return phrase in text


def compute_phrase_support(
    corpus_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    doc_phrase_map_df: pd.DataFrame,
    doc_id_col: str = "doc_id",
    title_col: str = "title",
    abstract_col: str = "abstract",
    claims_col: str = "claims",
) -> SupportScoringResult:
    """
    후보 phrase별로 title/claims support를 계산한다.
    후보 생성은 title+abstract에서 했지만, support는 title/claims를 별도로 본다.
    """
    if candidates_df.empty:
        empty_cols = list(candidates_df.columns) + [
            "title_df", "claims_df", "title_ratio", "claims_ratio",
            "title_boost", "claims_support_score", "final_score"
        ]
        return SupportScoringResult(
            scored_candidates=pd.DataFrame(columns=empty_cols),
            support_details=pd.DataFrame(columns=[doc_id_col, "phrase", "in_title", "in_claims"]),
        )

    corpus_lookup = corpus_df[[doc_id_col, title_col, abstract_col, claims_col]].copy()
    corpus_lookup[doc_id_col] = corpus_lookup[doc_id_col].astype(str)

    # phrase가 실제로 어느 문서 후보였는지 map 기준으로만 확인
    merged = doc_phrase_map_df.merge(
        corpus_lookup,
        on=doc_id_col,
        how="left",
    ).copy()

    merged["in_title"] = merged.apply(
        lambda r: _contains_phrase(r[title_col], r["phrase"]),
        axis=1,
    )
    merged["in_claims"] = merged.apply(
        lambda r: _contains_phrase(r[claims_col], r["phrase"]),
        axis=1,
    )

    support_details = merged[[doc_id_col, "phrase", "in_title", "in_claims"]].copy()

    support_agg = (
        support_details
        .groupby("phrase", as_index=False)
        .agg(
            title_df=("in_title", "sum"),
            claims_df=("in_claims", "sum"),
        )
    )

    scored = candidates_df.merge(support_agg, on="phrase", how="left").copy()
    scored["title_df"] = scored["title_df"].fillna(0).astype(int)
    scored["claims_df"] = scored["claims_df"].fillna(0).astype(int)

    # 비율화
    scored["title_ratio"] = (scored["title_df"] / scored["df"].clip(lower=1)).round(4)
    scored["claims_ratio"] = (scored["claims_df"] / scored["df"].clip(lower=1)).round(4)

    # title boost: 제목에 자주 직접 뜨는 용어는 가점
    scored["title_boost"] = (
        scored["title_ratio"] * 1.2
        + (scored["title_df"].clip(upper=10) / 10.0) * 0.6
    ).round(4)

    # claims support: 청구항 재등장은 구현/법적 범위 지지로 해석
    scored["claims_support_score"] = (
        scored["claims_ratio"] * 1.0
        + (scored["claims_df"].clip(upper=15) / 15.0) * 0.5
    ).round(4)

    # 너무 generic한 단일어가 claims에서만 센 경우 과대평가 방지
    onegram_generic_penalty = (
        (scored["ngram"] == 1)
        & (scored["char_len"] <= 3)
        & (~scored["is_protected"])
        & (scored["title_df"] == 0)
        & (scored["claims_df"] > 0)
    )
    scored["support_penalty"] = 0.0
    scored.loc[onegram_generic_penalty, "support_penalty"] = 0.4

    scored["final_score"] = (
        scored["base_quality"]
        + scored["title_boost"] * 1.2
        + scored["claims_support_score"] * 0.8
        - scored["support_penalty"]
    ).round(4)

    scored = scored.sort_values(
        by=["final_score", "base_quality", "df", "tf"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return SupportScoringResult(
        scored_candidates=scored,
        support_details=support_details,
    )


def summarize_scored_candidates(scored_df: pd.DataFrame, top_n: int = 40) -> None:
    print(f"[support_scoring] total scored candidates: {len(scored_df):,}")

    if scored_df.empty:
        print("[support_scoring] no scored candidates.")
        return

    preview_cols = [
        "phrase",
        "df",
        "tf",
        "title_df",
        "claims_df",
        "title_ratio",
        "claims_ratio",
        "base_quality",
        "final_score",
    ]
    print("[support_scoring] top scored candidates:")
    print(scored_df[preview_cols].head(top_n).to_string(index=False))