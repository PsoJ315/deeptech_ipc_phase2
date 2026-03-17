from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from kiwipiepy import Kiwi
from tqdm import tqdm

from src.phrase_quality import (
    base_quality_score,
    completeness_penalty,
    is_number_prefix_noise,
    is_numeric_fragment_noise,
    is_protected_term,
    normalize_candidate_text,
    patent_noise_penalty,
)


TOKEN_BLACKLIST = {
    "본", "발명", "관한", "위한", "이용한", "통한", "에서", "으로", "하는", "되는",
    "및", "또는", "상기", "적어도", "하나의", "하나", "이상", "이하",
    "가능", "실시", "개시", "이용", "구비",
    "것", "예", "상세", "정의", "특징", "용도", "이때", "내", "양", "부",
    "포함",
}

ONEGRAM_SOFT_BLOCK = {
    "형성", "사용", "실시", "개시", "제조", "복수", "이용", "구비",
    "위치", "생성", "연결", "제어", "배치", "방향", "정보",
    "영역", "채널", "부분", "방법", "장치", "시스템", "재료", "물질",
    "것", "예", "상세", "정의", "특징", "용도", "이때", "내", "양", "부",
    "조성", "결합", "기초", "기반", "데이터", "수행", "발생", "결정",
    "저장", "향상", "사용자", "설치", "이동", "상태", "대응", "내부", "처리", "선택",
}

POS_ALLOW_PREFIX = ("N", "SL", "SN", "XR")
POS_ALLOW_EXACT = {"XPN"}

MIN_TOKEN_LEN = 1
MAX_NGRAM = 5


@dataclass
class PhraseMiningResult:
    candidates: pd.DataFrame
    doc_phrase_map: pd.DataFrame


def is_candidate_token(form: str, tag: str) -> bool:
    form = form.strip()
    if not form:
        return False
    if form in TOKEN_BLACKLIST:
        return False
    if len(form) < MIN_TOKEN_LEN:
        return False
    if tag in POS_ALLOW_EXACT:
        return True
    return tag.startswith(POS_ALLOW_PREFIX)


def clean_surface_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def kiwi_tokenize_for_phrases(text: str, kiwi: Kiwi) -> list[str]:
    text = clean_surface_text(text)
    tokens: list[str] = []

    try:
        analyzed = kiwi.tokenize(text)
    except Exception:
        return tokens

    for tok in analyzed:
        form = tok.form.strip()
        tag = tok.tag
        if is_candidate_token(form, tag):
            tokens.append(form)

    return tokens


def generate_ngrams(tokens: list[str], max_ngram: int = MAX_NGRAM) -> list[str]:
    phrases: list[str] = []
    n_tokens = len(tokens)

    for n in range(1, max_ngram + 1):
        for i in range(n_tokens - n + 1):
            chunk = tokens[i:i+n]
            phrase = " ".join(chunk).strip()
            phrase = normalize_candidate_text(phrase)
            if phrase:
                phrases.append(phrase)

    return phrases


def filter_phrase_surface(phrase: str) -> bool:
    if not phrase:
        return False

    compact = phrase.replace(" ", "")
    toks = phrase.split()

    if len(compact) <= 1:
        return False

    if compact.isdigit():
        return False

    if is_number_prefix_noise(phrase) and not is_protected_term(phrase):
        return False

    if is_numeric_fragment_noise(phrase) and not is_protected_term(phrase):
        return False

    if phrase.endswith(("에", "의", "및", "또는")) and not is_protected_term(phrase):
        return False

    # orphan alpha token 제거: a, b, g, m 같은 찌꺼기
    if any(tok.lower() in {"a", "b", "c", "d", "e", "f", "g", "m"} for tok in toks):
        if not is_protected_term(phrase):
            return False

    # 숫자+일반한글 강결합 찌꺼기 제거
    if re.search(r"\d(?:\.\d+)?[가-힣]{2,}", compact) and not is_protected_term(phrase):
        return False

    if re.search(r"[가-힣]{2,}\d+$", compact) and not is_protected_term(phrase):
        return False

    if len(toks) == 1:
        if phrase in ONEGRAM_SOFT_BLOCK and not is_protected_term(phrase):
            return False

        if re.match(r"^제\d+$", compact):
            return False

        if re.match(r"^\d+[가-힣]{1,2}$", compact) and not is_protected_term(phrase):
            return False

    bad_suffix_tokens = {"것", "예", "상세", "정의", "특징", "용도", "이때", "내", "양", "부"}
    if toks and toks[-1] in bad_suffix_tokens and not is_protected_term(phrase):
        return False
    if toks and toks[0] in bad_suffix_tokens and not is_protected_term(phrase):
        return False

    # 마지막 토큰이 너무 generic한 multi-gram 제거
    if len(toks) >= 2 and toks[-1] in {"활성", "입자", "부피", "프레임", "워크"}:
        if not is_protected_term(phrase):
            return False

    # 3gram 이상인데 domain anchor가 전혀 없으면 제거
    if len(toks) >= 3:
        if not any(k in phrase for k in {
            "전지", "배터리", "반도체", "전해질", "전극", "코팅",
            "합금", "카본", "고분자", "세공", "프레임워크", "조성물",
            "노즐", "프린팅"
        }):
            return False

    if phrase in {"및", "또는", "상기"}:
        return False

    return True


def extract_candidate_phrases(
    corpus_df: pd.DataFrame,
    text_col: str = "title_abstract",
    doc_id_col: str = "doc_id",
    max_ngram: int = MAX_NGRAM,
) -> PhraseMiningResult:
    kiwi = Kiwi()
    tf_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()
    phrase_len_map: dict[str, int] = {}
    protected_map: dict[str, bool] = {}
    doc_phrase_rows: list[dict] = []

    iterable = corpus_df[[doc_id_col, text_col]].fillna("").itertuples(index=False, name=None)

    for doc_id, text in tqdm(iterable, total=len(corpus_df), desc="Extracting phrase candidates"):
        tokens = kiwi_tokenize_for_phrases(text, kiwi)
        phrases = generate_ngrams(tokens, max_ngram=max_ngram)

        filtered_phrases = []
        for phrase in phrases:
            if filter_phrase_surface(phrase):
                filtered_phrases.append(phrase)

        unique_doc_phrases = set(filtered_phrases)

        for phrase in filtered_phrases:
            tf_counter[phrase] += 1
            phrase_len_map[phrase] = len(phrase.split())
            protected_map[phrase] = is_protected_term(phrase)

        for phrase in unique_doc_phrases:
            df_counter[phrase] += 1
            doc_phrase_rows.append({"doc_id": str(doc_id), "phrase": phrase})

    rows = []
    for phrase, tf in tf_counter.items():
        df = df_counter.get(phrase, 0)
        rows.append(
            {
                "phrase": phrase,
                "tf": int(tf),
                "df": int(df),
                "ngram": int(phrase_len_map.get(phrase, len(phrase.split()))),
                "char_len": len(phrase.replace(" ", "")),
                "is_protected": bool(protected_map.get(phrase, False)),
                "noise_penalty": patent_noise_penalty(phrase),
                "completeness_penalty": completeness_penalty(phrase),
                "base_quality": base_quality_score(phrase, df=df, tf=tf),
            }
        )

    cand_df = pd.DataFrame(rows)
    if cand_df.empty:
        cand_df = pd.DataFrame(
            columns=[
                "phrase", "tf", "df", "ngram", "char_len",
                "is_protected", "noise_penalty", "completeness_penalty", "base_quality"
            ]
        )
    else:
        cand_df = cand_df.sort_values(
            by=["base_quality", "df", "tf", "char_len"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    doc_phrase_df = pd.DataFrame(doc_phrase_rows)
    return PhraseMiningResult(candidates=cand_df, doc_phrase_map=doc_phrase_df)


def summarize_candidates(cand_df: pd.DataFrame, top_n: int = 30) -> None:
    print(f"[phrase_mining] total candidates: {len(cand_df):,}")

    if cand_df.empty:
        print("[phrase_mining] no candidates extracted.")
        return

    preview_cols = ["phrase", "df", "tf", "ngram", "is_protected", "base_quality"]
    print("[phrase_mining] top candidates:")
    print(cand_df[preview_cols].head(top_n).to_string(index=False))