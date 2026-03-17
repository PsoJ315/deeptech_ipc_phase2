from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


GENERIC_SINGLE_TERMS = {
    "조성물", "장치", "시스템", "방법", "프로그램", "디바이스", "기기", "부재",
    "기판", "회로", "구조", "모듈", "물질", "재료", "부품", "표면", "기술",
    "데이터", "정보", "신호", "처리", "분석", "형성", "제조", "생성", "제공",
    "제어", "전지", "배터리", "전극", "전해질", "화합물", "조직", "입자",
    "필름", "튜브", "섬유", "수지", "합금", "조명", "표시", "센서",
    "네트워크", "차량", "선박", "항공기", "컴퓨터", "서버", "단말", "용액",
    "조성", "기재", "유닛", "소자", "부", "층", "부분", "유체", "장비",
    "소재", "장치부", "장치용", "용도",
}

PROTECTED_SINGLE_TERMS = {
    "반도체", "촉매", "그래핀", "흑연", "실리콘", "규소", "리튬",
    "산화물", "질화물", "카본", "탄소", "전해질", "양극", "음극",
    "애노드", "캐소드",
}

ALLOWED_SHORT_TOKENS = {
    "3D", "2D", "AI", "AR", "VR", "IoT", "ICT", "MEMS",
}

UNIT_PATTERNS = [
    re.compile(r"^\d+$"),
    re.compile(r"^\d+(?:\.\d+)?(?:nm|um|μm|mm|cm|m|wt%|vol%|mol%|mAh/g|cm3/g|m2/g)$", re.I),
    re.compile(r"^[A-Za-z]?\d+[A-Za-z]?$"),
    re.compile(r"^[^\w가-힣]+$"),
]

FRAGMENT_PATTERNS = [
    re.compile(r"^[가-힣]{1,2}\s+[가-힣]{2,}"),
    re.compile(r"^\d+[A-Za-z가-힣]{0,2}\s+[가-힣A-Za-z]"),
    re.compile(r"^[A-Za-z]\s+[가-힣A-Za-z]"),
]

DOMAIN_KEEP_PATTERNS = [
    re.compile(r"\b3D\b", re.I),
    re.compile(r"리튬|나트륨|금속|배터리|전지|전극|전해질|반도체|촉매|합금|카본|그래핀|실리콘|규소"),
]


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_phrase_basic(phrase: str) -> str:
    phrase = str(phrase)
    phrase = phrase.replace("–", "-").replace("—", "-").replace("−", "-")
    phrase = phrase.replace("/", " / ")
    phrase = normalize_spaces(phrase)
    return phrase


def compact(phrase: str) -> str:
    return phrase.replace(" ", "")


def has_unit_or_symbol_noise(phrase: str) -> bool:
    target = compact(phrase)
    return any(pat.match(target) for pat in UNIT_PATTERNS)


def token_is_suspicious(tok: str) -> bool:
    if tok in ALLOWED_SHORT_TOKENS:
        return False
    if len(tok) == 1 and re.search(r"[가-힣A-Za-z]", tok):
        return True
    if re.fullmatch(r"\d+[A-Za-z가-힣]{0,2}", tok):
        return True
    return False


def is_fragment_like(phrase: str) -> bool:
    phrase = normalize_phrase_basic(phrase)
    if not phrase:
        return True

    tokens = phrase.split()
    if not tokens:
        return True

    if phrase.count("(") != phrase.count(")"):
        return True

    if len(tokens) >= 2:
        if token_is_suspicious(tokens[0]) or token_is_suspicious(tokens[-1]):
            return True

    for pat in FRAGMENT_PATTERNS:
        if pat.search(phrase):
            return True

    return False


def is_generic_single_term(phrase: str, ngram: int) -> bool:
    return ngram == 1 and phrase in GENERIC_SINGLE_TERMS


def is_protected_single_term(phrase: str, ngram: int) -> bool:
    return ngram == 1 and phrase in PROTECTED_SINGLE_TERMS


def has_domain_signal(phrase: str) -> bool:
    return any(p.search(phrase) for p in DOMAIN_KEEP_PATTERNS)


def compute_generic_penalty(row: pd.Series) -> float:
    phrase = row["phrase"]
    ngram = int(row["ngram"])
    df = int(row["df"])
    char_len = int(row["char_len"])

    if not is_generic_single_term(phrase, ngram):
        return 0.0

    penalty = 1.2
    if df >= 10:
        penalty += 0.4
    if df >= 20:
        penalty += 0.3
    if char_len <= 3:
        penalty += 0.2
    return penalty


def should_drop_row(row: pd.Series) -> tuple[bool, str]:
    phrase = row["phrase"]
    ngram = int(row["ngram"])
    df = int(row["df"])
    char_len = int(row["char_len"])
    base_score = float(row["final_score"])

    if not phrase:
        return True, "empty"

    if has_unit_or_symbol_noise(phrase):
        return True, "unit_or_symbol_noise"

    if is_fragment_like(phrase):
        return True, "fragment_like"

    if ngram == 1 and char_len <= 2 and not is_protected_single_term(phrase, ngram):
        return True, "short_unigram"

    if is_generic_single_term(phrase, ngram) and df <= 2:
        return True, "weak_generic_single"

    if df == 1:
        if ngram == 1:
            return True, "singleton_unigram"
        if ngram >= 2 and char_len >= 5 and base_score >= 5.5 and has_domain_signal(phrase):
            return False, ""
        return True, "singleton_weak_phrase"

    return False, ""


def clean_phrase_candidates(
    scored_df: pd.DataFrame,
    min_cleaned_score: float = 3.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = scored_df.copy()

    required_cols = {"phrase", "tf", "df", "ngram", "char_len", "final_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["phrase"] = df["phrase"].astype(str).map(normalize_phrase_basic)

    drop_flags = df.apply(should_drop_row, axis=1, result_type="expand")
    df["drop_flag"] = drop_flags[0]
    df["drop_reason"] = drop_flags[1]

    df["generic_penalty"] = df.apply(compute_generic_penalty, axis=1)
    df["cleaned_score"] = df["final_score"] - df["generic_penalty"]

    cleaned = df.loc[~df["drop_flag"]].copy()
    cleaned = cleaned.loc[cleaned["cleaned_score"] >= min_cleaned_score].copy()

    cleaned = cleaned.sort_values(
        by=["cleaned_score", "df", "tf", "ngram", "char_len"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    return cleaned, df


def resolve_input_file(input_dir: str | Path, filename: str) -> Path:
    input_dir = Path(input_dir)
    project_root = input_dir.parent if input_dir.name == "data" else input_dir
    cwd = Path.cwd()

    candidates = [
        input_dir / filename,
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


def run_step1_clean_phrases(
    input_dir: str | Path,
    output_dir: str | Path,
    min_cleaned_score: float = 3.5,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_path = resolve_input_file(input_dir, "candidate_phrases_scored.csv")
    cleaned_path = output_dir / "candidate_phrases_cleaned.csv"
    drop_log_path = output_dir / "candidate_phrases_cleaned_drop_log.csv"

    print(f"[STEP 1] scored_path resolved to: {scored_path}")

    df = pd.read_csv(scored_path)
    cleaned, full_log = clean_phrase_candidates(df, min_cleaned_score=min_cleaned_score)

    cleaned.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    full_log.to_csv(drop_log_path, index=False, encoding="utf-8-sig")

    print("=== STEP 1 COMPLETE ===")
    print(f"input rows   : {len(df):,}")
    print(f"output rows  : {len(cleaned):,}")
    print(f"removed rows : {len(df) - len(cleaned):,}")
    print()
    print("[Top 20 cleaned]")
    print(
        cleaned[
            [
                "phrase", "df", "tf", "ngram", "char_len",
                "final_score", "generic_penalty", "cleaned_score"
            ]
        ].head(20).to_string(index=False)
    )
    print()
    print("[Drop reasons]")
    print(full_log.loc[full_log["drop_flag"], "drop_reason"].value_counts().to_string())

    return cleaned