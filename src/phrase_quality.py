from __future__ import annotations

import re


KOR_PATTERN = re.compile(r"[가-힣]")
ENG_PATTERN = re.compile(r"[A-Za-z]")
DIGIT_PATTERN = re.compile(r"\d")

PATENT_NOISE_WORDS = {
    "포함", "제공", "방법", "장치", "단계", "형태", "구성", "부분",
    "상기", "적어도", "하나", "이상", "이하", "범위",
    "형성", "가능", "사용", "실시", "개시", "제조", "복수",
    "이용", "구비", "위치", "생성", "연결", "제어", "배치", "방향", "정보",
    "영역", "채널", "종", "개",
}

GENERIC_DOMAIN_WORDS = {
    "전지", "배터리", "전극", "재료", "물질", "시스템", "구조", "장치"
}

PROTECTED_PATTERNS = [
    re.compile(r"^\d+[dD]$"),                 # 3D
    re.compile(r"^\d+차전지$"),               # 2차전지
    re.compile(r"^\d+차$"),                   # 2차
    re.compile(r"^[A-Za-z]{2,}\d*$"),         # AI, OLED
    re.compile(r"^\d+nm$", re.IGNORECASE),    # 20nm
]

NUMBER_PREFIX_NOISE = [
    re.compile(r"^\d+[가-힣]{1,2}$"),         # 1제, 2개, 1종
    re.compile(r"^\d+(영역|채널|부분|방향|전극|개|종)$"),
    re.compile(r"^제\d+$"),                   # 제1, 제2
    re.compile(r"^제\d+[가-힣]+$"),           # 제1방향, 제1전극
    re.compile(r"^\d+제\d*$"),                # 1제2
]

ATTACHED_TERM_RULES = [
    (re.compile(r"\b이\s+차\s+전지\b"), "2차전지"),
    (re.compile(r"\b이\s+차\b"), "2차"),
    (re.compile(r"\b차\s+전지\b"), "2차전지"),
    (re.compile(r"\b삼\s*디\b", re.IGNORECASE), "3D"),
    (re.compile(r"\b3\s*d\b", re.IGNORECASE), "3D"),
]


def normalize_candidate_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"(?<=\d)\s+(?=[A-Za-z])", "", text)
    text = re.sub(r"(?<=[A-Za-z])\s+(?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s+(?=[가-힣])", "", text)
    text = re.sub(r"(?<=[가-힣])\s+(?=\d)", "", text)

    for pattern, repl in ATTACHED_TERM_RULES:
        text = pattern.sub(repl, text)

    return text.strip()


def token_shape_score(text: str) -> float:
    score = 0.0
    if KOR_PATTERN.search(text):
        score += 1.0
    if ENG_PATTERN.search(text):
        score += 0.7
    if DIGIT_PATTERN.search(text):
        score += 0.4
    if "-" in text or "/" in text:
        score += 0.2
    return score


def is_protected_term(text: str) -> bool:
    compact = text.replace(" ", "")
    for pat in PROTECTED_PATTERNS:
        if pat.match(compact):
            return True

    protected_keywords = {
        "프린팅", "전지", "배터리", "전해질", "반도체", "노즐", "합금",
        "전극층", "양극", "음극", "코팅", "분리막", "카본", "고분자",
    }
    if any(k in text for k in protected_keywords):
        return True

    return False


def is_number_prefix_noise(text: str) -> bool:
    compact = text.replace(" ", "")
    for pat in NUMBER_PREFIX_NOISE:
        if pat.match(compact):
            return True
    return False


def patent_noise_penalty(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 1.0

    hits = sum(1 for tok in tokens if tok in PATENT_NOISE_WORDS)
    penalty = hits / len(tokens)

    if text in PATENT_NOISE_WORDS:
        penalty += 0.7

    if is_number_prefix_noise(text):
        penalty += 1.0

    return min(penalty, 1.5)


def completeness_penalty(text: str) -> float:
    penalty = 0.0

    bad_endings = ("에", "의", "및", "또는", "를", "을", "이", "가", "은", "는", "와", "과")
    if text.endswith(bad_endings):
        penalty += 0.4

    one_char_count = sum(1 for tok in text.split() if len(tok) == 1)
    if one_char_count >= 2:
        penalty += 0.3

    if len(text.replace(" ", "")) <= 1:
        penalty += 1.0

    return min(penalty, 1.5)


def generic_domain_penalty(text: str) -> float:
    if text in GENERIC_DOMAIN_WORDS and not is_protected_term(text):
        return 0.8

    if text in GENERIC_DOMAIN_WORDS:
        return 0.25

    return 0.0


def base_quality_score(text: str, df: int, tf: int) -> float:
    score = 0.0
    score += min(df / 8.0, 1.2)
    score += min(tf / 15.0, 1.2)
    score += token_shape_score(text)

    score -= patent_noise_penalty(text) * 1.5
    score -= completeness_penalty(text) * 1.0
    score -= generic_domain_penalty(text) * 1.0

    if is_protected_term(text):
        score += 0.8

    if len(text.split()) >= 2:
        score += 0.3

    return round(score, 4)