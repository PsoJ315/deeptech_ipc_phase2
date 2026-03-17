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
    "영역", "채널", "종", "개", "것", "예", "상세", "정의", "특징", "용도",
    "이때", "내", "양", "부",
}

GENERIC_DOMAIN_WORDS = {
    "전지", "배터리", "전극", "재료", "물질", "시스템", "구조", "장치"
}

PROTECTED_PATTERNS = [
    re.compile(r"^\d+[dD]$"),                  # 3D
    re.compile(r"^\d+차전지$"),                # 2차전지
    re.compile(r"^\d+차$"),                    # 2차
    re.compile(r"^[A-Za-z]{2,}\d*$"),          # AI, OLED, D50
    re.compile(r"^\d+(\.\d+)?nm$", re.I),      # 20nm
    re.compile(r"^[dD]\d+$"),                  # D50
]

MEASUREMENT_PROTECTED_PATTERNS = [
    re.compile(r"^\d+(\.\d+)?nm$", re.I),      # 20nm
    re.compile(r"^[dD]\d+$"),                  # D50
    re.compile(r"^\d+[dD]$"),                  # 3D
]

NUMERIC_FRAGMENT_NOISE_PATTERNS = [
    re.compile(r"^\d+(\.\d+)?$"),                      # 0.6, 0.1
    re.compile(r"^[a-zA-Z]?\d+(\.\d+)?[a-zA-Z/]*$"),  # a0.6cm3g
    re.compile(r"^(cm|mm|nm|m|g|kg|wt|wt%|mol)$", re.I),
]

NUMBER_PREFIX_NOISE = [
    re.compile(r"^\d+[가-힣]{1,2}$"),
    re.compile(r"^\d+(영역|채널|부분|방향|전극|개|종)$"),
    re.compile(r"^제\d+$"),
    re.compile(r"^제\d+[가-힣]+$"),
    re.compile(r"^\d+제\d*$"),
]

ATTACHED_TERM_RULES = [
    (re.compile(r"\b이\s+차\s+전지\b"), "2차전지"),
    (re.compile(r"\b이\s+차\b"), "2차"),
    (re.compile(r"\b차\s+전지\b"), "2차전지"),
    (re.compile(r"\b삼\s*디\b", re.IGNORECASE), "3D"),
    (re.compile(r"\b3\s*d\b", re.IGNORECASE), "3D"),
    (re.compile(r"\b조성\s+물\b"), "조성물"),
    (re.compile(r"\b프레임\s+워크\b"), "프레임워크"),
    (re.compile(r"\b메소\s+세공\b"), "메소세공"),
    (re.compile(r"\b메소세\s+공\b"), "메소세공"),
    (re.compile(r"\b마이크로\s+세공\b"), "마이크로세공"),
    (re.compile(r"\b전기\s+활성\s+물질\b"), "전기활성물질"),
    (re.compile(r"\b입자\s+크기\b"), "입자크기"),
    (re.compile(r"\b세공\s+부피\b"), "세공부피"),
    (re.compile(r"\b카본\s+프레임\s+워크\b"), "카본프레임워크"),
]

BAD_ENDINGS = ("에", "의", "및", "또는", "를", "을", "이", "가", "은", "는", "와", "과")

ORPHAN_ALPHA_TOKENS = {"a", "b", "c", "d", "e", "f", "g", "m"}

DOMAIN_ANCHOR_KEYWORDS = {
    "전지", "배터리", "반도체", "전해질", "전극", "전극층", "분리막",
    "코팅", "합금", "카본", "고분자", "세공", "프레임워크", "조성물",
    "노즐", "프린팅", "입자크기", "전기활성물질",
}

GENERIC_END_TOKENS = {
    "활성", "입자", "부피", "프레임", "워크", "직경", "크기", "복합"
}


def has_domain_anchor(text: str) -> bool:
    return any(k in text for k in DOMAIN_ANCHOR_KEYWORDS)


def has_orphan_alpha_token(text: str) -> bool:
    toks = text.split()
    for tok in toks:
        if tok.lower() in ORPHAN_ALPHA_TOKENS:
            return True
    return False


def has_bad_numeric_attachment(text: str) -> bool:
    compact = text.replace(" ", "")

    # 0.75다공 / 0.9범위20nm 같은 타입
    if re.search(r"\d(?:\.\d+)?[가-힣]{2,}", compact):
        if not is_measurement_protected(compact):
            return True

    # 워크20 같이 일반어 뒤 숫자 꼬리
    if re.search(r"[가-힣]{2,}\d+$", compact):
        if not is_measurement_protected(compact):
            return True

    return False

def normalize_candidate_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"(?<=\d)\s+(?=[A-Za-z])", "", text)
    text = re.sub(r"(?<=[A-Za-z])\s+(?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s+(?=[가-힣])", "", text)
    text = re.sub(r"(?<=[가-힣])\s+(?=\d)", "", text)

    for pattern, repl in ATTACHED_TERM_RULES:
        text = pattern.sub(repl, text)

    text = re.sub(r"\s+", " ", text)
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


def is_measurement_protected(text: str) -> bool:
    compact = text.replace(" ", "")
    for pat in MEASUREMENT_PROTECTED_PATTERNS:
        if pat.match(compact):
            return True
    return False


def is_protected_term(text: str) -> bool:
    compact = text.replace(" ", "")
    for pat in PROTECTED_PATTERNS:
        if pat.match(compact):
            return True

    protected_keywords = {
        "프린팅", "전지", "배터리", "전해질", "반도체", "노즐", "합금",
        "전극층", "양극", "음극", "코팅", "분리막", "카본", "고분자",
        "세공", "프레임워크", "조성물",
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


def is_numeric_fragment_noise(text: str) -> bool:
    compact = text.replace(" ", "")
    if is_measurement_protected(compact):
        return False

    for pat in NUMERIC_FRAGMENT_NOISE_PATTERNS:
        if pat.match(compact):
            return True

    # 숫자가 너무 많고 한글 정보가 거의 없으면 노이즈 가능성 큼
    digit_count = sum(ch.isdigit() for ch in compact)
    alpha_count = sum(ch.isalpha() for ch in compact)
    kor_count = len(re.findall(r"[가-힣]", compact))
    if digit_count >= 2 and kor_count == 0 and alpha_count <= 3 and not is_measurement_protected(compact):
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

    if is_numeric_fragment_noise(text):
        penalty += 1.0

    if has_orphan_alpha_token(text):
        penalty += 0.8

    if has_bad_numeric_attachment(text):
        penalty += 1.0

    return min(penalty, 1.8)


def completeness_penalty(text: str) -> float:
    penalty = 0.0

    if text.endswith(BAD_ENDINGS):
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

    if len(text.split()) >= 3 and not has_domain_anchor(text):
        score -= 0.8

    tokens = text.split()
    if tokens and tokens[-1] in GENERIC_END_TOKENS and not is_protected_term(text):
        score -= 0.6

    return round(score, 4)