from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch


# =========================
# Runtime / GPU Config
# =========================
USE_GPU = True
USE_FP16_FOR_EMBEDDING = True
GPU_DEVICE = "cuda:0"

# torch matmul 최적화
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# =========================
# Model / Pipeline Config
# =========================
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE = 64

SPLIT_LARGE_SIZE_THRESHOLD = 18
SPLIT_LARGE_COHESION_THRESHOLD = 0.58

# 1차 split bad topics는 좀 더 느슨하게
SPLIT_BAD_MIN_COMPONENT_SIZE = 2
SPLIT_BAD_SIMILARITY_THRESHOLD_PASS1 = 0.47
SPLIT_BAD_SIMILARITY_THRESHOLD_PASS2 = 0.50
SPLIT_BAD_MAX_PHRASES_FOR_SPLIT = 256

REFINE_SIM_MARGIN = 0.035
REFINE_MIN_SIMILARITY = 0.30
REFINE_MAX_ITER = 5
REFINE_MAX_ITER_FINAL = 3


# =========================
# Generic / Context Rules
# =========================
HARD_GENERIC_TERMS = {
    "조성물",
    "혼합물",
    "복합체",
    "장치",
    "모듈",
    "유닛",
    "부재",
    "구조체",
    "어셈블리",
    "파라미터",
    "스트림",
    "이벤트",
    "포인트",
    "데이터",
    "정보",
    "신호",
    "기판",
    "기재",
    "부품",
    "수단",
    "방법",
    "시스템",
    "공정",
}

GENERIC_SUFFIXES = [
    "조성물",
    "혼합물",
    "복합체",
    "장치",
    "시스템",
    "공정",
    "방법",
    "모듈",
    "유닛",
]

WEAK_PREFIX_TOKENS = {
    "운영",
    "관리",
    "제어",
    "처리",
    "구성",
    "분석",
    "제조",
    "제공",
    "생성",
    "수집",
    "저장",
    "전달",
    "표시",
    "구동",
    "실행",
    "지원",
    "활용",
    "적용",
    "개선",
    "최적화",
    "예측",
    "판단",
    "동작",
    "응답",
    "출력",
    "입력",
    "변환",
}

BRIDGE_LIKE_TERMS = {
    "디스플레이",
    "프로세스",
    "파라미터",
    "스트림",
    "어셈블리",
    "텍스트",
    "트레이",
    "포인트",
    "플레이트",
    "단말기",
    "사용자 단말",
    "이벤트",
    "와이어",
    "게이트",
}



TECH_CONTEXT_HINTS = {
    "ai", "a.i.", "ml", "llm",
    "인공지능", "머신러닝", "딥러닝", "신경망",
    "배터리", "이차전지", "전고체", "전극", "음극", "양극", "전해질", "분리막",
    "반도체", "식각", "증착", "포토레지스트", "리소그래피", "트랜지스터", "웨이퍼",
    "자율", "자율주행", "항법", "비행", "로켓", "위성", "우주", "추진",
    "센서", "라이다", "레이더", "영상", "복원", "검출", "인식",
    "진단", "예후", "바이오", "유전자", "단백질", "세포",
    "3d", "프린팅", "복합재", "탄소섬유", "금속분말", "적층",
    "수소", "연료전지", "촉매", "전기화학",
    "클라우드", "분산", "암호", "보안", "네트워크", "통신",
    "rf", "pdcch", "harq", "csi",
}

ENG_TECH_PATTERN = re.compile(
    r"\b(?:ai|ml|llm|gpu|cpu|fpga|asic|rf|lidar|radar|gnss|iot|5g|6g|3d|pdcch|harq|csi)\b",
    re.I,
)
ALNUM_TECH_PATTERN = re.compile(r"[A-Za-z]+[0-9]+|[0-9]+[A-Za-z]+")
ONLY_PUNCT_OR_SPACE = re.compile(r"^[\W_]+$")


# =========================
# Utility
# =========================
def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).strip().split())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_blank(text: str) -> bool:
    t = normalize_spaces(text)
    return t == "" or bool(ONLY_PUNCT_OR_SPACE.fullmatch(t))


def get_runtime_device() -> torch.device:
    if USE_GPU and torch.cuda.is_available():
        return torch.device(GPU_DEVICE)
    return torch.device("cpu")


def log_runtime_status(device: torch.device) -> None:
    log(f"[RUNTIME] torch.__version__={torch.__version__}")
    log(f"[RUNTIME] torch.cuda.is_available()={torch.cuda.is_available()}")
    log(f"[RUNTIME] selected_device={device}")
    if device.type == "cuda":
        try:
            idx = device.index if device.index is not None else 0
            log(f"[RUNTIME] cuda_device_name={torch.cuda.get_device_name(idx)}")
        except Exception:
            pass


def maybe_cuda_cleanup(device: torch.device) -> None:
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def resolve_input_file(base_dir: Path, filename: str) -> Path:
    candidates = [
        base_dir / filename,
        base_dir / "phase2_postprocess" / filename,
        base_dir / "interim" / filename,
        base_dir.parent / "phase2_postprocess" / filename,
        base_dir.parent / "interim" / filename,
        Path.cwd() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"Could not find {filename}. Tried: {candidates}")


# =========================
# Genericity / Term Context
# =========================
def term_has_technical_hint(text: str) -> bool:
    t = normalize_spaces(str(text))
    lower = t.lower()

    if any(h in lower for h in TECH_CONTEXT_HINTS):
        return True
    if ENG_TECH_PATTERN.search(lower):
        return True
    if ALNUM_TECH_PATTERN.search(t):
        return True
    return False


def classify_term_genericity(text: str) -> Literal["generic", "maybe_generic", "specific"]:
    t = normalize_spaces(str(text))
    if is_blank(t):
        return "generic"

    tokens = t.split()

    if t in HARD_GENERIC_TERMS:
        return "generic"

    matched_suffix = None
    for sfx in GENERIC_SUFFIXES:
        if t.endswith(sfx):
            matched_suffix = sfx
            break

    if matched_suffix is None:
        if len(tokens) == 1 and len(t) <= 2:
            return "maybe_generic"
        return "specific"

    prefix = t[: -len(matched_suffix)].strip()
    prefix_tokens = prefix.split()

    if not prefix_tokens:
        return "generic"

    if term_has_technical_hint(t) or term_has_technical_hint(prefix):
        return "specific"

    if len(prefix_tokens) == 1 and prefix_tokens[0] in WEAK_PREFIX_TOKENS:
        return "generic"
    
    if len(prefix_tokens) == 1:
        return "maybe_generic"

    return "maybe_generic"


def generic_weight(text: str) -> float:
    cls = classify_term_genericity(text)
    if cls == "generic":
        return 1.0
    if cls == "maybe_generic":
        return 0.5
    return 0.0
def is_bridge_like_term(text: str) -> bool:
    t = normalize_spaces(str(text))
    return t in BRIDGE_LIKE_TERMS

# =========================
# Embedding
# =========================
def embed_texts(
    texts: list[str],
    device: torch.device,
    model_name: str = EMBED_MODEL_NAME,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    log(f"[EMBED] device={device}")
    model = SentenceTransformer(model_name, device=str(device))

    if device.type == "cuda" and USE_FP16_FOR_EMBEDDING:
        try:
            model.half()
            log("[EMBED] fp16 enabled for embedding model")
        except Exception as e:
            log(f"[EMBED] fp16 enable skipped: {type(e).__name__}: {e}")

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        convert_to_tensor=False,
    )

    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass

    return np.asarray(emb, dtype=np.float32)


def build_embedding_maps(
    phrases: list[str],
    embeddings: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
    phrase_embeddings: dict[str, np.ndarray] = {}
    phrase_tensors: dict[str, torch.Tensor] = {}

    for i, phrase in enumerate(phrases):
        vec = np.asarray(embeddings[i], dtype=np.float32)
        phrase_embeddings[phrase] = vec
        phrase_tensors[phrase] = torch.from_numpy(vec).to(device=device, dtype=torch.float32)

    return phrase_embeddings, phrase_tensors


def stack_phrase_tensors(
    phrases: list[str],
    phrase_tensors: dict[str, torch.Tensor],
) -> tuple[list[str], torch.Tensor | None]:
    valid_phrases = []
    vecs = []

    for p in phrases:
        t = phrase_tensors.get(p)
        if t is not None:
            valid_phrases.append(p)
            vecs.append(t)

    if not vecs:
        return [], None

    return valid_phrases, torch.stack(vecs, dim=0)


# =========================
# Data Build
# =========================
def build_phrase_topic_map_from_doc_map(doc_map: pd.DataFrame) -> pd.DataFrame:
    df = doc_map.copy()

    required = ["phrase_canonical", "topic_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in doc map: {missing}")

    if "phrase_original_in_doc" in df.columns:
        df["phrase_original_in_doc"] = df["phrase_original_in_doc"].astype(str).map(normalize_spaces)
    else:
        df["phrase_original_in_doc"] = df["phrase_canonical"]

    df["phrase_canonical"] = df["phrase_canonical"].astype(str).map(normalize_spaces)
    df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce")
    df = df[df["topic_id"].notna()].copy()
    df["topic_id"] = df["topic_id"].astype(int)

    if "topic_label" not in df.columns:
        df["topic_label"] = ""

    if "doc_id" not in df.columns:
        df["doc_id"] = np.arange(len(df))

    grouped = (
        df.groupby(["phrase_canonical", "topic_id"], as_index=False)
        .agg(
            doc_count=("doc_id", "nunique"),
            tf=("doc_id", "size"),
            topic_label=("topic_label", "first"),
        )
        .rename(columns={"phrase_canonical": "phrase"})
    )

    grouped["phrase"] = grouped["phrase"].astype(str).map(normalize_spaces)
    grouped["doc_count"] = pd.to_numeric(grouped["doc_count"], errors="coerce").fillna(1)
    grouped["tf"] = pd.to_numeric(grouped["tf"], errors="coerce").fillna(grouped["doc_count"])

    return grouped


def standardize_canonical_map(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "phrase_canonical" not in out.columns:
        raise KeyError("phrase_canonical not found in canonical map")

    if "representative_phrase" in out.columns:
        out["canonical_phrase"] = out["representative_phrase"]
    else:
        out["canonical_phrase"] = out["phrase_canonical"]

    if "phrase_original" in out.columns:
        out["phrase"] = out["phrase_original"]
    elif "phrase" in out.columns:
        out["phrase"] = out["phrase"]
    else:
        out["phrase"] = out["phrase_canonical"]

    out["phrase"] = out["phrase"].astype(str).map(normalize_spaces)
    out["phrase_canonical"] = out["phrase_canonical"].astype(str).map(normalize_spaces)
    out["canonical_phrase"] = out["canonical_phrase"].astype(str).map(normalize_spaces)

    keep_cols = ["phrase", "phrase_canonical", "canonical_phrase"]
    for optional in ["df", "tf", "canonical_df_proxy", "canonical_score_proxy"]:
        if optional in out.columns:
            keep_cols.append(optional)

    out = out[keep_cols].drop_duplicates(subset=["phrase_canonical"], keep="first").copy()
    return out


def attach_canonical_phrase(
    phrase_topic_map: pd.DataFrame,
    canonical_map: pd.DataFrame,
) -> pd.DataFrame:
    df = phrase_topic_map.copy()

    cm = canonical_map[["phrase_canonical", "canonical_phrase"]].drop_duplicates()
    df = df.merge(
        cm,
        left_on="phrase",
        right_on="phrase_canonical",
        how="left",
    )

    df["canonical_phrase"] = df["canonical_phrase"].fillna(df["phrase"])
    df["canonical_phrase"] = df["canonical_phrase"].astype(str).map(normalize_spaces)
    df = df.drop(columns=["phrase_canonical"], errors="ignore")
    return df


# =========================
# Topic Semantics / Quality
# =========================
def choose_topic_representative(group: pd.DataFrame) -> str:
    score_df = (
        group.groupby("canonical_phrase", as_index=False)
        .agg(
            doc_count=("doc_count", "sum"),
            tf=("tf", "sum"),
        )
        .reset_index(drop=True)
    )

    if len(score_df) == 0:
        return "unknown"

    score_df["genericity"] = score_df["canonical_phrase"].map(classify_term_genericity)
    score_df["is_bridge"] = score_df["canonical_phrase"].map(is_bridge_like_term)

    generic_rank = {"specific": 0, "maybe_generic": 1, "generic": 2}
    score_df["generic_rank"] = score_df["genericity"].map(generic_rank)

    score_df = score_df.sort_values(
        ["generic_rank", "is_bridge", "doc_count", "tf", "canonical_phrase"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)

    return str(score_df.iloc[0]["canonical_phrase"])


def topic_generic_ratio(group: pd.DataFrame) -> float:
    phrases = group["canonical_phrase"].drop_duplicates().tolist()
    if not phrases:
        return 1.0
    return float(np.mean([generic_weight(p) for p in phrases]))


def topic_coherence_stats(
    group: pd.DataFrame,
    phrase_tensors: dict[str, torch.Tensor],
    sample_size: int = 96,
) -> dict:
    phrases = group["canonical_phrase"].drop_duplicates().tolist()
    generic_ratio = topic_generic_ratio(group)

    valid_phrases, mat = stack_phrase_tensors(phrases, phrase_tensors)
    if mat is None or len(valid_phrases) < 2:
        return {
            "n_phrases": len(phrases),
            "mean_sim": 1.0,
            "generic_ratio": generic_ratio,
        }

    if len(valid_phrases) > sample_size:
        idx = torch.randperm(len(valid_phrases), device=mat.device)[:sample_size]
        mat = mat[idx]

    sim = mat @ mat.T
    n = sim.shape[0]
    off_diag_mean = (sim.sum() - torch.trace(sim)) / max(n * (n - 1), 1)

    return {
        "n_phrases": len(phrases),
        "mean_sim": float(off_diag_mean.item()),
        "generic_ratio": generic_ratio,
    }


def topic_single_token_ratio(group: pd.DataFrame) -> float:
    phrases = group["canonical_phrase"].drop_duplicates().tolist()
    if not phrases:
        return 1.0
    return float(np.mean([1.0 if len(normalize_spaces(p).split()) == 1 else 0.0 for p in phrases]))


def topic_bridge_ratio(group: pd.DataFrame) -> float:
    phrases = group["canonical_phrase"].drop_duplicates().tolist()
    if not phrases:
        return 0.0
    return float(np.mean([1.0 if is_bridge_like_term(p) else 0.0 for p in phrases]))


def is_bad_topic(
    group: pd.DataFrame,
    phrase_tensors: dict[str, torch.Tensor],
    label_term: str | None = None,
) -> bool:
    phrases = group["canonical_phrase"].drop_duplicates().tolist()
    n_phrases = len(phrases)
    generic_ratio = topic_generic_ratio(group)
    single_token_ratio = topic_single_token_ratio(group)
    bridge_ratio = topic_bridge_ratio(group)

    label_term = label_term or choose_topic_representative(group)
    label_cls = classify_term_genericity(label_term)

    if n_phrases <= 3:
        return False

    stats = topic_coherence_stats(group, phrase_tensors, sample_size=96)
    mean_sim = stats["mean_sim"]

    # 아주 큰 토픽은 거의 무조건 분해
    if n_phrases >= 40:
        return True

    # 단일 명사 위주로 많이 뭉친 경우
    if n_phrases >= 20 and single_token_ratio >= 0.65:
        return True

    # 연결어가 많이 섞인 경우
    if n_phrases >= 18 and bridge_ratio >= 0.15:
        return True

    # 의미 응집도 낮은 중대형 토픽
    if n_phrases >= 20 and mean_sim < 0.48:
        return True

    if n_phrases >= 14 and generic_ratio >= 0.22 and mean_sim < 0.58:
        return True

    if label_cls == "generic" and n_phrases >= 7:
        return True

    if label_cls == "maybe_generic" and n_phrases >= 12:
        return True

    return False


# =========================
# Topic Split Helpers
# =========================
def connected_components_from_similarity(
    phrases: list[str],
    phrase_tensors: dict[str, torch.Tensor],
    threshold: float,
    max_phrases_for_matrix: int,
) -> list[list[str]]:
    valid_phrases, mat = stack_phrase_tensors(phrases, phrase_tensors)

    if mat is None or len(valid_phrases) == 0:
        return []
    if len(valid_phrases) == 1:
        return [valid_phrases]

    if len(valid_phrases) > max_phrases_for_matrix:
        log(
            f"[connected_components] skip matrix build: "
            f"{len(valid_phrases)} > {max_phrases_for_matrix}"
        )
        return [valid_phrases]

    sim = mat @ mat.T

    visited: set[int] = set()
    components: list[list[str]] = []

    for i in range(len(valid_phrases)):
        if i in visited:
            continue

        stack = [i]
        comp: list[str] = []
        visited.add(i)

        while stack:
            cur = stack.pop()
            comp.append(valid_phrases[cur])

            nbrs = torch.nonzero(sim[cur] >= threshold, as_tuple=False).flatten().tolist()
            for nb in nbrs:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(int(nb))

        components.append(comp)

    return components


def bisect_phrases_by_extremes(
    phrases: list[str],
    phrase_tensors: dict[str, torch.Tensor],
) -> set[str] | None:
    valid_phrases, mat = stack_phrase_tensors(phrases, phrase_tensors)
    if mat is None or len(valid_phrases) < 4:
        return None

    centroid = mat.mean(dim=0)
    centroid = centroid / centroid.norm().clamp_min(1e-12)

    sims = mat @ centroid
    idx_low = int(torch.argmin(sims).item())
    idx_high = int(torch.argmax(sims).item())

    seed_a = mat[idx_low]
    seed_b = mat[idx_high]

    seed_a = seed_a / seed_a.norm().clamp_min(1e-12)
    seed_b = seed_b / seed_b.norm().clamp_min(1e-12)

    sim_a = mat @ seed_a
    sim_b = mat @ seed_b
    assign_b = sim_b > sim_a

    n_b = int(assign_b.sum().item())
    if n_b == 0 or n_b == len(valid_phrases):
        return None

    flags = assign_b.tolist()
    return {valid_phrases[i] for i, flag in enumerate(flags) if flag}


# =========================
# Topic Split / Refine
# =========================
def split_bad_topics(
    df: pd.DataFrame,
    phrase_tensors: dict[str, torch.Tensor],
    min_component_size: int,
    similarity_threshold: float,
    max_phrases_for_split: int,
) -> pd.DataFrame:
    out = df.copy()
    current_max_topic = int(out["topic_id"].max()) if len(out) else 0

    topic_ids = sorted(out["topic_id"].dropna().astype(int).unique().tolist())
    log(f"[split_bad_topics] total topics: {len(topic_ids)}")

    for idx, topic_id in enumerate(topic_ids, start=1):
        group = out[out["topic_id"] == topic_id].copy()
        if len(group) == 0:
            continue

        label_term = choose_topic_representative(group)
        phrases = group["canonical_phrase"].drop_duplicates().tolist()
        n_phrases = len(phrases)
        g_ratio = topic_generic_ratio(group)

        log(
            f"[split_bad_topics] ({idx}/{len(topic_ids)}) "
            f"topic={topic_id}, label='{label_term}', n_rows={len(group)}, "
            f"n_phrases={n_phrases}, generic_ratio={g_ratio:.3f}"
        )

        bad_flag = is_bad_topic(group, phrase_tensors, label_term=label_term)
        log(f"[split_bad_topics] topic={topic_id} -> bad_flag={bad_flag}")

        if n_phrases >= 40:
            phrases_b = bisect_phrases_by_extremes(phrases, phrase_tensors)
            if phrases_b and 1 < len(phrases_b) < n_phrases:
                current_max_topic += 1
                mask = (out["topic_id"] == topic_id) & (out["canonical_phrase"].isin(phrases_b))
                out.loc[mask, "topic_id"] = current_max_topic
                log(
                    f"[split_bad_topics] topic={topic_id} -> forced giant-topic bisect "
                    f"new_topic={current_max_topic}, moved_phrases={len(phrases_b)}"
                )
                continue


        if not bad_flag:
            continue

        # 매우 큰 토픽이면 바로 bisect fallback
        if n_phrases > max_phrases_for_split:
            phrases_b = bisect_phrases_by_extremes(phrases, phrase_tensors)
            if phrases_b and 1 < len(phrases_b) < n_phrases:
                current_max_topic += 1
                mask = (out["topic_id"] == topic_id) & (out["canonical_phrase"].isin(phrases_b))
                out.loc[mask, "topic_id"] = current_max_topic
                log(
                    f"[split_bad_topics] topic={topic_id} -> fallback bisect "
                    f"new_topic={current_max_topic}, moved_phrases={len(phrases_b)}"
                )
            continue

        try:
            components = connected_components_from_similarity(
                phrases=phrases,
                phrase_tensors=phrase_tensors,
                threshold=similarity_threshold,
                max_phrases_for_matrix=max_phrases_for_split,
            )
        except Exception as e:
            log(
                f"[split_bad_topics] ERROR on topic {topic_id} ('{label_term}') "
                f"with n_phrases={n_phrases}: {type(e).__name__}: {e}"
            )
            components = []

        components = [c for c in components if len(c) >= min_component_size]

        if len(components) <= 1:
            phrases_b = bisect_phrases_by_extremes(phrases, phrase_tensors)
            if phrases_b and 1 < len(phrases_b) < n_phrases:
                current_max_topic += 1
                mask = (out["topic_id"] == topic_id) & (out["canonical_phrase"].isin(phrases_b))
                out.loc[mask, "topic_id"] = current_max_topic
                log(
                    f"[split_bad_topics] topic={topic_id} -> fallback bisect "
                    f"new_topic={current_max_topic}, moved_phrases={len(phrases_b)}"
                )
            continue

        components = sorted(components, key=len, reverse=True)

        log(
            f"[split_bad_topics] topic {topic_id} ('{label_term}') "
            f"-> {len(components)} components"
        )

        for comp in components[1:]:
            current_max_topic += 1
            mask = (out["topic_id"] == topic_id) & (out["canonical_phrase"].isin(comp))
            out.loc[mask, "topic_id"] = current_max_topic

    return out


def split_large_clusters(
    df: pd.DataFrame,
    phrase_tensors: dict[str, torch.Tensor],
    size_threshold: int,
    cohesion_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    current_max_topic = int(out["topic_id"].max()) if len(out) else 0

    changed = True
    while changed:
        changed = False

        for topic_id, group in out.groupby("topic_id"):
            phrase_stats = (
                group.groupby("canonical_phrase", as_index=False)
                .agg(
                    doc_count=("doc_count", "sum"),
                    tf=("tf", "sum"),
                )
            )

            if len(phrase_stats) < size_threshold:
                continue

            phrases = phrase_stats["canonical_phrase"].tolist()
            valid_phrases, mat = stack_phrase_tensors(phrases, phrase_tensors)
            if mat is None or len(valid_phrases) < 4:
                continue

            centroid = mat.mean(dim=0)
            centroid = centroid / centroid.norm().clamp_min(1e-12)
            sims = mat @ centroid
            cohesion = float(sims.mean().item())

            if cohesion >= cohesion_threshold:
                continue

            idx_low = int(torch.argmin(sims).item())
            idx_high = int(torch.argmax(sims).item())

            seed_a = mat[idx_low]
            seed_b = mat[idx_high]

            seed_a = seed_a / seed_a.norm().clamp_min(1e-12)
            seed_b = seed_b / seed_b.norm().clamp_min(1e-12)

            sim_a = mat @ seed_a
            sim_b = mat @ seed_b
            assign_b = sim_b > sim_a

            n_b = int(assign_b.sum().item())
            if n_b == 0 or n_b == len(valid_phrases):
                continue

            flags = assign_b.tolist()
            phrases_b = {valid_phrases[i] for i, flag in enumerate(flags) if flag}

            current_max_topic += 1
            mask = (out["topic_id"] == topic_id) & (out["canonical_phrase"].isin(phrases_b))
            out.loc[mask, "topic_id"] = current_max_topic
            changed = True
            log(f"[split_large_clusters] split topic {topic_id} -> {topic_id}, {current_max_topic}")
            break

    return out


def compute_topic_centroids(
    df: pd.DataFrame,
    phrase_tensors: dict[str, torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, str]]:
    centroids: dict[int, torch.Tensor] = {}
    labels: dict[int, str] = {}

    for topic_id, group in df.groupby("topic_id"):
        phrase_stats = (
            group.groupby("canonical_phrase", as_index=False)
            .agg(
                doc_count=("doc_count", "sum"),
                tf=("tf", "sum"),
            )
        )

        vecs = []
        weights = []

        for _, row in phrase_stats.iterrows():
            p = row["canonical_phrase"]
            t = phrase_tensors.get(p)
            if t is None:
                continue

            gw = generic_weight(p)
            weight = float(row["doc_count"]) * (1.0 - 0.35 * gw)
            weight = max(weight, 1e-6)

            vecs.append(t)
            weights.append(weight)

        if not vecs:
            continue

        mat = torch.stack(vecs, dim=0)
        w = torch.tensor(weights, dtype=torch.float32, device=mat.device).unsqueeze(1)

        centroid = (mat * w).sum(dim=0) / w.sum().clamp_min(1e-12)
        centroid = centroid / centroid.norm().clamp_min(1e-12)

        centroids[int(topic_id)] = centroid
        labels[int(topic_id)] = choose_topic_representative(group)

    return centroids, labels


def refine_assignments(
    df: pd.DataFrame,
    phrase_tensors: dict[str, torch.Tensor],
    sim_margin: float,
    min_similarity: float,
    max_iter: int,
) -> pd.DataFrame:
    out = df.copy()

    if "topic_label" not in out.columns:
        out["topic_label"] = ""

    for iteration in range(1, max_iter + 1):
        moved = 0
        centroids, labels = compute_topic_centroids(out, phrase_tensors)
        topic_ids = sorted(centroids.keys())

        if len(topic_ids) <= 1:
            log("[refine_assignments] only one topic present, skipping")
            break

        centroid_matrix = torch.stack([centroids[t] for t in topic_ids], dim=0)
        topic_index = {tid: idx for idx, tid in enumerate(topic_ids)}

        phrase_topic = (
            out.groupby(["canonical_phrase", "topic_id"], as_index=False)
            .agg(doc_count=("doc_count", "sum"))
            .sort_values(["canonical_phrase", "doc_count"], ascending=[True, False])
            .drop_duplicates(subset=["canonical_phrase"], keep="first")
        )

        phrase_new_topic: dict[str, int] = {}

        for _, row in phrase_topic.iterrows():
            p = row["canonical_phrase"]
            cur_topic = int(row["topic_id"])
            vec = phrase_tensors.get(p)

            if vec is None:
                phrase_new_topic[p] = cur_topic
                continue

            sims = centroid_matrix @ vec
            best_idx = int(torch.argmax(sims).item())
            best_topic = topic_ids[best_idx]
            best_sim = float(sims[best_idx].item())

            cur_idx = topic_index.get(cur_topic)
            cur_sim = float(sims[cur_idx].item()) if cur_idx is not None else -1.0

            if best_topic != cur_topic and best_sim >= min_similarity and (best_sim - cur_sim) >= sim_margin:
                phrase_new_topic[p] = best_topic
                moved += 1
            else:
                phrase_new_topic[p] = cur_topic

        out["topic_id"] = out["canonical_phrase"].map(phrase_new_topic).fillna(out["topic_id"]).astype(int)
        log(f"[refine_assignments] iter={iteration}, moved={moved}")

        if moved == 0:
            break

    _, labels = compute_topic_centroids(out, phrase_tensors)
    out["topic_label"] = out["topic_id"].map(labels).fillna(out["topic_label"])

    return out


# =========================
# Summaries / Exports
# =========================
def summarize_topics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for topic_id, group in df.groupby("topic_id"):
        rep = choose_topic_representative(group)

        phrases_sorted = (
            group.groupby("canonical_phrase", as_index=False)
            .agg(
                doc_count=("doc_count", "sum"),
                tf=("tf", "sum"),
            )
        )

        phrases_sorted["generic_rank"] = phrases_sorted["canonical_phrase"].map(
            lambda x: {"specific": 0, "maybe_generic": 1, "generic": 2}[classify_term_genericity(x)]
        )

        phrases_sorted = phrases_sorted.sort_values(
            ["generic_rank", "doc_count", "tf", "canonical_phrase"],
            ascending=[True, False, False, True],
        )

        rows.append(
            {
                "topic_id": int(topic_id),
                "topic_label": rep,
                "n_phrases": int(group["canonical_phrase"].nunique()),
                "sum_doc_count": float(group["doc_count"].sum()),
                "generic_ratio": topic_generic_ratio(group),
                "label_genericity": classify_term_genericity(rep),
                "phrases": " | ".join(phrases_sorted["canonical_phrase"].tolist()[:20]),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["sum_doc_count", "n_phrases", "topic_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def rebuild_doc_map_with_refined_topics(
    original_doc_map: pd.DataFrame,
    refined_phrase_topic_map: pd.DataFrame,
) -> pd.DataFrame:
    key = refined_phrase_topic_map[["phrase", "topic_id", "topic_label"]].drop_duplicates()
    key = key.rename(columns={"phrase": "phrase_canonical"})

    out = original_doc_map.drop(columns=["topic_id", "topic_label"], errors="ignore").merge(
        key,
        on="phrase_canonical",
        how="left",
    )

    if "concept_id" in out.columns:
        out = out.drop(columns=["concept_id"], errors="ignore")

    out["concept_id"] = (
        "BT" + out["topic_id"].fillna(-1).astype(int).map(lambda x: f"{x+1:04d}" if x >= 0 else "9999")
    )

    return out


def demote_tiny_generic_topics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    bad_topic_ids = []
    for topic_id, group in out.groupby("topic_id"):
        phrases = group["canonical_phrase"].drop_duplicates().tolist()
        if len(phrases) <= 2 and all(classify_term_genericity(p) == "generic" for p in phrases):
            bad_topic_ids.append(int(topic_id))

    if bad_topic_ids:
        log(f"[demote_tiny_generic_topics] removing topics: {bad_topic_ids}")
        out = out[~out["topic_id"].isin(bad_topic_ids)].copy()

    return out


def build_final_exports(
    refined_df: pd.DataFrame,
    original_doc_map: pd.DataFrame,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)

    refined_df = demote_tiny_generic_topics(refined_df)

    refined_phrase_topic_map = refined_df.sort_values(
        ["topic_id", "doc_count", "canonical_phrase"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    topic_summary = summarize_topics(refined_df)
    refined_doc_map = rebuild_doc_map_with_refined_topics(original_doc_map, refined_phrase_topic_map)

    refined_phrase_topic_map.to_csv(
        output_dir / "phrase_topic_map_ai_refined.csv",
        index=False,
        encoding="utf-8-sig",
    )
    topic_summary.to_csv(
        output_dir / "topic_summary_ai_refined.csv",
        index=False,
        encoding="utf-8-sig",
    )
    refined_doc_map.to_csv(
        output_dir / "doc_concept_map_ai_refined.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log(f"[EXPORT] saved: {output_dir / 'phrase_topic_map_ai_refined.csv'}")
    log(f"[EXPORT] saved: {output_dir / 'topic_summary_ai_refined.csv'}")
    log(f"[EXPORT] saved: {output_dir / 'doc_concept_map_ai_refined.csv'}")


# =========================
# Main
# =========================
def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_dir = data_dir / "phase2_postprocess"
    ensure_dir(output_dir)

    device = get_runtime_device()
    log_runtime_status(device)

    doc_map_path = resolve_input_file(output_dir, "doc_concept_map_bertopic.csv")
    canonical_map_path = resolve_input_file(output_dir, "phrase_canonical_map.csv")

    log("=== RUN AI CONCEPT REFINE ===")
    log(f"project_root : {project_root}")
    log(f"data_dir     : {data_dir}")
    log(f"output_dir   : {output_dir}")
    log(f"doc map      : {doc_map_path}")
    log(f"canonical map: {canonical_map_path}")

    doc_map_raw = pd.read_csv(doc_map_path)
    canonical_map_raw = pd.read_csv(canonical_map_path)

    log(f"[LOAD] doc_map shape: {doc_map_raw.shape}")
    log(f"[LOAD] doc_map columns: {doc_map_raw.columns.tolist()}")
    log(f"[LOAD] canonical_map shape: {canonical_map_raw.shape}")
    log(f"[LOAD] canonical_map columns: {canonical_map_raw.columns.tolist()}")

    phrase_topic_map = build_phrase_topic_map_from_doc_map(doc_map_raw)
    canonical_map = standardize_canonical_map(canonical_map_raw)
    df = attach_canonical_phrase(phrase_topic_map, canonical_map)

    log(f"[BUILD] phrase_topic_map shape: {phrase_topic_map.shape}")
    log(f"[BUILD] merged refine df shape: {df.shape}")

    unique_phrases = (
        df["canonical_phrase"]
        .dropna()
        .astype(str)
        .map(normalize_spaces)
        .drop_duplicates()
        .tolist()
    )
    log(f"[AI-REFINE] phrase count: {len(unique_phrases)}")

    log("[AI-REFINE] embedding start")
    embeddings = embed_texts(unique_phrases, device=device)
    log("[AI-REFINE] embedding done")

    _, phrase_tensors = build_embedding_maps(unique_phrases, embeddings, device=device)
    maybe_cuda_cleanup(device)

    log("[AI-REFINE] split large clusters start")
    df = split_large_clusters(
        df,
        phrase_tensors=phrase_tensors,
        size_threshold=SPLIT_LARGE_SIZE_THRESHOLD,
        cohesion_threshold=SPLIT_LARGE_COHESION_THRESHOLD,
    )
    log("[AI-REFINE] split large clusters done")

    log("[AI-REFINE] split bad topics start")
    df = split_bad_topics(
        df,
        phrase_tensors=phrase_tensors,
        min_component_size=SPLIT_BAD_MIN_COMPONENT_SIZE,
        similarity_threshold=SPLIT_BAD_SIMILARITY_THRESHOLD_PASS1,
        max_phrases_for_split=SPLIT_BAD_MAX_PHRASES_FOR_SPLIT,
    )
    log("[AI-REFINE] split bad topics done")

    log("[AI-REFINE] refine assignments start")
    df = refine_assignments(
        df,
        phrase_tensors=phrase_tensors,
        sim_margin=REFINE_SIM_MARGIN,
        min_similarity=REFINE_MIN_SIMILARITY,
        max_iter=REFINE_MAX_ITER,
    )
    log("[AI-REFINE] refine assignments done")

    log("[AI-REFINE] split bad topics (2nd pass) start")
    df = split_bad_topics(
        df,
        phrase_tensors=phrase_tensors,
        min_component_size=SPLIT_BAD_MIN_COMPONENT_SIZE,
        similarity_threshold=SPLIT_BAD_SIMILARITY_THRESHOLD_PASS2,
        max_phrases_for_split=SPLIT_BAD_MAX_PHRASES_FOR_SPLIT,
    )
    log("[AI-REFINE] split bad topics (2nd pass) done")

    log("[AI-REFINE] final refine start")
    df = refine_assignments(
        df,
        phrase_tensors=phrase_tensors,
        sim_margin=REFINE_SIM_MARGIN,
        min_similarity=REFINE_MIN_SIMILARITY,
        max_iter=REFINE_MAX_ITER_FINAL,
    )
    log("[AI-REFINE] final refine done")

    preview = summarize_topics(df)
    if len(preview):
        print(preview.head(15).to_string(index=False))

    build_final_exports(
        refined_df=df,
        original_doc_map=doc_map_raw,
        output_dir=output_dir,
    )

    maybe_cuda_cleanup(device)
    log("=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR] AI CONCEPT REFINE FAILED", flush=True)
        print(type(e).__name__, e, flush=True)
        traceback.print_exc()
        raise