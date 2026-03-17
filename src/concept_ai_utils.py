from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

GENERIC_TERMS = {
    "조성물", "장치", "시스템", "방법", "프로그램", "디바이스", "기기",
    "부재", "기판", "회로", "구조", "모듈", "물질", "재료", "부품",
    "표면", "용액", "기재", "유닛", "소자", "부분", "장비",
    "서비스", "콘텐츠", "베이스", "가이드", "프레임", "플랫폼",
    "애플리케이션", "프로세스", "테스트", "업데이트", "케이스",
    "컨테이너", "구조체", "구조물", "설비", "기술", "본체",
    "인터페이스", "패키지", "세트", "폼", "매체", "정보", "데이터",
}

WEAK_SINGLETON_TERMS = {
    "프레임", "플랫폼", "서비스", "프로그램", "콘텐츠", "시스템",
    "구조체", "구조물", "부재", "장치", "디바이스", "유닛",
    "케이스", "컨테이너", "모듈", "베이스", "가이드", "재료",
    "물질", "부분", "기판", "부품", "표면", "용액", "방법",
    "데이터", "정보", "기술", "장비", "기기", "오디오", "비디오",
    "이미지", "텍스트",
}

SIGNAL_TERMS = {
    "디지털", "트윈", "분석", "신호", "반도체", "센서", "배터리", "전지",
    "전극", "전해질", "촉매", "합금", "리튬", "카본", "탄소",
    "실리콘", "규소", "음극", "양극", "박막", "포토레지스트",
    "트랜지스터", "전해액", "고체전해질", "복합재", "초음파",
    "광학", "바이오", "전력", "고분자", "나노", "분리막",
    "인공지능", "머신러닝", "전자파", "차폐", "rf", "에너지",
    "알루미늄", "산화물",
}

TOKEN_PATTERN = r"(?u)\b[\w가-힣][\w가-힣\-\+/]*\b"
NUM_ONLY_PATTERN = re.compile(r"^\d+([.,]\d+)?$")
ALNUM_SHORT_PATTERN = re.compile(r"^[A-Za-z0-9]{1,2}$")


def get_default_config() -> dict[str, Any]:
    return {
        "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "cross_encoder_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "use_cross_encoder": False,
        "top_k_candidate_clusters": 4,
        "low_fit_threshold": 0.42,
        "reassign_margin": 0.08,
        "split_large_cluster_threshold": 12,
        "split_max_subclusters": 4,
        "min_cluster_size_after_refine": 2,
        "outlier_topic_id": -1,
    }


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize_text(text: str) -> list[str]:
    return [t for t in re.findall(TOKEN_PATTERN, normalize_spaces(text)) if t]


def contains_signal_term(text: str) -> bool:
    text = normalize_spaces(text).lower()
    return any(sig in text for sig in SIGNAL_TERMS)


def phrase_quality_score(text: str) -> float:
    text = normalize_spaces(text)
    tokens = tokenize_text(text)
    if not tokens:
        return 0.0

    informative = [t for t in tokens if t not in GENERIC_TERMS and not NUM_ONLY_PATTERN.match(t)]
    weak = [t for t in tokens if t in GENERIC_TERMS]

    score = 0.35
    score += min(len(informative) * 0.18, 0.54)
    score -= min(len(weak) * 0.10, 0.30)

    if contains_signal_term(text):
        score += 0.22
    if len(tokens) >= 2:
        score += 0.12
    if len(text) >= 8:
        score += 0.08
    if len(text) >= 14:
        score += 0.05
    if len(tokens) == 1 and tokens[0].lower() in WEAK_SINGLETON_TERMS:
        score -= 0.30
    if NUM_ONLY_PATTERN.match(text):
        score -= 0.50
    if ALNUM_SHORT_PATTERN.match(text) and not contains_signal_term(text):
        score -= 0.20

    return max(0.0, min(1.0, score))


def duplicate_penalty(text: str) -> float:
    tokens = [t.lower() for t in tokenize_text(text)]
    if not tokens:
        return 0.15
    cnt = Counter(tokens)
    repeated = sum(v - 1 for v in cnt.values() if v > 1)
    penalty = min(repeated * 0.12, 0.35)
    if len(tokens) == 1 and tokens[0] in WEAK_SINGLETON_TERMS:
        penalty += 0.18
    return min(0.5, penalty)


def lexical_overlap_score(a: str, b: str) -> float:
    ta = {t.lower() for t in tokenize_text(a) if t not in GENERIC_TERMS}
    tb = {t.lower() for t in tokenize_text(b) if t not in GENERIC_TERMS}
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def select_representative_phrases(grp: pd.DataFrame, top_k: int = 5) -> list[str]:
    scored = []
    for _, row in grp.iterrows():
        phrase = normalize_spaces(str(row["phrase_canonical"]))
        score = phrase_quality_score(phrase)

        for col, w, cap in [
            ("canonical_score_proxy", 0.02, 0.25),
            ("cleaned_score", 0.02, 0.20),
            ("df", 0.01, 0.15),
            ("tf", 0.005, 0.10),
        ]:
            if col in row and pd.notna(row[col]):
                score += min(float(row[col]) * w, cap)

        if len(tokenize_text(phrase)) >= 2:
            score += 0.08
        if contains_signal_term(phrase):
            score += 0.12
        score -= duplicate_penalty(phrase)

        scored.append((phrase, score))

    scored = sorted(scored, key=lambda x: (-x[1], -len(x[0])))
    out = []
    seen = set()
    for phrase, _ in scored:
        if phrase not in seen:
            out.append(phrase)
            seen.add(phrase)
        if len(out) >= top_k:
            break
    return out


def build_cluster_label(grp: pd.DataFrame) -> str:
    reps = select_representative_phrases(grp, top_k=4)
    if not reps:
        return "UNLABELED"

    strong_multi = [p for p in reps if len(tokenize_text(p)) >= 2]
    if len(strong_multi) >= 2:
        return f"{strong_multi[0]} / {strong_multi[1]}"
    if len(strong_multi) == 1:
        return strong_multi[0]
    return reps[0]


def split_large_clusters(df: pd.DataFrame, embeddings: np.ndarray, config: dict[str, Any]) -> pd.DataFrame:
    from sklearn.cluster import AgglomerativeClustering

    work = df.copy()
    outlier_topic_id = config["outlier_topic_id"]

    unique_topics = sorted(t for t in work["topic_id"].dropna().unique().tolist() if t != outlier_topic_id)
    next_topic_id = (max(unique_topics) + 1) if unique_topics else 0

    for topic_id, grp in work.groupby("topic_id"):
        if topic_id == outlier_topic_id:
            continue
        if len(grp) < config["split_large_cluster_threshold"]:
            continue

        idxs = grp.index.to_list()
        split_target = min(max(2, len(grp) // 5), config["split_max_subclusters"])
        if split_target <= 1:
            continue

        model = AgglomerativeClustering(
            n_clusters=split_target,
            metric="cosine",
            linkage="average",
        )
        sub_labels = model.fit_predict(embeddings[idxs])
        counts = Counter(sub_labels)
        if len(counts) <= 1:
            continue

        dominant_label, _ = counts.most_common(1)[0]
        for idx, sub_lab in zip(idxs, sub_labels):
            if sub_lab == dominant_label:
                continue
            work.at[idx, "topic_id"] = next_topic_id + int(sub_lab)
        next_topic_id += split_target

    non_out = sorted(t for t in work["topic_id"].unique().tolist() if t != outlier_topic_id)
    remap = {old: new for new, old in enumerate(non_out)}
    work["topic_id"] = work["topic_id"].map(
        lambda x: outlier_topic_id if x == outlier_topic_id else remap[x]
    )
    return work


def refine_assignments(
    phrase_df: pd.DataFrame,
    embeddings: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = phrase_df.copy().reset_index(drop=True)

    cluster_rows = []
    for topic_id, grp in work.groupby("topic_id"):
        idxs = grp.index.to_list()
        centroid = embeddings[idxs].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        reps = select_representative_phrases(grp, top_k=5)
        label = build_cluster_label(grp)
        cluster_rows.append({
            "topic_id": int(topic_id),
            "centroid": centroid,
            "cluster_label": label,
            "representative_phrases": reps,
        })

    profiles = pd.DataFrame(cluster_rows)
    centroids = np.vstack(profiles["centroid"].to_list())
    centroid_topics = profiles["topic_id"].tolist()
    sim_mat = cosine_similarity_matrix(embeddings, centroids)
    profile_map = {int(r["topic_id"]): r for _, r in profiles.iterrows()}

    refined_rows = []
    new_topics = []

    for i, row in work.iterrows():
        phrase = normalize_spaces(row["phrase_canonical"])
        current_topic = int(row["topic_id"])

        top_idx = np.argsort(sim_mat[i])[::-1][: config["top_k_candidate_clusters"]]
        candidates = []

        for j in top_idx:
            topic_id = int(centroid_topics[j])
            prof = profile_map[topic_id]

            centroid_sim = float(np.dot(embeddings[i], prof["centroid"]))
            label_overlap = lexical_overlap_score(phrase, prof["cluster_label"])
            rep_overlap = max([lexical_overlap_score(phrase, r) for r in prof["representative_phrases"]] or [0.0])
            pq = phrase_quality_score(phrase)
            penalty = duplicate_penalty(phrase)

            fit = (
                0.42 * centroid_sim
                + 0.20 * label_overlap
                + 0.16 * rep_overlap
                + 0.18 * pq
                - 0.16 * penalty
            )
            fit = max(0.0, min(1.0, fit))

            candidates.append({
                "candidate_topic_id": topic_id,
                "centroid_similarity": centroid_sim,
                "label_overlap": label_overlap,
                "rep_overlap": rep_overlap,
                "phrase_quality_score": pq,
                "duplicate_penalty": penalty,
                "final_fit_score": fit,
            })

        candidates = sorted(candidates, key=lambda x: x["final_fit_score"], reverse=True)
        best = candidates[0]
        current = next((c for c in candidates if c["candidate_topic_id"] == current_topic), None)
        current_score = current["final_fit_score"] if current is not None else -1.0

        assigned_topic = current_topic
        decision = "keep"

        if best["final_fit_score"] < config["low_fit_threshold"]:
            assigned_topic = config["outlier_topic_id"]
            decision = "outlier"
        elif best["candidate_topic_id"] != current_topic and (best["final_fit_score"] - current_score) >= config["reassign_margin"]:
            assigned_topic = best["candidate_topic_id"]
            decision = "reassign"

        new_topics.append(assigned_topic)
        refined_rows.append({
            "phrase_canonical": phrase,
            "old_topic_id": current_topic,
            "new_topic_id": assigned_topic,
            "decision": decision,
            "best_candidate_topic_id": best["candidate_topic_id"],
            "current_fit_score": current_score,
            "best_fit_score": best["final_fit_score"],
            "best_centroid_similarity": best["centroid_similarity"],
            "best_label_overlap": best["label_overlap"],
            "best_rep_overlap": best["rep_overlap"],
            "phrase_quality_score": best["phrase_quality_score"],
            "duplicate_penalty": best["duplicate_penalty"],
        })

    work["topic_id"] = new_topics
    debug_df = pd.DataFrame(refined_rows)

    counts = work["topic_id"].value_counts().to_dict()
    work["topic_id"] = work["topic_id"].map(
        lambda x: config["outlier_topic_id"]
        if x != config["outlier_topic_id"] and counts.get(x, 0) < config["min_cluster_size_after_refine"]
        else x
    )

    non_out = sorted(t for t in work["topic_id"].unique().tolist() if t != config["outlier_topic_id"])
    remap = {old: new for new, old in enumerate(non_out)}
    work["topic_id"] = work["topic_id"].map(
        lambda x: config["outlier_topic_id"] if x == config["outlier_topic_id"] else remap[x]
    )

    return work, debug_df


def build_final_exports(refined_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = refined_df.copy()

    topic_label_map = {}
    topic_terms_rows = []

    for topic_id, grp in work.groupby("topic_id"):
        label = "OUTLIER" if topic_id == -1 else build_cluster_label(grp)
        topic_label_map[int(topic_id)] = label
        reps = select_representative_phrases(grp, top_k=12)
        for rank, term in enumerate(reps, start=1):
            topic_terms_rows.append({
                "topic_id": int(topic_id),
                "topic_label": label,
                "term_rank": rank,
                "term": term,
            })

    work["topic_label"] = work["topic_id"].map(topic_label_map)

    rows = []
    seq = 1
    for topic_id, grp in work.groupby("topic_id"):
        label = topic_label_map[int(topic_id)]
        reps = select_representative_phrases(grp, top_k=5)
        members = sorted(set(map(str, grp["phrase_canonical"].tolist())))
        concept_id = "OUTLIER" if topic_id == -1 else f"AI{seq:04d}"
        if topic_id != -1:
            seq += 1
        rows.append({
            "topic_id": int(topic_id),
            "topic_label": label,
            "concept_id": concept_id,
            "concept_size": len(grp),
            "member_phrases": members,
            "representative_phrases": reps,
            "member_phrases_str": " || ".join(members),
            "representative_phrases_str": " || ".join(reps),
        })

    concept_df = pd.DataFrame(rows).sort_values(["topic_id"], ascending=True).reset_index(drop=True)
    topic_terms_df = pd.DataFrame(topic_terms_rows)
    return concept_df, topic_terms_df
