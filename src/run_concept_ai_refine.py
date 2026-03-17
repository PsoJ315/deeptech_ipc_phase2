from __future__ import annotations

import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


GENERIC_BAN_TERMS = {
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
}

GENERIC_CONTAINS_PATTERNS = [
    r"조성물$",
    r"혼합물$",
    r"복합체$",
    r"장치$",
    r"유닛$",
    r"프로세스$",
    r"공정$",
    r"방법$",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).strip().split())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("l2_normalize expects a 2D array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return x / norms


def is_generic_term(text: str) -> bool:
    t = normalize_spaces(str(text))
    if t in GENERIC_BAN_TERMS:
        return True
    for pat in GENERIC_CONTAINS_PATTERNS:
        if re.search(pat, t):
            return True
    return False


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


def embed_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 32,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


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


def choose_topic_representative(group: pd.DataFrame) -> str:
    score_df = (
        group.groupby("canonical_phrase", as_index=False)["doc_count"]
        .sum()
        .sort_values(["doc_count", "canonical_phrase"], ascending=[False, True])
        .reset_index(drop=True)
    )

    if len(score_df) == 0:
        return "unknown"

    score_df["is_generic"] = score_df["canonical_phrase"].map(is_generic_term)

    non_generic = score_df[~score_df["is_generic"]].copy()
    if len(non_generic) > 0:
        return non_generic.iloc[0]["canonical_phrase"]

    return score_df.iloc[0]["canonical_phrase"]


def topic_coherence_stats(
    group: pd.DataFrame,
    phrase_embeddings: dict[str, np.ndarray],
    max_phrases_for_full_sim: int = 250,
    sample_size: int = 120,
) -> dict:
    phrases = group["canonical_phrase"].drop_duplicates().tolist()
    vecs = [phrase_embeddings[p] for p in phrases if p in phrase_embeddings]

    generic_ratio = float(np.mean([is_generic_term(p) for p in phrases])) if phrases else 1.0

    if len(vecs) < 2:
        return {
            "n_phrases": len(phrases),
            "mean_sim": 1.0,
            "generic_ratio": generic_ratio,
        }

    # 너무 크면 샘플링 기반 근사
    if len(vecs) > max_phrases_for_full_sim:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(vecs), size=min(sample_size, len(vecs)), replace=False)
        vecs = [vecs[i] for i in idx]

    mat = np.vstack(vecs).astype(np.float32)
    sim = mat @ mat.T

    n = sim.shape[0]
    off_diag_mean = (sim.sum() - np.trace(sim)) / max(n * (n - 1), 1)

    return {
        "n_phrases": len(phrases),
        "mean_sim": float(off_diag_mean),
        "generic_ratio": generic_ratio,
    }


def is_bad_topic(
    group: pd.DataFrame,
    phrase_embeddings: dict[str, np.ndarray],
    label_term: str | None = None,
) -> bool:
    stats = topic_coherence_stats(group, phrase_embeddings)
    label_term = label_term or choose_topic_representative(group)

    if stats["n_phrases"] >= 15 and stats["mean_sim"] < 0.42:
        return True

    if stats["n_phrases"] >= 12 and stats["generic_ratio"] >= 0.30:
        return True

    if is_generic_term(label_term) and stats["n_phrases"] >= 8:
        return True

    return False


def connected_components_from_similarity(
    phrases: list[str],
    phrase_embeddings: dict[str, np.ndarray],
    threshold: float = 0.50,
    max_phrases_for_matrix: int = 300,
) -> list[list[str]]:
    valid_phrases = [p for p in phrases if p in phrase_embeddings]

    if len(valid_phrases) == 0:
        return []
    if len(valid_phrases) == 1:
        return [valid_phrases]

    # 안전장치: 너무 큰 경우 NxN similarity matrix 생성 금지
    if len(valid_phrases) > max_phrases_for_matrix:
        log(
            f"[connected_components_from_similarity] skip matrix build: "
            f"{len(valid_phrases)} phrases > max_phrases_for_matrix={max_phrases_for_matrix}"
        )
        return [valid_phrases]

    mat = np.vstack([phrase_embeddings[p] for p in valid_phrases]).astype(np.float32)
    sim = mat @ mat.T

    visited = set()
    components = []

    for i in range(len(valid_phrases)):
        if i in visited:
            continue

        stack = [i]
        comp = []
        visited.add(i)

        while stack:
            cur = stack.pop()
            comp.append(valid_phrases[cur])

            nbrs = np.where(sim[cur] >= threshold)[0].tolist()
            for nb in nbrs:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        components.append(comp)

    return components

def split_bad_topics(
    df: pd.DataFrame,
    phrase_embeddings: dict[str, np.ndarray],
    min_component_size: int = 2,
    similarity_threshold: float = 0.50,
    max_phrases_for_split: int = 300,
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

        log(
            f"[split_bad_topics] ({idx}/{len(topic_ids)}) "
            f"topic={topic_id}, label='{label_term}', n_rows={len(group)}, n_phrases={n_phrases}"
        )

        if not is_bad_topic(group, phrase_embeddings, label_term=label_term):
            continue

        # 안전장치: 너무 큰 토픽은 여기서 억지로 NxN 분해하지 않음
        if n_phrases > max_phrases_for_split:
            log(
                f"[split_bad_topics] skip topic {topic_id} ('{label_term}') "
                f"because n_phrases={n_phrases} > max_phrases_for_split={max_phrases_for_split}"
            )
            continue

        try:
            components = connected_components_from_similarity(
                phrases=phrases,
                phrase_embeddings=phrase_embeddings,
                threshold=similarity_threshold,
                max_phrases_for_matrix=max_phrases_for_split,
            )
        except Exception as e:
            log(
                f"[split_bad_topics] ERROR on topic {topic_id} ('{label_term}') "
                f"with n_phrases={n_phrases}: {type(e).__name__}: {e}"
            )
            continue

        components = [c for c in components if len(c) >= min_component_size]
        if len(components) <= 1:
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


def compute_topic_centroids(
    df: pd.DataFrame,
    phrase_embeddings: dict[str, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    centroids: dict[int, np.ndarray] = {}
    labels: dict[int, str] = {}

    for topic_id, group in df.groupby("topic_id"):
        vecs = []
        weights = []
        for _, row in group.iterrows():
            p = row["canonical_phrase"]
            if p not in phrase_embeddings:
                continue
            vecs.append(phrase_embeddings[p])
            weights.append(float(row.get("doc_count", 1.0)))

        if not vecs:
            continue

        arr = np.vstack(vecs)
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        centroid = (arr * w).sum(axis=0) / max(w.sum(), 1e-12)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)

        centroids[int(topic_id)] = centroid.astype(np.float32)
        labels[int(topic_id)] = choose_topic_representative(group)

    return centroids, labels


def split_large_clusters(
    df: pd.DataFrame,
    phrase_embeddings: dict[str, np.ndarray],
    size_threshold: int = 18,
    cohesion_threshold: float = 0.58,
) -> pd.DataFrame:
    out = df.copy()
    current_max_topic = int(out["topic_id"].max()) if len(out) else 0

    changed = True
    while changed:
        changed = False
        for topic_id, group in out.groupby("topic_id"):
            if len(group) < size_threshold:
                continue

            phrases = group["canonical_phrase"].tolist()
            vecs = []
            valid_phrases = []
            for p in phrases:
                if p in phrase_embeddings:
                    vecs.append(phrase_embeddings[p])
                    valid_phrases.append(p)

            if len(vecs) < 4:
                continue

            mat = np.vstack(vecs)
            centroid = mat.mean(axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
            sims = mat @ centroid
            cohesion = float(np.mean(sims))

            if cohesion >= cohesion_threshold:
                continue

            idx_low = int(np.argmin(sims))
            idx_high = int(np.argmax(sims))
            seed_a = mat[idx_low]
            seed_b = mat[idx_high]

            sim_a = mat @ (seed_a / max(np.linalg.norm(seed_a), 1e-12))
            sim_b = mat @ (seed_b / max(np.linalg.norm(seed_b), 1e-12))
            assign_b = sim_b > sim_a

            if assign_b.sum() == 0 or assign_b.sum() == len(assign_b):
                continue

            phrases_b = {valid_phrases[i] for i, flag in enumerate(assign_b) if flag}
            current_max_topic += 1
            mask = (out["topic_id"] == topic_id) & (out["canonical_phrase"].isin(phrases_b))
            out.loc[mask, "topic_id"] = current_max_topic
            changed = True
            log(f"[split_large_clusters] split topic {topic_id} -> {topic_id}, {current_max_topic}")
            break

    return out


def refine_assignments(
    df: pd.DataFrame,
    phrase_embeddings: dict[str, np.ndarray],
    sim_margin: float = 0.035,
    min_similarity: float = 0.30,
    max_iter: int = 5,
) -> pd.DataFrame:
    out = df.copy()

    if "topic_label" not in out.columns:
        out["topic_label"] = ""

    for iteration in range(1, max_iter + 1):
        moved = 0
        centroids, labels = compute_topic_centroids(out, phrase_embeddings)
        topic_ids = sorted(centroids.keys())

        if len(topic_ids) <= 1:
            log("[refine_assignments] only one topic present, skipping")
            break

        topic_index = {tid: idx for idx, tid in enumerate(topic_ids)}
        centroid_matrix = np.vstack([centroids[t] for t in topic_ids])

        new_topic_ids = []
        for _, row in out.iterrows():
            p = row["canonical_phrase"]
            cur_topic = int(row["topic_id"])
            vec = phrase_embeddings.get(p)

            if vec is None:
                new_topic_ids.append(cur_topic)
                continue

            sims = centroid_matrix @ vec
            best_idx = int(np.argmax(sims))
            best_topic = topic_ids[best_idx]
            best_sim = float(sims[best_idx])

            cur_idx = topic_index.get(cur_topic)
            cur_sim = float(sims[cur_idx]) if cur_idx is not None else -1.0

            if best_topic != cur_topic and best_sim >= min_similarity and (best_sim - cur_sim) >= sim_margin:
                new_topic_ids.append(best_topic)
                moved += 1
            else:
                new_topic_ids.append(cur_topic)

        out["topic_id"] = new_topic_ids
        log(f"[refine_assignments] iter={iteration}, moved={moved}")

        if moved == 0:
            break

    centroids, labels = compute_topic_centroids(out, phrase_embeddings)
    out["topic_label"] = out["topic_id"].map(labels).fillna(out["topic_label"])

    return out


def summarize_topics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic_id, group in df.groupby("topic_id"):
        rep = choose_topic_representative(group)

        phrases_sorted = (
            group.groupby("canonical_phrase", as_index=False)["doc_count"]
            .sum()
            .sort_values(["doc_count", "canonical_phrase"], ascending=[False, True])
        )

        rows.append(
            {
                "topic_id": int(topic_id),
                "topic_label": rep,
                "n_phrases": int(group["canonical_phrase"].nunique()),
                "sum_doc_count": float(group["doc_count"].sum()),
                "generic_ratio": float(np.mean(phrases_sorted["canonical_phrase"].map(is_generic_term))),
                "phrases": " | ".join(phrases_sorted["canonical_phrase"].tolist()[:20]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["topic_id", "topic_label", "n_phrases", "sum_doc_count", "generic_ratio", "phrases"]
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

    summary = (
        out.groupby("topic_id")["canonical_phrase"]
        .nunique()
        .reset_index(name="n_phrases")
    )

    bad_topic_ids = []
    for topic_id in summary["topic_id"].tolist():
        group = out[out["topic_id"] == topic_id]
        phrases = group["canonical_phrase"].drop_duplicates().tolist()
        if len(phrases) <= 2 and all(is_generic_term(p) for p in phrases):
            bad_topic_ids.append(topic_id)

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


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_dir = data_dir / "phase2_postprocess"
    ensure_dir(output_dir)

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
    embeddings = embed_texts(unique_phrases)
    log("[AI-REFINE] embedding done")

    phrase_embeddings = {p: embeddings[i] for i, p in enumerate(unique_phrases)}

    log("[AI-REFINE] split large clusters start")
    df = split_large_clusters(
        df,
        phrase_embeddings,
        size_threshold=18,
        cohesion_threshold=0.58,
    )
    log("[AI-REFINE] split large clusters done")

    log("[AI-REFINE] split bad topics start")
    df = split_bad_topics(
        df,
        phrase_embeddings,
        min_component_size=2,
        similarity_threshold=0.50,
        max_phrases_for_split=300,
    )
    log("[AI-REFINE] split bad topics done")

    log("[AI-REFINE] refine assignments start")
    df = refine_assignments(
        df,
        phrase_embeddings,
        sim_margin=0.035,
        min_similarity=0.30,
        max_iter=5,
    )
    log("[AI-REFINE] refine assignments done")

    log("[AI-REFINE] split bad topics (2nd pass) start")
    df = split_bad_topics(
        df,
        phrase_embeddings,
        min_component_size=2,
        similarity_threshold=0.53,
        max_phrases_for_split=300,
    )
    log("[AI-REFINE] split bad topics (2nd pass) done")

    preview = summarize_topics(df)
    if len(preview):
        print(preview.head(10).to_string(index=False))

    build_final_exports(
        refined_df=df,
        original_doc_map=doc_map_raw,
        output_dir=output_dir,
    )

    log("=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR] AI CONCEPT REFINE FAILED", flush=True)
        print(type(e).__name__, e, flush=True)
        traceback.print_exc()
        raise