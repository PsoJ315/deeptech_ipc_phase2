from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


GENERIC_LABEL_STOPWORDS = {
    "조성물", "장치", "시스템", "방법", "프로그램", "디바이스", "기기",
    "부재", "기판", "회로", "구조", "모듈", "물질", "재료", "부품",
    "표면", "용액", "기재", "유닛", "소자", "부분", "장비",
    "서비스", "콘텐츠", "베이스", "가이드", "프레임",
}

PREFERRED_SIGNAL_TERMS = {
    "디지털", "트윈", "이미지", "영상", "분석", "처리", "신호", "데이터",
    "반도체", "센서", "배터리", "전지", "전극", "전해질", "촉매", "합금",
    "리튬", "그래핀", "카본", "탄소", "실리콘", "규소",
}

TOKEN_PATTERN = r"(?u)\b[\w가-힣][\w가-힣\-\+/]*\b"

# small-scale fallback threshold
SMALL_DATA_FALLBACK_THRESHOLD = 300


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def resolve_input_file(
    input_dir: str | Path,
    output_dir: str | Path,
    filename: str,
) -> Path:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    project_root = input_dir.parent if input_dir.name == "data" else input_dir
    cwd = Path.cwd()

    candidates = [
        output_dir / filename,
        input_dir / filename,
        project_root / "data" / filename,
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


def safe_import_bertopic_stack():
    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
        from hdbscan import HDBSCAN
    except ImportError as e:
        raise ImportError(
            "BERTopic stack is not installed.\n"
            "Required packages: bertopic, sentence-transformers, umap-learn, hdbscan, scikit-learn\n"
            "Install them in your patents environment, then rerun run_phase2_postprocess.py"
        ) from e

    return BERTopic, CountVectorizer, UMAP, HDBSCAN


def safe_import_fallback_stack():
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as e:
        raise ImportError(
            "Fallback clustering stack is not installed.\n"
            "Required packages: sentence-transformers, scikit-learn"
        ) from e

    return SentenceTransformer, AgglomerativeClustering


def build_topic_label_from_terms(terms: list[tuple[str, float]], topic_id: int) -> str:
    if topic_id == -1:
        return "OUTLIER"

    cleaned_terms: list[str] = []
    fallback_terms: list[str] = []

    for term, _score in terms:
        term = normalize_spaces(term)
        if not term:
            continue
        fallback_terms.append(term)
        if term not in GENERIC_LABEL_STOPWORDS:
            cleaned_terms.append(term)

    chosen = cleaned_terms[:3] if cleaned_terms else fallback_terms[:3]
    if not chosen:
        return f"TOPIC_{topic_id}"

    chosen = sorted(
        chosen,
        key=lambda x: (0 if any(sig in x for sig in PREFERRED_SIGNAL_TERMS) else 1, -len(x))
    )
    return " | ".join(chosen[:3])


def build_topic_label_from_phrases(phrases: list[str], topic_id: int) -> str:
    if topic_id == -1:
        return "OUTLIER"

    cleaned = []
    fallback = []

    for phrase in phrases:
        phrase = normalize_spaces(phrase)
        if not phrase:
            continue
        fallback.append(phrase)

        tokens = [t for t in re.findall(TOKEN_PATTERN, phrase) if t not in GENERIC_LABEL_STOPWORDS]
        cleaned.extend(tokens if tokens else [phrase])

    chosen_pool = cleaned if cleaned else fallback
    if not chosen_pool:
        return f"TOPIC_{topic_id}"

    # unique preserve order
    seen = set()
    uniq = []
    for x in chosen_pool:
        if x not in seen:
            seen.add(x)
            uniq.append(x)

    uniq = sorted(
        uniq,
        key=lambda x: (0 if any(sig in x for sig in PREFERRED_SIGNAL_TERMS) else 1, -len(x))
    )
    return " | ".join(uniq[:3]) if uniq else f"TOPIC_{topic_id}"


def detect_phrase_col(df: pd.DataFrame) -> str:
    for cand in ["phrase", "phrase_canonical", "matched_phrase"]:
        if cand in df.columns:
            return cand
    raise ValueError("Could not find phrase column. Expected one of: phrase, phrase_canonical, matched_phrase")


def detect_doc_col(df: pd.DataFrame) -> str:
    for cand in ["doc_id", "document_id", "id"]:
        if cand in df.columns:
            return cand
    raise ValueError("Could not find doc id column. Expected one of: doc_id, document_id, id")


def prepare_phrase_docs(canonical_df: pd.DataFrame) -> pd.DataFrame:
    df = canonical_df.copy()

    required = {"phrase_canonical"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in phrase_canonical_map.csv: {sorted(missing)}")

    keep_cols = [c for c in [
        "phrase_original",
        "phrase_canonical",
        "representative_phrase",
        "df",
        "tf",
        "cleaned_score",
        "canonical_score_proxy",
    ] if c in df.columns]

    df = df[keep_cols].copy()

    sort_cols = [c for c in ["canonical_score_proxy", "cleaned_score", "df", "tf"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    phrase_docs = df.drop_duplicates(subset=["phrase_canonical"]).copy().reset_index(drop=True)
    if "representative_phrase" not in phrase_docs.columns:
        phrase_docs["representative_phrase"] = phrase_docs["phrase_canonical"]

    phrase_docs["doc_text"] = phrase_docs["phrase_canonical"].astype(str).map(normalize_spaces)
    phrase_docs["doc_index"] = range(len(phrase_docs))

    return phrase_docs


def fit_bertopic_on_phrases(phrase_docs: pd.DataFrame):
    BERTopic, CountVectorizer, UMAP, HDBSCAN = safe_import_bertopic_stack()

    docs = phrase_docs["doc_text"].tolist()
    n_docs = len(docs)

    if n_docs < 5:
        raise ValueError(f"Not enough phrase docs for BERTopic: {n_docs}. Need at least 5.")

    n_neighbors = max(2, min(8, n_docs - 1))
    n_components = 5 if n_docs >= 30 else 2

    print(f"[STEP 3.5] BERTopic mode")
    print(f"[STEP 3.5] n_docs={n_docs}, n_neighbors={n_neighbors}, n_components={n_components}")

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 3),
        token_pattern=TOKEN_PATTERN,
        min_df=1,
    )

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        prediction_data=True,
    )

    topic_model = BERTopic(
        language="multilingual",
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        low_memory=True,
        verbose=True,
        nr_topics=None,
    )

    try:
        print("[STEP 3.5] BERTopic fit_transform start")
        topics, _probs = topic_model.fit_transform(docs)
        print("[STEP 3.5] BERTopic fit_transform done")
    except Exception as e:
        print("[STEP 3.5] BERTopic failed during fit_transform")
        print(traceback.format_exc())
        raise RuntimeError(f"BERTopic fit_transform failed: {e}") from e

    if -1 in topics and len(set(topics)) > 1:
        try:
            print("[STEP 3.5] reducing outliers")
            reduced_topics = topic_model.reduce_outliers(docs, topics)
            topic_model.update_topics(
                docs,
                topics=reduced_topics,
                vectorizer_model=vectorizer_model,
            )
            topics = reduced_topics
            print("[STEP 3.5] reduce_outliers done")
        except Exception as e:
            print(f"[STEP 3.5] reduce_outliers skipped due to: {e}")

    return topic_model, list(topics)


def fit_fallback_clustering_on_phrases(phrase_docs: pd.DataFrame):
    SentenceTransformer, AgglomerativeClustering = safe_import_fallback_stack()

    docs = phrase_docs["doc_text"].tolist()
    n_docs = len(docs)

    if n_docs < 2:
        raise ValueError(f"Not enough phrase docs for fallback clustering: {n_docs}. Need at least 2.")

    print(f"[STEP 3.5] fallback mode: embedding + agglomerative clustering")
    print(f"[STEP 3.5] n_docs={n_docs}")

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder = SentenceTransformer(model_name)

    print("[STEP 3.5] embedding phrases for fallback")
    embeddings = embedder.encode(
        docs,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # small n -> conservative cluster count
    n_clusters = max(2, min(int(np.sqrt(n_docs)), max(2, n_docs // 8)))
    n_clusters = min(n_clusters, n_docs - 1) if n_docs > 2 else 2

    print(f"[STEP 3.5] fallback clustering with n_clusters={n_clusters}")

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    return None, list(labels), embeddings


def build_topic_exports_from_fallback(
    phrase_docs: pd.DataFrame,
    topics: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phrase_map = phrase_docs.copy()
    phrase_map["topic_id"] = [int(t) for t in topics]
    phrase_map["is_outlier_topic"] = False

    topic_terms_rows = []
    topic_label_map: dict[int, str] = {}

    for topic_id, grp in phrase_map.groupby("topic_id"):
        phrase_candidates = grp["phrase_canonical"].astype(str).tolist()
        topic_label = build_topic_label_from_phrases(phrase_candidates, int(topic_id))
        topic_label_map[int(topic_id)] = topic_label

        score_candidates = []
        for _, row in grp.iterrows():
            score = 0.0
            if "canonical_score_proxy" in row and pd.notna(row["canonical_score_proxy"]):
                score = float(row["canonical_score_proxy"])
            elif "cleaned_score" in row and pd.notna(row["cleaned_score"]):
                score = float(row["cleaned_score"])
            elif "df" in row and pd.notna(row["df"]):
                score = float(row["df"])
            elif "tf" in row and pd.notna(row["tf"]):
                score = float(row["tf"])
            score_candidates.append((normalize_spaces(str(row["phrase_canonical"])), score))

        score_candidates = sorted(score_candidates, key=lambda x: (-x[1], -len(x[0])))
        for rank, (term, score) in enumerate(score_candidates[:15], start=1):
            topic_terms_rows.append({
                "topic_id": int(topic_id),
                "topic_label": topic_label,
                "term_rank": rank,
                "term": term,
                "term_score": score,
            })

    topic_terms_df = pd.DataFrame(topic_terms_rows)
    phrase_map["topic_label"] = phrase_map["topic_id"].map(topic_label_map)

    agg_dict = {
        "phrase_canonical": lambda s: sorted(set(s)),
        "representative_phrase": lambda s: sorted(set(s)),
    }
    if "phrase_original" in phrase_map.columns:
        agg_dict["phrase_original"] = lambda s: sorted(set(s))
    if "df" in phrase_map.columns:
        agg_dict["df"] = "max"
    if "tf" in phrase_map.columns:
        agg_dict["tf"] = "sum"
    if "cleaned_score" in phrase_map.columns:
        agg_dict["cleaned_score"] = "max"
    if "canonical_score_proxy" in phrase_map.columns:
        agg_dict["canonical_score_proxy"] = "max"

    concept_df = (
        phrase_map.groupby(["topic_id", "topic_label", "is_outlier_topic"], as_index=False)
        .agg(agg_dict)
        .rename(columns={
            "phrase_canonical": "member_phrases",
            "representative_phrase": "member_representatives",
            "phrase_original": "member_original_phrases",
            "df": "concept_df_proxy",
            "tf": "concept_tf_sum",
            "cleaned_score": "concept_best_phrase_score",
            "canonical_score_proxy": "concept_score_proxy",
        })
    )

    concept_df["concept_size"] = concept_df["member_phrases"].map(len)
    sort_cols = [c for c in ["concept_size", "concept_df_proxy", "concept_score_proxy", "concept_tf_sum"] if c in concept_df.columns]
    concept_df = concept_df.sort_values(
        by=sort_cols,
        ascending=[False] * len(sort_cols)
    ).reset_index(drop=True)
    concept_df["concept_id"] = [f"BT{idx:04d}" for idx in range(1, len(concept_df) + 1)]

    topic_to_concept_id = dict(zip(concept_df["topic_id"], concept_df["concept_id"]))
    phrase_map["concept_id"] = phrase_map["topic_id"].map(topic_to_concept_id)

    for col in ["member_phrases", "member_representatives", "member_original_phrases"]:
        if col in concept_df.columns:
            concept_df[f"{col}_str"] = concept_df[col].map(lambda x: " || ".join(map(str, x)))

    return concept_df, phrase_map, topic_terms_df


def build_topic_exports(
    topic_model,
    phrase_docs: pd.DataFrame,
    topics: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phrase_map = phrase_docs.copy()
    phrase_map["topic_id"] = topics
    phrase_map["is_outlier_topic"] = phrase_map["topic_id"].eq(-1)

    info_df = topic_model.get_topic_info().copy()

    topic_terms_rows = []
    topic_label_map: dict[int, str] = {}

    topic_ids = info_df["Topic"].tolist() if "Topic" in info_df.columns else sorted(set(topics))
    for topic_id in topic_ids:
        terms = topic_model.get_topic(topic_id)
        if topic_id == -1:
            topic_label = "OUTLIER"
        else:
            topic_label = build_topic_label_from_terms(terms or [], topic_id)
        topic_label_map[int(topic_id)] = topic_label

        if terms:
            for rank, (term, score) in enumerate(terms, start=1):
                topic_terms_rows.append({
                    "topic_id": int(topic_id),
                    "topic_label": topic_label,
                    "term_rank": rank,
                    "term": term,
                    "term_score": score,
                })

    topic_terms_df = pd.DataFrame(topic_terms_rows)
    phrase_map["topic_label"] = phrase_map["topic_id"].map(topic_label_map)

    agg_dict = {
        "phrase_canonical": lambda s: sorted(set(s)),
        "representative_phrase": lambda s: sorted(set(s)),
    }
    if "phrase_original" in phrase_map.columns:
        agg_dict["phrase_original"] = lambda s: sorted(set(s))
    if "df" in phrase_map.columns:
        agg_dict["df"] = "max"
    if "tf" in phrase_map.columns:
        agg_dict["tf"] = "sum"
    if "cleaned_score" in phrase_map.columns:
        agg_dict["cleaned_score"] = "max"
    if "canonical_score_proxy" in phrase_map.columns:
        agg_dict["canonical_score_proxy"] = "max"

    concept_df = (
        phrase_map.groupby(["topic_id", "topic_label", "is_outlier_topic"], as_index=False)
        .agg(agg_dict)
        .rename(columns={
            "phrase_canonical": "member_phrases",
            "representative_phrase": "member_representatives",
            "phrase_original": "member_original_phrases",
            "df": "concept_df_proxy",
            "tf": "concept_tf_sum",
            "cleaned_score": "concept_best_phrase_score",
            "canonical_score_proxy": "concept_score_proxy",
        })
    )

    concept_df["concept_size"] = concept_df["member_phrases"].map(len)

    sort_cols = [c for c in ["is_outlier_topic", "concept_size", "concept_df_proxy", "concept_score_proxy", "concept_tf_sum"] if c in concept_df.columns]
    concept_df = concept_df.sort_values(
        by=sort_cols,
        ascending=[True] + [False] * (len(sort_cols) - 1)
    ).reset_index(drop=True)
    concept_df["concept_id"] = [f"BT{idx:04d}" for idx in range(1, len(concept_df) + 1)]

    topic_to_concept_id = dict(zip(concept_df["topic_id"], concept_df["concept_id"]))
    phrase_map["concept_id"] = phrase_map["topic_id"].map(topic_to_concept_id)

    for col in ["member_phrases", "member_representatives", "member_original_phrases"]:
        if col in concept_df.columns:
            concept_df[f"{col}_str"] = concept_df[col].map(lambda x: " || ".join(map(str, x)))

    return concept_df, phrase_map, topic_terms_df


def build_doc_concept_map_bertopic(
    doc_phrase_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    phrase_topic_map_df: pd.DataFrame,
) -> pd.DataFrame:
    phrase_col = detect_phrase_col(doc_phrase_df)
    doc_col = detect_doc_col(doc_phrase_df)

    work = doc_phrase_df.copy()
    work["phrase_norm"] = work[phrase_col].astype(str).map(normalize_spaces)

    canonical_pairs = canonical_df[["phrase_original", "phrase_canonical"]].drop_duplicates().copy()
    canonical_pairs["phrase_original"] = canonical_pairs["phrase_original"].astype(str).map(normalize_spaces)
    canonical_pairs["phrase_canonical"] = canonical_pairs["phrase_canonical"].astype(str).map(normalize_spaces)

    orig_to_canonical = dict(zip(canonical_pairs["phrase_original"], canonical_pairs["phrase_canonical"]))

    topic_lookup_original = {}
    topic_lookup_canonical = {}

    for _, row in phrase_topic_map_df.iterrows():
        canon = normalize_spaces(row["phrase_canonical"])
        record = {
            "phrase_canonical": canon,
            "topic_id": row["topic_id"],
            "topic_label": row["topic_label"],
            "concept_id": row["concept_id"],
            "is_outlier_topic": row["is_outlier_topic"],
        }
        topic_lookup_canonical[canon] = record

        if "phrase_original" in row and pd.notna(row["phrase_original"]):
            topic_lookup_original[normalize_spaces(row["phrase_original"])] = record

    def resolve_topic_record(phrase: str) -> Optional[dict]:
        phrase = normalize_spaces(phrase)
        if phrase in topic_lookup_original:
            return topic_lookup_original[phrase]
        if phrase in topic_lookup_canonical:
            return topic_lookup_canonical[phrase]

        canon = orig_to_canonical.get(phrase)
        if canon and canon in topic_lookup_canonical:
            return topic_lookup_canonical[canon]

        return None

    work["topic_record"] = work["phrase_norm"].map(resolve_topic_record)
    mapped = work.loc[work["topic_record"].notna()].copy()

    mapped["phrase_canonical"] = mapped["topic_record"].map(lambda x: x["phrase_canonical"])
    mapped["topic_id"] = mapped["topic_record"].map(lambda x: x["topic_id"])
    mapped["topic_label"] = mapped["topic_record"].map(lambda x: x["topic_label"])
    mapped["concept_id"] = mapped["topic_record"].map(lambda x: x["concept_id"])
    mapped["is_outlier_topic"] = mapped["topic_record"].map(lambda x: x["is_outlier_topic"])

    out = mapped[[doc_col, phrase_col, "phrase_canonical", "topic_id", "topic_label", "concept_id", "is_outlier_topic"]].copy()
    out = out.rename(columns={
        doc_col: "doc_id",
        phrase_col: "phrase_original_in_doc",
    }).drop_duplicates().reset_index(drop=True)

    return out


def try_build_hierarchy(topic_model, docs: list[str]) -> pd.DataFrame:
    if topic_model is None:
        return pd.DataFrame()

    try:
        hier = topic_model.hierarchical_topics(docs)
        if isinstance(hier, pd.DataFrame):
            return hier
    except Exception as e:
        print(f"[STEP 3.5] hierarchical_topics skipped due to: {e}")
    return pd.DataFrame()


def build_fallback_topic_info(
    concept_df: pd.DataFrame,
    topic_terms_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    term_count_map = topic_terms_df.groupby("topic_id").size().to_dict() if not topic_terms_df.empty else {}

    for _, row in concept_df.iterrows():
        rows.append({
            "Topic": row["topic_id"],
            "Count": row["concept_size"],
            "Name": row["topic_label"],
            "Representative_Docs": row.get("member_representatives_str", ""),
            "Representative_Terms_Count": term_count_map.get(row["topic_id"], 0),
            "topic_label": row["topic_label"],
        })

    return pd.DataFrame(rows)


def run_step35_bertopic_rebuild(
    input_dir: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_map_path = resolve_input_file(input_dir, output_dir, "phrase_canonical_map.csv")
    doc_phrase_map_path = resolve_input_file(input_dir, output_dir, "doc_phrase_map.csv")

    phrase_topic_map_path = output_dir / "phrase_topic_map_bertopic.csv"
    topic_info_path = output_dir / "bertopic_topic_info.csv"
    topic_terms_path = output_dir / "bertopic_topic_terms.csv"
    concept_families_path = output_dir / "concept_families_bertopic.csv"
    doc_concept_map_path = output_dir / "doc_concept_map_bertopic.csv"
    hierarchy_path = output_dir / "bertopic_hierarchy.csv"

    print(f"[STEP 3.5] canonical_map_path resolved to: {canonical_map_path}")
    print(f"[STEP 3.5] doc_phrase_map_path resolved to: {doc_phrase_map_path}")

    canonical_df = pd.read_csv(canonical_map_path)
    phrase_docs = prepare_phrase_docs(canonical_df)

    print(f"[STEP 3.5] phrase docs for BERTopic: {len(phrase_docs):,}")

    topic_model = None
    topics = None
    run_mode = None

    try:
        if len(phrase_docs) < SMALL_DATA_FALLBACK_THRESHOLD:
            raise RuntimeError(
                f"Using fallback because phrase_docs={len(phrase_docs)} < {SMALL_DATA_FALLBACK_THRESHOLD}"
            )

        topic_model, topics = fit_bertopic_on_phrases(phrase_docs)
        concept_df, phrase_topic_map_df, topic_terms_df = build_topic_exports(
            topic_model=topic_model,
            phrase_docs=phrase_docs,
            topics=topics,
        )
        topic_info_df = topic_model.get_topic_info().copy()
        run_mode = "bertopic"

    except Exception as e:
        print(f"[STEP 3.5] BERTopic path not used or failed: {e}")
        print("[STEP 3.5] switching to fallback clustering")
        print(traceback.format_exc())

        topic_model, topics, _embeddings = fit_fallback_clustering_on_phrases(phrase_docs)
        concept_df, phrase_topic_map_df, topic_terms_df = build_topic_exports_from_fallback(
            phrase_docs=phrase_docs,
            topics=topics,
        )
        topic_info_df = build_fallback_topic_info(concept_df, topic_terms_df)
        run_mode = "fallback"

    if "Topic" in topic_info_df.columns:
        topic_label_map = (
            phrase_topic_map_df[["topic_id", "topic_label"]]
            .drop_duplicates()
            .rename(columns={"topic_id": "Topic"})
        )
        topic_info_df = topic_info_df.merge(topic_label_map, on="Topic", how="left", suffixes=("", "_mapped"))
        if "topic_label_mapped" in topic_info_df.columns:
            topic_info_df["topic_label"] = topic_info_df["topic_label_mapped"].combine_first(topic_info_df.get("topic_label"))
            topic_info_df = topic_info_df.drop(columns=["topic_label_mapped"])

    doc_phrase_df = pd.read_csv(doc_phrase_map_path)
    doc_concept_map_df = build_doc_concept_map_bertopic(
        doc_phrase_df=doc_phrase_df,
        canonical_df=canonical_df,
        phrase_topic_map_df=phrase_topic_map_df,
    )

    hierarchy_df = try_build_hierarchy(topic_model, phrase_docs["doc_text"].tolist())

    phrase_topic_map_df.to_csv(phrase_topic_map_path, index=False, encoding="utf-8-sig")
    topic_info_df.to_csv(topic_info_path, index=False, encoding="utf-8-sig")
    topic_terms_df.to_csv(topic_terms_path, index=False, encoding="utf-8-sig")
    doc_concept_map_df.to_csv(doc_concept_map_path, index=False, encoding="utf-8-sig")
    if not hierarchy_df.empty:
        hierarchy_df.to_csv(hierarchy_path, index=False, encoding="utf-8-sig")

    concept_df_for_save = concept_df.copy()
    drop_cols = [c for c in ["member_phrases", "member_representatives", "member_original_phrases"] if c in concept_df_for_save.columns]
    concept_df_for_save = concept_df_for_save.drop(columns=drop_cols)
    concept_df_for_save["run_mode"] = run_mode
    concept_df_for_save.to_csv(concept_families_path, index=False, encoding="utf-8-sig")

    print("=== STEP 3.5 COMPLETE ===")
    print(f"run mode          : {run_mode}")
    print(f"topic count       : {topic_info_df.shape[0]:,}")
    print(f"concept families  : {concept_df_for_save.shape[0]:,}")
    print(f"phrase-topic rows : {phrase_topic_map_df.shape[0]:,}")
    print(f"doc-concept rows  : {doc_concept_map_df.shape[0]:,}")
    print()

    preview_cols = [
        c for c in [
            "concept_id",
            "topic_id",
            "topic_label",
            "concept_size",
            "concept_df_proxy",
            "concept_tf_sum",
            "concept_best_phrase_score",
            "concept_score_proxy",
            "is_outlier_topic",
            "member_phrases_str",
            "run_mode",
        ]
        if c in concept_df_for_save.columns
    ]
    print("[Top 30 concept families]")
    print(concept_df_for_save[preview_cols].head(30).to_string(index=False))

    return concept_df_for_save, phrase_topic_map_df, doc_concept_map_df, topic_info_df