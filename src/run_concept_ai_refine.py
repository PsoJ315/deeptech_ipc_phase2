from __future__ import annotations

from pathlib import Path
import traceback

import pandas as pd

from concept_ai_utils import (
    build_final_exports,
    embed_texts,
    get_default_config,
    normalize_spaces,
    refine_assignments,
    split_large_clusters,
)

import traceback
import time

print("[AI-REFINE] refine assignments start")
t0 = time.time()

try:
    refined_df = refine_assignments(
        phrase_topic_map=phrase_topic_map,
        canonical_map=canonical_map,
        embeddings=embeddings,
        config=config,
    )
    print(f"[AI-REFINE] refine assignments done in {time.time()-t0:.2f}s")
    print("[AI-REFINE] refined_df shape:", None if refined_df is None else refined_df.shape)
    if refined_df is not None:
        print("[AI-REFINE] refined_df columns:", refined_df.columns.tolist())
except Exception as e:
    print("[AI-REFINE] refine assignments FAILED")
    print(type(e).__name__, e)
    traceback.print_exc()
    raise

def resolve_input_file(base_dir: Path, filename: str) -> Path:
    candidates = [
        base_dir / filename,
        base_dir.parent / filename,
        base_dir / "phase2_postprocess" / filename,
        Path.cwd() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"Could not find {filename}. Tried: {candidates}")


def main():
    try:
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"
        output_dir = data_dir / "phase2_postprocess"
        output_dir.mkdir(parents=True, exist_ok=True)

        phrase_topic_map_path = resolve_input_file(output_dir, "phrase_topic_map_bertopic.csv")
        canonical_map_path = resolve_input_file(output_dir, "phrase_canonical_map.csv")

        print("=== RUN AI CONCEPT REFINE ===")
        print(f"project_root : {project_root}")
        print(f"data_dir     : {data_dir}")
        print(f"output_dir   : {output_dir}")
        print(f"phrase map   : {phrase_topic_map_path}")
        print(f"canonical map: {canonical_map_path}")

        phrase_topic_df = pd.read_csv(phrase_topic_map_path)
        canonical_df = pd.read_csv(canonical_map_path)

        if "phrase_canonical" not in phrase_topic_df.columns:
            raise ValueError("phrase_topic_map_bertopic.csv must contain phrase_canonical")
        if "topic_id" not in phrase_topic_df.columns:
            raise ValueError("phrase_topic_map_bertopic.csv must contain topic_id")

        canon_cols = [
            c for c in [
                "phrase_canonical",
                "phrase_original",
                "canonical_score_proxy",
                "cleaned_score",
                "df",
                "tf",
            ]
            if c in canonical_df.columns
        ]
        canon_meta = canonical_df[canon_cols].drop_duplicates(subset=["phrase_canonical"]).copy()

        merged = phrase_topic_df.merge(canon_meta, on="phrase_canonical", how="left", suffixes=("", "_canon"))

        if "phrase_original" not in merged.columns and "phrase_original_canon" in merged.columns:
            merged["phrase_original"] = merged["phrase_original_canon"]

        merged["phrase_canonical"] = merged["phrase_canonical"].astype(str).map(normalize_spaces)
        merged = merged.drop_duplicates(subset=["phrase_canonical"]).reset_index(drop=True)

        config = get_default_config()
        config["use_cross_encoder"] = False

        print(f"[AI-REFINE] phrase count: {len(merged):,}")

        print("[AI-REFINE] embedding start")
        embeddings = embed_texts(
            merged["phrase_canonical"].astype(str).tolist(),
            model_name=config["embedding_model_name"],
        )
        print("[AI-REFINE] embedding done")

        print("[AI-REFINE] split large clusters start")
        split_df = split_large_clusters(merged, embeddings, config)
        print("[AI-REFINE] split large clusters done")

        print("[AI-REFINE] refine assignments start")
        refined_df, debug_df = refine_assignments(split_df, embeddings, config)
        print("[AI-REFINE] refine assignments done")
        print("[refine_assignments] entered")
        print("[refine_assignments] phrase_topic_map shape:", phrase_topic_map.shape)
        print("[refine_assignments] canonical_map shape:", canonical_map.shape if canonical_map is not None else None)
        print("[refine_assignments] embeddings shape:", getattr(embeddings, "shape", None))

        print("[AI-REFINE] build final exports start")
        concept_df, topic_terms_df = build_final_exports(refined_df)
        print("[AI-REFINE] build final exports done")

        topic_to_concept = dict(zip(concept_df["topic_id"], concept_df["concept_id"]))
        topic_to_label = dict(zip(concept_df["topic_id"], concept_df["topic_label"]))

        refined_df["concept_id"] = refined_df["topic_id"].map(topic_to_concept)
        refined_df["topic_label"] = refined_df["topic_id"].map(topic_to_label)

        concept_path = output_dir / "concept_families_ai_refined.csv"
        phrase_path = output_dir / "phrase_topic_map_ai_refined.csv"
        terms_path = output_dir / "topic_terms_ai_refined.csv"
        debug_path = output_dir / "phrase_refine_debug_scores.csv"

        print("[AI-REFINE] saving files")
        concept_df.to_csv(concept_path, index=False, encoding="utf-8-sig")
        refined_df.to_csv(phrase_path, index=False, encoding="utf-8-sig")
        topic_terms_df.to_csv(terms_path, index=False, encoding="utf-8-sig")
        debug_df.to_csv(debug_path, index=False, encoding="utf-8-sig")
        print("[AI-REFINE] saving files done")

        print("=== AI CONCEPT REFINE COMPLETE ===")
        print(f"concept families : {len(concept_df):,}")
        print(f"phrase rows      : {len(refined_df):,}")
        print(f"debug rows       : {len(debug_df):,}")
        print()

        preview_cols = [
            c for c in [
                "concept_id",
                "topic_id",
                "topic_label",
                "concept_size",
                "representative_phrases_str",
                "member_phrases_str",
            ]
            if c in concept_df.columns
        ]
        print("[Top 20 refined concept families]")
        print(concept_df[preview_cols].head(20).to_string(index=False))

        print()
        print("[Top 20 suspicious phrases]")
        suspicious = debug_df.sort_values(
            by=["best_fit_score", "duplicate_penalty", "best_label_overlap"],
            ascending=[True, False, True],
        ).head(20)
        print(suspicious.to_string(index=False))

    except Exception as e:
        print("[AI-REFINE] FAILED")
        print(str(e))
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

