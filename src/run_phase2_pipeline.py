from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.phrase_mining import extract_candidate_phrases, summarize_candidates
from src.preprocess import standardize_patent_dataframe, summarize_preprocessed_df
from src.support_scoring import compute_phrase_support, summarize_scored_candidates
from src.utils.config import get_paths, load_pipeline_config
from src.utils.io import choose_raw_file, print_df_overview, read_table, write_table


def step1_preprocess() -> Path:
    paths = get_paths()
    config = load_pipeline_config()

    raw_path = choose_raw_file(paths.data_raw, preferred_name=config.raw_file)
    print(f"[STEP 1] raw file selected: {raw_path.name}")

    raw_df = read_table(raw_path)
    print_df_overview(raw_df, "raw")

    pre_df, mapping = standardize_patent_dataframe(raw_df)

    print("[STEP 1] inferred columns:")
    print(f"  doc_id   = {mapping.doc_id}")
    print(f"  title    = {mapping.title}")
    print(f"  abstract = {mapping.abstract}")
    print(f"  claims   = {mapping.claims}")

    summarize_preprocessed_df(pre_df)

    out_path = paths.data_interim / "preprocessed_corpus.parquet"
    write_table(pre_df, out_path)
    print(f"[STEP 1] saved: {out_path}")

    debug_csv_path = paths.data_interim / "preprocessed_corpus.csv"
    write_table(pre_df, debug_csv_path)
    print(f"[STEP 1] saved: {debug_csv_path}")

    return out_path


def step2_phrase_mining(preprocessed_path: Path) -> tuple[Path, Path]:
    paths = get_paths()

    print(f"[STEP 2] loading preprocessed corpus: {preprocessed_path}")
    pre_df = read_table(preprocessed_path)
    print_df_overview(pre_df, "preprocessed_loaded")

    result = extract_candidate_phrases(
        corpus_df=pre_df,
        text_col="title_abstract",
        doc_id_col="doc_id",
        max_ngram=5,
    )

    summarize_candidates(result.candidates, top_n=40)

    cand_path = paths.data_interim / "candidate_phrases_raw.csv"
    doc_map_path = paths.data_interim / "doc_phrase_map.csv"

    write_table(result.candidates, cand_path)
    write_table(result.doc_phrase_map, doc_map_path)

    print(f"[STEP 2] saved candidates: {cand_path}")
    print(f"[STEP 2] saved doc-phrase map: {doc_map_path}")

    return cand_path, doc_map_path


def step3_support_scoring(
    preprocessed_path: Path,
    candidate_path: Path,
    doc_map_path: Path,
) -> Path:
    paths = get_paths()

    print(f"[STEP 3] loading preprocessed corpus: {preprocessed_path}")
    pre_df = read_table(preprocessed_path)

    print(f"[STEP 3] loading candidates: {candidate_path}")
    cand_df = read_table(candidate_path)

    print(f"[STEP 3] loading doc phrase map: {doc_map_path}")
    doc_map_df = read_table(doc_map_path)

    result = compute_phrase_support(
        corpus_df=pre_df,
        candidates_df=cand_df,
        doc_phrase_map_df=doc_map_df,
        doc_id_col="doc_id",
        title_col="title",
        abstract_col="abstract",
        claims_col="claims",
    )

    summarize_scored_candidates(result.scored_candidates, top_n=40)

    scored_path = paths.data_interim / "candidate_phrases_scored.csv"
    support_detail_path = paths.data_interim / "phrase_support_details.csv"

    write_table(result.scored_candidates, scored_path)
    write_table(result.support_details, support_detail_path)

    print(f"[STEP 3] saved scored candidates: {scored_path}")
    print(f"[STEP 3] saved support details: {support_detail_path}")

    return scored_path


def main() -> None:
    print("=== Phase-2 Pipeline ===")
    preprocessed_path = step1_preprocess()
    candidate_path, doc_map_path = step2_phrase_mining(preprocessed_path)
    step3_support_scoring(preprocessed_path, candidate_path, doc_map_path)
    print("=== Done: STEP 1-3 completed ===")


if __name__ == "__main__":
    main()