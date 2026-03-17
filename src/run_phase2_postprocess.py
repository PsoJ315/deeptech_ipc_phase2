from __future__ import annotations

from pathlib import Path

from step1_clean_phrases import run_step1_clean_phrases
from step2_canonicalize_phrases import run_step2_canonicalize_phrases
from step3_build_concepts import run_step3_build_concepts
from step35_bertopic_rebuild import run_step35_bertopic_rebuild

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "phase2_postprocess"

# =========================
# RUN SWITCHES
# =========================
RUN_STEP_1 = False
RUN_STEP_2 = False
RUN_STEP_3 = False
RUN_STEP_35 = True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== RUN PHASE 02 POSTPROCESS ===")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"data_dir    : {DATA_DIR}")
    print(f"output_dir  : {OUTPUT_DIR}")
    print(f"cwd         : {Path.cwd()}")

    if RUN_STEP_1:
        print("\n=== STEP 1: Clean Phrase Candidates ===")
        run_step1_clean_phrases(
            input_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            min_cleaned_score=3.5,
        )

    if RUN_STEP_2:
        print("\n=== STEP 2: Canonicalize Phrases ===")
        run_step2_canonicalize_phrases(
            output_dir=OUTPUT_DIR,
        )

    if RUN_STEP_3:
        print("\n=== STEP 3: Build Concept Families (Rule-based) ===")
        run_step3_build_concepts(
            input_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
        )

    if RUN_STEP_35:
        print("\n=== STEP 3.5: Rebuild Concept Families with BERTopic ===")
        run_step35_bertopic_rebuild(
            input_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
        )

    print("\n=== PHASE 02 POSTPROCESS DONE ===")


if __name__ == "__main__":
    main()