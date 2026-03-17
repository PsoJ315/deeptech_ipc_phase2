from __future__ import annotations

from pathlib import Path

from step1_clean_phrases import run_step1_clean_phrases
from step2_canonicalize_phrases import run_step2_canonicalize_phrases
from step3_build_concepts import run_step3_build_concepts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "phase2_postprocess"

START_STEP = 1
END_STEP = 3


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== RUN PHASE 02 POSTPROCESS ===")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"data_dir    : {DATA_DIR}")
    print(f"output_dir  : {OUTPUT_DIR}")
    print(f"cwd         : {Path.cwd()}")
    print(f"step range  : {START_STEP} ~ {END_STEP}")

    if START_STEP <= 1 <= END_STEP:
        print("\n=== STEP 1: Clean Phrase Candidates ===")
        run_step1_clean_phrases(
            input_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            min_cleaned_score=3.5,
        )

    if START_STEP <= 2 <= END_STEP:
        print("\n=== STEP 2: Canonicalize Phrases ===")
        run_step2_canonicalize_phrases(
            output_dir=OUTPUT_DIR,
        )

    if START_STEP <= 3 <= END_STEP:
        print("\n=== STEP 3: Build Concept Families ===")
        run_step3_build_concepts(
            input_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
        )

    if START_STEP <= 4 <= END_STEP:
        print("\n=== STEP 4: Make Review Sheet ===")
        raise NotImplementedError("STEP 4 is not implemented yet.")

    print("\n=== PHASE 02 POSTPROCESS DONE ===")


if __name__ == "__main__":
    main()