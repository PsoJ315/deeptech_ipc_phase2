from pathlib import Path
import pandas as pd


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).strip().split())


def classify_review_status(row):
    topic_label = normalize_spaces(row["topic_label"])
    n_phrases = int(row["n_phrases"])
    sum_doc_count = float(row["sum_doc_count"])
    generic_ratio = float(row["generic_ratio"])
    label_genericity = normalize_spaces(row["label_genericity"])
    phrases = normalize_spaces(row["phrases"])

    label_tokens = topic_label.split()
    is_single_token_label = len(label_tokens) == 1

    # 1. 대형 혼합군
    if n_phrases >= 20:
        return "MIXED_LARGE", "대형 residual cluster 가능성 높음. Phase 2.5 재분해 후보.", 1

    # 2. alias-heavy 패밀리
    if generic_ratio >= 0.45 and label_genericity == "specific":
        return "ALIAS_HEAVY", "동일 계열 alias가 많이 섞인 패밀리. split보다 표시 정제 권장.", 2

    # 3. 너무 넓은 단일어
    if n_phrases == 1 and is_single_token_label and sum_doc_count >= 5:
        return "TOO_BROAD_SINGLE", "단일 broad term. 상위 seed로는 유효하지만 독립 concept 해석은 주의.", 2

    # 4. 저신뢰 소토픽
    if n_phrases <= 2 and sum_doc_count <= 3:
        return "LOW_CONFIDENCE", "빈도와 구성 모두 작아 후속 검토 필요.", 3

    # 5. maybe_generic 단일어
    if label_genericity == "maybe_generic" and is_single_token_label:
        return "TOO_BROAD_SINGLE", "대표어가 다소 넓음. 하위 문맥 확인 필요.", 2

    return "OK", "현 단계에서 바로 사용 가능한 1차 concept family.", 4


project_root = Path(r"C:\Users\psoj3\OneDrive\문서\ASC\ICP\deeptech_ipc_phase2")
summary_path = project_root / "data" / "phase2_postprocess" / "topic_summary_ai_refined.csv"
out_path = project_root / "data" / "phase2_postprocess" / "topic_summary_ai_refined_review.csv"

df = pd.read_csv(summary_path)

review = df.apply(classify_review_status, axis=1, result_type="expand")
review.columns = ["review_status", "review_note", "priority"]

df = pd.concat([df, review], axis=1)
df = df.sort_values(["priority", "sum_doc_count", "n_phrases"], ascending=[True, False, False]).reset_index(drop=True)

df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[SAVED] {out_path}")
print(df.head(20).to_string(index=False))