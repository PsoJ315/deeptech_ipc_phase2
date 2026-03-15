from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".parquet"}

HEADER_KEYWORDS = [
    "발명의명칭",
    "요약",
    "청구항",
    "ipc분류",
    "출원번호",
    "title",
    "abstract",
    "claims",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_raw_files(raw_dir: Path) -> list[Path]:
    files = [
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files)


def choose_raw_file(raw_dir: Path, preferred_name: str | None = None) -> Path:
    files = list_raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No raw data files found in: {raw_dir}")

    if preferred_name:
        preferred_path = raw_dir / preferred_name
        if preferred_path.exists():
            return preferred_path
        raise FileNotFoundError(f"Preferred raw file not found: {preferred_path}")

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to read CSV: {path}\nLast error: {last_error}")


def _score_header_row(row_values: list[str]) -> int:
    score = 0
    normalized = [str(v).strip().lower() for v in row_values if str(v).strip()]

    for cell in normalized:
        for kw in HEADER_KEYWORDS:
            if kw.lower() in cell:
                score += 1

    return score


def detect_header_row_csv(path: Path, preview_rows: int = 20) -> int:
    preview_df = _read_csv_with_fallback(path, header=None, nrows=preview_rows)

    best_idx = 0
    best_score = -1

    for i in range(len(preview_df)):
        row_values = preview_df.iloc[i].fillna("").astype(str).tolist()
        score = _score_header_row(row_values)

        # 열이 너무 적으면 헤더 가능성 낮음
        non_empty = sum(1 for v in row_values if str(v).strip())
        if non_empty >= 5:
            score += min(non_empty, 10)

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        header_row = detect_header_row_csv(path)
        print(f"[read_table] detected header row: {header_row}")

        df = _read_csv_with_fallback(path, header=header_row)

        # 완전 비어있는 열 제거
        df = df.dropna(axis=1, how="all").copy()

        # pandas가 이상한 unnamed 열을 만들었더라도 최대한 정리
        cleaned_columns = []
        for i, col in enumerate(df.columns):
            col_str = str(col).strip()
            if not col_str or col_str.lower().startswith("unnamed"):
                cleaned_columns.append(f"col_{i}")
            else:
                cleaned_columns.append(col_str)

        df.columns = cleaned_columns
        return df

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return

    raise ValueError(f"Unsupported output file type: {path}")


def print_df_overview(df: pd.DataFrame, name: str = "DataFrame") -> None:
    print(f"[{name}] shape={df.shape}")
    print(f"[{name}] columns={list(df.columns)}")