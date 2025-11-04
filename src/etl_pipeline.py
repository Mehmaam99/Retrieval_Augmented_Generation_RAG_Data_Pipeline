import os
import pandas as pd

"""Minimal ETL for news CSVs.

Steps:
- Read one or both raw CSV files from data/raw.
- Normalize and pick a single text column.
- Remove blanks/duplicates and save to data/cleaned/cleaned_news.csv.
"""

#ensure directory exists
def ensure_directory(path: str) -> None:
    """Create directory if it does not exist (no-op if it does)."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

#load raw data
def load_raw_data() -> pd.DataFrame:
    """Load available raw CSVs from data/raw and concatenate if multiple.

    Returns a DataFrame with all raw rows found, or raises if none.
    """
    raw_dir = os.path.join('data', 'raw')
    candidates = [
        os.path.join(raw_dir, 'news_summary.csv'),
        os.path.join(raw_dir, 'news_summary_more.csv'),
    ]

    dataframes = []
    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                df = pd.read_csv(candidate, encoding='latin-1')
                dataframes.append(df)
            except Exception as e:
                print(f"Failed to read {candidate}: {e}")

    if not dataframes:
        raise FileNotFoundError(
            "No raw CSV files found. Expected one of: data/raw/news_summary.csv, data/raw/news_summary_more.csv"
        )

    if len(dataframes) == 1:
        return dataframes[0]
    return pd.concat(dataframes, ignore_index=True)

#basic clean data
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns and extract a single 'text' column.

    - Lowercase/underscore column names
    - Choose a likely text column (fallback to first object column)
    - Trim/drop empty, deduplicate, cap length to avoid huge rows
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    possible_text_cols = [
        'text',
        'summary',
        'article',
        'content',
        'headline',
        'title',
        'short_text',
        'short_summary',
    ]

    existing = [c for c in possible_text_cols if c in df.columns]
    if not existing:
        object_cols = [c for c in df.columns if df[c].dtype == 'object']
        if not object_cols:
            raise ValueError("No textual columns found to clean.")
        existing = [object_cols[0]]

    cleaned = df[existing].copy()
    cleaned = cleaned.rename(columns={existing[0]: 'text'})
    cleaned['text'] = cleaned['text'].astype(str).str.strip()
    cleaned = cleaned.dropna(subset=['text'])
    cleaned = cleaned[cleaned['text'].str.len() > 0]
    cleaned = cleaned.drop_duplicates(subset=['text'])
    cleaned['text'] = cleaned['text'].str.slice(0, 5000)

    return cleaned.reset_index(drop=True)

#run etl pipeline
def run_etl() -> str:
    """Execute ETL and write cleaned CSV. Returns output path."""
    print("[ETL] Loading raw data...")
    df_raw = load_raw_data()
    print(f"[ETL] Loaded {len(df_raw)} raw rows.")

    print("[ETL] Cleaning data...")
    df_clean = basic_clean(df_raw)
    print(f"[ETL] Cleaned down to {len(df_clean)} rows.")

    cleaned_dir = os.path.join('data', 'cleaned')
    ensure_directory(cleaned_dir)
    output_path = os.path.join(cleaned_dir, 'cleaned_news.csv')

    print(f"[ETL] Writing cleaned data to {output_path}...")
    df_clean.to_csv(output_path, index=False, encoding='utf-8')
    print("[ETL] Done.")
    return output_path


if __name__ == '__main__':
    run_etl()