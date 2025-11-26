# src/data_loading.py
#
# Utilities for loading the True–False dataset from multiple topic CSV files.
#
# Expected directory structure:
#   ~/true-false-dataset/
#       topic1.csv
#       topic2.csv
#       ...
#
# Each CSV must contain:
#   - a statement-like column: one of ['statement','text','question','prompt']
#   - a label-like column:    one of ['label','answer','is_true','truth']
#
# Returns a DataFrame with columns:
#   - 'statement' (str)
#   - 'label'    (int: 1=true, 0=false)
#   - 'topic'    (filename stem)
#
# Used by: collect_activations.py, honesty_filter_tuned_improved_prompt.py, etc.

import os
import glob
import pandas as pd
from typing import Optional


# Directory where your topic-wise CSVs live
TRUE_FALSE_DIR = os.path.join(os.path.expanduser("~"), "true-false-dataset")


def _normalize_tf_frame(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Normalize a single topic CSV to have columns:
      - 'statement' (string)
      - 'label'    (int: 1=true, 0=false)
    Adds a 'topic' column from the filename (without .csv).
    """
    original_cols = df.columns.tolist()
    cols_lower = {c.lower(): c for c in original_cols}

    # --- Find statement-like column ---
    if "statement" in original_cols:
        stmt_col = "statement"
    elif "text" in cols_lower:
        stmt_col = cols_lower["text"]
    elif "question" in cols_lower:
        stmt_col = cols_lower["question"]
    elif "prompt" in cols_lower:
        stmt_col = cols_lower["prompt"]
    else:
        raise RuntimeError(
            f"Could not find a statement/text/question/prompt column in {filename}. "
            f"Columns were: {original_cols}"
        )

    # --- Find label-like column ---
    if "label" in original_cols:
        lab_col = "label"
    elif "answer" in cols_lower:
        lab_col = cols_lower["answer"]
    elif "is_true" in cols_lower:
        lab_col = cols_lower["is_true"]
    elif "truth" in cols_lower:
        lab_col = cols_lower["truth"]
    else:
        raise RuntimeError(
            f"Could not find a label/answer/is_true/truth column in {filename}. "
            f"Columns were: {original_cols}"
        )

    out = df[[stmt_col, lab_col]].copy()
    out = out.rename(columns={stmt_col: "statement", lab_col: "label"})

    # --- Convert label to 0/1 ---
    if out["label"].dtype == bool:
        out["label"] = out["label"].astype(int)
    elif out["label"].dtype == object:
        # Normalize string labels
        vals = out["label"].astype(str).str.strip().str.lower()
        uniq = set(vals.unique())
        if uniq <= {"true", "false"}:
            out["label"] = vals.map({"true": 1, "false": 0})
        elif uniq <= {"0", "1"}:
            out["label"] = vals.astype(int)
        else:
            raise RuntimeError(
                f"Unrecognized string label format in {filename}: {uniq}"
            )
    else:
        # Assume numeric 0/1
        out["label"] = out["label"].astype(int)

    out["label"] = out["label"].astype(int)

    # --- Attach topic from filename ---
    topic = os.path.basename(filename).rsplit(".", 1)[0]
    out["topic"] = topic

    return out


def load_true_false_dataset(
    max_examples: Optional[int] = None,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Load the True–False dataset from a directory of per-topic CSVs.

    Returns a DataFrame with columns:
      - 'statement'
      - 'label' (int: 1=true, 0=false)
      - 'topic'

    If max_examples is not None, randomly sample that many rows
    from the concatenated pool, using the given random seed.
    """
    if not os.path.isdir(TRUE_FALSE_DIR):
        raise FileNotFoundError(
            f"True–False dataset directory not found at {TRUE_FALSE_DIR}.\n"
            f"Create it or update TRUE_FALSE_DIR in data_loading.py."
        )

    # Grab all CSVs in that directory
    csv_paths = sorted(glob.glob(os.path.join(TRUE_FALSE_DIR, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {TRUE_FALSE_DIR}.\n"
            f"Make sure your topic CSVs (e.g., history.csv, geography.csv) "
            f"are inside that folder."
        )

    dfs = []
    for path in csv_paths:
        try:
            df_i = pd.read_csv(path)
            df_norm = _normalize_tf_frame(df_i, path)
            dfs.append(df_norm)
        except Exception as e:
            raise RuntimeError(f"Error while processing {path}: {e}")

    full = pd.concat(dfs, ignore_index=True)

    if max_examples is not None and len(full) > max_examples:
        full = full.sample(n=max_examples, random_state=seed).reset_index(drop=True)
    else:
        full = full.reset_index(drop=True)

    return full
