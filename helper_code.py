"""
Disneyland Reviews Complaint Patterns Analysis Pipeline (Parts 1-4)
BDCC Lab 1 Core Module

This script contains all modular functions, constants, data cleaning pipelines,
EDA functions, and vectorization logic for analyzing Disneyland reviews.
"""

import hashlib
import html
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

SEED = 42
DATA_PATH = Path("/mnt/data/public/disneyland-reviews/DisneylandReviews.csv")
PLACEHOLDERS = {"", "missing", "unknown", "n/a", "na", "null", "none"}
REQUIRED_COLUMNS = [
    "Review_ID",
    "Rating",
    "Year_Month",
    "Reviewer_Location",
    "Review_Text",
    "Branch",
]


# PART 1: DATA LOADING & VERIFICATION

def resolve_data_path(custom_path=None):
    """Resolve and verify existence of input dataset path."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
    if DATA_PATH.exists():
        return DATA_PATH
    local_p = Path("DisneylandReviews.csv")
    if local_p.exists():
        return local_p
    raise FileNotFoundError("DisneylandReviews.csv not found locally or at public path.")


def load_raw_data(data_path=None):
    """Load raw dataset with latin-1 encoding and string Review_ID."""
    path = resolve_data_path(data_path)
    raw = pd.read_csv(path, encoding="latin-1", dtype={"Review_ID": "string"})
    if not set(REQUIRED_COLUMNS).issubset(raw.columns):
        raise ValueError(f"Missing required columns. Found: {list(raw.columns)}")
    return raw, path


def compute_file_digest(file_path):
    """Compute SHA-256 hash digest of input CSV."""
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()


def inspect_raw_schema(raw_df):
    """Return dataset shape, head, and data types."""
    return {
        "shape": raw_df.shape,
        "head": raw_df.head(3),
        "dtypes": raw_df.dtypes.rename("data_type").to_frame(),
    }


# PART 2: CLEANING & QUALITY CHECKS

def prepare_text(value):
    """Preserve tone while removing markup and repairing common broken negations."""
    if pd.isna(value):
        return ""
    text = html.unescape(str(value)).replace("’", "'")
    text = re.sub(r"<[^>]+>|https?://\S+|www\.\S+", " ", text)
    # Fix broken negation forms common in dataset
    text = re.sub(
        r"\b(can|won|isn|aren|wasn|weren|don|doesn|didn|haven|hasn|hadn|"
        r"couldn|wouldn|shouldn|mustn)\s+t\b",
        r"\1't",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:dont|doesnt|didnt|isnt|arent|wasnt|werent|havent|hasnt|"
        r"hadnt|couldnt|wouldnt|shouldnt|mustnt)\b",
        lambda match: match.group()[:-1] + "'t",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


def clean_for_terms(text):
    """Return lowercase tokens while expanding and retaining negation words."""
    text = text.lower().replace("can't", "cannot").replace("won't", "will not")
    text = re.sub(r"n't\b", " not", text)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def assess_data_quality(raw_df):
    """Compute missing values and placeholder frequencies across required columns."""
    quality_rows = []
    for column in REQUIRED_COLUMNS:
        values = raw_df[column].astype("string").str.strip()
        missing = values.isna() | values.str.lower().isin(PLACEHOLDERS)
        quality_rows.append(
            {
                "column": column,
                "nulls": raw_df[column].isna().sum(),
                "missing_including_placeholders": missing.sum(),
                "missing_pct": round(100 * missing.mean(), 2),
            }
        )
    return pd.DataFrame(quality_rows)


def run_cleaning_pipeline(raw_df):
    """Deduplicate, clean placeholders, parse ratings/dates, and normalize text."""
    exact_duplicates = int(raw_df.duplicated().sum())
    
    # Drop exact duplicates and set row keys
    df = raw_df.drop_duplicates().copy()
    df.insert(0, "row_id", df.index)
    df = df.reset_index(drop=True)

    for col in REQUIRED_COLUMNS:
        if col != "Rating":
            values = df[col].astype("string").str.strip()
            df[col] = values.mask(values.str.lower().isin(PLACEHOLDERS))

    numeric_rating = pd.to_numeric(df["Rating"], errors="coerce")
    df["rating"] = numeric_rating.where(numeric_rating.isin([1, 2, 3, 4, 5]))
    df["review_month"] = pd.to_datetime(df["Year_Month"], format="%Y-%m", errors="coerce")
    
    for col in ["Branch", "Reviewer_Location"]:
        df[col] = df[col].fillna("Unknown")

    df["repeated_id"] = df["Review_ID"].notna() & df["Review_ID"].duplicated(keep=False)

    df["sentiment_text"] = df["Review_Text"].apply(prepare_text)
    df["clean_text"] = df["sentiment_text"].apply(clean_for_terms)
    df["word_count"] = df["clean_text"].str.split().str.len()
    df["text_eligible"] = df["clean_text"].str.contains(r"\w", regex=True)

    text_df = df.loc[df["text_eligible"]].copy().reset_index(drop=True)
    
    return df, text_df, exact_duplicates


# PART 3: EXPLORATORY DATA ANALYSIS (EDA)

def compute_branch_summary(dataframe):
    """Compute review count, rating metrics, date span, and market share by branch."""
    summary = dataframe.groupby("Branch").agg(
        reviews=("Review_ID", "count"),
        valid_ratings=("rating", "count"),
        mean_rating=("rating", "mean"),
        median_rating=("rating", "median"),
        first_month=("review_month", "min"),
        last_month=("review_month", "max"),
    )
    summary["share_pct"] = (100.0 * summary["reviews"] / len(dataframe)).round(2)
    summary["mean_rating"] = summary["mean_rating"].round(2)
    return summary


def compute_rating_distribution(dataframe):
    """Compute percentage distribution of ratings (1-5) per branch."""
    crosstab = pd.crosstab(dataframe["Branch"], dataframe["rating"], normalize="index") * 100.0
    return crosstab.round(1)


def compute_location_distribution(dataframe, top_n=10):
    """Compute cross-tabulation of top reviewer locations across branches."""
    top_locs = dataframe["Reviewer_Location"].value_counts().head(top_n).index
    filtered = dataframe[dataframe["Reviewer_Location"].isin(top_locs)]
    return pd.crosstab(filtered["Reviewer_Location"], filtered["Branch"])


def plot_eda_charts(df, rating_dist):
    """Generate dual EDA plot: Rating Distribution and Monthly Volume Over Time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    rating_dist.plot(kind="bar", stacked=False, ax=axes[0], colormap="viridis")
    axes[0].set_title("Rating Distribution by Branch (%)")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_xlabel("Branch")
    axes[0].tick_params(axis="x", rotation=15)

    monthly_trends = (
        df.dropna(subset=["review_month"])
        .groupby(["review_month", "Branch"])
        .size()
        .unstack(fill_value=0)
    )
    monthly_trends.plot(ax=axes[1], linewidth=1.5)
    axes[1].set_title("Monthly Review Volume Over Time")
    axes[1].set_ylabel("Number of Reviews")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    return fig


# PART 4: VOCABULARY & TEXT REPRESENTATION

def calculate_sparsity(matrix):
    """Calculate percentage of zero elements in a sparse matrix."""
    non_zero = matrix.nnz
    total_elements = matrix.shape[0] * matrix.shape[1]
    return 100.0 * (1.0 - (non_zero / total_elements))


def extract_top_tfidf_terms(tfidf_matrix, feature_names, doc_idx=0, top_n=10):
    """Extract top N TF-IDF terms for a single document index."""
    row = tfidf_matrix[doc_idx].toarray().flatten()
    top_indices = row.argsort()[::-1][:top_n]
    top_terms = [(feature_names[i], row[i]) for i in top_indices if row[i] > 0]
    return pd.DataFrame(top_terms, columns=["term", "tfidf_score"])


def compute_corpus_top_terms(tfidf_matrix, feature_names, top_n=15):
    """Compute top terms across the full corpus sorted by mean TF-IDF score."""
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[::-1][:top_n]
    return pd.DataFrame(
        {"term": feature_names[top_indices], "mean_tfidf": mean_scores[top_indices]}
    )


def build_text_representations(text_df, max_features=5000):
    """Build Bag-of-Words and TF-IDF matrices using CountVectorizer and TfidfVectorizer."""
    count_vec = CountVectorizer(max_features=max_features, stop_words="english")
    tfidf_vec = TfidfVectorizer(max_features=max_features, stop_words="english")

    bow_matrix = count_vec.fit_transform(text_df["clean_text"])
    tfidf_matrix = tfidf_vec.fit_transform(text_df["clean_text"])

    feature_names = tfidf_vec.get_feature_names_out()
    sparsity = calculate_sparsity(tfidf_matrix)

    return {
        "count_vec": count_vec,
        "tfidf_vec": tfidf_vec,
        "bow_matrix": bow_matrix,
        "tfidf_matrix": tfidf_matrix,
        "feature_names": feature_names,
        "sparsity": sparsity,
    }
