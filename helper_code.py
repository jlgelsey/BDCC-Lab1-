"""
Disneyland Reviews Complaint Patterns Analysis Pipeline (Parts 1-10)
BDCC Lab 1 Core Module

Parts 1-4 retain the group helper's interfaces, while Parts 5-10 convert
the checked Sections5-10 notebook into functions with explicit inputs.
Apriori and single-consequent rules follow the supplied class notebook.
Phrase rules are editable screening rules, not trained sentiment models.

Install pandas, numpy, matplotlib, scikit-learn and vaderSentiment first.
Importing this module does not read the dataset or run the analysis.

Example
-------
>>> raw, path = load_raw_data("DisneylandReviews.csv")
>>> cleaned, text_df, duplicate_count = run_cleaning_pipeline(raw)
>>> vectors = build_text_representations(text_df)
>>> results = run_parts_5_to_10(text_df)
>>> checks = validate_analysis(text_df, results, vectors)
>>> results["complaint_summary"].head(5)

Call the individual section functions when working one section at a time.
Use export_analysis(results, input_path=path) only when ready to save.
Run test.py for independent TF-IDF and mining checks, or test.py --full
to also check the complete dataset. Language accuracy still needs an audit.
"""

import hashlib
import html
import json
from itertools import combinations
from math import ceil
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    TfidfTransformer,
)

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
    """Find the CSV, using an explicit path or the hosted/local defaults.
    An invalid explicit path raises an error instead of loading another file.

    Parameters
    ----------
    custom_path : str or Path, optional
        Explicit CSV path; omitted means try the hosted and local defaults.

    Returns
    -------
    Path
        Existing CSV path.
    """
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Dataset not found: {p}")
    if DATA_PATH.is_file():
        return DATA_PATH
    local_p = Path("DisneylandReviews.csv")
    if local_p.is_file():
        return local_p
    raise FileNotFoundError(
        "DisneylandReviews.csv not found locally or at public path."
    )


def load_raw_data(data_path=None):
    """Read the Latin-1 CSV and check that the six required columns exist.
    Review identifiers stay strings so leading zeros are preserved.

    Parameters
    ----------
    data_path : str or Path, optional
        CSV location forwarded to resolve_data_path.

    Returns
    -------
    tuple
        (raw DataFrame, resolved Path); no rows are cleaned yet.
    """
    path = resolve_data_path(data_path)
    raw = pd.read_csv(path, encoding="latin-1", dtype={"Review_ID": "string"})
    if not set(REQUIRED_COLUMNS).issubset(raw.columns):
        raise ValueError(
            f"Missing required columns. Found: {list(raw.columns)}"
        )
    return raw, path


def compute_file_digest(file_path):
    """Hash the file bytes with SHA-256 to identify the input used in a run.

    Parameters
    ----------
    file_path : str or Path
        Existing file whose bytes will be hashed.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()


def inspect_raw_schema(raw_df):
    """Collect basic schema information without changing the input table.

    Parameters
    ----------
    raw_df : DataFrame
        Raw reviews with the six columns in REQUIRED_COLUMNS.

    Returns
    -------
    dict
        Shape, first five rows under head, and a data_type table under
        dtypes.
    """
    return {
        "shape": raw_df.shape,
        "head": raw_df.head(5),
        "dtypes": raw_df.dtypes.rename("data_type").to_frame(),
    }


# PART 2: CLEANING & QUALITY CHECKS


def prepare_text(value):
    """Remove markup and repair common broken negations while retaining
    capitalization and punctuation for sentiment analysis.

    Parameters
    ----------
    value : object
        One review value, which may be missing.

    Returns
    -------
    str
        Prepared text, or an empty string for a missing value.
    """
    if pd.isna(value):
        return ""
    text = html.unescape(str(value)).replace("’", "'")
    text = re.sub(r"<[^>]+>|https?://\S+|www\.\S+", " ", text)
    # fix broken negation forms common in dataset
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
    """Lowercase text, expand negations, and remove punctuation for tokens.
    The original prepared text remains available for sentiment analysis.

    Parameters
    ----------
    text : str
        Prepared review text, with case and punctuation where needed.

    Returns
    -------
    str
        Whitespace-normalized text for vectorization.
    """
    text = text.lower().replace("can't", "cannot").replace("won't", "will not")
    text = re.sub(r"n't\b", " not", text)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def assess_data_quality(raw_df):
    """Count nulls and the placeholder words in each required column.
    Percentages use all input rows; empty inputs have undefined percentages.

    Parameters
    ----------
    raw_df : DataFrame
        Raw reviews with the six columns in REQUIRED_COLUMNS.

    Returns
    -------
    DataFrame
        Column names, null counts, placeholder-inclusive counts and
        missing_pct.
    """
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
    """Remove exact duplicates and assign original row positions as unique
    keys. Parse ratings and dates, replace placeholders, and prepare separate
    text versions for sentiment and vectorization. Repeated IDs are flagged,
    not silently discarded; input data is copied rather than modified.

    Parameters
    ----------
    raw_df : DataFrame
        Raw reviews with the six columns in REQUIRED_COLUMNS.

    Returns
    -------
    tuple
        (all cleaned rows, usable text rows, exact duplicate count).
    """
    exact_duplicates = int(raw_df.duplicated().sum())

    # drop exact duplicates and set row keys
    df = raw_df.drop_duplicates().copy()
    df.insert(0, "row_id", np.flatnonzero(~raw_df.duplicated()))
    df = df.reset_index(drop=True)

    for col in REQUIRED_COLUMNS:
        if col != "Rating":
            values = df[col].astype("string").str.strip()
            df[col] = values.mask(values.str.lower().isin(PLACEHOLDERS))

    numeric_rating = pd.to_numeric(df["Rating"], errors="coerce")
    df["rating"] = numeric_rating.where(numeric_rating.isin([1, 2, 3, 4, 5]))
    df["review_month"] = pd.to_datetime(
        df["Year_Month"], format="%Y-%m", errors="coerce"
    )

    for col in ["Branch", "Reviewer_Location"]:
        df[col] = df[col].fillna("Unknown")

    df["repeated_id"] = df["Review_ID"].notna() & df["Review_ID"].duplicated(
        keep=False
    )

    df["sentiment_text"] = df["Review_Text"].apply(prepare_text)
    df["clean_text"] = df["sentiment_text"].apply(clean_for_terms)
    df["word_count"] = df["clean_text"].str.split().str.len()
    df["text_eligible"] = df["clean_text"].str.contains(r"\w", regex=True)

    text_df = df.loc[df["text_eligible"]].copy().reset_index(drop=True)

    return df, text_df, exact_duplicates


# PART 3: EXPLORATORY DATA ANALYSIS (EDA)


def compute_branch_summary(dataframe):
    """Aggregate every review row by branch, including rows without an ID.
    Rating summaries use valid ratings; share_pct describes this dataset,
    not the market share of a park.

    Parameters
    ----------
    dataframe : DataFrame
        Cleaned review table returned by run_cleaning_pipeline.

    Returns
    -------
    DataFrame
        Branch-indexed counts, ratings, date range and dataset share
        percentage.
    """
    summary = dataframe.groupby("Branch").agg(
        reviews=("Review_ID", "size"),
        valid_ratings=("rating", "count"),
        mean_rating=("rating", "mean"),
        median_rating=("rating", "median"),
        first_month=("review_month", "min"),
        last_month=("review_month", "max"),
    )
    summary["share_pct"] = (100.0 * summary["reviews"] / len(dataframe)).round(
        2
    )
    summary["mean_rating"] = summary["mean_rating"].round(2)
    return summary


def compute_rating_distribution(dataframe):
    """Calculate within-branch percentages using only nonmissing ratings.

    Parameters
    ----------
    dataframe : DataFrame
        Cleaned review table returned by run_cleaning_pipeline.

    Returns
    -------
    DataFrame
        Branch-by-rating percentages; absent rating columns may be omitted.
    """
    crosstab = (
        pd.crosstab(
            dataframe["Branch"], dataframe["rating"], normalize="index"
        )
        * 100.0
    )
    return crosstab.round(1)


def compute_location_distribution(dataframe, top_n=10):
    """Select the most frequent reviewer locations and count their branches.

    Parameters
    ----------
    dataframe : DataFrame
        Cleaned review table returned by run_cleaning_pipeline.
    top_n : int, optional
        Maximum number of ranked rows to return; must be positive.

    Returns
    -------
    DataFrame
        Location-by-branch counts for the selected locations.
    """
    top_locs = dataframe["Reviewer_Location"].value_counts().head(top_n).index
    filtered = dataframe[dataframe["Reviewer_Location"].isin(top_locs)]
    return pd.crosstab(filtered["Reviewer_Location"], filtered["Branch"])


def plot_eda_charts(df, rating_dist):
    """Plot rating percentages and monthly review counts on two axes.

    Parameters
    ----------
    df : DataFrame
        Cleaned reviews with Branch and review_month columns.
    rating_dist : DataFrame
        Branch-by-rating percentage table from compute_rating_distribution.

    Returns
    -------
    Figure
        Matplotlib figure; the caller decides when to display or save it.
    """
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
    """Use stored nonzero counts to calculate the percentage of zero cells.
    No dense copy is made; a matrix with no cells returns NaN.

    Parameters
    ----------
    matrix : sparse matrix
        Numeric matrix with shape and nnz attributes.

    Returns
    -------
    float
        Percentage from 0 to 100, or NaN for an empty shape.
    """
    non_zero = matrix.nnz
    total_elements = matrix.shape[0] * matrix.shape[1]
    if not total_elements:
        return np.nan
    return 100.0 * (1.0 - (non_zero / total_elements))


def extract_top_tfidf_terms(tfidf_matrix, feature_names, doc_idx=0, top_n=10):
    """Sort one review by TF-IDF and retain only positive-weight terms.

    Parameters
    ----------
    tfidf_matrix : sparse matrix
        TF-IDF rows for reviews and columns for the supplied vocabulary.
    feature_names : sequence of str
        Vocabulary in exactly the matrix column order.
    doc_idx : int, optional
        Zero-based review position in the TF-IDF matrix.
    top_n : int, optional
        Maximum number of ranked rows to return; must be positive.

    Returns
    -------
    DataFrame
        Up to top_n rows with term and tfidf_score columns.
    """
    row = tfidf_matrix[doc_idx].toarray().flatten()
    top_indices = row.argsort()[::-1][:top_n]
    top_terms = [(feature_names[i], row[i]) for i in top_indices if row[i] > 0]
    return pd.DataFrame(top_terms, columns=["term", "tfidf_score"])


def compute_corpus_top_terms(tfidf_matrix, feature_names, top_n=15):
    """Average each term over all reviews, then sort by its mean TF-IDF.

    Parameters
    ----------
    tfidf_matrix : sparse matrix
        TF-IDF rows for reviews and columns for the supplied vocabulary.
    feature_names : sequence of str
        Vocabulary in exactly the matrix column order.
    top_n : int, optional
        Maximum number of ranked rows to return; must be positive.

    Returns
    -------
    DataFrame
        Up to top_n rows with term and mean_tfidf columns.
    """
    feature_names = np.asarray(feature_names)
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[::-1][:top_n]
    return pd.DataFrame(
        {
            "term": feature_names[top_indices],
            "mean_tfidf": mean_scores[top_indices],
        }
    )


def build_text_representations(text_df, max_features=5000):
    """Fit count and TF-IDF vectorizers with the same 5,000-term default
    and English stopwords used in Parts 1-4. TF-IDF uses smoothed IDF and
    L2 normalization. Keep sentiment_text for negation-sensitive analysis
    because the English stopword list removes some negation words. Empty
    text or a corpus without usable tokens raises ValueError.

    Parameters
    ----------
    text_df : DataFrame
        Usable-text table returned by run_cleaning_pipeline.
    max_features : int, optional
        Maximum vocabulary size for both vectorizers.

    Returns
    -------
    dict
        count_vec, tfidf_vec, bow_matrix, tfidf_matrix, feature_names and
        sparsity.
    """
    if text_df.empty:
        raise ValueError("No usable reviews are available for vectorization.")
    count_vec = CountVectorizer(
        max_features=max_features, stop_words="english"
    )
    tfidf_vec = TfidfVectorizer(
        max_features=max_features, stop_words="english"
    )

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


# PART 5: MENTIONS, COMPLAINTS AND CONTEXTUAL SENTIMENT


# store named attractions so generic words don't identify them

ATTRACTIONS = {
    "Space Mountain": ["space mountain", "hyperspace mountain"],
    "It's a Small World": [
        "it's a small world",
        "its a small world",
        "small world",
    ],
    "Pirates of the Caribbean": ["pirates of the caribbean"],
    "Big Thunder Mountain": ["big thunder mountain", "thunder mountain"],
    "Buzz Lightyear": ["buzz lightyear"],
    "Haunted Mansion": ["haunted mansion"],
    "Phantom Manor": ["phantom manor"],
    "Jungle Cruise": ["jungle cruise", "jungle river cruise"],
    "Iron Man Experience": ["iron man experience", "ironman experience"],
    "Mystic Manor": ["mystic manor"],
    "Ratatouille": ["ratatouille"],
    "Dumbo": ["dumbo"],
    "Railroad": ["disneyland railroad", "disney railroad"],
}

# store service categories

SERVICES = {
    "Staff / customer service": [
        "staff",
        "cast member",
        "cast members",
        "customer service",
        "employees",
    ],
    "Food / dining": [
        "food",
        "restaurant",
        "restaurants",
        "food counter",
        "food counters",
        "meal",
        "meals",
        "dining",
    ],
    "Tickets / admission": ["ticket", "tickets", "admission", "entry fee"],
    "Queue / fast pass": [
        "queue",
        "queues",
        "waiting time",
        "wait time",
        "fastpass",
        "fast pass",
    ],
    "Cleanliness / toilets": [
        "toilet",
        "toilets",
        "restroom",
        "restrooms",
        "bathroom",
        "bathrooms",
        "cleanliness",
    ],
    "Shows / entertainment": ["parade", "parades", "fireworks", "shows"],
    "Transport / parking": ["parking", "shuttle", "shuttles", "transport"],
}

SERVICES["Hotel / accommodation"] = ["hotel", "hotels", "room", "rooms"]
SERVICES["Transport / access"] = [
    "shuttle",
    "bus",
    "buses",
    "mtr",
    "train",
    "parking",
]


def compile_aliases(dictionary):
    """Escape alias text and compile case-insensitive whole-word patterns.
    Longer aliases are tried first to retain the more specific phrase.

    Parameters
    ----------
    dictionary : dict
        Mapping from labels to nonempty lists of literal text aliases.

    Returns
    -------
    dict
        One compiled regex per label.
    """
    patterns = {}
    for label, aliases in dictionary.items():
        alternatives = [
            re.escape(alias)
            for alias in sorted(aliases, key=len, reverse=True)
        ]
        patterns[label] = re.compile(
            r"\b(?:" + "|".join(alternatives) + r")\b", re.I
        )
    return patterns


def find_labels(text, patterns):
    """Search each compiled alias pattern once in the supplied segment.

    Parameters
    ----------
    text : str
        Prepared review text, with case and punctuation where needed.
    patterns : dict or list
        Compiled label patterns, or a list of target patterns with cue
        groups.

    Returns
    -------
    list of str
        Matched labels in dictionary order, without repeated labels.
    """
    return [
        label for label, pattern in patterns.items() if pattern.search(text)
    ]


def nearby_patterns(topic, cue):
    """Connect a topic and problem cue in either order through at most four
    allowed linking words. The patterns cannot jump across another aspect
    or a comma, although these rules cannot understand every sentence.

    Parameters
    ----------
    topic : str
        Regex alternatives for the attraction or service words.
    cue : str
        Regex alternatives for explicit problem words.

    Returns
    -------
    list
        Two compiled regex patterns, each with a named cue group.
    """
    links = (
        r"is|are|was|were|be|been|being|very|quite|really|so|too|"
        r"all|the|a|bit|generally|extremely|absolutely|which|that|"
        r"either|still|often|sometimes|not|never|only|"
        r"at|in|of|quality|attitude"
    )
    # restrict the gap so it cannot jump across another aspect or a comma.
    gap = r"(?:\s+(?:" + links + r")){0,4}\s+"
    first = r"\b(?:" + topic + r")\b" + gap + r"(?P<cue>" + cue + r")\b"
    second = r"\b(?P<cue>" + cue + r")\b" + gap + r"(?:" + topic + r")\b"
    return [re.compile(first, re.I), re.compile(second, re.I)]


attraction_patterns = compile_aliases(ATTRACTIONS)
service_patterns = compile_aliases(SERVICES)
ride_topics = r"rides?|attractions?|roller\s*coasters?|" + "|".join(
    re.escape(alias) for aliases in ATTRACTIONS.values() for alias in aliases
)
# each category requires an explicit cue, rather than general negativity.
COMPLAINT_RULES = {
    "Long queues": nearby_patterns(
        r"queues?|lines?|waits?|waiting",
        r"long|lengthy|slow|huge|massive|horrendous|awful",
    ),
    "Closures / breakdowns": nearby_patterns(
        ride_topics,
        r"closed|broken\s+down|broke\s+down|not\s+working|"
        r"out\s+of\s+service|under\s+maintenance|shut\s+down",
    ),
    "Poor staff service": nearby_patterns(
        r"staff|cast\s+members?|employees?|service",
        r"rude|unhelpful|unfriendly|impolite|poor|terrible|awful",
    ),
    "High prices / poor value": [
        re.compile(
            r"\b(?P<cue>too\s+expensive|very\s+expensive|overpriced|"
            r"poor\s+value|"
            r"waste\s+of\s+money|not\s+worth\s+(?:the\s+)?"
            r"(?:money|price|cost))\b",
            re.I,
        )
    ],
    "Food quality / dining": nearby_patterns(
        r"food|meals?|restaurants?|lunch|dinner|burgers?|pizza",
        r"cold|stale|undercooked|poor|bad|terrible|awful|disgusting",
    ),
    "Crowding": [
        re.compile(
            r"\b(?P<cue>overcrowded|too\s+crowded|very\s+crowded|"
            r"extremely\s+crowded|uncomfortably\s+crowded)\b",
            re.I,
        )
    ],
    "Cleanliness": nearby_patterns(
        r"park|toilets?|restrooms?|bathrooms?|tables?|floors?|restaurant",
        r"dirty|filthy|unclean|disgusting",
    ),
}
COMPLAINT_RULES["Crowding"] += nearby_patterns(
    r"park|parks|crowds?|areas?", r"packed|heaving|rammed|huge"
)
COMPLAINT_RULES["Hotel problems"] = nearby_patterns(
    r"hotels?|rooms?", r"dirty|filthy|noisy|poor|terrible|awful"
)
COMPLAINT_RULES["Transport problems"] = nearby_patterns(
    r"shuttles?|buses?|trains?|mtr|parking",
    r"delayed|unreliable|overcrowded|poor|terrible|awful",
)
PRIMARY_CATEGORIES = sorted(COMPLAINT_RULES)


def negated_cue(text, start):
    """Inspect nearby preceding words for denial, while treating not only as
    emphasis. This is a limited phrase rule, not a grammatical parser.

    Parameters
    ----------
    text : str
        Prepared review text, with case and punctuation where needed.
    start : int
        Character offset at which the matched cue begins.

    Returns
    -------
    bool
        True when the nearby context appears to deny the matched cue.
    """
    prefix = text[:start].lower()
    prefix = re.split(r"[.!?;:]|\b(?:but|however|although|yet)\b", prefix)[-1]
    if re.search(r"\bnot\s+that\b", prefix):
        return True
    denial = re.search(
        r"\b(?:don't|doesn't|didn't|do\s+not|does\s+not)\s+have\s+to\b|"
        r"\bno\s+need\s+to\b",
        prefix[-100:],
    )
    if denial:
        return True
    if re.search(
        r"\b(?:didn't|don't|doesn't|not|never)\b[^.!?;]{0,70}"
        r"\b(?:any\s+)?more\s+$",
        prefix,
    ):
        return True
    prefix = re.sub(r"\bnot\s+only\b", "", prefix)
    words = re.findall(r"[\w']+", prefix)[-4:]
    negatives = {
        "no",
        "not",
        "never",
        "without",
        "hardly",
        "cannot",
        "wasn't",
        "weren't",
        "isn't",
        "aren't",
        "don't",
        "didn't",
    }
    return any(word in negatives for word in words)


def detect_complaints(segment):
    """Match explicit problem phrases and reject nearby denial, hypothetical
    statements and scheduled closing times. Return at most one match per
    category. Dining co-mentions are marked primary=False and excluded from
    primary mining to avoid creating a category from another category.

    Parameters
    ----------
    segment : str
        One prepared sentence or contrast clause to inspect.

    Returns
    -------
    list of dict
        category, evidence, cue and primary for each accepted phrase.
    """
    hits = []
    for category, patterns in COMPLAINT_RULES.items():
        for pattern in patterns:
            accepted = None
            for match in pattern.finditer(segment):
                # advice elsewhere must not erase a factual complaint.
                local_prefix = re.split(r"[,;.!?]", segment[: match.start()])[
                    -1
                ]
                if re.search(
                    r"^\s*(?:if|unless)\b|\b(?:can|could|would)\s+"
                    r"(?:be|get|see)\b|\bwould\s+(?:probably\s+)?say\b",
                    local_prefix,
                    flags=re.I,
                ):
                    continue
                if negated_cue(segment, match.start("cue")):
                    continue
                # scheduled closing times are not sufficient closure evidence.
                cue_end = match.end("cue")
                tail = segment[cue_end:]
                if category == "Closures / breakdowns" and re.match(
                    r"\s+at\s+\d", tail, flags=re.I
                ):
                    continue
                accepted = match
                break
            if accepted:
                hits.append(
                    {
                        "category": category,
                        "evidence": accepted.group(),
                        "cue": accepted.group("cue"),
                        "primary": True,
                    }
                )
                break
    dining = service_patterns["Food / dining"].search(segment)
    if dining and any(
        hit["category"] in {"Long queues", "Crowding"} for hit in hits
    ):
        hits.append(
            {
                "category": "Dining with queue/crowd mentions",
                "evidence": segment,
                "cue": "dining with queue/crowding cue",
                "primary": False,
            }
        )
    return hits


def split_segments(text):
    """Split at sentence endings and contrast words while retaining the
    case and punctuation that help VADER interpret the surrounding text.

    Parameters
    ----------
    text : str
        Prepared review text, with case and punctuation where needed.

    Returns
    -------
    list of str
        Nonempty sentence or contrast-clause segments.
    """
    pieces = re.split(
        r"(?<=[.!?;])\s+|(?<=[.!?;])(?=[A-Z])|"
        r"\b(?:but|however|although|yet)\b",
        text,
        flags=re.I,
    )
    return [piece.strip(" ,;") for piece in pieces if piece.strip(" ,;")]


def sentiment_label(score):
    """Label a VADER compound score: positive at 0.05 or above, negative
    at -0.05 or below, and neutral between those limits.

    Parameters
    ----------
    score : float
        VADER compound score between -1 and 1.

    Returns
    -------
    str
        positive, negative or neutral; this labels context, not target
        truth.
    """
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


TARGET_RULES = {
    "Dining:queue": nearby_patterns(
        r"food\s+(?:queues?|lines?)|restaurant\s+(?:queues?|lines?)",
        r"long|slow|huge|horrendous",
    ),
    "Dining:crowding": nearby_patterns(
        r"food\s+counters?|restaurants?|dining\s+areas?",
        r"overcrowded|too\s+crowded|packed",
    ),
    "Tickets:price": nearby_patterns(
        r"tickets?|admission|entry", r"overpriced|too\s+expensive"
    ),
    "Food:price": nearby_patterns(
        r"food|meals?|restaurants?", r"overpriced|too\s+expensive"
    ),
    "Hotel:price": nearby_patterns(
        r"hotels?|rooms?", r"overpriced|too\s+expensive"
    ),
    "Merchandise:price": nearby_patterns(
        r"souvenirs?|merchandise|gifts?", r"overpriced|too\s+expensive"
    ),
}
for attraction, aliases in ATTRACTIONS.items():
    topic = "|".join(re.escape(alias) for alias in aliases)
    TARGET_RULES[attraction + ":closure"] = nearby_patterns(
        topic, r"closed|broken\s+down|not\s+working|under\s+maintenance"
    )


def find_target_match(segment, patterns, target):
    """Search all occurrences for a direct target-problem phrase, skipping
    negation, hypothetical statements and scheduled attraction closures.

    Parameters
    ----------
    segment : str
        One prepared sentence or contrast clause to inspect.
    patterns : dict or list
        Compiled label patterns, or a list of target patterns with cue
        groups.
    target : str
        Target label; labels ending in :closure receive a schedule check.

    Returns
    -------
    re.Match or None
        First acceptable match, or None when no phrase qualifies.
    """
    for pattern in patterns:
        for match in pattern.finditer(segment):
            if negated_cue(segment, match.start("cue")):
                continue
            match_start = match.start()
            prefix = re.split(r"[,;.!?]", segment[:match_start])[-1]
            hypothetical = re.search(
                r"^\s*(?:if|unless)\b|\b(?:can|could|would)\s+"
                r"(?:be|get|see)\b|\bwould\s+(?:probably\s+)?say\b",
                prefix,
                flags=re.I,
            )
            cue_end = match.end("cue")
            scheduled = target.endswith(":closure") and re.match(
                r"\s+at\s+\d", segment[cue_end:], flags=re.I
            )
            if not hypothetical and not scheduled:
                return match
    return None


MIN_SUPPORT = 0.01
MIN_COUNT = 30
MIN_CONFIDENCE = 0.30
MIN_LIFT = 1.20


def extract_review_evidence(text_df, analyzer=None):
    # keep the matched words so we can check why a label was assigned
    """Find phrase evidence and score its surrounding clauses. Each review has
    equal weight in an aspect average, even when it mentions that aspect
    several times. Complaint percentages include all usable branch reviews.
    A VADER label describes context, not the sentiment of a specific target.
    Candidate attraction links and audit sheets still need human review.

    Parameters
    ----------
    text_df : DataFrame
        Usable-text table returned by run_cleaning_pipeline.
    analyzer : object, optional
        Object with polarity_scores(text); defaults to vaderSentiment
        VADER.

    Returns
    -------
    dict
        Tables and settings keyed by: mentions, complaints,
        primary_complaints, aspect_reviews, aspect_summary,
        complaint_summary, extraction_audit, unmatched_audit.
    """
    required = {
        "row_id",
        "Review_ID",
        "Branch",
        "sentiment_text",
        "Review_Text",
    }
    if not required.issubset(text_df.columns):
        raise ValueError("Use text_df returned by run_cleaning_pipeline.")
    if text_df["row_id"].isna().any() or not text_df["row_id"].is_unique:
        raise ValueError("Each review needs a unique, nonmissing row_id.")
    if analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
    mention_rows = []
    complaint_rows = []
    for position, row in enumerate(text_df.itertuples(index=False), start=1):
        for segment_id, segment in enumerate(
            split_segments(row.sentiment_text)
        ):
            attractions = find_labels(segment, attraction_patterns)
            services = find_labels(segment, service_patterns)
            hits = detect_complaints(segment)
            if not attractions and (not services) and (not hits):
                continue
            score = analyzer.polarity_scores(segment)["compound"]
            common = {
                "row_id": row.row_id,
                "Review_ID": row.Review_ID,
                "Branch": row.Branch,
                "segment_id": segment_id,
                "segment": segment,
                "compound": score,
                "context_sentiment": sentiment_label(score),
            }
            for kind, labels in [
                ("attraction", attractions),
                ("service", services),
            ]:
                for aspect in labels:
                    mention_rows.append(
                        {
                            **common,
                            "aspect_type": kind,
                            "aspect": aspect,
                            "multiple_aspects": len(attractions)
                            + len(services)
                            > 1,
                        }
                    )
            for hit in hits:
                complaint_rows.append(
                    {
                        **common,
                        **hit,
                        "candidate_attraction": (
                            attractions[0] if len(attractions) == 1 else None
                        ),
                        "link_status": (
                            "candidate_same_segment"
                            if len(attractions) == 1
                            else (
                                "ambiguous"
                                if len(attractions) > 1
                                else "unlinked"
                            )
                        ),
                    }
                )
    common_cols = [
        "row_id",
        "Review_ID",
        "Branch",
        "segment_id",
        "segment",
        "compound",
        "context_sentiment",
    ]
    mentions = pd.DataFrame(
        mention_rows,
        columns=common_cols + ["aspect_type", "aspect", "multiple_aspects"],
    )
    complaints = pd.DataFrame(
        complaint_rows,
        columns=common_cols
        + [
            "category",
            "evidence",
            "cue",
            "primary",
            "candidate_attraction",
            "link_status",
        ],
    )
    primary_complaints = complaints.loc[complaints["primary"].eq(True)].copy()
    aspect_reviews = (
        mentions.groupby(["row_id", "Branch", "aspect_type", "aspect"])
        .agg(
            mean_context_score=("compound", "mean"),
            positive_context=(
                "compound",
                lambda values: values.ge(0.05).any(),
            ),
            negative_context=(
                "compound",
                lambda values: values.le(-0.05).any(),
            ),
        )
        .reset_index()
    )
    aspect_reviews["mixed_context"] = (
        aspect_reviews["positive_context"] & aspect_reviews["negative_context"]
    )
    aspect_summary = (
        aspect_reviews.groupby(["Branch", "aspect_type", "aspect"])
        .agg(
            reviews=("row_id", "nunique"),
            mean_context_score=("mean_context_score", "mean"),
            mixed_context_reviews=("mixed_context", "sum"),
        )
        .reset_index()
        .sort_values("reviews", ascending=False)
    )
    denominators = text_df.groupby("Branch").size().rename("text_reviews")
    all_pairs = pd.MultiIndex.from_product(
        [denominators.index, PRIMARY_CATEGORIES], names=["Branch", "category"]
    )
    complaint_summary = (
        primary_complaints.groupby(["Branch", "category"])["row_id"]
        .nunique()
        .reindex(all_pairs, fill_value=0)
        .rename("reviews")
        .reset_index()
        .merge(denominators, on="Branch")
    )
    complaint_summary["review_pct"] = (
        100 * complaint_summary["reviews"] / complaint_summary["text_reviews"]
    )
    audit_parts = []
    for _, group in primary_complaints.groupby(["Branch", "category"]):
        audit_parts.append(
            group.sample(n=min(10, len(group)), random_state=SEED)
        )
    extraction_audit = (
        pd.concat(audit_parts, ignore_index=True)
        if audit_parts
        else primary_complaints.copy()
    )
    extraction_audit["reviewer_judgment"] = ""
    extraction_audit["notes"] = ""
    unmatched = text_df.loc[
        ~text_df["row_id"].isin(primary_complaints["row_id"])
    ]
    unmatched_audit = unmatched.sample(
        n=min(150, len(unmatched)), random_state=SEED
    )[["row_id", "Branch", "Review_Text"]].copy()
    unmatched_audit["missed_categories"] = ""
    attraction_review_count = mentions.loc[
        mentions["aspect_type"].eq("attraction"), "row_id"
    ].nunique()
    return {
        "mentions": mentions,
        "complaints": complaints,
        "primary_complaints": primary_complaints,
        "aspect_reviews": aspect_reviews,
        "aspect_summary": aspect_summary,
        "complaint_summary": complaint_summary,
        "extraction_audit": extraction_audit,
        "unmatched_audit": unmatched_audit,
    }


def review_vocabulary(top_terms):
    """Explain how inspected terms informed the complaint dictionary.

    Common words help us choose topics to inspect, but their frequency or
    TF-IDF weight does not make them complaints. This function annotates
    the supplied terms without creating or changing any complaint labels.

    Parameters
    ----------
    top_terms : DataFrame
        Term table with a term column, such as compute_corpus_top_terms.
        Additional columns, including branch scores, are retained.

    Returns
    -------
    DataFrame
        Copy of top_terms with a decision column explaining dictionary use.
    """
    # a topic word still needs a problem phrase before we label a complaint
    decisions = {
        "hotel": "Require hotel or room beside a problem cue",
        "hotels": "Require hotel or room beside a problem cue",
        "staff": "Require an explicit staff-service problem cue",
        "queues": "Require a queue problem expression",
        "mtr": "Require a transport problem cue",
        "shuttle": "Require a transport problem cue",
    }
    reviewed = top_terms.copy()
    reviewed["decision"] = (
        reviewed["term"]
        .map(decisions)
        .fillna("Context or place term; no automatic complaint label")
    )
    return reviewed


def analyze_evidence_quality(text_df, evidence):
    # a complaint and a positive sentence can appear together
    """Check context labels and extract direct target-problem phrases. A
    positive VADER context does not cancel an explicit complaint. Separate
    target labels prevent a parent category from implying its own subtype.
    The blank audit columns are for human judgments, not measured accuracy.

    Parameters
    ----------
    text_df : DataFrame
        Usable-text table returned by run_cleaning_pipeline.
    evidence : dict
        Result of extract_review_evidence with mention and complaint
        tables.

    Returns
    -------
    dict
        Tables and settings keyed by: sentiment_diagnostic,
        target_evidence, target_summary, review_audit.
    """
    mentions = evidence["mentions"]
    complaints = evidence["complaints"]
    primary_complaints = evidence["primary_complaints"]
    common_cols = [
        "row_id",
        "Review_ID",
        "Branch",
        "segment_id",
        "segment",
        "compound",
        "context_sentiment",
    ]
    denominators = text_df.groupby("Branch").size().rename("text_reviews")
    sentiment_diagnostic = (
        primary_complaints.assign(
            positive_context=primary_complaints["context_sentiment"].eq(
                "positive"
            )
        )
        .groupby(["Branch", "category"])
        .agg(
            matched_segments=("row_id", "size"),
            positive_segments=("positive_context", "sum"),
        )
        .reset_index()
    )
    sentiment_diagnostic["positive_pct"] = (
        100
        * sentiment_diagnostic["positive_segments"]
        / sentiment_diagnostic["matched_segments"]
    )
    target_rows = []
    segments = pd.concat([mentions[common_cols], complaints[common_cols]])
    for row in segments.drop_duplicates(["row_id", "segment_id"]).itertuples():
        for target, patterns in TARGET_RULES.items():
            match = find_target_match(row.segment, patterns, target)
            if match:
                target_rows.append(
                    {
                        "row_id": row.row_id,
                        "Branch": row.Branch,
                        "target_item": target,
                        "phrase": match.group(),
                        "segment": row.segment,
                        "context_sentiment": row.context_sentiment,
                    }
                )
    target_evidence = pd.DataFrame(
        target_rows,
        columns=[
            "row_id",
            "Branch",
            "target_item",
            "phrase",
            "segment",
            "context_sentiment",
        ],
    ).drop_duplicates(["row_id", "target_item", "segment"])
    target_summary = (
        target_evidence.groupby(["Branch", "target_item"])
        .agg(reviews=("row_id", "nunique"))
        .reset_index()
        .merge(denominators, on="Branch")
    )
    target_summary["review_pct"] = (
        100 * target_summary["reviews"] / target_summary["text_reviews"]
    )
    target_summary["small_count"] = target_summary["reviews"].lt(20)
    audit_groups = [
        group.sample(n=min(50, len(group)), random_state=SEED + 1)
        for _, group in text_df.groupby("Branch")
    ]
    review_audit = (
        pd.concat(audit_groups, ignore_index=True)[
            ["row_id", "Branch", "Review_Text"]
        ].copy()
        if audit_groups
        else text_df[["row_id", "Branch", "Review_Text"]].copy()
    )
    for category in PRIMARY_CATEGORIES:
        review_audit["human: " + category] = ""
    review_audit["reviewer_notes"] = ""
    return {
        "sentiment_diagnostic": sentiment_diagnostic,
        "target_evidence": target_evidence,
        "target_summary": target_summary,
        "review_audit": review_audit,
    }


# PART 6: REVIEW BASKETS AND CLASS-STYLE APRIORI


def support_count(db, itemset):
    """Count baskets containing every requested item, including empty
    baskets in the database. Repeating an item does not increase support.

    Parameters
    ----------
    db : list of sets
        Review baskets; all items must be comparable strings.
    itemset : iterable of str
        Items that must all occur in a matching basket.

    Returns
    -------
    int
        Absolute number of matching baskets.
    """
    wanted = set(itemset)
    return sum(wanted.issubset(transaction) for transaction in db)


def apriori(db, minsup):
    """Follow the class candidate-generation method: count singletons, join
    frequent sets, and prune candidates with an infrequent subset. Return
    absolute counts ordered by count, set size and then item names.

    Parameters
    ----------
    db : list of sets
        Review baskets; all items must be comparable strings.
    minsup : int
        Minimum absolute support count, not a percentage.

    Returns
    -------
    list of tuple
        (itemset tuple, absolute support) pairs, including frequent
        singletons.
    """
    if (
        isinstance(minsup, bool)
        or not isinstance(minsup, (int, np.integer))
        or minsup < 1
    ):
        raise ValueError(
            "Minimum absolute support must be a positive integer."
        )
    items = sorted({item for transaction in db for item in transaction})
    candidates = [(item,) for item in items]
    frequent = []
    size = 1
    while candidates:
        level = []
        for itemset in candidates:
            count = support_count(db, itemset)
            if count >= minsup:
                level.append((itemset, count))
        frequent.extend(level)
        if not level:
            break
        previous = {itemset for itemset, count in level}
        next_candidates = set()
        for first, second in combinations(sorted(previous), 2):
            if first[:-1] != second[:-1]:
                continue
            joined = tuple(sorted(set(first) | set(second)))
            if len(joined) == size + 1 and all(
                subset in previous for subset in combinations(joined, size)
            ):
                next_candidates.add(joined)
        candidates = sorted(next_candidates)
        size += 1
    return sorted(
        frequent, key=lambda result: (-result[1], -len(result[0]), result[0])
    )


def assoc(fi, minconf, db_size):
    """Use frequent-set counts to form rules with a single consequent.
    Confidence is joint count divided by antecedent count, and lift divides
    confidence by consequent prevalence across all database baskets.

    Parameters
    ----------
    fi : list of tuple
        Complete frequent itemsets and counts returned by apriori.
    minconf : float
        Minimum confidence between zero and one.
    db_size : int
        Number of baskets, including empty baskets, used to obtain fi.

    Returns
    -------
    list of dict
        antecedent, consequent, support count, confidence and lift per
        rule.
    """
    if not 0 <= minconf <= 1:
        raise ValueError("Minimum confidence must be between zero and one.")
    if db_size <= 0:
        return []
    counts = dict(fi)
    rules = []
    for itemset, count in fi:
        if len(itemset) < 2:
            continue
        for consequent in itemset:
            antecedent = tuple(item for item in itemset if item != consequent)
            confidence = count / counts[antecedent]
            if confidence >= minconf:
                rules.append(
                    {
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support": count,
                        "confidence": confidence,
                        "lift": confidence / (counts[(consequent,)] / db_size),
                    }
                )
    return sorted(
        rules,
        key=lambda rule: (
            -rule["lift"],
            -rule["confidence"],
            -rule["support"],
            rule["consequent"],
            rule["antecedent"],
        ),
    )


def build_complaint_baskets(
    text_df, primary_complaints, min_support=MIN_SUPPORT, min_count=MIN_COUNT
):
    # keep empty baskets because they belong in the denominator
    """Make one set of primary complaints per usable review. Repeated matches
    count once, and reviews without matches keep an empty basket so support
    uses all usable reviews rather than only reviews with complaints.

    Parameters
    ----------
    text_df : DataFrame
        Usable-text table returned by run_cleaning_pipeline.
    primary_complaints : DataFrame
        Primary-only complaint evidence from extract_review_evidence.
    min_support : float, optional
        Minimum support fraction, greater than zero and at most one.
    min_count : int, optional
        Minimum absolute support; must be a positive integer.

    Returns
    -------
    dict
        Tables and settings keyed by: baskets_df, basket_coverage.
    """
    validate_thresholds(min_support, min_count, 0, 0)
    MIN_SUPPORT = min_support
    MIN_COUNT = min_count
    category_lists = primary_complaints.groupby("row_id")["category"].agg(
        lambda values: sorted(set(values))
    )
    baskets_df = text_df[
        [
            "row_id",
            "Review_ID",
            "Branch",
            "Reviewer_Location",
            "rating",
            "word_count",
            "repeated_id",
        ]
    ].copy()
    baskets_df["basket"] = baskets_df["row_id"].apply(
        lambda row_id: set(category_lists.get(row_id, []))
    )
    baskets_df["basket_size"] = baskets_df["basket"].map(len).astype(int)
    basket_coverage = baskets_df.groupby("Branch").agg(
        reviews=("row_id", "size"),
        empty=("basket_size", lambda sizes: sizes.eq(0).sum()),
        singleton=("basket_size", lambda sizes: sizes.eq(1).sum()),
        multiple=("basket_size", lambda sizes: sizes.ge(2).sum()),
    )
    basket_coverage["min_count"] = basket_coverage["reviews"].apply(
        lambda total: max(MIN_COUNT, ceil(MIN_SUPPORT * total))
    )
    return {"baskets_df": baskets_df, "basket_coverage": basket_coverage}


def mine_branch_patterns(
    baskets_df,
    min_support=MIN_SUPPORT,
    min_count=MIN_COUNT,
    min_confidence=MIN_CONFIDENCE,
    min_lift=MIN_LIFT,
):
    # apply the same cutoffs to every branch
    """Mine each known branch using the class Apriori and association methods.
    An itemset must meet both the percentage and absolute count cutoffs.
    Rules have one consequent, and must also pass confidence and lift.
    Support is a count; relative_support is a fraction between zero and one.

    Parameters
    ----------
    baskets_df : DataFrame
        Review-level table from build_complaint_baskets, including empty
        sets.
    min_support : float, optional
        Minimum support fraction, greater than zero and at most one.
    min_count : int, optional
        Minimum absolute support; must be a positive integer.
    min_confidence : float, optional
        Minimum rule confidence, between zero and one.
    min_lift : float, optional
        Minimum rule lift, a finite nonnegative number.

    Returns
    -------
    dict
        Tables and settings keyed by: branch_baskets, frequent_itemsets,
        selected_rules.
    """
    MIN_SUPPORT = min_support
    MIN_COUNT = min_count
    MIN_CONFIDENCE = min_confidence
    MIN_LIFT = min_lift
    validate_thresholds(min_support, min_count, min_confidence, min_lift)
    branch_baskets = {
        branch: group["basket"].tolist()
        for branch, group in baskets_df.groupby("Branch")
        if branch != "Unknown"
    }
    itemset_rows = []
    rule_rows = []
    for branch, db in branch_baskets.items():
        minimum = max(MIN_COUNT, ceil(MIN_SUPPORT * len(db)))
        frequent = apriori(db, minimum)
        for itemset, count in frequent:
            itemset_rows.append(
                {
                    "Branch": branch,
                    "itemset": itemset,
                    "size": len(itemset),
                    "support": count,
                    "relative_support": count / len(db),
                    "total_reviews": len(db),
                }
            )
        for rule in assoc(frequent, MIN_CONFIDENCE, len(db)):
            if rule["lift"] >= MIN_LIFT:
                rule_rows.append(
                    {
                        "Branch": branch,
                        **rule,
                        "relative_support": rule["support"] / len(db),
                        "total_reviews": len(db),
                    }
                )
    frequent_itemsets = pd.DataFrame(
        itemset_rows,
        columns=[
            "Branch",
            "itemset",
            "size",
            "support",
            "relative_support",
            "total_reviews",
        ],
    )
    selected_rules = pd.DataFrame(
        rule_rows,
        columns=[
            "Branch",
            "antecedent",
            "consequent",
            "support",
            "confidence",
            "lift",
            "relative_support",
            "total_reviews",
        ],
    )
    selected_rules["rule"] = [
        " + ".join(a) + " → " + b
        for a, b in zip(
            selected_rules["antecedent"], selected_rules["consequent"]
        )
    ]
    return {
        "branch_baskets": branch_baskets,
        "frequent_itemsets": frequent_itemsets,
        "selected_rules": selected_rules,
    }


# PART 7: BRANCH COMPARISONS AND SENSITIVITY CHECKS


def compare_branch_patterns(
    baskets_df,
    mining,
    min_support=MIN_SUPPORT,
    min_count=MIN_COUNT,
    min_confidence=MIN_CONFIDENCE,
    min_lift=MIN_LIFT,
):
    # a pattern missing from one branch still needs a count there
    """Recount selected combinations in every branch, including zero matches.
    Check both directions of frequent pairs, try 0.5%, 1%, and 2% support,
    and rerun after excluding repeated review IDs. These are descriptive
    checks, not significance tests or proof of a branch effect. Pass the
    same thresholds used to create mining.

    Parameters
    ----------
    baskets_df : DataFrame
        Review-level table from build_complaint_baskets, including empty
        sets.
    mining : dict
        Result of mine_branch_patterns for the same baskets and settings.
    min_support : float, optional
        Minimum support fraction, greater than zero and at most one.
    min_count : int, optional
        Minimum absolute support; must be a positive integer.
    min_confidence : float, optional
        Minimum rule confidence, between zero and one.
    min_lift : float, optional
        Minimum rule lift, a finite nonnegative number.

    Returns
    -------
    dict
        Tables and settings keyed by: itemset_comparison, rule_comparison,
        sensitivity, duplicate_sensitivity.
    """
    MIN_SUPPORT = min_support
    MIN_COUNT = min_count
    MIN_CONFIDENCE = min_confidence
    MIN_LIFT = min_lift
    validate_thresholds(min_support, min_count, min_confidence, min_lift)
    branch_baskets = mining["branch_baskets"]
    frequent_itemsets = mining["frequent_itemsets"]
    selected_rules = mining["selected_rules"]
    shared_sets = sorted(
        set(frequent_itemsets.loc[frequent_itemsets["size"].ge(2), "itemset"])
    )
    comparison_rows = []
    for itemset in shared_sets:
        for branch, db in branch_baskets.items():
            count = support_count(db, itemset)
            comparison_rows.append(
                {
                    "Branch": branch,
                    "itemset": itemset,
                    "pattern": " + ".join(itemset),
                    "support": count,
                    "relative_support": count / len(db),
                    "total_reviews": len(db),
                }
            )
    itemset_comparison = pd.DataFrame(
        comparison_rows,
        columns=[
            "Branch",
            "itemset",
            "pattern",
            "support",
            "relative_support",
            "total_reviews",
        ],
    )
    rule_keys = sorted(
        {
            ((first,), second)
            for pair in shared_sets
            if len(pair) == 2
            for first, second in [pair, pair[::-1]]
        }
    )
    rule_comparison_rows = []
    for antecedent, consequent in rule_keys:
        for branch, db in branch_baskets.items():
            joint = support_count(db, set(antecedent) | {consequent})
            a_count = support_count(db, antecedent)
            b_count = support_count(db, {consequent})
            confidence = joint / a_count if a_count else np.nan
            lift = (
                confidence / (b_count / len(db))
                if a_count and b_count
                else np.nan
            )
            rule_comparison_rows.append(
                {
                    "Branch": branch,
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "support": joint,
                    "antecedent_count": a_count,
                    "consequent_count": b_count,
                    "relative_support": joint / len(db),
                    "confidence": confidence,
                    "lift": lift,
                    "passes_thresholds": joint
                    >= max(MIN_COUNT, ceil(MIN_SUPPORT * len(db)))
                    and confidence >= MIN_CONFIDENCE
                    and (lift >= MIN_LIFT),
                }
            )
    rule_comparison = pd.DataFrame(
        rule_comparison_rows,
        columns=[
            "Branch",
            "antecedent",
            "consequent",
            "support",
            "antecedent_count",
            "consequent_count",
            "relative_support",
            "confidence",
            "lift",
            "passes_thresholds",
        ],
    )
    sensitivity_rows = []
    duplicate_sensitivity_rows = []
    for branch, db in branch_baskets.items():
        for threshold in [0.005, 0.01, 0.02]:
            frequent = apriori(db, max(MIN_COUNT, ceil(threshold * len(db))))
            rules = assoc(frequent, MIN_CONFIDENCE, len(db))
            sensitivity_rows.append(
                {
                    "Branch": branch,
                    "min_relative_support": threshold,
                    "multi_itemsets": sum(
                        (len(items) >= 2 for items, count in frequent)
                    ),
                    "selected_rules": sum(
                        (rule["lift"] >= MIN_LIFT for rule in rules)
                    ),
                }
            )
        clean_db = baskets_df.loc[
            baskets_df["Branch"].eq(branch) & ~baskets_df["repeated_id"],
            "basket",
        ].tolist()
        clean_fi = apriori(
            clean_db, max(MIN_COUNT, ceil(MIN_SUPPORT * len(clean_db)))
        )
        clean_keys = {
            (rule["antecedent"], rule["consequent"])
            for rule in assoc(clean_fi, MIN_CONFIDENCE, len(clean_db))
            if rule["lift"] >= MIN_LIFT
        }
        original = selected_rules.loc[selected_rules["Branch"].eq(branch)]
        original_keys = set(
            zip(original["antecedent"], original["consequent"])
        )
        duplicate_sensitivity_rows.append(
            {
                "Branch": branch,
                "original_rules": len(original_keys),
                "rules_without_repeated_ids": len(clean_keys),
                "removed_rules": len(original_keys - clean_keys),
                "new_rules": len(clean_keys - original_keys),
            }
        )
    sensitivity = pd.DataFrame(
        sensitivity_rows,
        columns=[
            "Branch",
            "min_relative_support",
            "multi_itemsets",
            "selected_rules",
        ],
    )
    duplicate_sensitivity = pd.DataFrame(
        duplicate_sensitivity_rows,
        columns=[
            "Branch",
            "original_rules",
            "rules_without_repeated_ids",
            "removed_rules",
            "new_rules",
        ],
    )
    return {
        "itemset_comparison": itemset_comparison,
        "rule_comparison": rule_comparison,
        "sensitivity": sensitivity,
        "duplicate_sensitivity": duplicate_sensitivity,
    }


# PART 8: OPTIONAL WAYS TO LOOK AT THE BASKETS


def analyze_favorable_reviews(
    baskets_df, min_support=MIN_SUPPORT, min_count=MIN_COUNT
):
    # a good overall rating does not mean the review has no complaints
    """Mine pairs in four- and five-star reviews, then recount them in one-
    and two-star reviews. Each percentage uses its own rating-group total.
    Missing low-rating groups have NaN support rather than a misleading zero.

    Parameters
    ----------
    baskets_df : DataFrame
        Review-level table from build_complaint_baskets, including empty
        sets.
    min_support : float, optional
        Minimum support fraction, greater than zero and at most one.
    min_count : int, optional
        Minimum absolute support; must be a positive integer.

    Returns
    -------
    dict
        Tables and settings keyed by: rating_coverage, favorable_pairs.
    """
    validate_thresholds(min_support, min_count, 0, 0)
    MIN_SUPPORT = min_support
    MIN_COUNT = min_count
    favorable_pair_rows = []
    rating_coverage = (
        baskets_df.assign(
            rating_group=baskets_df["rating"]
            .map({1: "low", 2: "low", 3: "middle", 4: "high", 5: "high"})
            .fillna("missing")
        )
        .groupby(["Branch", "rating_group"])
        .size()
        .unstack(fill_value=0)
    )
    for branch, group in baskets_df.groupby("Branch"):
        if branch == "Unknown":
            continue
        high_db = group.loc[group["rating"].ge(4), "basket"].tolist()
        low_db = group.loc[group["rating"].le(2), "basket"].tolist()
        if not high_db:
            continue
        frequent = apriori(
            high_db, max(MIN_COUNT, ceil(MIN_SUPPORT * len(high_db)))
        )
        for itemset, high_count in frequent:
            if len(itemset) != 2:
                continue
            low_count = support_count(low_db, itemset)
            favorable_pair_rows.append(
                {
                    "Branch": branch,
                    "itemset": itemset,
                    "pattern": " + ".join(itemset),
                    "high_count": high_count,
                    "high_total": len(high_db),
                    "high_support": high_count / len(high_db),
                    "low_count": low_count,
                    "low_total": len(low_db),
                    "low_support": (
                        low_count / len(low_db) if low_db else np.nan
                    ),
                }
            )
    favorable_pairs = pd.DataFrame(
        favorable_pair_rows,
        columns=[
            "Branch",
            "itemset",
            "pattern",
            "high_count",
            "high_total",
            "high_support",
            "low_count",
            "low_total",
            "low_support",
        ],
    )
    favorable_pairs["small_low_count"] = favorable_pairs["low_count"].lt(20)
    return {
        "rating_coverage": rating_coverage,
        "favorable_pairs": favorable_pairs,
    }


def compare_review_context(baskets_df, text_df):
    # compare like groups before interpreting a branch difference
    """Compare category percentages within review-length bands, the top three
    reviewer locations per branch, and review years. Missing locations or
    dates follow the cleaning output; missing years are omitted. These
    summaries describe the sample and do not adjust for confounding.

    Parameters
    ----------
    baskets_df : DataFrame
        Review-level table from build_complaint_baskets, including empty
        sets.
    text_df : DataFrame
        Usable-text table returned by run_cleaning_pipeline.

    Returns
    -------
    dict
        Tables and settings keyed by: context_comparison.
    """
    context_df = baskets_df.merge(
        text_df[["row_id", "review_month"]], on="row_id", validate="one_to_one"
    )
    context_df["length_group"] = pd.cut(
        context_df["word_count"],
        bins=[-1, 50, 150, np.inf],
        labels=["1–50 words", "51–150 words", "151+ words"],
    )
    context_df["year"] = context_df["review_month"].dt.year.astype("Int64")
    context_rows = []
    for branch, group in context_df.groupby("Branch"):
        top_locations = group["Reviewer_Location"].value_counts().head(3).index
        for dimension in ["length_group", "Reviewer_Location", "year"]:
            selected = group
            if dimension == "Reviewer_Location":
                selected = group.loc[group[dimension].isin(top_locations)]
            for value, subgroup in selected.groupby(dimension, observed=True):
                for category in PRIMARY_CATEGORIES:
                    count = sum(
                        (category in basket for basket in subgroup["basket"])
                    )
                    context_rows.append(
                        {
                            "Branch": branch,
                            "dimension": dimension,
                            "group": str(value),
                            "category": category,
                            "reviews": len(subgroup),
                            "detected": count,
                            "review_pct": 100 * count / len(subgroup),
                            "small_count": count < 20,
                        }
                    )
    context_comparison = pd.DataFrame(
        context_rows,
        columns=[
            "Branch",
            "dimension",
            "group",
            "category",
            "reviews",
            "detected",
            "review_pct",
            "small_count",
        ],
    )
    return {"context_comparison": context_comparison}


def analyze_target_and_rating_patterns(
    baskets_df,
    target_evidence,
    min_support=MIN_SUPPORT,
    min_count=MIN_COUNT,
    min_confidence=MIN_CONFIDENCE,
    min_lift=MIN_LIFT,
):
    # keep direct targets separate so categories cannot imply themselves
    """Mine direct target phrases separately from the primary complaint sets.
    Then test single-category or pair antecedents against a low rating,
    using only reviews with valid ratings. Rating rules returned here meet
    joint support; passes_rule_cutoffs marks confidence and lift eligibility.
    An association is not evidence that a complaint caused the rating.

    Parameters
    ----------
    baskets_df : DataFrame
        Review-level table from build_complaint_baskets, including empty
        sets.
    target_evidence : DataFrame
        Direct target phrase table from analyze_evidence_quality.
    min_support : float, optional
        Minimum support fraction, greater than zero and at most one.
    min_count : int, optional
        Minimum absolute support; must be a positive integer.
    min_confidence : float, optional
        Minimum rule confidence, between zero and one.
    min_lift : float, optional
        Minimum rule lift, a finite nonnegative number.

    Returns
    -------
    dict
        Tables and settings keyed by: target_itemsets, rating_rules.
    """
    MIN_SUPPORT = min_support
    MIN_COUNT = min_count
    MIN_CONFIDENCE = min_confidence
    MIN_LIFT = min_lift
    validate_thresholds(min_support, min_count, min_confidence, min_lift)
    target_lists = target_evidence.groupby("row_id")["target_item"].agg(set)
    target_itemset_rows = []
    for branch, group in baskets_df.groupby("Branch"):
        if branch == "Unknown":
            continue
        db = [set(target_lists.get(key, set())) for key in group["row_id"]]
        minimum = max(MIN_COUNT, ceil(MIN_SUPPORT * len(db)))
        for itemset, count in apriori(db, minimum):
            target_itemset_rows.append(
                {
                    "Branch": branch,
                    "itemset": itemset,
                    "size": len(itemset),
                    "support": count,
                    "relative_support": count / len(db),
                }
            )
    target_itemsets = pd.DataFrame(
        target_itemset_rows,
        columns=["Branch", "itemset", "size", "support", "relative_support"],
    )
    rating_rule_rows = []
    for branch, group in baskets_df.groupby("Branch"):
        if branch == "Unknown":
            continue
        valid = group.loc[group["rating"].between(1, 5)]
        if valid.empty:
            continue
        db = valid["basket"].tolist()
        low_db = valid.loc[valid["rating"].le(2), "basket"].tolist()
        baseline = len(low_db) / len(db)
        minimum = max(MIN_COUNT, ceil(MIN_SUPPORT * len(db)))
        for antecedent, count in apriori(db, minimum):
            if len(antecedent) > 2:
                continue
            joint = support_count(low_db, antecedent)
            if joint < minimum:
                continue
            confidence = joint / count
            lift = confidence / baseline if baseline else np.nan
            rating_rule_rows.append(
                {
                    "Branch": branch,
                    "antecedent": antecedent,
                    "consequent": "Rating:low",
                    "antecedent_count": count,
                    "joint_count": joint,
                    "total_reviews": len(db),
                    "relative_support": joint / len(db),
                    "low_rating_baseline": baseline,
                    "confidence": confidence,
                    "lift": lift,
                    "passes_rule_cutoffs": confidence >= MIN_CONFIDENCE
                    and lift >= MIN_LIFT,
                }
            )
    rating_rules = pd.DataFrame(
        rating_rule_rows,
        columns=[
            "Branch",
            "antecedent",
            "consequent",
            "antecedent_count",
            "joint_count",
            "total_reviews",
            "relative_support",
            "low_rating_baseline",
            "confidence",
            "lift",
            "passes_rule_cutoffs",
        ],
    )
    return {"target_itemsets": target_itemsets, "rating_rules": rating_rules}


# PART 9: EVIDENCE-BASED FOLLOW-UP SUGGESTIONS


ACTION_BY_CATEGORY = {
    "Long queues": (
        "compare posted and observed waits and inspect queue handling"
    ),
    "Closures / breakdowns": (
        "check whether closure notices and available alternatives were clear"
    ),
    "Poor staff service": (
        "review the described staff interactions and service-recovery process"
    ),
    "High prices / poor value": (
        "check price communication against the experience described"
    ),
    "Food quality / dining": (
        "inspect the food and dining conditions mentioned in the reviews"
    ),
    "Crowding": (
        "examine crowd movement and capacity management at the cited locations"
    ),
    "Cleanliness": (
        "review cleaning coverage around the locations and times described"
    ),
}
ACTION_BY_CATEGORY.update(
    {
        "Hotel problems": "inspect the described room and hotel conditions",
        "Transport problems": (
            "review the described access and shuttle problems"
        ),
    }
)


def build_recommendations(
    text_df, baskets_df, mining, comparisons, target_summary
):
    # tie each suggestion to a counted pattern we can inspect
    """Suggest one follow-up per branch with a frequent pair, or a frequent
    single category if no pair qualifies. Rank positive percentage-point
    gaps against pooled other branches, otherwise use the largest count.
    These are leads for checking original reviews, not verified causes.
    Also return sample reviews and supported direct target phrases.

    Parameters
    ----------
    text_df : DataFrame
        Usable-text table returned by run_cleaning_pipeline.
    baskets_df : DataFrame
        Review-level table from build_complaint_baskets, including empty
        sets.
    mining : dict
        Result of mine_branch_patterns for the same baskets and settings.
    comparisons : dict
        Result of compare_branch_patterns for the same mining run.
    target_summary : DataFrame
        Target review counts from analyze_evidence_quality.

    Returns
    -------
    dict
        Tables and settings keyed by: recommendations, evidence_reviews,
        target_follow_up.
    """
    branch_baskets = mining["branch_baskets"]
    frequent_itemsets = mining["frequent_itemsets"]
    itemset_comparison = comparisons["itemset_comparison"]
    recommendation_rows = []
    for branch in branch_baskets:
        pairs = frequent_itemsets.loc[
            frequent_itemsets["Branch"].eq(branch)
            & frequent_itemsets["size"].eq(2)
        ]
        if pairs.empty:
            pairs = frequent_itemsets.loc[
                frequent_itemsets["Branch"].eq(branch)
                & frequent_itemsets["size"].eq(1)
            ]
        if pairs.empty:
            continue
        pairs = pairs.copy()
        other_db = [
            basket
            for other, db in branch_baskets.items()
            if other != branch
            for basket in db
        ]
        pairs["other_support"] = pairs["itemset"].apply(
            lambda items: (
                support_count(other_db, items) / len(other_db)
                if other_db
                else np.nan
            )
        )
        pairs["gap_pp"] = 100 * (
            pairs["relative_support"] - pairs["other_support"]
        )
        distinctive = pairs["gap_pp"].gt(0).any()
        ranking = ["gap_pp", "support"] if distinctive else ["support"]
        top = pairs.sort_values(ranking, ascending=False).iloc[0]
        actions = " and ".join(
            (ACTION_BY_CATEGORY[item] for item in top["itemset"])
        )
        recommendation_rows.append(
            {
                "Branch": branch,
                "evidence_pair": " + ".join(top["itemset"]),
                "other_review_pct": 100 * top["other_support"],
                "gap_pp": top["gap_pp"],
                "distinctive": distinctive,
                "review_count": top["support"],
                "review_pct": 100 * top["relative_support"],
                "proposed_follow_up": actions
                + "; first confirm that the comments describe related"
                " circumstances.",
            }
        )
    recommendations = pd.DataFrame(
        recommendation_rows,
        columns=[
            "Branch",
            "evidence_pair",
            "other_review_pct",
            "gap_pp",
            "distinctive",
            "review_count",
            "review_pct",
            "proposed_follow_up",
        ],
    )
    evidence_reviews = text_df.head(0).copy()
    if not itemset_comparison.empty:
        chosen = itemset_comparison.sort_values(
            "relative_support", ascending=False
        ).iloc[0]
        matching_ids = baskets_df.loc[
            baskets_df["Branch"].eq(chosen["Branch"])
            & baskets_df["basket"].apply(
                lambda basket: set(chosen["itemset"]).issubset(basket)
            ),
            "row_id",
        ]
        example_ids = matching_ids.sample(
            n=min(3, len(matching_ids)), random_state=SEED
        )
        evidence_reviews = text_df.loc[text_df["row_id"].isin(example_ids)]
    target_follow_up = (
        target_summary.loc[target_summary["reviews"].ge(20)]
        .sort_values("reviews", ascending=False)
        .groupby("Branch", sort=False)
        .head(1)
    )
    return {
        "recommendations": recommendations,
        "evidence_reviews": evidence_reviews,
        "target_follow_up": target_follow_up,
    }


# PART 10: VALIDATION AND OPTIONAL EXPORTS


def validate_thresholds(min_support, min_count, min_confidence, min_lift):
    """Check support, count, confidence and lift settings before mining.

    Parameters
    ----------
    min_support : float, optional
        Minimum support fraction, greater than zero and at most one.
    min_count : int, optional
        Minimum absolute support; must be a positive integer.
    min_confidence : float, optional
        Minimum rule confidence, between zero and one.
    min_lift : float, optional
        Minimum rule lift, a finite nonnegative number.

    Returns
    -------
    None
        Returns normally for valid settings; otherwise raises ValueError.
    """
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be greater than 0 and at most 1.")
    if (
        isinstance(min_count, bool)
        or not isinstance(min_count, (int, np.integer))
        or min_count < 1
    ):
        raise ValueError("min_count must be a positive integer.")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1.")
    if not np.isfinite(min_lift) or min_lift < 0:
        raise ValueError("min_lift must be finite and nonnegative.")


def validate_analysis(text_df, results, representations=None):
    """Check row alignment and count bounds, and optionally reconstruct
    TF-IDF from the count matrix. This checks computations, not language
    accuracy. A failed condition raises ValueError even outside test runs.

    Parameters
    ----------
    text_df : DataFrame
        Usable-text table returned by run_cleaning_pipeline.
    results : dict
        Combined analysis output returned by run_parts_5_to_10.
    representations : dict, optional
        Output of build_text_representations for the same review order.

    Returns
    -------
    DataFrame
        Named checks and Boolean passed values, or ValueError on failure.
    """
    baskets = results["baskets_df"]
    checks = {
        "unique_review_keys": text_df["row_id"].is_unique,
        "one_basket_per_review": baskets["row_id"].tolist()
        == text_df["row_id"].tolist(),
        "unique_basket_keys": baskets["row_id"].is_unique,
        "valid_itemset_counts": results["frequent_itemsets"]["support"]
        .between(1, results["frequent_itemsets"]["total_reviews"])
        .all(),
        "valid_confidence": results["selected_rules"]["confidence"]
        .between(0, 1)
        .all(),
        "valid_complaint_counts": results["complaint_summary"]["reviews"]
        .between(0, results["complaint_summary"]["text_reviews"])
        .all(),
    }
    if representations is not None:
        bow = representations["bow_matrix"]
        tfidf = representations["tfidf_matrix"]
        checks["matrix_rows_match_reviews"] = (
            len(text_df) == bow.shape[0] == tfidf.shape[0]
        )
        checks["matching_vocabularies"] = np.array_equal(
            representations["count_vec"].get_feature_names_out(),
            representations["tfidf_vec"].get_feature_names_out(),
        )
        difference = TfidfTransformer().fit_transform(bow) - tfidf
        checks["tfidf_matches_counts"] = (
            difference.nnz == 0 or np.max(np.abs(difference.data)) < 1e-10
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Analysis checks failed: " + ", ".join(failed))
    return pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )


def save_table(table, filename, list_columns=()):
    """Copy a result table, encode collection columns as JSON arrays, and
    write a UTF-8 CSV. An existing file at filename is replaced.

    Parameters
    ----------
    table : DataFrame
        Complete result table to save, not just its first five rows.
    filename : str or Path
        Destination CSV path, including the file name.
    list_columns : sequence of str, optional
        Columns containing sets, tuples or lists to encode as JSON.

    Returns
    -------
    Path
        Path of the saved CSV; the input DataFrame is unchanged.
    """
    output = table.copy()
    for column in list_columns:
        output[column] = output[column].apply(
            lambda value: json.dumps(sorted(value))
        )
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_analysis(
    results, output_dir="outputs/helper_analysis", input_path=None
):
    """Write complete result DataFrames to a new timestamped folder and
    record thresholds and matching dictionaries. Optionally hash the input
    CSV. Nothing is exported until this function is explicitly called.

    Parameters
    ----------
    results : dict
        Combined analysis output returned by run_parts_5_to_10.
    output_dir : str or Path, optional
        Parent directory in which a new run folder will be created.
    input_path : str or Path, optional
        Original CSV to record and hash in the export metadata.

    Returns
    -------
    Path
        New run folder containing CSV tables and run_metadata.json.
    """
    run_dir = Path(output_dir) / pd.Timestamp.now().strftime(
        "run_%Y%m%d_%H%M%S_%f"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    for name, table in results.items():
        if isinstance(table, pd.DataFrame):
            list_columns = [
                col
                for col in table
                if any(
                    isinstance(value, (set, tuple, list))
                    for value in table[col]
                )
            ]
            save_table(
                (
                    table.reset_index()
                    if name in {"basket_coverage", "rating_coverage"}
                    else table
                ),
                run_dir / (name + ".csv"),
                list_columns,
            )
    metadata = {
        "thresholds": results.get("settings", {}),
        "seed": SEED,
        "primary_categories": PRIMARY_CATEGORIES,
        "attraction_aliases": ATTRACTIONS,
        "service_aliases": SERVICES,
        "complaint_regex": {
            name: [pattern.pattern for pattern in patterns]
            for name, patterns in COMPLAINT_RULES.items()
        },
        "target_regex": {
            name: [pattern.pattern for pattern in patterns]
            for name, patterns in TARGET_RULES.items()
        },
        "label_status": (
            "Rule-based candidates; human accuracy not established."
        ),
    }
    if input_path is not None:
        metadata.update(
            input_path=str(Path(input_path).resolve()),
            sha256=compute_file_digest(input_path),
        )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return run_dir


def run_parts_5_to_10(
    text_df,
    analyzer=None,
    min_support=MIN_SUPPORT,
    min_count=MIN_COUNT,
    min_confidence=MIN_CONFIDENCE,
    min_lift=MIN_LIFT,
    include_optional=True,
):
    """Run extraction, baskets, branch mining, comparisons and follow-up
    suggestions in order. Optional Part 8 analyses can be switched off.
    Results are returned in full, without printing, plotting or writing files.
    The same threshold settings are passed throughout the analysis.

    Parameters
    ----------
    text_df : DataFrame
        Usable-text table returned by run_cleaning_pipeline.
    analyzer : object, optional
        Object with polarity_scores(text); defaults to vaderSentiment
        VADER.
    min_support : float, optional
        Minimum support fraction, greater than zero and at most one.
    min_count : int, optional
        Minimum absolute support; must be a positive integer.
    min_confidence : float, optional
        Minimum rule confidence, between zero and one.
    min_lift : float, optional
        Minimum rule lift, a finite nonnegative number.
    include_optional : bool, optional
        Whether to run the extra Part 8 comparisons; defaults to True.

    Returns
    -------
    dict
        Combined section result tables, branch_baskets, settings and
        validation.
    """
    settings = dict(
        min_support=min_support,
        min_count=min_count,
        min_confidence=min_confidence,
        min_lift=min_lift,
    )
    validate_thresholds(**settings)
    evidence = extract_review_evidence(text_df, analyzer)
    quality = analyze_evidence_quality(text_df, evidence)
    baskets = build_complaint_baskets(
        text_df, evidence["primary_complaints"], min_support, min_count
    )
    mining = mine_branch_patterns(baskets["baskets_df"], **settings)
    comparisons = compare_branch_patterns(
        baskets["baskets_df"], mining, **settings
    )
    results = {
        **evidence,
        **quality,
        **baskets,
        **mining,
        **comparisons,
        "settings": settings,
    }
    if include_optional:
        results.update(
            analyze_favorable_reviews(
                baskets["baskets_df"], min_support, min_count
            )
        )
        results.update(compare_review_context(baskets["baskets_df"], text_df))
        results.update(
            analyze_target_and_rating_patterns(
                baskets["baskets_df"], quality["target_evidence"], **settings
            )
        )
    results.update(
        build_recommendations(
            text_df,
            baskets["baskets_df"],
            mining,
            comparisons,
            quality["target_summary"],
        )
    )
    results["validation"] = validate_analysis(text_df, results)
    return results
