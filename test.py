"""Check helper_code with independent calculations and small known examples.

Run the normal checks with ``python test.py``. Add ``--full`` to also run
the complete local CSV through Parts 5-10 and recount its mining results.
Use ``--data PATH`` to choose a different CSV for the full-data check.

Dependencies are the same as helper_code, plus pycodestyle for the style
check: pandas, numpy, matplotlib, scikit-learn, vaderSentiment, pycodestyle.
The tests never overwrite the dataset or export analysis files outside a
temporary directory. They test computations and selected language examples;
they do not establish how accurately the rules interpret all real reviews.
"""

import argparse
import ast
import hashlib
import inspect
from itertools import combinations
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

import helper_code as helper  # noqa: E402


def make_reviews():
    """Build a small raw dataset with no inputs.

    Returns a DataFrame containing two known branches, an unknown branch,
    negated complaints, repeated IDs, missing values and varied ratings.
    Each row is written explicitly so expected results can be checked.
    """
    rows = [
        ["001", 1, "2020-01", "UK", "Queues were long. Staff were rude.", "A"],
        ["002", 5, "2020-02", "UK", "Queues were long. Staff were rude.", "A"],
        ["003", 4, "2020-03", "US", "Queues were long.", "A"],
        ["004", 5, "2020-03", "US", "A lovely visit.", "A"],
        ["005", 2, "2020-01", "UK", "The food was cold.", "B"],
        ["005", 1, "2020-02", "US", "The tickets were too expensive.", "B"],
        [None, 5, "bad date", None, "Staff were not rude.", "B"],
        ["007", 8, "2020-04", "UK", "A lovely visit.", None],
        ["008", None, None, "UK", None, "B"],
    ]
    return pd.DataFrame(rows, columns=helper.REQUIRED_COLUMNS)


def exhaustive_itemsets(baskets, minimum):
    """Count all combinations without Apriori pruning.

    Parameters are baskets (a list of sets) and minimum (a support count).
    Returns a dictionary from itemset tuples to absolute counts. This slow
    approach is deliberately limited to tiny test databases so it provides
    an independent check of the faster candidate-generation method.
    """
    items = sorted(set().union(*baskets)) if baskets else []
    expected = {}
    for size in range(1, len(items) + 1):
        for itemset in combinations(items, size):
            count = sum(
                all(item in basket for item in itemset) for basket in baskets
            )
            if count >= minimum:
                expected[itemset] = count
    return expected


def check_mining_tables(case, results):
    """Independently recount returned mining tables.

    Parameters are case (a unittest.TestCase for assertions) and results
    (the dictionary returned by run_parts_5_to_10). Returns None and raises
    an assertion failure when counts or rule metrics disagree with baskets.
    """
    for row in results["frequent_itemsets"].itertuples():
        baskets = results["branch_baskets"][row.Branch]
        joint = sum(set(row.itemset) <= basket for basket in baskets)
        case.assertEqual(row.support, joint)
        case.assertAlmostEqual(row.relative_support, joint / len(baskets))
    for row in results["selected_rules"].itertuples():
        baskets = results["branch_baskets"][row.Branch]
        left = set(row.antecedent)
        joint = sum(left | {row.consequent} <= basket for basket in baskets)
        left_count = sum(left <= basket for basket in baskets)
        right_count = sum(row.consequent in basket for basket in baskets)
        confidence = joint / left_count
        case.assertEqual(row.support, joint)
        case.assertAlmostEqual(row.confidence, confidence)
        case.assertAlmostEqual(
            row.lift, confidence / (right_count / len(baskets))
        )


class TestDataPreparation(unittest.TestCase):
    """Check loading and cleaning without relying on the real dataset."""

    def test_loading_hash_and_schema(self):
        """No inputs; return None after checking CSV loading and SHA-256."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            make_reviews().to_csv(path, index=False, encoding="latin-1")
            raw, resolved = helper.load_raw_data(path)
            self.assertEqual(resolved, path)
            self.assertEqual(raw.loc[0, "Review_ID"], "001")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(helper.compute_file_digest(path), expected)
            schema = helper.inspect_raw_schema(raw)
            self.assertEqual(schema["shape"], (9, 6))
            self.assertEqual(len(schema["head"]), 5)
            self.assertEqual(len(schema["dtypes"]), 6)
            with self.assertRaises(FileNotFoundError):
                helper.load_raw_data(Path(directory) / "missing.csv")

    def test_cleaning_preserves_rows_and_input(self):
        """No inputs; return None after checking duplicates and row keys."""
        raw = make_reviews()
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        raw.index = [0] * len(raw)
        original = raw.copy(deep=True)
        cleaned, text, duplicate_count = helper.run_cleaning_pipeline(raw)
        pd.testing.assert_frame_equal(raw, original)
        self.assertEqual(duplicate_count, 1)
        self.assertEqual(len(cleaned), 9)
        self.assertEqual(len(text), 8)
        self.assertTrue(cleaned["row_id"].is_unique)
        self.assertEqual(cleaned["repeated_id"].sum(), 2)
        self.assertTrue(pd.isna(cleaned.loc[7, "rating"]))
        self.assertTrue(pd.isna(cleaned.loc[6, "review_month"]))
        self.assertEqual(cleaned.loc[7, "Branch"], "Unknown")

    def test_quality_and_branch_counts(self):
        """No inputs; return None after checking denominators and plots."""
        raw = make_reviews()
        quality = helper.assess_data_quality(raw).set_index("column")
        self.assertEqual(quality.loc["Review_ID", "nulls"], 1)
        cleaned, _, _ = helper.run_cleaning_pipeline(raw)
        summary = helper.compute_branch_summary(cleaned)
        self.assertEqual(summary.loc["B", "reviews"], 4)
        self.assertEqual(summary["reviews"].sum(), len(cleaned))
        self.assertAlmostEqual(summary["share_pct"].sum(), 100, places=1)
        ratings = helper.compute_rating_distribution(cleaned)
        np.testing.assert_allclose(ratings.sum(axis=1), 100, atol=0.2)
        locations = helper.compute_location_distribution(cleaned, top_n=1)
        self.assertEqual(locations.to_numpy().sum(), 5)
        figure = helper.plot_eda_charts(cleaned, ratings)
        self.assertEqual(len(figure.axes), 2)
        plt.close(figure)

    def test_text_preparation(self):
        """Take no inputs; return None after checking markup and negation."""
        text = helper.prepare_text("<b>Staff</b> weren   t rude &amp; cold.")
        self.assertEqual(text, "Staff weren't rude & cold.")
        self.assertEqual(helper.prepare_text(pd.NA), "")
        self.assertEqual(
            helper.clean_for_terms(text), "staff were not rude cold"
        )


class TestVectorization(unittest.TestCase):
    """Use hand-calculated counts and the documented TF-IDF formula."""

    def test_tfidf_against_manual_formula(self):
        """Take no inputs; return None after checking every matrix weight."""
        documents = ["cat cat dog", "dog bird", "bird"]
        result = helper.build_text_representations(
            pd.DataFrame({"clean_text": documents})
        )
        self.assertEqual(list(result["feature_names"]), ["bird", "cat", "dog"])
        counts = np.array([[0, 2, 1], [1, 0, 1], [1, 0, 0]], dtype=float)
        np.testing.assert_array_equal(result["bow_matrix"].toarray(), counts)
        # calculate smooth idf ourselves, then normalize each review
        document_frequency = (counts > 0).sum(axis=0)
        idf = np.array(
            [
                math.log((1 + len(documents)) / (1 + count)) + 1
                for count in document_frequency
            ]
        )
        weighted = counts * idf
        expected = weighted / np.sqrt((weighted**2).sum(axis=1))[:, None]
        np.testing.assert_allclose(
            result["tfidf_matrix"].toarray(), expected, atol=1e-12
        )
        np.testing.assert_allclose(result["tfidf_vec"].idf_, idf)
        self.assertAlmostEqual(result["sparsity"], 100 * 4 / 9)
        terms = helper.extract_top_tfidf_terms(
            result["tfidf_matrix"], result["feature_names"], top_n=1
        )
        self.assertEqual(terms.loc[0, "term"], "cat")
        top = helper.compute_corpus_top_terms(
            result["tfidf_matrix"], list(result["feature_names"]), top_n=3
        ).set_index("term")
        np.testing.assert_allclose(
            top.loc[["bird", "cat", "dog"], "mean_tfidf"],
            expected.mean(axis=0),
        )
        reviewed = helper.review_vocabulary(top.reset_index())
        self.assertEqual(len(reviewed), 3)
        self.assertTrue(
            reviewed["decision"].str.contains("no automatic").all()
        )

    def test_empty_and_stopword_only_corpora(self):
        """Take no inputs; return None after checking unusable text errors."""
        for documents in [[], ["the and or"]]:
            with self.assertRaises(ValueError):
                helper.build_text_representations(
                    pd.DataFrame({"clean_text": documents})
                )
        self.assertTrue(
            math.isnan(helper.calculate_sparsity(csr_matrix((0, 0))))
        )


class TestComplaintRules(unittest.TestCase):
    """Check explicit examples rather than claim measured language accuracy."""

    def test_complaint_boundaries(self):
        """No inputs; return None after checking expected phrase labels."""
        examples = {
            "Queues were long.": {"Long queues"},
            "The queue was not long.": set(),
            "There were no long queues.": set(),
            "The staff were rude.": {"Poor staff service"},
            "The staff weren't rude.": set(),
            "The hotel was dirty.": {"Hotel problems"},
            "The shuttle was unreliable.": {"Transport problems"},
            "The food was cold.": {"Food quality / dining"},
            "The food was not cold.": set(),
            "The tickets were too expensive.": {"High prices / poor value"},
            "Expensive but worth it.": set(),
            "The toilets were dirty.": {"Cleanliness"},
            "Space Mountain was closed.": {"Closures / breakdowns"},
            "Space Mountain was not closed.": set(),
            "The ride closed at 9.": set(),
            "If you encounter long queues, try another ride.": set(),
            "Queues were long, fastpass would be better.": {"Long queues"},
            "People were rude, cast members nearby.": set(),
        }
        for sentence, expected in examples.items():
            with self.subTest(sentence=sentence):
                actual = {
                    hit["category"]
                    for hit in helper.detect_complaints(sentence)
                    if hit["primary"]
                }
                self.assertEqual(actual, expected)

    def test_aliases_segments_and_sentiment(self):
        """Take no inputs; return None after checking small text helpers."""
        patterns = helper.compile_aliases({"ride": ["space mountain"]})
        self.assertEqual(
            helper.find_labels("SPACE MOUNTAIN", patterns), ["ride"]
        )
        self.assertEqual(helper.find_labels("space mountains", patterns), [])
        segments = helper.split_segments("Great food, but queues were long.")
        self.assertEqual(len(segments), 2)
        self.assertEqual(helper.sentiment_label(0.05), "positive")
        self.assertEqual(helper.sentiment_label(-0.05), "negative")
        self.assertEqual(helper.sentiment_label(0), "neutral")
        self.assertTrue(helper.negated_cue("not long", 4))

    def test_target_phrases_and_later_matches(self):
        """Take no inputs; return None after checking target attribution."""
        examples = [
            ("The food queues were long.", True),
            ("Food queues were not long; food queues were long later.", True),
            ("The food queues were not long.", False),
            ("If the food queues were long, go elsewhere.", False),
            ("Food was good and ride queues were long.", False),
        ]
        for sentence, expected in examples:
            match = helper.find_target_match(
                sentence, helper.TARGET_RULES["Dining:queue"], "Dining:queue"
            )
            self.assertEqual(match is not None, expected)


class TestFrequentItemsets(unittest.TestCase):
    """Compare class-style mining with exhaustive independent counting."""

    def test_apriori_against_exhaustive_search(self):
        """No inputs; return None after checking 120 random mining runs."""
        rng = random.Random(42)
        for _ in range(30):
            baskets = [
                {item for item in "abcd" if rng.random() < 0.5}
                for _ in range(20)
            ]
            for minimum in [1, 3, 10, 21]:
                self.assertEqual(
                    dict(helper.apriori(baskets, minimum)),
                    exhaustive_itemsets(baskets, minimum),
                )

    def test_known_support_confidence_lift(self):
        """Take no inputs; return None after checking a hand-counted rule."""
        baskets = [{"a", "b"}, {"a", "b"}, {"a"}, {"b"}, set()]
        self.assertEqual(helper.support_count(baskets, ("a", "b")), 2)
        rules = helper.assoc(helper.apriori(baskets, 2), 0.3, len(baskets))
        self.assertEqual(len(rules), 2)
        for rule in rules:
            self.assertEqual(rule["support"], 2)
            self.assertAlmostEqual(rule["confidence"], 2 / 3)
            self.assertAlmostEqual(rule["lift"], 10 / 9)
        self.assertEqual(helper.apriori([set(), set()], 1), [])
        self.assertEqual(helper.assoc([], 0.3, 0), [])

    def test_invalid_settings(self):
        """Take no inputs; return None after checking threshold errors."""
        for settings in [
            (0, 1, 0.3, 1),
            (1, 0, 0.3, 1),
            (1, 1, 2, 1),
            (1, 1, 0.3, float("nan")),
        ]:
            with self.assertRaises(ValueError):
                helper.validate_thresholds(*settings)
        with self.assertRaises(ValueError):
            helper.apriori([{"a"}], 0)
        with self.assertRaises(ValueError):
            helper.assoc([], -0.1, 1)


class TestCompletePipeline(unittest.TestCase):
    """Exercise section functions together, including empty results."""

    def test_sections_and_metrics(self):
        """Take no inputs; return None after independently checking tables."""
        _, text, _ = helper.run_cleaning_pipeline(make_reviews())
        unchanged = text.copy(deep=True)
        results = helper.run_parts_5_to_10(text, min_count=1)
        pd.testing.assert_frame_equal(text, unchanged)
        self.assertEqual(len(results["baskets_df"]), len(text))
        self.assertNotIn("Unknown", results["branch_baskets"])
        self.assertEqual(len(results["branch_baskets"]["A"]), 4)
        self.assertEqual(results["baskets_df"]["basket_size"].eq(0).sum(), 3)
        check_mining_tables(self, results)
        summary = results["complaint_summary"]
        queues = summary.loc[
            summary["Branch"].eq("A") & summary["category"].eq("Long queues")
        ].iloc[0]
        self.assertEqual(queues["reviews"], 3)
        self.assertEqual(queues["review_pct"], 75)
        pair = results["favorable_pairs"].iloc[0]
        self.assertEqual(pair["Branch"], "A")
        self.assertAlmostEqual(pair["high_support"], 1 / 3)
        self.assertEqual(pair["low_support"], 1)
        comparison = results["itemset_comparison"]
        self.assertEqual(
            comparison.loc[comparison["Branch"].eq("B"), "support"].sum(), 0
        )
        follow_up = results["recommendations"].set_index("Branch").loc["A"]
        self.assertEqual(follow_up["review_count"], 2)
        self.assertEqual(follow_up["gap_pp"], 50)
        # each subgroup has its own denominator, including unmatched reviews
        for row in results["rating_rules"].itertuples():
            valid = results["baskets_df"].loc[
                results["baskets_df"]["Branch"].eq(row.Branch)
                & results["baskets_df"]["rating"].between(1, 5)
            ]
            matched = valid["basket"].map(lambda b: set(row.antecedent) <= b)
            low = valid["rating"].le(2)
            self.assertEqual(row.joint_count, (matched & low).sum())
            self.assertAlmostEqual(
                row.confidence, (matched & low).sum() / matched.sum()
            )
            self.assertAlmostEqual(row.lift, row.confidence / low.mean())
        vectorized = helper.build_text_representations(text)
        self.assertTrue(
            helper.validate_analysis(text, results, vectorized)["passed"].all()
        )
        results["baskets_df"] = results["baskets_df"].iloc[::-1]
        with self.assertRaises(ValueError):
            helper.validate_analysis(text, results)

    def test_empty_no_match_and_unknown_branch(self):
        """Take no inputs; return None after checking sparse edge cases."""
        raw = make_reviews()
        for sample in [raw.iloc[:0], raw.iloc[[3]], raw.iloc[[7]]]:
            _, text, _ = helper.run_cleaning_pipeline(sample)
            results = helper.run_parts_5_to_10(text)
            self.assertTrue(results["selected_rules"].empty)
            self.assertTrue(results["validation"]["passed"].all())
            self.assertEqual(len(results["baskets_df"]), len(text))

    def test_optional_switch_and_exports(self):
        """Take no inputs; return None after checking opt-in disk exports."""
        _, text, _ = helper.run_cleaning_pipeline(make_reviews())
        results = helper.run_parts_5_to_10(text, include_optional=False)
        self.assertNotIn("rating_rules", results)
        with tempfile.TemporaryDirectory() as directory:
            saved = helper.export_analysis(results, directory)
            exported = pd.read_csv(saved / "baskets_df.csv")
            self.assertEqual(len(exported), len(text))
            self.assertEqual(
                set(json.loads(exported.loc[0, "basket"])),
                {"Long queues", "Poor staff service"},
            )
            metadata = json.loads((saved / "run_metadata.json").read_text())
            self.assertEqual(metadata["thresholds"]["min_count"], 30)

    def test_unique_keys_required(self):
        """No inputs; return None after rejecting ambiguous review keys."""
        _, text, _ = helper.run_cleaning_pipeline(make_reviews())
        text["row_id"] = 0
        with self.assertRaises(ValueError):
            helper.extract_review_evidence(text)


class TestDocumentation(unittest.TestCase):
    """Check executable style and documented public parameters."""

    def test_docstrings_cover_parameters_and_returns(self):
        """No inputs; return None after inspecting every helper function."""
        for name, function in inspect.getmembers(helper, inspect.isfunction):
            if function.__module__ != helper.__name__:
                continue
            doc = inspect.getdoc(function) or ""
            self.assertIn("Returns", doc, name)
            for parameter in inspect.signature(function).parameters:
                self.assertIn(parameter + " :", doc, (name, parameter))
        source = Path(helper.__file__).read_text(encoding="utf-8")
        ast.parse(source)

    def test_pep8(self):
        """Take no inputs; return None after pycodestyle checks both files."""
        try:
            import pycodestyle
        except ImportError:
            self.fail("Install pycodestyle to run the required PEP 8 check.")
        report = pycodestyle.StyleGuide(quiet=True).check_files(
            [helper.__file__, __file__]
        )
        self.assertEqual(report.total_errors, 0, report.get_statistics())


class TestFullDataset(unittest.TestCase):
    """Run only with --full, because full text extraction takes longer."""

    data_path = None
    enabled = False

    def test_full_dataset(self):
        """Take no inputs; return None after recounting real-data results."""
        if not self.enabled:
            self.skipTest("Use python test.py --full for the real-data check.")
        raw, _ = helper.load_raw_data(self.data_path)
        _, text, _ = helper.run_cleaning_pipeline(raw)
        results = helper.run_parts_5_to_10(text)
        vectorized = helper.build_text_representations(text)
        checks = helper.validate_analysis(text, results, vectorized)
        self.assertTrue(checks["passed"].all())
        check_mining_tables(self, results)
        for branch, baskets in results["branch_baskets"].items():
            minimum = max(30, math.ceil(0.01 * len(baskets)))
            expected = exhaustive_itemsets(baskets, minimum)
            actual = results["frequent_itemsets"].loc[
                results["frequent_itemsets"]["Branch"].eq(branch)
            ]
            self.assertEqual(
                dict(zip(actual["itemset"], actual["support"])), expected
            )
        print(
            f"\nFull data: {len(raw):,} raw rows, "
            f"{len(text):,} usable reviews, "
            f"{len(results['frequent_itemsets'])} frequent itemsets, "
            f"{len(results['selected_rules'])} selected rules."
        )


def main():
    """Read command-line flags, run selected checks and return an exit code.

    No function parameters are required. --full enables the CSV integration
    test and --data supplies its path. Returns 0 when all checks pass or 1
    when any fail, so this file can also be used in an automated check.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--data", default=None)
    arguments = parser.parse_args()
    loader = unittest.TestLoader()
    classes = [
        TestDataPreparation,
        TestVectorization,
        TestComplaintRules,
        TestFrequentItemsets,
        TestCompletePipeline,
        TestDocumentation,
    ]
    if arguments.full:
        TestFullDataset.data_path = arguments.data
        TestFullDataset.enabled = True
        classes.append(TestFullDataset)
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(cls) for cls in classes
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
