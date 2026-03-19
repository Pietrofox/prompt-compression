"""
tests/test_compression.py
==========================
Unit tests — all offline, mock LLM only.

Run:
    python -m pytest tests/ -v
    python tests/test_compression.py
"""

import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_compression import (
    PromptCompressor, CompressionResult,
    ClassificationEvaluator, QAEvaluator,
    count_tokens, compression_ratio, token_cost_usd,
)
from prompt_compression.utils import mock_llm


LONG_PROMPT = (
    "You are a helpful sentiment analysis assistant. Your task is to carefully "
    "read the provided text and determine the overall emotional tone expressed "
    "by the author. Please classify the sentiment as positive, negative, or neutral. "
    "Respond with only the label."
)

SENTIMENT_EXAMPLES = [
    ("Great product!", "positive"),
    ("Terrible service.", "negative"),
    ("The item arrived.", "neutral"),
]

QA_EXAMPLES = [
    ("Capital of France?", "Paris"),
    ("How many sides in a hexagon?", "6"),
]


class TestTokens(unittest.TestCase):

    def test_count_tokens_positive(self):
        self.assertGreater(count_tokens("Hello world"), 0)

    def test_count_tokens_empty(self):
        self.assertGreaterEqual(count_tokens(""), 0)

    def test_compression_ratio_range(self):
        original = "This is a long prompt with many words."
        compressed = "Short prompt."
        ratio = compression_ratio(original, compressed)
        self.assertGreater(ratio, 0)
        self.assertLess(ratio, 1)

    def test_compression_ratio_no_change(self):
        text = "Same text."
        ratio = compression_ratio(text, text)
        self.assertAlmostEqual(ratio, 0.0, places=1)

    def test_token_cost_usd(self):
        cost = token_cost_usd(1_000_000)
        self.assertAlmostEqual(cost, 0.10, places=5)


class TestEvaluators(unittest.TestCase):

    def setUp(self):
        self.llm = mock_llm(seed=0)

    def test_classification_score_range(self):
        ev = ClassificationEvaluator(SENTIMENT_EXAMPLES)
        score = ev.score("Classify sentiment.", self.llm)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_classification_evaluate_returns_result(self):
        from prompt_compression import EvalResult
        ev = ClassificationEvaluator(SENTIMENT_EXAMPLES)
        result = ev.evaluate("Classify.", self.llm)
        self.assertIsInstance(result, EvalResult)
        self.assertEqual(result.n_examples, 3)

    def test_qa_score_range(self):
        ev = QAEvaluator(QA_EXAMPLES)
        score = ev.score("Answer briefly.", self.llm)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_composite_evaluator(self):
        from prompt_compression import CompositeEvaluator
        comp = CompositeEvaluator([
            (ClassificationEvaluator(SENTIMENT_EXAMPLES), 0.6),
            (QAEvaluator(QA_EXAMPLES), 0.4),
        ])
        score = comp.score("Answer the task.", self.llm)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_empty_examples_returns_zero(self):
        ev = ClassificationEvaluator([])
        self.assertEqual(ev.score("any prompt", self.llm), 0.0)


class TestCompressor(unittest.TestCase):

    def setUp(self):
        self.llm = mock_llm(seed=42)
        self.evaluator = ClassificationEvaluator(SENTIMENT_EXAMPLES)

    def test_compress_returns_result(self):
        comp = PromptCompressor(self.llm, self.evaluator, strategy="balanced", verbose=False)
        result = comp.compress(LONG_PROMPT)
        self.assertIsInstance(result, CompressionResult)

    def test_compressed_prompt_not_empty(self):
        comp = PromptCompressor(self.llm, self.evaluator, strategy="balanced", verbose=False)
        result = comp.compress(LONG_PROMPT)
        self.assertGreater(len(result.compressed_prompt), 0)

    def test_tokens_saved_non_negative(self):
        comp = PromptCompressor(self.llm, self.evaluator, strategy="balanced", verbose=False)
        result = comp.compress(LONG_PROMPT)
        self.assertGreaterEqual(result.tokens_saved, 0)

    def test_compression_ratio_in_range(self):
        comp = PromptCompressor(self.llm, self.evaluator, strategy="aggressive", verbose=False)
        result = comp.compress(LONG_PROMPT)
        self.assertGreaterEqual(result.compression_ratio, 0.0)
        self.assertLessEqual(result.compression_ratio, 1.0)

    def test_all_strategies_work(self):
        for strategy in ["conservative", "balanced", "aggressive"]:
            comp = PromptCompressor(self.llm, self.evaluator,
                                    strategy=strategy, verbose=False)
            result = comp.compress(LONG_PROMPT)
            self.assertIsInstance(result, CompressionResult)

    def test_compress_iter_yields_steps(self):
        comp = PromptCompressor(self.llm, self.evaluator, strategy="balanced", verbose=False)
        steps = list(comp.compress_iter(LONG_PROMPT))
        self.assertGreater(len(steps), 0)
        for step in steps:
            self.assertGreaterEqual(step.quality, 0.0)
            self.assertLessEqual(step.quality, 1.0)

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            PromptCompressor(self.llm, self.evaluator, strategy="nonexistent")

    def test_summary_string(self):
        comp = PromptCompressor(self.llm, self.evaluator, verbose=False)
        result = comp.compress(LONG_PROMPT)
        summary = result.summary()
        self.assertIn("Compression Result", summary)
        self.assertIn("tokens", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
