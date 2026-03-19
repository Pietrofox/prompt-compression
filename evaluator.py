"""
evaluator.py
============
Task-specific evaluators that score a prompt on a held-out example set.

An evaluator answers: "How well does this prompt perform on this task?"
It drives the compression loop — the compressor only accepts a
compressed prompt if the evaluator says quality is still acceptable.

Built-in evaluators:
  ClassificationEvaluator  — exact label match (sentiment, categories)
  QAEvaluator              — substring match on short factual answers
  GenerationEvaluator      — LLM-as-judge for open-ended generation
  CompositeEvaluator       — weighted average of multiple evaluators
"""

from __future__ import annotations
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

LLMFn = Callable[[str], str]
Example = tuple[str, str]   # (input_text, expected_output)


# ------------------------------------------------------------------ #
# Base class                                                          #
# ------------------------------------------------------------------ #

@dataclass
class EvalResult:
    score: float            # 0.0 – 1.0
    n_examples: int
    details: list[dict]     # per-example breakdown

    def __str__(self) -> str:
        return f"EvalResult(score={self.score:.3f}, n={self.n_examples})"


class Evaluator(ABC):
    """Abstract base for all evaluators."""

    name: str = "base"

    @abstractmethod
    def score(self, prompt: str, llm: LLMFn) -> float:
        """Return a quality score in [0.0, 1.0]."""

    def evaluate(self, prompt: str, llm: LLMFn) -> EvalResult:
        """Full evaluation with per-example breakdown."""
        raise NotImplementedError


# ------------------------------------------------------------------ #
# 1. Classification evaluator                                         #
# ------------------------------------------------------------------ #

class ClassificationEvaluator(Evaluator):
    """
    Scores a prompt on a classification task (sentiment, topic, intent).
    Score = fraction of correct labels (case-insensitive, stripped).

    examples: list of (input_text, expected_label)

    Usage:
        eval = ClassificationEvaluator(examples=[
            ("Great movie!", "positive"),
            ("Terrible service.", "negative"),
        ])
    """

    name = "classification"

    def __init__(self, examples: list[Example]):
        self.examples = examples

    def score(self, prompt: str, llm: LLMFn) -> float:
        correct = 0
        for inp, expected in self.examples:
            response = llm(f"{prompt}\n\nText: {inp}\nLabel:").strip().lower()
            # Accept if expected label appears anywhere in response
            if expected.strip().lower() in response:
                correct += 1
        return correct / len(self.examples) if self.examples else 0.0

    def evaluate(self, prompt: str, llm: LLMFn) -> EvalResult:
        details = []
        correct = 0
        for inp, expected in self.examples:
            response = llm(f"{prompt}\n\nText: {inp}\nLabel:").strip()
            ok = expected.strip().lower() in response.lower()
            if ok:
                correct += 1
            details.append({
                "input": inp[:60],
                "expected": expected,
                "got": response[:60],
                "correct": ok,
            })
        score = correct / len(self.examples) if self.examples else 0.0
        return EvalResult(score=score, n_examples=len(self.examples), details=details)


# ------------------------------------------------------------------ #
# 2. QA evaluator                                                     #
# ------------------------------------------------------------------ #

class QAEvaluator(Evaluator):
    """
    Scores a prompt on question-answering tasks.
    Score = fraction of answers where expected appears in response.

    examples: list of (question, expected_answer)
    """

    name = "qa"

    def __init__(self, examples: list[Example]):
        self.examples = examples

    def score(self, prompt: str, llm: LLMFn) -> float:
        correct = 0
        for question, expected in self.examples:
            response = llm(f"{prompt}\n\nQuestion: {question}\nAnswer:").strip().lower()
            if expected.strip().lower() in response:
                correct += 1
        return correct / len(self.examples) if self.examples else 0.0

    def evaluate(self, prompt: str, llm: LLMFn) -> EvalResult:
        details = []
        correct = 0
        for question, expected in self.examples:
            response = llm(f"{prompt}\n\nQuestion: {question}\nAnswer:").strip()
            ok = expected.strip().lower() in response.lower()
            if ok:
                correct += 1
            details.append({
                "question": question[:60],
                "expected": expected,
                "got": response[:60],
                "correct": ok,
            })
        score = correct / len(self.examples) if self.examples else 0.0
        return EvalResult(score=score, n_examples=len(self.examples), details=details)


# ------------------------------------------------------------------ #
# 3. Generation evaluator (LLM-as-judge)                             #
# ------------------------------------------------------------------ #

class GenerationEvaluator(Evaluator):
    """
    Uses a second LLM call to judge quality of open-ended generation.
    The judge scores 0–10 on: relevance, completeness, clarity.

    examples: list of (instruction_input, quality_criterion)
    where quality_criterion describes what a good response looks like.
    """

    name = "generation"

    JUDGE_TEMPLATE = (
        "Rate the quality of the following response on a scale from 0 to 10.\n\n"
        "Task instruction: {prompt}\n"
        "Input: {inp}\n"
        "Quality criterion: {criterion}\n"
        "Response: {response}\n\n"
        "Reply with only a single integer from 0 to 10. No explanation."
    )

    def __init__(self, examples: list[tuple[str, str]]):
        """examples: [(input_text, quality_criterion), ...]"""
        self.examples = examples

    def score(self, prompt: str, llm: LLMFn) -> float:
        total = 0.0
        for inp, criterion in self.examples:
            response = llm(f"{prompt}\n\nInput: {inp}").strip()
            judge = self.JUDGE_TEMPLATE.format(
                prompt=prompt, inp=inp, criterion=criterion, response=response
            )
            raw = llm(judge).strip()
            match = re.search(r"\d+", raw)
            score = int(match.group()) if match else 5
            total += min(max(score, 0), 10) / 10.0
        return total / len(self.examples) if self.examples else 0.0

    def evaluate(self, prompt: str, llm: LLMFn) -> EvalResult:
        details = []
        total = 0.0
        for inp, criterion in self.examples:
            response = llm(f"{prompt}\n\nInput: {inp}").strip()
            judge = self.JUDGE_TEMPLATE.format(
                prompt=prompt, inp=inp, criterion=criterion, response=response
            )
            raw = llm(judge).strip()
            match = re.search(r"\d+", raw)
            score = int(match.group()) if match else 5
            score = min(max(score, 0), 10)
            total += score / 10.0
            details.append({
                "input": inp[:60],
                "criterion": criterion[:60],
                "response_preview": response[:80],
                "score": score,
            })
        final = total / len(self.examples) if self.examples else 0.0
        return EvalResult(score=final, n_examples=len(self.examples), details=details)


# ------------------------------------------------------------------ #
# 4. Composite evaluator                                             #
# ------------------------------------------------------------------ #

class CompositeEvaluator(Evaluator):
    """
    Weighted average of multiple evaluators.
    Useful when a prompt must perform well on heterogeneous tasks.

    Usage:
        composite = CompositeEvaluator([
            (ClassificationEvaluator(cls_examples), 0.5),
            (QAEvaluator(qa_examples), 0.5),
        ])
    """

    name = "composite"

    def __init__(self, evaluators: list[tuple[Evaluator, float]]):
        self.evaluators = evaluators
        total_weight = sum(w for _, w in evaluators)
        # Normalise weights
        self.evaluators = [(e, w / total_weight) for e, w in evaluators]

    def score(self, prompt: str, llm: LLMFn) -> float:
        return sum(e.score(prompt, llm) * w for e, w in self.evaluators)

    def evaluate(self, prompt: str, llm: LLMFn) -> EvalResult:
        total = 0.0
        all_details = []
        n = 0
        for evaluator, weight in self.evaluators:
            result = evaluator.evaluate(prompt, llm)
            total += result.score * weight
            all_details.extend(result.details)
            n += result.n_examples
        return EvalResult(score=total, n_examples=n, details=all_details)
