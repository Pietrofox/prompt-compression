"""
compressor.py
=============
Iterative prompt compressor.

The core idea: repeatedly ask an LLM to shorten a prompt,
then score the compressed version on a held-out evaluation set.
Stop when quality drops below a threshold or the target ratio is reached.

Three compression strategies:
  - aggressive  : maximise token reduction, accept small quality loss
  - balanced    : good tradeoff between size and quality (default)
  - conservative: preserve quality at all costs, minimal compression

The compressor also exposes a step-by-step iterator so you can
plot the compression curve token-by-token.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .tokens import count_tokens, compression_ratio, token_cost_usd
from .evaluator import Evaluator, EvalResult


LLMFn = Callable[[str], str]


# ------------------------------------------------------------------ #
# Result types                                                        #
# ------------------------------------------------------------------ #

@dataclass
class CompressionStep:
    """Snapshot at one compression iteration."""
    iteration: int
    prompt: str
    tokens: int
    quality: float          # 0.0 – 1.0
    compression_ratio: float  # fraction of original tokens removed
    accepted: bool          # whether this step was kept


@dataclass
class CompressionResult:
    """
    Final output of a compression run.

    original_prompt   : the input prompt
    compressed_prompt : the best compressed version found
    steps             : full history of compression attempts
    final_quality     : score on eval set after compression
    tokens_saved      : absolute token reduction
    cost_saved_usd    : estimated API cost saving
    """
    original_prompt: str
    compressed_prompt: str
    steps: list[CompressionStep]
    final_quality: float
    original_tokens: int
    compressed_tokens: int

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def compression_ratio(self) -> float:
        return 1.0 - (self.compressed_tokens / self.original_tokens)

    @property
    def cost_saved_usd(self) -> float:
        return token_cost_usd(self.tokens_saved)

    def summary(self) -> str:
        return (
            f"\n{'='*52}\n"
            f"  Compression Result\n"
            f"{'='*52}\n"
            f"  Original tokens   : {self.original_tokens}\n"
            f"  Compressed tokens : {self.compressed_tokens}\n"
            f"  Tokens saved      : {self.tokens_saved} "
            f"({self.compression_ratio*100:.1f}%)\n"
            f"  Quality retained  : {self.final_quality*100:.1f}%\n"
            f"  Est. cost saving  : ${self.cost_saved_usd*1000:.4f} per 1000 calls\n"
            f"  Iterations        : {len(self.steps)}\n"
            f"{'='*52}\n"
            f"  Compressed prompt:\n"
            f"  {self.compressed_prompt}\n"
            f"{'='*52}"
        )


# ------------------------------------------------------------------ #
# Compression strategies                                              #
# ------------------------------------------------------------------ #

STRATEGIES = {
    "aggressive": {
        "min_quality":     0.70,   # accept up to 30% quality drop
        "target_ratio":    0.60,   # aim for 60% token reduction
        "max_iterations":  8,
        "temperature":     1.0,
    },
    "balanced": {
        "min_quality":     0.85,   # accept up to 15% quality drop
        "target_ratio":    0.40,   # aim for 40% token reduction
        "max_iterations":  6,
        "temperature":     0.8,
    },
    "conservative": {
        "min_quality":     0.95,   # accept only 5% quality drop
        "target_ratio":    0.20,   # aim for 20% token reduction
        "max_iterations":  4,
        "temperature":     0.5,
    },
}

# LLM instructions for each compression level
_COMPRESSION_PROMPTS = {
    1: (
        "Rewrite the following instruction to be more concise. "
        "Remove filler words and redundancy. Preserve all meaning."
    ),
    2: (
        "Compress the following instruction significantly. "
        "Keep only the essential information needed to perform the task. "
        "You may use abbreviations and telegraphic style."
    ),
    3: (
        "Compress this instruction to its absolute minimum. "
        "Use the shortest possible phrasing. "
        "Sacrifice stylistic clarity for brevity."
    ),
}


# ------------------------------------------------------------------ #
# Main compressor                                                     #
# ------------------------------------------------------------------ #

class PromptCompressor:
    """
    Iteratively compresses a prompt while monitoring quality on an eval set.

    Parameters
    ----------
    llm : callable
        (prompt: str) -> str. Used both for compression and evaluation.
    evaluator : Evaluator
        Scores a prompt against the eval set.
    strategy : str
        "aggressive", "balanced", or "conservative".
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        llm: LLMFn,
        evaluator: Evaluator,
        strategy: str = "balanced",
        verbose: bool = True,
    ):
        self.llm = llm
        self.evaluator = evaluator
        self.verbose = verbose

        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose: {list(STRATEGIES)}")
        self.config = STRATEGIES[strategy]
        self.strategy = strategy

    # ---------------------------------------------------------------- #
    # Public API                                                        #
    # ---------------------------------------------------------------- #

    def compress(self, prompt: str) -> CompressionResult:
        """
        Run the full compression loop.
        Returns the best compressed prompt found within quality constraints.
        """
        original_tokens = count_tokens(prompt)
        baseline_quality = self.evaluator.score(prompt, self.llm)

        if self.verbose:
            print(f"\n[Compressor] Strategy: {self.strategy}")
            print(f"[Compressor] Original: {original_tokens} tokens, "
                  f"quality={baseline_quality:.3f}")

        steps: list[CompressionStep] = []
        current_prompt = prompt
        best_prompt = prompt
        best_quality = baseline_quality

        for iteration in range(1, self.config["max_iterations"] + 1):
            # Choose compression intensity (escalates over iterations)
            intensity = min(3, 1 + (iteration - 1) // 2)
            compressed = self._compress_once(current_prompt, intensity)

            if not compressed or len(compressed) < 5:
                if self.verbose:
                    print(f"  Iter {iteration}: empty response, stopping.")
                break

            quality = self.evaluator.score(compressed, self.llm)
            ratio = compression_ratio(prompt, compressed)
            tokens = count_tokens(compressed)
            accepted = quality >= self.config["min_quality"]

            step = CompressionStep(
                iteration=iteration,
                prompt=compressed,
                tokens=tokens,
                quality=quality,
                compression_ratio=ratio,
                accepted=accepted,
            )
            steps.append(step)

            if self.verbose:
                status = "✓" if accepted else "✗"
                print(f"  Iter {iteration} {status}: {tokens} tokens "
                      f"(-{ratio*100:.0f}%), quality={quality:.3f}")

            if accepted:
                current_prompt = compressed
                if quality > best_quality * 0.98:  # keep best quality-adjusted
                    best_prompt = compressed
                    best_quality = quality

            # Stop if we hit the target ratio
            if ratio >= self.config["target_ratio"]:
                if self.verbose:
                    print(f"  Target ratio reached ({ratio*100:.0f}%), stopping.")
                break

        return CompressionResult(
            original_prompt=prompt,
            compressed_prompt=best_prompt,
            steps=steps,
            final_quality=best_quality,
            original_tokens=original_tokens,
            compressed_tokens=count_tokens(best_prompt),
        )

    def compress_iter(self, prompt: str) -> Iterator[CompressionStep]:
        """
        Generator version — yields one CompressionStep per iteration.
        Useful for live progress displays or early stopping.

        Usage:
            for step in compressor.compress_iter(prompt):
                print(step.tokens, step.quality)
                if step.quality < 0.8:
                    break
        """
        current = prompt
        for iteration in range(1, self.config["max_iterations"] + 1):
            intensity = min(3, 1 + (iteration - 1) // 2)
            compressed = self._compress_once(current, intensity)
            if not compressed:
                return
            quality = self.evaluator.score(compressed, self.llm)
            ratio = compression_ratio(prompt, compressed)
            step = CompressionStep(
                iteration=iteration,
                prompt=compressed,
                tokens=count_tokens(compressed),
                quality=quality,
                compression_ratio=ratio,
                accepted=quality >= self.config["min_quality"],
            )
            yield step
            if step.accepted:
                current = compressed
            if ratio >= self.config["target_ratio"]:
                return

    # ---------------------------------------------------------------- #
    # Internal                                                          #
    # ---------------------------------------------------------------- #

    def _compress_once(self, prompt: str, intensity: int) -> str:
        """Ask the LLM to compress the prompt at a given intensity level."""
        instruction = _COMPRESSION_PROMPTS[intensity]
        meta_prompt = (
            f"{instruction}\n\n"
            f"PROMPT TO COMPRESS:\n{prompt}\n\n"
            f"Output only the compressed prompt, nothing else."
        )
        return self.llm(meta_prompt).strip()
