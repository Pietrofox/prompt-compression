"""
examples/demo.py
================
Demonstrates prompt compression on three task types:
  1. Sentiment classification
  2. Question answering
  3. Open-ended generation (LLM-as-judge)

Run:
    GROQ_API_KEY=your_key python examples/demo.py
    python examples/demo.py --mock          # offline, no API
    python examples/demo.py --task qa       # single task
    python examples/demo.py --strategy aggressive
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_compression import (
    PromptCompressor,
    ClassificationEvaluator,
    QAEvaluator,
    GenerationEvaluator,
    CompositeEvaluator,
    count_tokens,
)
from prompt_compression.utils import groq_llm, mock_llm


# ─────────────────────────────────────────────────────────────────── #
# Task 1: Sentiment classification                                    #
# ─────────────────────────────────────────────────────────────────── #

SENTIMENT_PROMPT = """
You are a sentiment analysis assistant. Your task is to carefully read
the provided text and determine the overall emotional tone expressed by
the author. You should consider the specific words used, any implied
meaning, and the general context of the statement.

Please classify the sentiment as one of the following three categories:
- positive: the text expresses a favourable, happy, or optimistic view
- negative: the text expresses an unfavourable, unhappy, or pessimistic view
- neutral: the text is factual, objective, or does not express a clear emotion

Respond with only the label: positive, negative, or neutral.
""".strip()

SENTIMENT_EXAMPLES = [
    ("I absolutely loved the restaurant, the food was incredible!", "positive"),
    ("The service was absolutely terrible and the food was cold.", "negative"),
    ("The package was delivered on Tuesday afternoon.", "neutral"),
    ("What an amazing experience, I'll definitely come back!", "positive"),
    ("I'm deeply disappointed with this product, total waste of money.", "negative"),
    ("The report contains data from the last fiscal quarter.", "neutral"),
    ("This is hands down the best coffee I've ever had.", "positive"),
    ("The app keeps crashing and support is unresponsive.", "negative"),
]


# ─────────────────────────────────────────────────────────────────── #
# Task 2: Question answering                                          #
# ─────────────────────────────────────────────────────────────────── #

QA_PROMPT = """
You are a knowledgeable assistant that provides accurate, concise answers
to factual questions. When answering, you should draw upon your knowledge
to provide the most accurate information possible. Please keep your answers
brief and to the point — typically one sentence or a few words is sufficient.
Do not include lengthy explanations unless specifically requested.
Focus on providing the factual answer directly without unnecessary preamble
such as "The answer is..." or "Based on my knowledge...".
""".strip()

QA_EXAMPLES = [
    ("What is the capital of France?", "Paris"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the chemical symbol for water?", "H2O"),
    ("In what year did World War II end?", "1945"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
]


# ─────────────────────────────────────────────────────────────────── #
# Task 3: Open-ended generation                                       #
# ─────────────────────────────────────────────────────────────────── #

GENERATION_PROMPT = """
You are a helpful writing assistant specialised in creating clear,
professional summaries. When given a topic or a piece of text, you should
produce a concise one-paragraph summary that captures the main points
accurately and engagingly. Your summaries should be written in a neutral,
informative tone appropriate for a professional audience. Avoid jargon
where possible and ensure the summary stands on its own without requiring
additional context. The summary should be between two and four sentences long.
""".strip()

GENERATION_EXAMPLES = [
    (
        "Explain the water cycle",
        "Should mention evaporation, condensation, precipitation, and be 2-4 sentences"
    ),
    (
        "Describe how photosynthesis works",
        "Should mention sunlight, CO2, water, glucose production, be accurate and clear"
    ),
    (
        "Summarise what machine learning is",
        "Should be accurate, mention learning from data, be suitable for a non-expert"
    ),
]


# ─────────────────────────────────────────────────────────────────── #
# Runner                                                              #
# ─────────────────────────────────────────────────────────────────── #

def run_task(name, prompt, evaluator, llm, strategy, plot):
    print(f"\n{'━'*60}")
    print(f"  TASK: {name.upper()}")
    print(f"  Original prompt: {count_tokens(prompt)} tokens")
    print(f"{'━'*60}")

    compressor = PromptCompressor(
        llm=llm,
        evaluator=evaluator,
        strategy=strategy,
        verbose=True,
    )
    result = compressor.compress(prompt)
    print(result.summary())

    if plot:
        try:
            from prompt_compression.visualise import plot_compression_curve
            plot_compression_curve(result, title=name, save_path=f"{name.lower().replace(' ', '_')}.png")
        except ImportError:
            print("  (matplotlib not installed — skipping plot)")

    return result


def run_strategy_comparison(llm, plot):
    """Compare all three strategies on the sentiment task."""
    print(f"\n{'━'*60}")
    print("  STRATEGY COMPARISON")
    print(f"{'━'*60}")

    evaluator = ClassificationEvaluator(SENTIMENT_EXAMPLES)
    results = {}
    for strategy in ["conservative", "balanced", "aggressive"]:
        comp = PromptCompressor(llm=llm, evaluator=evaluator,
                                strategy=strategy, verbose=False)
        result = comp.compress(SENTIMENT_PROMPT)
        results[strategy] = result
        print(f"  {strategy:<14}: -{result.compression_ratio*100:.0f}% tokens, "
              f"quality={result.final_quality:.2f}")

    if plot:
        try:
            from prompt_compression.visualise import plot_multi_strategy
            plot_multi_strategy(results, save_path="strategy_comparison.png")
        except ImportError:
            pass

    return results


def main():
    plot = "--no-plot" not in sys.argv
    use_mock = "--mock" in sys.argv
    strategy = "balanced"
    task_filter = None

    for arg in sys.argv[1:]:
        if arg.startswith("--strategy="):
            strategy = arg.split("=")[1]
        if arg.startswith("--task="):
            task_filter = arg.split("=")[1]

    if use_mock:
        print("Running with MOCK LLM (no API calls)\n")
        llm = mock_llm(seed=42)
    else:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Error: set GROQ_API_KEY or use --mock for offline demo.")
            sys.exit(1)
        llm = groq_llm(model="llama3-8b-8192", temperature=0.3)

    tasks = {
        "sentiment": (
            SENTIMENT_PROMPT,
            ClassificationEvaluator(SENTIMENT_EXAMPLES),
        ),
        "qa": (
            QA_PROMPT,
            QAEvaluator(QA_EXAMPLES),
        ),
        "generation": (
            GENERATION_PROMPT,
            GenerationEvaluator(GENERATION_EXAMPLES),
        ),
    }

    if task_filter:
        tasks = {k: v for k, v in tasks.items() if k == task_filter}

    for name, (prompt, evaluator) in tasks.items():
        run_task(name, prompt, evaluator, llm, strategy, plot)

    if not task_filter:
        run_strategy_comparison(llm, plot)


if __name__ == "__main__":
    main()
