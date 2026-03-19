# prompt-compression

> **Iterative prompt compressor with quality-aware stopping — reduce LLM costs without sacrificing accuracy.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

Prompts are expensive. A well-engineered prompt often contains 200–500 tokens of context, instructions, and examples. Over millions of API calls, that adds up fast.

`prompt-compression` iteratively rewrites a prompt to be shorter, scores the result on a held-out evaluation set, and stops when quality drops below a threshold. It finds the **Pareto frontier** between token cost and task quality.

---

## How it works

```
Original prompt (400 tokens)
        │
        ▼
┌─────────────────────────────────┐
│  Compression step               │
│  LLM rewrites prompt shorter    │
│  (3 intensity levels)           │
└────────────┬────────────────────┘
             │
        ▼
┌─────────────────────────────────┐
│  Quality evaluation             │
│  Score on held-out examples     │
│  (classification / QA / judge)  │
└────────────┬────────────────────┘
             │
     quality ≥ threshold?
       yes → keep, iterate
       no  → stop, return best
```

Three strategies control the tradeoff:

| Strategy | Min quality | Target reduction | Use when |
|----------|-------------|-----------------|----------|
| `conservative` | 95% | 20% | Quality-critical tasks |
| `balanced` | 85% | 40% | Default, most tasks |
| `aggressive` | 70% | 60% | Cost-critical, tolerant tasks |

---

## Quick start

```bash
git clone https://github.com/your-username/prompt-compression
cd prompt-compression
pip install groq   # only if using Groq API
```

**With Groq:**
```python
import os
from prompt_compression import PromptCompressor, ClassificationEvaluator
from prompt_compression.utils import groq_llm

llm = groq_llm()   # set GROQ_API_KEY env var

long_prompt = """
You are a sentiment analysis assistant. Your task is to carefully read
the provided text and determine the overall emotional tone expressed by
the author. Please classify the sentiment as positive, negative, or neutral.
Respond with only the label.
""".strip()

evaluator = ClassificationEvaluator(examples=[
    ("Great product!", "positive"),
    ("Terrible service.", "negative"),
    ("Item arrived Tuesday.", "neutral"),
])

compressor = PromptCompressor(llm=llm, evaluator=evaluator, strategy="balanced")
result = compressor.compress(long_prompt)
print(result.summary())

# Original tokens   : 67
# Compressed tokens : 12
# Tokens saved      : 55 (82.1%)
# Quality retained  : 87.5%
# Est. cost saving  : $0.0055 per 1000 calls
#
# Compressed prompt:
# Classify sentiment: positive, negative, or neutral.
```

**Offline demo (no API key):**
```bash
python examples/demo.py --mock
```

---

## Three task types

### Classification
```python
from prompt_compression import ClassificationEvaluator

evaluator = ClassificationEvaluator(examples=[
    ("Great movie!", "positive"),
    ("Terrible service.", "negative"),
])
```

### Question answering
```python
from prompt_compression import QAEvaluator

evaluator = QAEvaluator(examples=[
    ("Capital of France?", "Paris"),
    ("Sides of a hexagon?", "6"),
])
```

### Open-ended generation (LLM-as-judge)
```python
from prompt_compression import GenerationEvaluator

evaluator = GenerationEvaluator(examples=[
    ("Explain photosynthesis", "Should mention sunlight, CO2, glucose"),
])
```

### Composite (multiple tasks)
```python
from prompt_compression import CompositeEvaluator

evaluator = CompositeEvaluator([
    (ClassificationEvaluator(cls_examples), 0.6),
    (QAEvaluator(qa_examples), 0.4),
])
```

---

## Architecture

```
prompt_compression/
├── __init__.py      public API
├── compressor.py    PromptCompressor — iterative compression loop
├── evaluator.py     4 evaluator types + base class
├── tokens.py        token counting (tiktoken or approximation)
└── visualise.py     compression curve plots (matplotlib)
```

---

## Step-by-step iterator

For live monitoring or custom stopping logic:

```python
for step in compressor.compress_iter(prompt):
    print(f"iter {step.iteration}: {step.tokens} tokens, quality={step.quality:.2f}")
    if step.quality < 0.80:
        break
```

---

## Running examples

```bash
python examples/demo.py --mock                     # offline
GROQ_API_KEY=... python examples/demo.py           # all tasks
GROQ_API_KEY=... python examples/demo.py --task=qa
GROQ_API_KEY=... python examples/demo.py --strategy=aggressive
```

---

## Running tests

```bash
python -m pytest tests/ -v
python tests/test_compression.py
```

18 tests, all offline.

---

## License

MIT
