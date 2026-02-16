# Evaluation Guide

The evaluation harness measures how well the GraphRAG system answers questions. It scores three dimensions: **groundedness**, **relevance**, and **refusal correctness**.

## Quick Start

```bash
# Run the built-in sample dataset
python scripts/eval.py --suite sample_qna --output reports/

# Run a custom dataset
python scripts/eval.py --dataset path/to/my_dataset.yaml --output reports/
```

Reports are saved in three formats:

- `reports/<suite>.json` -- machine-readable results
- `reports/<suite>.html` -- visual report with metrics and per-question table
- `reports/<suite>.txt` -- plain-text summary

---

## Metrics

### Groundedness

Measures whether the answer text is supported by the retrieved chunks. The scorer splits the answer into sentences, computes the token overlap between each sentence and the combined chunk content, and returns the fraction of sentences that exceed a support threshold (default 0.3).

- **Score range**: 0.0 (nothing supported) to 1.0 (fully supported)
- **Target**: > 90%

### Relevance

Measures whether the answer addresses the original question. The scorer extracts meaningful tokens from the question (after removing stop-words), checks which appear in the answer, and computes a weighted recall score. Rarer terms receive higher weight via an IDF-like formula. An optional length penalty reduces the score when the answer is very short relative to the question.

- **Score range**: 0.0 (completely off-topic) to 1.0 (all question keywords addressed)
- **Target**: > 85%

### Refusal Correctness

Checks whether the system's decision to answer or refuse matched the expected outcome defined in the dataset. There are four possible outcomes:

| Expected | Actual  | Verdict                        |
|----------|---------|--------------------------------|
| Answer   | Answer  | Correct (true negative)        |
| Refuse   | Refuse  | Correct (true positive)        |
| Answer   | Refuse  | Incorrect (false positive)     |
| Refuse   | Answer  | Incorrect (false negative)     |

- **Score**: fraction of questions where the decision was correct
- **Target**: > 95%

---

## Evaluation Datasets

A dataset is a YAML or JSON file containing a list of questions with expected outcomes.

### Built-in Datasets

| Suite Name   | File                                | Questions | Description                     |
|--------------|-------------------------------------|-----------|---------------------------------|
| `sample_qna` | `src/eval/datasets/sample_qna.yaml` | 10        | Mix of factual, procedural, and refusal questions |

### Dataset Schema

```yaml
name: my_dataset
description: Optional human-readable description
version: "1.0"

questions:
  - question: "What is federalism?"
    expected_answer_contains: ["system of government", "power"]
    expected_citations_min: 1
    expected_refusal: false
    tags: ["factual", "government"]
    difficulty: easy

  - question: "What is the price of Bitcoin today?"
    expected_refusal: true
    refusal_reason: "not in corpus — live data"
    tags: ["refusal", "finance"]
    difficulty: easy
```

### Question Fields

| Field                       | Type       | Required | Default | Description                                           |
|-----------------------------|------------|----------|---------|-------------------------------------------------------|
| `question`                  | string     | yes      | --      | The question to ask (must be non-empty)               |
| `expected_answer_contains`  | list[str]  | no       | `[]`    | Substrings the answer must contain (case-insensitive) |
| `expected_citations_min`    | int        | no       | `0`     | Minimum citations expected                            |
| `expected_refusal`          | bool       | no       | `false` | Whether the system should refuse                      |
| `refusal_reason`            | string     | no       | `null`  | Documentary note (not used for scoring)               |
| `tags`                      | list[str]  | no       | `[]`    | Free-form tags for filtering                          |
| `difficulty`                | string     | no       | `null`  | `easy`, `medium`, or `hard`                           |

**Validation rules:**
- `question` must not be empty
- `expected_citations_min` must be >= 0
- `difficulty` must be `easy`, `medium`, or `hard` (or omitted)
- When `expected_refusal` is `true`, `expected_answer_contains` must be empty

---

## Creating a Custom Dataset

1. Create a YAML file following the schema above:

```yaml
name: my_custom_eval
description: Evaluation for my specific corpus
version: "1.0"

questions:
  - question: "What causes climate change?"
    expected_answer_contains: ["greenhouse", "carbon"]
    expected_citations_min: 1
    tags: ["factual", "science"]
    difficulty: medium

  - question: "What is the best pizza topping?"
    expected_refusal: true
    refusal_reason: "subjective — not in corpus"
    tags: ["refusal"]
```

2. Run the evaluation:

```bash
python scripts/eval.py --dataset my_custom_eval.yaml --output reports/
```

3. View the report:

```bash
# Open the HTML report in a browser
open reports/my_custom_eval.html

# Or inspect the JSON
cat reports/my_custom_eval.json | python -m json.tool
```

---

## CLI Options

```
python scripts/eval.py [OPTIONS]

Options:
  --suite TEXT       Built-in suite name (e.g. sample_qna)
  --dataset TEXT     Path to a custom dataset (YAML or JSON)
  --output TEXT      Directory to write report files
  --format TEXT      Output format: text, json, html, or all (default: all)
  --baseline TEXT    Path to a baseline report JSON for comparison
  --verbose          Enable debug logging
```

### Baseline Comparison

Compare current results against a previous run:

```bash
# First run — save as baseline
python scripts/eval.py --suite sample_qna --output reports/

# Later run — compare
python scripts/eval.py --suite sample_qna --baseline reports/sample_qna.json
```

Output includes deltas:

```
  Avg groundedness:   0.9200 (+0.0300)
  Avg relevance:      0.8800 (-0.0100)
  Refusal accuracy:   100.00% (=)
```

---

## Programmatic Usage

```python
from src.eval import EvalRunner, load_dataset

# Load a dataset
dataset = load_dataset("src/eval/datasets/sample_qna.yaml")

# Run evaluation (graph is any object with an invoke() method)
runner = EvalRunner()
report = runner.run(dataset, graph)

# Inspect results
print(report.metrics.avg_groundedness)
print(report.metrics.avg_relevance)
print(report.metrics.refusal_accuracy)

# Per-question detail
for r in report.results:
    print(f"{r.question[:50]}  G={r.groundedness}  R={r.relevance}")

# Export
report.to_json()   # JSON string
report.to_html()   # self-contained HTML page
report.to_dict()   # plain dict
```

### Custom Scorers

```python
from src.eval.scorers import GroundednessScorer, RelevanceScorer

# Use a stricter groundedness threshold
runner = EvalRunner(
    groundedness_scorer=GroundednessScorer(threshold=0.5),
    relevance_scorer=RelevanceScorer(length_penalty=False),
)
report = runner.run(dataset, graph)
```

### Filtering Questions

```python
dataset = load_dataset("src/eval/datasets/sample_qna.yaml")

# Filter by tag
factual = dataset.filter_by_tag("factual")

# Filter by difficulty
hard = dataset.filter_by_difficulty("hard")

# Refusal / answerable subsets
refusals = dataset.refusal_questions
answerable = dataset.answerable_questions
```

---

## Report Structure

The JSON report has this structure:

```json
{
  "suite_name": "sample_qna",
  "created_at": "2025-01-15T10:30:00+00:00",
  "metrics": {
    "total_questions": 10,
    "answered_count": 7,
    "refused_count": 3,
    "error_count": 0,
    "avg_groundedness": 0.9200,
    "avg_relevance": 0.8800,
    "refusal_accuracy": 1.0000,
    "avg_latency_ms": 1250.00
  },
  "results": [
    {
      "question": "What is federalism?",
      "answer": "Federalism is a system of government...",
      "refusal_reason": null,
      "groundedness": 0.95,
      "relevance": 0.90,
      "refusal_correct": true,
      "expected_refusal": false,
      "latency_ms": 1100.50,
      "error": null
    }
  ]
}
```

---

## Troubleshooting

**All scores are 0.0**: The graph may not be returning chunks in the state. Ensure your graph populates the `chunks` key with `Chunk` objects.

**Refusal accuracy is low**: Check that your dataset correctly marks which questions should be refused (`expected_refusal: true`).

**Evaluation errors**: If questions fail with errors, the runner captures them per-question and continues. Check `report.results[i].error` for details.

**Slow evaluation**: Each question invokes the full pipeline (retrieval + LLM). Reduce the dataset size or use a faster LLM provider for initial development.
