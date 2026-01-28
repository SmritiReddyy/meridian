/Users/usmri/Desktop/clm/meridian/tests/test_evaluation.py

```python
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluation.ragas_eval import EvalResult, RAGASEvaluator
```

The file covers:

**TestRAGASEvaluator**
- `test_evaluate_returns_scores` — patches `ragas.evaluate` (via `evaluation.ragas_eval.evaluate`) to return a pandas-backed mock and patches `anthropic.Anthropic` so `score_hallucination` returns 0.88; asserts every field of `EvalResult` is populated correctly.
- `test_hallucination_scorer_high` — Claude mock returns `"0.95"`; asserts `score_hallucination` returns `0.95`.
- `test_hallucination_scorer_low` — Claude mock returns `"0.1"`; asserts `score_hallucination` returns `0.1`.
- `test_overall_score_weighted` — runs two sub-cases (with and without `ground_truth`) using exact known inputs and validates the weighted formula from the source (`0.35/0.30/0.20/0.15` and `0.40/0.35/0.25`).

**TestEvaluationRunner**
- `test_run_single_logs_to_mlflow` — patches `graph.ainvoke`, `RAGASEvaluator`, and `mlflow`; asserts `mlflow.log_metric` or `mlflow.log_metrics` was called.
- `test_run_benchmark_processes_all` — writes a two-line JSONL to a temp file, patches `run_single`, and asserts both questions were dispatched and two results returned.