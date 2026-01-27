/Users/usmri/Desktop/clm/meridian/evaluation/runner.py

```python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
```

Key design decisions:

- `EvalResult` is a plain dataclass (no Pydantic dependency in the eval layer) holding question, answer, contexts, ground_truth, scores dict, MLflow run_id, and an optional error string.
- `__init__` lazily imports `RAGASEvaluator` and `mlflow` so missing optional deps surface with clear import errors rather than silent failures at startup.
- `run_single` builds a fresh `ResearchState` with a UUID session_id, calls `graph.ainvoke(state)`, extracts `final_answer` and `all_findings[*].content` as contexts, then calls `evaluator.evaluate`. MLflow logging is wrapped in a broad try/except so a tracking failure never blocks an eval run.
- `run_benchmark` streams progress (`[i/N] question...`) to stdout before printing the summary table.
- `print_results` dynamically discovers all metric column names across all results, prints a fixed-width table, and appends an AVERAGE row.
- CLI supports `--question`, `--ground-truth`, `--benchmark`, and `--config`. Exits with code 1 if a single-question run errors.