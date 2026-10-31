"""Summarize runs and scores per configuration, write results/summary.json, log to MLflow."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _mean_ci(values: list[float]) -> dict:
    v = [x for x in values if x is not None]
    if not v:
        return {"mean": None, "ci95": None, "n": 0}
    m = statistics.mean(v)
    half = 1.96 * statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
    return {"mean": round(m, 3), "ci95": round(half, 3), "n": len(v)}


def summarize(config: str) -> dict:
    runs = {r["id"]: r for r in _load(RESULTS / "runs" / f"{config}.jsonl")}
    scores = {s["id"]: s for s in _load(RESULTS / "scores" / f"{config}.jsonl")}
    if not runs:
        return {}
    tokens = [sum(u["prompt_tokens"] + u["completion_tokens"] for u in r["usage"].values()) for r in runs.values()]
    calls = [sum(u["calls"] for u in r["usage"].values()) for r in runs.values()]
    secs = sorted(r["seconds"] for r in runs.values())
    cited = [len(r["citations"]) for r in runs.values()]
    invalid = sum(len(r["invalid_citations"]) for r in runs.values())
    total_cites = sum(cited) + invalid
    out = {
        "questions": len(runs),
        "scored": len(scores),
        "faithfulness": _mean_ci([s["faithfulness"] for s in scores.values()]),
        "answer_relevancy": _mean_ci([s["answer_relevancy"] for s in scores.values()]),
        "context_recall": _mean_ci([s["context_recall"] for s in scores.values()]),
        "seconds_p50": secs[len(secs) // 2],
        "seconds_p90": secs[min(len(secs) - 1, int(0.9 * len(secs)))],
        "llm_calls_per_question": round(statistics.mean(calls), 1),
        "tokens_per_question": round(statistics.mean(tokens)),
        "sources_cited_per_answer": round(statistics.mean(cited), 1),
        "invalid_citation_rate": round(invalid / total_cites, 4) if total_cites else 0.0,
        "critic_loops_triggered": sum(1 for r in runs.values() if r["critic_iterations"] > 1),
        "tool_errors": sum(len(r["tool_errors"]) for r in runs.values()),
    }
    if out["faithfulness"]["mean"] is not None:
        out["hallucination_rate"] = round(1 - out["faithfulness"]["mean"], 3)
    return out


def report(settings, log_mlflow: bool = True) -> dict:
    summary = {c: summarize(c) for c in ("critic", "no_critic")}
    summary["models"] = settings.get("llm.models")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    if log_mlflow:
        import mlflow

        mlflow.set_tracking_uri(f"file://{ROOT / 'data' / 'mlruns'}")
        mlflow.set_experiment("meridian-eval")
        for config in ("critic", "no_critic"):
            s = summary[config]
            if not s:
                continue
            with mlflow.start_run(run_name=config):
                mlflow.log_params({"config": config, **{f"model_{k}": v for k, v in settings.get("llm.models").items()},
                                   "questions": s["questions"]})
                for k, v in s.items():
                    if isinstance(v, dict) and v.get("mean") is not None:
                        mlflow.log_metric(k, v["mean"])
                    elif isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                mlflow.log_artifact(str(RESULTS / "summary.json"))
    return summary
