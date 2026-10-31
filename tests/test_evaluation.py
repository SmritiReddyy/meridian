import json

import evaluation.report as report


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_summary_aggregates_runs_and_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    usage = {"planner": {"calls": 1, "cached": 0, "prompt_tokens": 100, "completion_tokens": 50},
             "synthesizer": {"calls": 1, "cached": 0, "prompt_tokens": 900, "completion_tokens": 300}}
    _write(tmp_path / "runs" / "critic.jsonl", [
        {"id": "q1", "citations": [{}, {}], "invalid_citations": [], "critic_iterations": 2, "tool_errors": [],
         "seconds": 10.0, "usage": usage},
        {"id": "q2", "citations": [{}], "invalid_citations": [7], "critic_iterations": 1, "tool_errors": ["x"],
         "seconds": 20.0, "usage": usage},
    ])
    _write(tmp_path / "scores" / "critic__faithfulness.jsonl", [{"id": "q1", "faithfulness": 1.0},
                                                                 {"id": "q2", "faithfulness": 0.8}])
    _write(tmp_path / "scores" / "critic__answer_relevancy.jsonl", [{"id": "q1", "answer_relevancy": 0.9}])
    _write(tmp_path / "scores" / "critic__context_recall.jsonl", [{"id": "q1", "context_recall": 0.5},
                                                                   {"id": "q2", "context_recall": 1.0}])
    s = report.summarize("critic")
    assert s["faithfulness"]["mean"] == 0.9 and s["faithfulness"]["n"] == 2
    assert s["answer_relevancy"]["n"] == 1          # a metric the judge could not score is left out, not zeroed
    assert s["hallucination_rate"] == 0.1
    assert s["tokens_per_question"] == 1350 and s["llm_calls_per_question"] == 2
    assert s["invalid_citation_rate"] == round(1 / 4, 4)
    assert s["critic_loops_triggered"] == 1 and s["tool_errors"] == 1
    assert report.summarize("no_critic") == {}
