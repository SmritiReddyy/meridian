"""python -m evaluation run|score|report [--limit N] [--config critic|no_critic]"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging

from evaluation.report import report
from evaluation.run import CONFIGS, load_questions, run_config
from llm import load_settings


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m evaluation")
    ap.add_argument("step", choices=["run", "score", "report", "all"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--config", choices=list(CONFIGS), action="append", help="default: both")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    configs = args.config or list(CONFIGS)

    if args.step in ("run", "all"):
        from api.service import build_services

        svc = build_services(settings)
        questions = load_questions(args.limit)

        async def run_all() -> None:   # one event loop: the clients' locks belong to it
            for c in configs:
                await run_config(svc, c, questions)

        asyncio.run(run_all())
    if args.step in ("score", "all"):
        from evaluation.score import score_config

        for c in configs:
            score_config(settings, c)
    if args.step in ("report", "all"):
        print(json.dumps(report(settings), indent=2))


if __name__ == "__main__":
    main()
