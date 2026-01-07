# Meridian

**Meridian** is a multi-agent LLM research assistant that plans, retrieves, critiques, and synthesizes answers to complex research questions with full source attribution.

Meridian orchestrates a pipeline of specialized AI agents to tackle research tasks that require more than a single model pass. Given a research question, the system decomposes it into sub-queries, retrieves evidence from the web, academic databases, and local documents, subjects that evidence to adversarial critique, and finally synthesizes a coherent, cited answer. Every step is observable, logged to MLflow, and evaluated against RAGAS metrics so you can track quality over time.

## Architecture

Meridian is built on a four-agent pipeline coordinated by LangGraph, with each agent implemented as a stateful node in a directed graph:

1. **Planner** — Receives the raw research question and decomposes it into a prioritized list of sub-queries. It decides which tools each sub-query requires and emits a structured research plan that drives the rest of the pipeline.

2. **Researcher** — Executes the plan by calling external tools (Tavily web search, ArXiv, Wikipedia, PDF ingestion, sandboxed code execution) and retrieves raw evidence chunks. Retrieved passages are embedded and indexed in FAISS for fast semantic lookup; persistent context and session history are stored in PostgreSQL.

3. **Critic** — Receives the Researcher's evidence and interrogates it for gaps, contradictions, and low-confidence claims. The Critic can loop back to the Researcher with follow-up queries (the Researcher ↔ Critic cycle) until the evidence meets a confidence threshold or a maximum iteration budget is reached.

4. **Synthesizer** — Consumes the verified evidence set and produces the final answer: a structured report with inline citations, a source table, and an uncertainty statement. All agents call Claude via the Anthropic API. Evaluation results are logged to MLflow using the RAGAS harness.

## Project Structure

```
meridian/
├── agents/           # LangGraph agent nodes
├── memory/           # Episodic, semantic, procedural memory
├── tools/            # Web search, ArXiv, Wikipedia, PDF, code exec
├── evaluation/       # RAGAS harness, MLflow logging
├── api/              # FastAPI SSE backend
├── frontend/         # Next.js UI
├── infra/            # Docker Compose, DB migrations
└── tests/            # Unit and integration tests
```

## Quick Start

### Prerequisites

- Python 3.11+, Node 20+, Docker
- API keys: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`

### Setup

```bash
cp .env.example .env
# fill in API keys
docker compose up -d db
pip install -r requirements.txt
cd frontend && npm install
```

### Run

```bash
# Start backend
uvicorn api.main:app --reload --port 8000

# Start frontend (new terminal)
cd frontend && npm run dev
```

Open http://localhost:3000

## Agents

| Agent | Role | Key Tools |
|---|---|---|
| Planner | Decomposes question into sub-queries and emits a research plan | Claude (claude-sonnet-4-5), query decomposer |
| Researcher | Retrieves and indexes evidence from external sources | Tavily, ArXiv, Wikipedia, PDF loader, FAISS, PostgreSQL |
| Critic | Identifies gaps and contradictions; loops back to Researcher if needed | Claude, confidence scorer, claim verifier |
| Synthesizer | Produces the final cited answer from verified evidence | Claude, citation formatter, uncertainty estimator |

## Memory System

- **Episodic memory** — Per-session conversation history and retrieved evidence chunks stored in PostgreSQL, enabling the agent to reference earlier turns and avoid redundant retrievals within a session.
- **Semantic memory** — Long-term vector store backed by FAISS that persists embeddings across sessions, allowing frequently accessed documents to be retrieved without re-fetching from external APIs.
- **Procedural memory** — Stored research plans and tool-call strategies that the Planner can reuse as templates when it encounters question types similar to ones it has solved before.

## Evaluation

Run the RAGAS evaluation harness against any research question using a config file:

```bash
python -m evaluation.run --question "What are the long-term effects of sleep deprivation on cognitive performance?" --config configs/default.yaml
```

Results are logged automatically to MLflow. Launch the MLflow UI to browse runs:

```bash
mlflow ui --port 5000
```

## Sample Eval Results

| Question | Faithfulness | Answer Relevancy | Context Recall |
|---|---|---|---|
| What are the mechanisms behind mRNA vaccine efficacy? | 0.91 | 0.88 | 0.85 |
| How does transformer attention scale with sequence length? | 0.95 | 0.93 | 0.90 |
| What is the evidence for neuroplasticity in adult brains? | 0.87 | 0.85 | 0.82 |

## License

MIT