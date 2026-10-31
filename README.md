# Meridian

A multi-agent research assistant. Ask a question; a **Planner** splits it into
sub-questions, a **Researcher** gathers sourced evidence with tools, a
**Critic** checks every sub-question against that evidence and sends gaps back
for more research, and a **Synthesizer** writes the answer with inline
citations. Agent steps stream live to a Next.js UI.

Built on LangGraph and FastAPI, and evaluated with RAGAS: the Critic's effect
is measured by answering the same 25 questions with it switched on and off.

## Results

25 research questions with reference answers (`evaluation/questions.jsonl`):
ML, distributed systems, science, history. Each was answered twice, Critic on
and Critic off, from scratch, with a fresh memory, using Wikipedia and arXiv.
Scored with RAGAS. Raw data is in `results/`.

**Quality** (paired: the same questions in both configurations)

| | Critic on | Critic off | Change | Critic better / worse / tied |
|---|---|---|---|---|
| **Faithfulness** (answer claims supported by the sources) | **0.867** | 0.795 | **+0.072** | 10 / 2 / 9 (n = 21) |
| **Context recall** (reference facts present in the sources) | **0.767** | 0.673 | **+0.093** | 4 / 0 / 21 (n = 25) |
| Answer relevancy | 0.898 | 0.904 | −0.006 | 7 / 7 / 11 (n = 25) |

With the Critic, the **hallucination rate** (1 − faithfulness, unpaired means) is
**13.8%**, against 20.3% without it. The Critic asked for more research on 11 of
25 questions. Those extra passes find evidence the first pass missed, and the
answers stick more closely to their sources, with no loss of relevancy.

**Cost and behaviour**

| | Critic on | Critic off |
|---|---|---|
| Tokens per question | 8,924 | 3,017 |
| Model calls per question | 7.9 | 5.5 |
| Median time per question | 43 s (p90 101 s) | * |
| Sources cited per answer | 3.4 | 3.3 |
| Citations to a non-existent source | 0% | 0% |

The Critic costs about 3× the tokens.

\* The Critic-off runs hit Groq's daily token cap and spent most of their time
waiting, so their wall-clock times measure the rate limit, not the system.

**Caveats**
- **The judges are open models.** Each RAGAS metric has its own judge, the same
  one for every answer: faithfulness by `qwen3.8-27b` (never the model that
  wrote the answers), context recall by `gpt-oss-20b`, answer relevancy by
  `gpt-oss-120b`. Treat absolute scores as indicative; the paired on/off
  comparison is the robust part.
- **Faithfulness covers 46 of the 50 answers.** The four longest (two per
  configuration) produce verdict lists that exceed the judge's 8k tokens per
  request on Groq's free tier. The paired comparison uses the 21 questions
  scored in both configurations.
- **Small sample.** 25 questions, and the reference answers were written for
  this set. 95% confidence intervals are in `results/summary.json`.

## How it works

```
question → Planner → Researcher → Critic ─┬─ gaps, budget left → Researcher
                                          └─ otherwise ────────→ Synthesizer → answer
```

- **Planner:** up to 4 independently researchable sub-questions.
- **Researcher:**
  - For each sub-question, it reuses earlier findings when semantic memory has a
    close match. Otherwise it asks the model which tools to call, with what
    input. The model sees each tool's track record.
  - It runs the calls concurrently, and identical calls within a session run once.
  - It keeps the passages most similar to the sub-question, with at most 2 per
    page.
- **Critic:**
  - Checks *each* sub-question for a passage that actually states the answer,
    not one merely on the topic.
  - For every unanswered one, it writes a new sub-question that takes a
    different route. It sees the searches already tried, and paraphrases of
    them are filtered out.
  - At most 2 extra research rounds.
- **Synthesizer:** answers only from numbered sources, citing each claim. If the
  sources don't cover something, it says so instead of guessing. Citations to
  sources that don't exist are dropped and counted.

**Tools:**
- Wikipedia (full article text, in passages).
- arXiv (abstracts; strict keyword search with a loose fallback).
- A PDF reader.
- A restricted Python runner: separate interpreter, empty environment, CPU,
  file-size and time limits.
- Web search: Tavily if `TAVILY_API_KEY` is set, else You.com's free endpoint.
  It has a **hard cap of 100 searches per day**, counted in SQLite, and is
  dropped from the tool list once the cap is spent.

**Memory:**
- **Episodic** (SQLite): every session and every streamed event, so a stream
  can be replayed.
- **Semantic** (FAISS with local `bge-small` embeddings): findings by
  sub-question, reused when a new sub-question is close enough (cosine ≥ 0.92).
- **Procedural** (SQLite): each tool's success rate, hit rate, latency and how
  often its findings get cited. It decides tool order and is shown to the
  Researcher.

**Models:** any OpenAI-compatible provider, set in `configs/default.yaml`.

| Role | Model (on Groq's free tier) |
|---|---|
| Planner, Critic, Synthesizer | `openai/gpt-oss-120b` |
| Researcher (writes search queries) | `qwen/qwen3.8-27b` |

The client caches every response on disk, retries on rate limits using the
provider's own wait times, and counts tokens per role.

## Found along the way

- **Wikipedia returned empty text for the top hit.** The extracts API returns
  full text for only one page per request, so in a multi-page request the best
  match (e.g. "Raft (algorithm)") came back empty. Each page is now fetched
  separately.
- **arXiv found nothing for exact paper searches.** The strict search required
  years like "2020" as keywords, which arXiv doesn't index, so
  "…knowledge-intensive NLP tasks Lewis 2020" matched nothing. With years
  dropped, the original RAG paper is found and cited.
- **One question took 298 s.** Three sub-tasks embedded the same 78 passages in
  parallel threads. With shared tool calls, an embedding cache and a lock, it
  takes 12 s.
- **The Critic gave up on gaps it had found.** It marked a sub-question
  unanswered, then re-asked it word for word, so nothing new was searched.
  Showing it the searches already tried, and filtering only true paraphrases
  (cosine ≥ 0.90; a new route to the same fact scores about 0.85), made it find
  new angles.

## Running it

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY (and TAVILY_API_KEY, optional)
.venv/bin/uvicorn api.main:app --port 8000
cd frontend && npm install && npm run dev     # http://localhost:3000
```

```bash
.venv/bin/python -m pytest      # 25 tests, no network or API key needed
.venv/bin/python -m evaluation run       # answer the 25 questions, Critic on and off
.venv/bin/python -m evaluation score     # RAGAS (--metric to run one judge per process)
.venv/bin/python -m evaluation report    # results/summary.json, and MLflow (data/mlflow.db)
```

Every evaluation step resumes where it stopped. On Groq's free tier (200k
tokens a day per model) the full run took about a day.

## Limitations

- **Small evaluation.** 25 questions with open-model judges.
- **Wall-clock numbers come with a caveat.** On the free tier they include
  rate-limit waits; see above.
- **The Python runner has resource limits but is not a network sandbox.**
- **Web search is excluded from the evaluation,** to stay within its daily cap
  and keep the evaluation reproducible.
