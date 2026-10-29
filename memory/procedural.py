"""Procedural memory: how well each tool has worked, used to rank tools.

For every call: did it fail, did it return nothing, how long did it take; and
after synthesis, how many of its findings were actually cited. The researcher
sees these figures when choosing tools, and tries tools in descending order of
usefulness.
"""
from __future__ import annotations

from memory.store import Store


class ProceduralMemory:
    def __init__(self, store: Store) -> None:
        self.s = store

    def record(self, tool: str, ok: bool, empty: bool, ms: float) -> None:
        with self.s.lock:
            self.s.db.execute("INSERT INTO tool_stats (tool) VALUES (?) ON CONFLICT(tool) DO NOTHING", (tool,))
            self.s.db.execute("UPDATE tool_stats SET calls = calls + 1, failures = failures + ?, empty = empty + ?, "
                              "total_ms = total_ms + ? WHERE tool = ?", (0 if ok else 1, 1 if empty else 0, ms, tool))
            self.s.db.commit()

    def record_cited(self, counts: dict[str, int]) -> None:
        with self.s.lock:
            for tool, n in counts.items():
                self.s.db.execute("INSERT INTO tool_stats (tool) VALUES (?) ON CONFLICT(tool) DO NOTHING", (tool,))
                self.s.db.execute("UPDATE tool_stats SET cited = cited + ? WHERE tool = ?", (n, tool))
            self.s.db.commit()

    def stats(self) -> dict[str, dict[str, float]]:
        with self.s.lock:
            rows = self.s.db.execute("SELECT * FROM tool_stats").fetchall()
        out = {}
        for r in rows:
            calls = max(r["calls"], 1)
            out[r["tool"]] = {
                "calls": r["calls"],
                "success_rate": round(1 - r["failures"] / calls, 3),
                "hit_rate": round(1 - (r["failures"] + r["empty"]) / calls, 3),
                "avg_ms": round(r["total_ms"] / calls, 1),
                "cited_per_call": round(r["cited"] / calls, 3),
            }
        return out

    def rank(self, tools: list[str]) -> list[str]:
        """Tools in order of expected usefulness; untried tools keep their given order, first."""
        st = self.stats()
        def key(t: str) -> tuple[int, float]:
            s = st.get(t)
            if not s or s["calls"] < 3:
                return (0, 0.0)
            return (1, -(s["hit_rate"] + s["cited_per_call"]))
        return sorted(tools, key=key)

    def describe(self, tools: list[str]) -> str:
        st = self.stats()
        lines = []
        for t in tools:
            s = st.get(t)
            lines.append(f"- {t}: " + ("no history yet" if not s else
                         f"{s['calls']} calls, {s['hit_rate']:.0%} returned results, {s['avg_ms']:.0f} ms avg, "
                         f"{s['cited_per_call']:.2f} citations per call"))
        return "\n".join(lines)
