"""Run a short Python snippet in a separate, restricted interpreter.

Isolation: a fresh `python -I` process (no user site-packages, no PYTHON*
environment), an empty environment and a throwaway working directory, with
limits on CPU time, memory, file size and wall-clock time. It does not block
network access, so treat it as a calculator for the researcher, not a sandbox
for untrusted code.
"""
from __future__ import annotations

import asyncio
import resource
import sys
import tempfile

from agents.state import Finding
from tools.base import ToolError

CPU_SECONDS = 5
MEMORY_BYTES = 512 * 1024 * 1024
OUTPUT_CHARS = 4000


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    try:  # not enforced on macOS, effective on Linux
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    except (ValueError, OSError):
        pass


class PythonTool:
    name = "python"
    description = "Run Python code and return what it prints. Use for arithmetic, unit conversion, quick calculations."

    def __init__(self, timeout_s: float = 10) -> None:
        self.timeout_s = timeout_s

    async def run(self, query: str) -> list[Finding]:
        out, err, code = await self.execute(query)
        if code != 0:
            raise ToolError(f"exit {code}: {err[-500:]}")
        return [Finding(title="Python computation", url="python://local", tool=self.name, query=query,
                        content=f"Code:\n{query}\n\nOutput:\n{out}")]

    async def execute(self, code: str) -> tuple[str, str, int]:
        with tempfile.TemporaryDirectory() as cwd:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-c", code, cwd=cwd, env={}, preexec_fn=_limits,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(), self.timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return "", "timed out", -9
        return out.decode(errors="replace")[:OUTPUT_CHARS], err.decode(errors="replace")[:OUTPUT_CHARS], proc.returncode
