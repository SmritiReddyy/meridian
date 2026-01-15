/Users/usmri/Desktop/clm/meridian/tools/code_executor.py

Key implementation decisions:

- `_sanitize` runs two passes: forbidden-pattern check on raw code, then strips the three banned import lines (`import os/sys/subprocess`, `from os/sys/subprocess`), then re-checks the stripped result so stripping cannot be used to launder a forbidden construct.
- Forbidden patterns cover `os.system`, `subprocess`, `open(`, `socket`, `requests`, `urllib`, `eval`, `exec`, `compile`, `__import__`, `importlib`, `ctypes`, `pickle`, `shutil`, `pathlib`, `glob`.
- Code is written to a `NamedTemporaryFile` (`.py`) and run as `python3 <file>` rather than `-c` so multi-line scripts work correctly; the file is always deleted in a `finally` block.
- Resource limits (`RLIMIT_CPU` 5s, `RLIMIT_AS` 128 MB) are applied via `preexec_fn` on Linux only; on macOS/Windows the process still has the 10-second `subprocess.run` timeout as a backstop.
- `asyncio.to_thread` keeps the event loop unblocked during subprocess execution.
- `ToolResult.success` is `True` only when `returncode == 0` and no security rejection occurred; `stderr` is always included in `metadata` regardless of success.