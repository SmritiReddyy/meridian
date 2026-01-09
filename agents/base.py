```python
import json
import re
from datetime import datetime, timezone


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response that may have markdown code fences."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code fences
    match = re.search(r'```(?:json)?\s*([\s\S]+?)```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Last resort: find first { ... }
    match = re.search(r'\{[\s\S]+\}', text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not extract JSON from: {text[:200]}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate(text: str, max_len: int = 500) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text
```