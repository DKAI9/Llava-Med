"""Utilities for LoRA target module parsing and matching."""
from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional, Sequence, Union

_REGEX_MARKERS: Sequence[str] = ("|", ".*", "(", ")", "[", "]", "^", "$", "\\")


def parse_lora_target_modules(raw: Optional[str]) -> Optional[Union[str, List[str]]]:
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped or stripped.lower() == "none":
        return None
    if stripped.lower().startswith("re:"):
        return stripped[3:]
    if stripped.lower().startswith("list:"):
        remainder = stripped[5:]
        return [token.strip() for token in remainder.split(",") if token.strip()]
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed
    if "," in stripped:
        return [token.strip() for token in stripped.split(",") if token.strip()]
    if any(marker in stripped for marker in _REGEX_MARKERS):
        return stripped
    return [stripped]


def match_lora_target_modules(
    module_names: Iterable[str],
    target_modules: Optional[Union[str, List[str]]],
) -> List[str]:
    if not target_modules:
        return []
    names = list(module_names)
    if isinstance(target_modules, str):
        pattern = re.compile(target_modules)
        return [name for name in names if pattern.search(name)]
    tokens = list(target_modules)
    return [
        name
        for name in names
        if any(name == token or name.endswith(token) for token in tokens)
    ]
