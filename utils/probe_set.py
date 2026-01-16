"""Utilities for deterministic probe set selection."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def select_probe(records: Iterable[Dict], size: int, seed: int = 42) -> List[Dict]:
    records = list(records)
    if size <= 0 or not records:
        return []
    if size >= len(records):
        return records

    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for record in records:
        answer_type = str(record.get("answer_type", ""))
        q_lang = str(record.get("q_lang", ""))
        groups[(answer_type, q_lang)].append(record)

    for items in groups.values():
        rng.shuffle(items)

    selected: List[Dict] = []
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    while len(selected) < size and group_keys:
        for key in list(group_keys):
            if not groups[key]:
                group_keys.remove(key)
                continue
            selected.append(groups[key].pop())
            if len(selected) >= size:
                break
    return selected
