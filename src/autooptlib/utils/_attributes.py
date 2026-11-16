"""Deterministic spelling aliases shared by settings and components."""

from __future__ import annotations

import re
from functools import lru_cache


def name_variants(name: str) -> list[str]:
    # Only spelling is static. Never cache attribute values or the winning
    # spelling: callbacks may replace either between Search events.
    return list(_name_variants(name))


@lru_cache(maxsize=256)
def normalized_name(name: str) -> str:
    """Cache spelling only; owners and current values never enter this cache."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


@lru_cache(maxsize=256)
def _name_variants(name: str) -> tuple[str, ...]:
    # Preserve a deterministic precedence order.  A set made alias resolution
    # depend on Python's hash seed when an object exposed more than one spelling
    # (for example both ``N`` and ``n``), so identical seeded runs could read a
    # different field in another process.
    variants = [name, name.lower(), name.upper()]
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    variants.append(snake)
    if "_" in name:
        parts = name.split("_")
        camel = parts[0].lower() + "".join(part.title() for part in parts[1:])
        pascal = "".join(part.title() for part in parts)
        variants.extend((camel, pascal))
    else:
        parts = re.findall(r"[A-Za-z][^A-Z]*", name)
        if parts:
            variants.append("_".join(part.lower() for part in parts))
            camel = parts[0].lower() + "".join(part.title() for part in parts[1:])
            pascal = "".join(part.title() for part in parts)
            variants.extend((camel, pascal))
    return tuple(dict.fromkeys(variant for variant in variants if variant))
