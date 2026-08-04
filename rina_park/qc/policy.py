from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .models import CheckResult, Severity, Status


@dataclass(frozen=True)
class PolicyTerm:
    label: str
    pattern: re.Pattern[str]


def _terms(items: Iterable[tuple[str, str]]) -> tuple[PolicyTerm, ...]:
    return tuple(PolicyTerm(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in items)


ALWAYS_PROHIBITED = _terms(
    (
        ("nudity", r"\b(?:nude|nudity|naked|topless|bare\s+(?:breasts?|genitals?))\b"),
        ("explicit anatomy", r"\b(?:nipples?|genitals?|vulva|penis)\b"),
        ("sexual act", r"\b(?:sex(?:ual)?\s+act|intercourse|masturbat(?:e|ion)|orgasm)\b"),
        ("sexual fluid", r"\b(?:semen|sexual\s+fluids?)\b"),
        ("coercion", r"\b(?:non[- ]?consensual|forced\s+sex|rape)\b"),
        ("minor implication", r"\b(?:minor|underage|teen(?:age)?)\b.{0,30}\b(?:sexy|sexual|nude)\b"),
        ("sexual service", r"\b(?:pay|tip|subscribe).{0,30}\b(?:sext|sexual|explicit|nudes?)\b"),
    )
)

PLATFORM_PROHIBITED = _terms(
    (
        ("lingerie", r"\b(?:lingerie|bra\s+and\s+pant(?:y|ies)|thong)\b"),
        ("sexualized wet-look", r"\b(?:sexuali[sz]ed\s+wet[- ]?look|glossy\s+wet\s+skin)\b"),
        ("seductive framing", r"\b(?:seductive|sultry|erotic|body[- ]?part[- ]?first)\b"),
        ("reveal copy", r"\b(?:full\s+reveal|uncensored|explicit\s+(?:cut|content|photos?))\b"),
    )
)


def _metadata_strings(value: Any, prefix: str = "metadata") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _metadata_strings(nested, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple, set)):
        for index, nested in enumerate(value):
            yield from _metadata_strings(nested, f"{prefix}[{index}]")
    elif isinstance(value, str):
        yield prefix, value


def inspect_text(
    prompt: str,
    caption: str,
    track: str,
    metadata: Mapping[str, Any] | None = None,
) -> CheckResult:
    matches: list[dict[str, str]] = []
    fields = (
        ("prompt", prompt),
        ("caption", caption),
        *_metadata_strings(metadata or {}),
    )
    terms = ALWAYS_PROHIBITED + (() if track == "mature_non_explicit" else PLATFORM_PROHIBITED)
    for field, value in fields:
        for term in terms:
            match = term.pattern.search(value or "")
            if match:
                matches.append({"field": field, "rule": term.label, "match": match.group(0)})

    if matches:
        labels = ", ".join(sorted({item["rule"] for item in matches}))
        return CheckResult(
            check="prohibited_content",
            status=Status.FAIL,
            severity=Severity.BLOCKING,
            detail=f"Prohibited prompt/caption content: {labels}",
            data={"matches": matches, "track": track},
        )
    return CheckResult(
        check="prohibited_content",
        status=Status.PASS,
        severity=Severity.BLOCKING,
        detail="Prompt and caption passed deterministic policy rules",
        data={"track": track},
    )
