"""Fixed SearchQA data contract, single-turn input and SQuAD-style metrics."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string


# SHA-256 of the published split's ordered IDs joined by LF without a trailing LF.
SPLITS = {
    "train": (400, "cded018312f19f27e1da3e019931fb4d18fbbdbc4fd6d1651b5a52f29dedd268"),
    "val": (200, "d2913aff1ba10bd9a577e79c24121108218b30136c0d5a7002eed36133babafe"),
    "test": (1400, "7fca98256e96c837eee2612da9b9d29fb9d4e5cb908c751ca9bfdfd2d10ab6fb"),
}


def load_splits(directory: Path) -> dict[str, list[dict]]:
    splits = {}
    for name, (count, expected_hash) in SPLITS.items():
        path = directory / name / "items.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
        if len(rows) != count or digest != expected_hash:
            raise ValueError(f"{path} does not match the fixed {count} {name} question IDs")
        for row in rows:
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("question", "context")):
                raise ValueError(f"Incomplete question or context for {name}/{row['id']}")
            answers = row.get("answers")
            if not isinstance(answers, list) or not answers or not all(isinstance(a, str) for a in answers):
                raise ValueError(f"Invalid references for {name}/{row['id']}")
        splits[name] = rows
    return splits


def format_question(row: dict) -> str:
    context = row["context"]
    if len(context) > 6000:
        excerpt = ""
        for passage in context.split("[DOC]"):
            candidate = excerpt + "[DOC]" + passage if excerpt else passage
            if len(candidate) > 6000:
                break
            excerpt = candidate
        context = excerpt or context[:6000] + "\n...[truncated]"
    return f"## Context\n{context}\n\n## Question\n{row['question']}"


def _tokens(text: str) -> list[str]:
    clean = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", clean).split()


def score_response(response: str, references: list[str]) -> dict:
    matches = re.findall(r"<answer>(.*?)</answer>", response, flags=re.IGNORECASE | re.DOTALL)
    final = matches[-1].strip() if matches else (response.strip().splitlines() or [""])[-1].strip()
    predicted = _tokens(final)
    em = f1 = 0.0
    for reference in references:
        gold = _tokens(reference)
        if predicted == gold:
            em = f1 = 1.0
            break
        overlap = sum((Counter(predicted) & Counter(gold)).values())
        if overlap:
            precision = overlap / len(predicted)
            recall = overlap / len(gold)
            f1 = max(f1, 2 * precision * recall / (precision + recall))
    return {"em": em, "f1": f1, "predicted_answer": final}
