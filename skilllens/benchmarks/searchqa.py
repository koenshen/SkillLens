"""Fixed benchmark manifests and the open-book QA input and metrics."""
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


FIXED_SPLITS = {
    "searchqa": SPLITS,
    "triviaqa": {
        "train": (1600, "f465cbaea5820ac4a01a9d0f5d6dc6b71668a78bf21cf783c687e3f181ac3445"),
        "val": (200, "c2b8957c0814fc54e381821f2efb6af200bae1e26605304ca3bac7c10747176c"),
        "test": (1000, "34c26854b2b9348c90fb740675bf45562937c1525f5a770824474e0b6df6397c"),
    },
    "hotpotqa": {
        "train": (1600, "925adec60c7d1488b925b5c5facaaeb9b0f2d60b155928db809fde778433a6d9"),
        "val": (200, "9022dabfa5ddd7a7df3a32eee1a8abbab8963ea807013f257dbefb1250d3d82b"),
        "test": (1000, "7b81664c3206ab4a9d4e3bd274a7bbe7b7d09976c74918970a02ecbb463b7439"),
    },
    "bird": {
        "train": (834, "bf49ab424f7627116e654bb8719fdee322031a35da199aa04b6c370ae0c281b3"),
        "val": (200, "2b3ab585858e168cb05ab1cf2ce8ceb4ecd24a4ae2d375d89ba9d94a362b19d1"),
        "test": (500, "706fcd80ba2577502ed013f49f22ceac5071af0dbfe58adb6fe22ee030b2487f"),
    },
}


def load_splits(directory: Path, benchmark: str = "searchqa") -> dict[str, list[dict]]:
    splits = {}
    for name, (count, expected_hash) in FIXED_SPLITS[benchmark].items():
        path = directory / name / "items.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
        if len(rows) != count or digest != expected_hash:
            raise ValueError(f"{path} does not match the fixed {count} {name} question IDs")
        for row in rows:
            fields = ("question", "prompt", "db_id", "SQL") if benchmark == "bird" else ("question", "context")
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in fields):
                raise ValueError(f"Incomplete {benchmark} item at {name}/{row['id']}")
            if benchmark == "bird":
                continue
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
