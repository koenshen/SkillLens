"""Extract a SkillLens skill from SearchQA train trajectories, then evaluate it."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil

from skilllens.benchmarks.searchqa import format_question, load_splits, score_response


REPO = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = """You are an expert question answering agent.

{skill_section}## Task Format
You will receive a CONTEXT containing document passages and a QUESTION.
Read the context carefully and answer the question based on the information provided.

## Answer Format
Think step by step, then provide your final answer inside <answer>...</answer> tags.
Keep your answer concise — typically a few words or a short phrase.
Do not repeat the question. Do not include unnecessary explanation in the answer tags.

Example:
<answer>Abraham Lincoln</answer>"""


async def evaluate(client, items, skill, workers, output):
    section = f"## Skill\n{skill.strip()}\n\n" if skill.strip() else ""
    system = SYSTEM_PROMPT.format(skill_section=section)
    semaphore = asyncio.Semaphore(workers)

    async def answer(item):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": format_question(item)}]
        async with semaphore:
            try:
                response, usage = await client.chat_with_usage(messages)
            except Exception as error:
                return {"id": item["id"], "response": "", "predicted_answer": "", "em": 0.0,
                        "f1": 0.0, "error": str(error), "usage": {}}
        return {"id": item["id"], "response": response, "usage": usage,
                **score_response(response, item["answers"])}

    output.mkdir()
    results = await asyncio.gather(*(answer(row) for row in items))
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    failures = sum("error" in row for row in results)
    if failures == len(results):
        raise RuntimeError(f"All {failures} calls failed; see {output / 'results.jsonl'}")
    summary = {"n": len(results), "errors": failures,
               "em": sum(row["em"] for row in results) / len(results),
               "f1": sum(row["f1"] for row in results) / len(results)}
    (output / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output.name, summary, flush=True)
    return results


def train_trajectories(items, results, target):
    trajectories = []
    for item, row in zip(items, results, strict=True):
        if item["id"] != row["id"]:
            raise ValueError("Training result does not match its input ID")
        outcome = "resolved" if row["em"] else "unresolved"
        trajectories.append({
            "id": item["id"], "task_id": item["id"], "task_name": item["question"],
            "benchmark": "searchqa", "agent": target, "source_format": "generic",
            "steps": [{"role": "user", "content": format_question(item)},
                      {"role": "agent", "content": row["response"]}],
            "final_answer": row["predicted_answer"], "reward": row["em"],
            "outcome": "error" if "error" in row else outcome,
        })
    return trajectories


async def run(args, splits):
    from skilllens.cli.extract import _add_extract_args, _run_extract
    from skilllens.client.openai_client import LLMClient
    from skilllens.config import build_config
    from skilllens.inference.seal0.run import _render_skill_to_file

    overrides = {}
    if args.provider:
        overrides["llm.provider"] = args.provider
    if args.api_style:
        overrides["llm.api_style"] = args.api_style
    target_config = build_config(args.extract_config, {
        **overrides, **({"llm.model": args.target_model} if args.target_model else {})
    }).llm
    client = LLMClient(target_config)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = await evaluate(client, splits["train"], "", args.workers, out / "train")
    pool = out / "train_trajectories.json"
    pool.write_text(json.dumps({"trajectories": train_trajectories(
        splits["train"], results, target_config.model
    )}, ensure_ascii=False), encoding="utf-8")

    extract_out = out / "extraction"
    extract_args = ["--config", str(args.extract_config), "--input", str(pool),
                    "--output", str(extract_out), "--set", "input.benchmark=searchqa"]
    if args.extract_model:
        overrides["llm.model"] = args.extract_model
    for key, value in overrides.items():
        extract_args.extend(["--set", f"{key}={value}"])
    extract_parser = argparse.ArgumentParser()
    _add_extract_args(extract_parser)
    await _run_extract(extract_parser.parse_args(extract_args))
    artifacts = list((extract_out / "searchqa").glob("*/skill_set.json"))
    if len(artifacts) != 1:
        raise RuntimeError(f"Expected one SkillLens extraction output, found {len(artifacts)}")
    frozen = out / "skill.md"
    shutil.move(_render_skill_to_file(str(artifacts[0])), frozen)
    skill = frozen.read_text(encoding="utf-8")
    for split in ("val", "test"):
        await evaluate(client, splits[split], skill, args.workers, out / split)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path(os.environ.get(
        "SEARCHQA_SPLIT_DIR", REPO.parent / "SkillOpt/data/searchqa_split")))
    parser.add_argument("--check", action="store_true", help="Validate the fixed IDs offline and exit")
    parser.add_argument("--output", type=Path, help="New directory for this run")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--extract-config", type=Path, default=REPO / "configs/examples/seal0_parallel.yaml")
    parser.add_argument("--target-model", help="Override the answering model from the config")
    parser.add_argument("--extract-model", help="Override the extraction model from the config")
    parser.add_argument("--provider", choices=["azure", "openai", "vllm"])
    parser.add_argument("--api-style", choices=["responses", "chat"])
    args = parser.parse_args(argv)
    splits = load_splits(args.split_dir)
    if args.check:
        print("Fixed SearchQA IDs verified:", {k: len(v) for k, v in splits.items()})
        return
    if not args.output or args.workers < 1 or not args.extract_config.is_file():
        parser.error("Set --output, use positive --workers, and supply an existing extraction config")
    asyncio.run(run(args, splits))


if __name__ == "__main__":
    main()
