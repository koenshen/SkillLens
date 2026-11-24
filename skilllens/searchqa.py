"""Extract a SkillLens skill from fixed train trajectories, then evaluate on dev and test."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil

from skilllens.benchmarks import bird
from skilllens.benchmarks.searchqa import FIXED_SPLITS, format_question, load_splits, score_response


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


def user_input(item, benchmark):
    return item["prompt"] if benchmark == "bird" else format_question(item)


async def evaluate(client, items, skill, workers, output, *, benchmark="searchqa", db_root=None):
    section = f"## Skill\n{skill.strip()}\n\n" if skill.strip() else ""
    system = skill.strip() if benchmark == "bird" else SYSTEM_PROMPT.format(skill_section=section)
    semaphore = asyncio.Semaphore(workers)

    async def answer(item):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user_input(item, benchmark)}]
        async with semaphore:
            try:
                response, usage = await client.chat_with_usage(messages)
            except Exception as error:
                grades = ({"db_id": item["db_id"], "ex": 0, "predicted_sql": "", "exec_status": "not_run"}
                          if benchmark == "bird" else {"predicted_answer": "", "em": 0.0, "f1": 0.0})
                return {"id": item["id"], "response": "", "error": str(error), "usage": {}, **grades}
            if benchmark == "bird":
                grades = await asyncio.to_thread(bird.score_response, response, item["SQL"],
                                                 bird.database_path(db_root, item["db_id"]))
                return {"id": item["id"], "db_id": item["db_id"], "response": response, "usage": usage, **grades}
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
    metrics = ("ex",) if benchmark == "bird" else ("em", "f1")
    summary = {"n": len(results), "errors": failures}
    summary.update({k: sum(row[k] for row in results) / len(results) for k in metrics})
    if benchmark == "bird":
        summary["sql_errors"] = sum(row["exec_status"] in {"timeout", "error"} for row in results)
    (output / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output.name, summary, flush=True)
    return results


def train_trajectories(items, results, target, benchmark="searchqa"):
    trajectories = []
    for item, row in zip(items, results, strict=True):
        if item["id"] != row["id"]:
            raise ValueError("Training result does not match its input ID")
        metric = row["ex"] if benchmark == "bird" else row["em"]
        outcome = "resolved" if metric else "unresolved"
        agent_step = {"role": "agent", "content": row["response"]}
        if benchmark == "bird":
            detail = row.get("exec_error") or row.get("error") or ""
            agent_step["observation"] = f"SQL execution: {row['exec_status']}; EX={metric}. {detail}".strip()
        trajectories.append({
            "id": item["id"], "task_id": item["id"], "task_name": item["question"],
            "benchmark": benchmark, "agent": target, "source_format": "generic",
            "steps": [{"role": "user", "content": user_input(item, benchmark)}, agent_step],
            "final_answer": row["predicted_sql"] if benchmark == "bird" else row["predicted_answer"],
            "reward": metric,
            "outcome": "error" if "error" in row else outcome,
        })
    return trajectories


def model_overrides(args, role):
    """Resolve the SearchQA model for answering or skill extraction."""
    model = args.target_model if role == "target" else args.extract_model
    base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
    provider = args.provider or ("openai" if base_url else None)
    settings = {}
    if base_url and provider == "openai":
        settings["llm.base_url"] = base_url
        if key := os.environ.get("OPENAI_COMPATIBLE_API_KEY"):
            settings["llm.api_key"] = key
        model = model or os.environ.get("OPENAI_COMPATIBLE_MODEL")
        raw = os.environ.get(f"{role.upper()}_OPENAI_COMPATIBLE_MAX_TOKENS")
        if raw:
            limit = int(raw)
            if limit < 1:
                raise ValueError(f"{role} max tokens must be positive")
            settings["llm.max_tokens"] = limit
    if provider:
        settings["llm.provider"] = provider
    api_style = args.api_style or ("chat" if base_url and provider == "openai" else None)
    if api_style:
        settings["llm.api_style"] = api_style
    if model:
        settings["llm.model"] = model
    return settings


async def run(args, splits):
    from skilllens.cli.extract import _add_extract_args, _run_extract
    from skilllens.client.openai_client import LLMClient
    from skilllens.config import build_config
    from skilllens.inference.seal0.run import _render_skill_to_file

    target_config = build_config(args.extract_config, model_overrides(args, "target")).llm
    extraction_overrides = model_overrides(args, "optimizer")
    client = LLMClient(target_config)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    task_args = {"benchmark": args.benchmark, "db_root": args.db_root}
    results = await evaluate(client, splits["train"], "", args.workers, out / "train", **task_args)
    pool = out / "train_trajectories.json"
    pool.write_text(json.dumps({"trajectories": train_trajectories(
        splits["train"], results, target_config.model, args.benchmark
    )}, ensure_ascii=False), encoding="utf-8")

    extract_out = out / "extraction"
    extract_args = ["--config", str(args.extract_config), "--input", str(pool),
                    "--output", str(extract_out), "--set", f"input.benchmark={args.benchmark}"]
    extract_parser = argparse.ArgumentParser()
    _add_extract_args(extract_parser)
    await _run_extract(extract_parser.parse_args(extract_args), model_overrides=extraction_overrides)
    artifacts = list((extract_out / args.benchmark).glob("*/skill_set.json"))
    if len(artifacts) != 1:
        raise RuntimeError(f"Expected one SkillLens extraction output, found {len(artifacts)}")
    frozen = out / "skill.md"
    shutil.move(_render_skill_to_file(str(artifacts[0])), frozen)
    skill = frozen.read_text(encoding="utf-8")
    for split in ("val", "test"):
        await evaluate(client, splits[split], skill, args.workers, out / split, **task_args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=tuple(FIXED_SPLITS), default="searchqa")
    parser.add_argument("--split-dir", type=Path, help="Fixed benchmark directory with train/val/test items.json")
    parser.add_argument("--db-root", type=Path, default=Path(os.getenv(
        "BIRD_DB_ROOT", REPO.parent / "bird-text-to-sql-260820/data_dev/dev_databases")),
        help="BIRD SQLite databases, each under DB_ID/DB_ID.sqlite")
    parser.add_argument("--check", action="store_true", help="Validate the fixed IDs offline and exit")
    parser.add_argument("--output", type=Path, help="New directory for this run")
    parser.add_argument("--workers", type=int, default=os.environ.get("SKILLOPT_WORKERS", "16"))
    parser.add_argument("--extract-config", type=Path, default=REPO / "configs/examples/seal0_parallel.yaml")
    parser.add_argument("--target-model", help="Override the answering model from the config")
    parser.add_argument("--extract-model", help="Override the extraction model from the config")
    parser.add_argument("--provider", choices=["azure", "openai", "vllm"])
    parser.add_argument("--api-style", choices=["responses", "chat"])
    args = parser.parse_args(argv)
    if args.split_dir is None:
        args.split_dir = Path(os.environ.get(f"{args.benchmark.upper()}_SPLIT_DIR")
                              or REPO.parent / "SkillOpt/data" / f"{args.benchmark}_split")
    splits = load_splits(args.split_dir, args.benchmark)
    if args.benchmark == "bird":
        for db_id in sorted({row["db_id"] for rows in splits.values() for row in rows}):
            path = bird.database_path(args.db_root, db_id)
            if not path.is_file():
                parser.error(f"Missing database for BIRD: {path}")
    if args.check:
        print(f"Fixed {args.benchmark} IDs verified:", {k: len(v) for k, v in splits.items()})
        return
    if not args.output or args.workers < 1 or not args.extract_config.is_file():
        parser.error("Set --output, use positive --workers, and supply an existing extraction config")
    asyncio.run(run(args, splits))


if __name__ == "__main__":
    main()
