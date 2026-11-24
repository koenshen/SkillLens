"""Offline checks for the fixed QA and SQL benchmark paths."""
import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from skilllens import searchqa
from skilllens.benchmarks import bird


class FixedBenchmarksTest(unittest.TestCase):
    def test_sql_extraction_and_execution(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "sample.sqlite"
            with closing(sqlite3.connect(db)) as conn:
                conn.executescript("CREATE TABLE t(s TEXT); INSERT INTO t VALUES ('New York'),('New York'),('Rome');")
            gold = "SELECT s FROM t"
            def grade(response, timeout=30):
                return bird.score_response(response, gold, db, timeout)
            sql = "SELECT DISTINCT s FROM t ORDER BY s DESC;"
            self.assertEqual(grade(f"reasoning\n```sqlite\n{sql}\n```")["ex"], 1)
            self.assertEqual(bird.extract_sql("text\nWITH x AS (SELECT 'a'';b') SELECT * FROM x; extra"),
                             "WITH x AS (SELECT 'a'';b') SELECT * FROM x;")
            self.assertEqual(grade("SELECT lower(s) FROM t")["ex"], 0)
            self.assertEqual(grade("SELECT missing FROM t")["exec_status"], "error")
            self.assertEqual(grade("DELETE FROM t")["exec_status"], "error")
            metadata = bird.score_response("SELECT name FROM pragma_table_info('t')", "SELECT 's'", db)
            self.assertEqual(metadata["ex"], 1)
            self.assertEqual(grade("PRAGMA query_only=OFF")["exec_status"], "error")
            attach = Path(tmp) / "other.sqlite"
            self.assertEqual(grade(f"ATTACH DATABASE '{attach}' AS other")["exec_status"], "error")
            self.assertFalse(attach.exists())
            endless = "WITH RECURSIVE x(n) AS (VALUES(0) UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x"
            self.assertEqual(grade(endless, timeout=0.001)["exec_status"], "timeout")
            self.assertEqual(grade("")["ex"], 0)
            self.assertEqual(grade(gold)["ex"], 1)

    def test_benchmark_reaches_train_extraction_and_both_eval_splits(self):
        for bench in ("searchqa", "triviaqa", "hotpotqa", "bird"):
            with self.subTest(bench=bench), TemporaryDirectory() as tmp:
                root = Path(tmp)
                split_data = {s: [{"id": s+"1", "question": "Question?", "context": "known",
                                   "answers": ["x", "right"], "db_id": "sample", "SQL": "SELECT 'A'",
                                   "prompt": "Schema facts\n" + "p"*6300 + "\nWrite SQL."}]
                              for s in ("train", "val", "test")}
                db = bird.database_path(root, "sample")
                db.parent.mkdir()
                with closing(sqlite3.connect(db)) as conn:
                    conn.execute("CREATE TABLE tiny(i)")
                out = root / "output"
                calls = []
                class Client:
                    async def chat_with_usage(self, messages):
                        calls.append(messages)
                        return ("SELECT 'A'" if bench == "bird" else "<answer>right</answer>"), {"tokens_in": 3}

                async def extract(args, **kwargs):
                    self.assertIn(f"input.benchmark={bench}", args.set)
                    trajectory, = json.loads(Path(args.input).read_text())["trajectories"]
                    self.assertEqual((trajectory["benchmark"], trajectory["reward"]), (bench, 1))
                    if bench == "bird":
                        self.assertIn("EX=1", trajectory["steps"][1]["observation"])
                    result = Path(args.output) / bench / "test-model" / "skill_set.json"
                    result.parent.mkdir(parents=True)
                    result.write_text(json.dumps({"skills": [{"name": "example", "body": "frozen guidance"}]}))

                with (patch.object(searchqa, "load_splits", return_value=split_data),
                      patch("skilllens.client.openai_client.LLMClient", return_value=Client()),
                      patch("skilllens.cli.extract._run_extract", side_effect=extract)):
                    searchqa.main(["--benchmark", bench, "--db-root", str(root), "--output", str(out)])
                self.assertEqual(len(calls), 3)
                for i,s in enumerate(("train", "val", "test")):
                    row, = map(json.loads,(out / s / "results.jsonl").read_text().splitlines())
                    self.assertEqual(row["id"], s+"1")
                    self.assertEqual(row["ex" if bench == "bird" else "em"], 1)
                    if bench == "bird":
                        self.assertEqual(calls[i][1]["content"], split_data[s][0]["prompt"])
                        self.assertNotIn(split_data[s][0]["SQL"], str(calls[i]))
                    if i: self.assertIn("frozen guidance", calls[i][0]["content"])

    def test_bird_reports_client_errors_separately_from_bad_sql(self):
        class Client:
            async def chat_with_usage(self, messages):
                if messages[1]["content"] == "fail": raise RuntimeError("gateway unavailable")
                return "SELECT (", {}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = bird.database_path(root, "sample")
            db.parent.mkdir()
            with closing(sqlite3.connect(db)) as conn:
                conn.execute("CREATE TABLE t(i)")
            items = [dict(id=msg,prompt=msg,db_id="sample",SQL="SELECT 1") for msg in ("fail","invalid sql")]
            asyncio.run(searchqa.evaluate(Client(),items,"",1,root/"out",benchmark="bird",db_root=root))
            summary=json.loads((root/"out/eval_summary.json").read_text())
            self.assertEqual(summary,{"n":2,"errors":1,"ex":0.0,"sql_errors":1})


if __name__ == "__main__":
    unittest.main()
