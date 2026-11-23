"""Exercise the result parser with the real 4,295-task paired plan."""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import report
import status


class ReportTest(unittest.TestCase):
    def test_common_pair_alignment_and_incomplete_label(self):
        source = Path(__file__).resolve().parents[1] / "smolvla_turing_setup" / "eval_kit" / "paired_plan_4295.tsv"
        rows = [line.split("\t") for line in source.read_text().splitlines() if line and not line.startswith("#")]
        self.assertEqual(len(rows), 4295)
        first, second = rows[:2]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            for arm in report.ARMS:
                run = root / "results" / arm
                run.mkdir(parents=True)
                shutil.copyfile(source, run / "plan.tsv")
                records = [{"suite": first[1], "task_id": int(first[2]), "success": arm != "clean"}]
                if arm != "base":
                    records.append({"suite": second[1], "task_id": int(second[2]), "success": arm == "clean"})
                (run / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
            with patch.object(report, "ROOT", root), contextlib.redirect_stdout(io.StringIO()) as stdout:
                report.main()
            output = stdout.getvalue()
            self.assertIn("PARTIAL common exact-task rollouts: 1/4295", output)
            self.assertIn("noise vs clean: noise-only successes=1", output)
            with (root / "paired_rollouts.csv").open(newline="") as handle:
                import csv
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 4295)
            self.assertEqual(records[0]["base"], "True")
            self.assertEqual(records[0]["clean"], "False")
            self.assertEqual(records[1]["base"], "")
            with patch("sys.argv", ["status.py", str(root / "results" / "base")]), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(status.main(), 1)


if __name__ == "__main__":
    unittest.main()
