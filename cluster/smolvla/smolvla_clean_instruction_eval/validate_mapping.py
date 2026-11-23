"""Validate frozen text against the installed benchmark's actual language function."""
import ast
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
tree = ast.parse(source)
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "grab_language_from_filename")
scope = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), "benchmark_language", "exec"), scope)
language = scope["grab_language_from_filename"]
rows = list(csv.DictReader((Path(__file__).parent / "frozen/instruction_map.csv").open()))
count = 0
for row in rows:
    if row["category"] == "Language_Instructions":
        assert not row["official_instruction"] and not row["clean_instruction"]
        continue
    assert language(row["suite"], row["task_name"] + ".bddl") == row["official_instruction"], row
    assert language(row["suite"], row["base_task_name"] + ".bddl") == row["clean_instruction"], row
    count += 1
print(f"PASS: {count} official and cleaned instructions match the benchmark language function")
