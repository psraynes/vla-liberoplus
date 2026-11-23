from pathlib import Path
import csv
ROOT=Path(__file__).resolve().parents[2]
expected={"lp_base_official":(10030,2681),"lp_combined_v3_official":(10030,4571),"lp_base_clean":(10030,5372),"lp_combined_v3_clean":(10030,4001),"lp_noise_official_4295":(4295,1213),"lp_clean_training_official_4295":(4295,1210),"lp_v3_spatial_repair_clean":(10030,5433),"lp_v3_spatial_repair_clean_4295":(4295,2098)}
for name,(count,successes) in expected.items():
 rows=list(csv.DictReader((ROOT/"results/smolvla"/name/"results.csv").open()))
 keys={(r["suite"],int(r["task_id"])) for r in rows}
 assert len(keys)==len(rows)==count,name
 assert sum(int(r["success"]) for r in rows)==successes,name
 assert all(r["success"] in ("0","1") and r["error"]==r["skipped"]=="0" for r in rows)
 if count==10030:assert sum(int(r["in_frozen_4295"]) for r in rows)==4295
 if name in ("lp_base_clean","lp_combined_v3_clean"):assert sum(int(r["reused_official_outcome"]) for r in rows)==1537
 print(f"PASS {name}: {successes}/{count}")
for name,successes in (("standard_base",1641),("standard_combined_v3",1058),("standard_v3_spatial_repair",1425)):
 rows=list(csv.DictReader((ROOT/"results/smolvla"/name/"results.csv").open()))
 assert len(rows)==len({(r["suite"],int(r["task_id"]),int(r["episode"])) for r in rows})==2000
 assert sum(int(r["success"]) for r in rows)==successes
 print(f"PASS {name}: {successes}/2000")
print("PASS: eight per-task exports and three standard per-episode exports; missing fields remain blank")
