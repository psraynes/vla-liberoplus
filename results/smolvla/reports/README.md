# SmolVLA result sources

The original six per-task runs, repaired clean full/subset, and base/v3/repaired standard per-episode exports are verified raw measurements. New comparison inputs are retained in source_exports; validation joins exact frozen task/episode keys and checks all published suite/category rates. Missing rollout seconds and some Language instruction text remain blank, not invented.

Shared-speed raw numeric JSON is now included for base and v3; private account roots/internal hostname are redacted, with original/public hashes retained. Historical 20-warmup/100-query profile CSVs remain different-input aggregate records. Noise/unaltered-demo/v2 standard results still have aggregate-only evidence.

paired_overall.csv includes nominal exact conditional McNemar tests and Holm correction across six overall tests. Paired suite/category outputs are exploratory. Variants share task families, the frozen set was inspected during development, full benchmark overlaps some teacher sources, and no multi-seed replication has been measured. Source inference limits apply even when p values are small.
