# Retrieved comparison exports

The three CSVs are the completed report script's original wide exports, with one paired row per task/episode. Timing JSONs are the completed shared-speed outputs, with account-specific path roots replaced by `${PROJECT_ROOT}` / `${LP_DIR}` and the internal DNS suffix removed. Numeric outcomes, all timing samples, checkpoint run identifiers and hashes are unchanged. `PROVENANCE.json` records original and public-copy checksums. No raw rollout time was present in the CSVs; missing `seconds` fields are not inferred.

Source: completed clean workers 2542993_0/1; standard 2542994; speed 2568735. The full clean set includes teacher source variants; the 4,295 subset is source-ID-held-out but reused during development and shares task families.
