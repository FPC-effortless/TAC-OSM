# LANGSCALE072 — original native vs capped position nine-run test: INCOMPLETE / rejected by strict cross-run replay

Original [draft PR #611](https://github.com/FPC-effortless/TAC-OSM/pull/611), [GitHub Actions #40873502139](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40873502139). This original preregistered strict test remains preserved, **not marked PASS**.

Five model/seed jobs passed and retained their source-locked cap/native evidence:
- p500k seed1701,1702,1703; p1000k seed1701,1703.
Four model/seed jobs **FAILED** before any verified cap result could be accepted:
- p250k seed1701 and1702: `ValueError: native 071 score did not replay bitwise numerically`.
- p250k seed1703 and p1000k seed1702: `ValueError: original 128 score/source target mismatch`.
All observed failure messages are at cross-run original NLL/metrics strict equality checks requiring native recomputation on new GitHub runner to match an earlier source benchmark at extremely small absolute tolerance 1e-10 or exact float equality. Whether these are ordinary platform floating-point differences or larger discrepancies was **not established from the originally printed failure logs**, as no per-error numerical drift was printed. The primary nine-result aggregator **FAILED** due missing four successful original receipts. A single successful example (p500k seed1703 cap256 2.220876 vs native256 2.364772, 6 focused tests passed) is a **subset diagnosis only, not a full validated result**.

Science interpretation: original 071 negative context scaling remains valid. 072 is a negative verification result: cannot claim the positional cap is generally helpful from 5/9 success or revise the pre-registered exact comparison posthoc. New separate diagnostic LANGSCALE073 may instrument numerical differences and use a separately frozen cross-host replay tolerance (e.g. NLL/Brier <=1e-6) with exact source owner/label/source-input hashes, exact same-host cap128 no-op, unchanged original weights and all nine artifact requirements. The tolerance must be set **before that new execution** and new results must not be relabelled as independent held-out evidence. Original source and model checkpoint unchanged; no training, GPU/Kaggle or protected TEST, merges or model promotions.
