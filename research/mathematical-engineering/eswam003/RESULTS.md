# ESWAM003 — real-world capability results and independent numeric replay

**2026-10-23; research draft, unmerged.** Original authentic-data training completed for five 101,226-parameter models. **Scientific verdict: developmental physical forecast gain, full capability gate FAIL / mixed trade-off. Action imitation FAIL. Strict external replay FAIL with preserved numeric exceptions. No architectural superiority, causal control or customer claim.**

## Original training and source integrity

Original publicly licensed BodenAI/RW-RL-HIL pinned source digest: `4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776`. Exact previously exposed 160 original FIT episodes, 120 training / 40 episode-disjoint original FIT-derived DEV. Five cold-start seeds 76001–76005, each 640 guarded AdamW updates at batch12 on four original FIT-only seeded sample rounds. Original model/optimizer/Torch/NumPy/Python RNG, sample digests, source IDs, weights, raw actual targets and original forecasts saved in Git. All tests and live 101226 active/deployed-parameter guard passed. **960 factual query origins per seed (84 observed moving, 876 observed stationary), only 40 distinct DEV episodes reused across the five fits.** No sealed original TEST or untouched physical cohort opened, no artificial trajectories.

## Real next-frame physical MAE and logged-command results

Lower is better; all five models trained separately on genuine observations. Columns are the **same 960 original real prediction cases per seed**, not independent episodes.

| Seed | All physical MAE | Moving MAE (84) | Stationary MAE (876) | Next command L1 | Movement AP |
|---|---:|---:|---:|---:|---:|
|76001|0.061972|0.576933|0.012592|0.010269|0.208326|
|76002|0.063034|0.546001|0.016722|0.010792|0.102717|
|76003|0.063864|0.564057|0.015900|0.010597|0.168625|
|76004|0.065700|0.538157|0.020396|0.010381|0.103489|
|76005|0.061576|0.566103|0.013197|0.010317|0.141632|
|**Mean**|**0.063229**|**0.558250**|**0.015762**|**0.010471**|**0.144958**|

Original source persistence: all MAE **0.068668**, moving **0.743241**, stationary **0.003983**. Frozen ES-WAM v1: overall **0.076625**, moving **0.747384**. New ESWAM003 improves original overall mean MAE by **7.9% against persistence** and **17.5% against ES-WAM v1**; moving MAE is **24.9% below persistence**. Every seed beats persistence in both all-case and moving groups. However, seed76004 (0.065700) misses the preregistered >=5% overall margin (MAE<=0.065235), so even that margin is not consistent across all five seeds.

**Full physical gate FAIL.** The preregistered moving-state MAE <=0.706079 passes on all five seeds, but stationary preservation <=0.001871 (1.10 times frozen ME077 stationary MAE0.001701) fails on all five seeds. Mean stationary 0.015762 is about 9.3x ME077 and 4x stationary persistence. This is the primary remaining trade-off.

**Command imitation FAIL.** Repeat-last-recorded command has exact same-case mean normalized L1 **0.009537**; the anchored next-command head scores **0.010471** (~9.8% worse). The earlier ES-WAM v1 absolute-action head had mean L1~0.80465, so anchoring fixes the catastrophic scaling problem but still does not produce a useful learned action controller. Future logged qpos is only a supervised target, never a known-at-time action input. No direct policy utility or causal intervention effect is established.

**Motion ranking remains weak.** Mean individual-model movement average precision **0.144958**, with true observed-positive rate84/960=0.0875 and original ME078/ME079 AP0.61717/0.60775. Mean proper motion Brier **0.147101**. This probability is a separately trained event head and is not an independently calibrated causal risk.

## Frozen historical benchmark context, not an architecture winner

Original previously published ME077 physical all MAE0.066476 / stationary0.001701; ME078 native all MAE0.061221 / moving0.421182; ME079 native all MAE0.060595 / moving0.433934. Different training objectives, budgets and model families mean these context figures are NOT a fair same-compute architecture comparison. ESWAM003 does not outperform the stronger moving-state baselines, and does not preserve ME077's near-zero stationary forecasts. No newly trained equal-opportunity GRU/Transformer controls or independent untouched source were used for ESWAM003.

Native exchange-structured finite role-memory remains present and trainable. But eight real frames from a single episode entity do not qualify long-history associative retrieval, multi-entity collision exclusion, paired-role verification benefit, CASM counterfactual reasoning, or actual safe policy intelligence. Structural tests are not trained capability evidence.

## Immutable original replay verification and exact exception

The original first training and Git checkpoint preservation job completed successfully. Independent clean-checkout restored all five original Git models and recomputed the same source cases with **zero optimizer updates**. All per-case physical prediction/label, forecast mean/scale, movement probability/Brier and recorded-command metric elements satisfy the original **rtol=1e-6, atol=1e-6** numeric contract.

**Strict independent cross-CPU replay FAIL:** 194 of the original **4,800 normalized Laplace NLL** elements exceeded the exact tolerance, with seed counts **54/27/34/66/13** and maximum absolute difference **3.814697265625e-5**. Every other audited element met the unchanged tolerance. Original checkpoints, original scores, full benchmark case identities, source hash, metric formula and tolerance were NOT modified or relaxed. The complete audit is Git-owned at `research/replay-bundles/ESWAM003/CROSS_CPU_REPLAY_AUDIT.json` and the original scores/checkpoints at `research/replay-bundles/ESWAM003/`.

Full software suite **1,144 tests PASS**, independent strict complete numeric replay **FAIL**. A later explanatory numeric audit **PASS as an audit process** does not change the scientific replay qualification.

## Research and commercial assessment

**Demonstrated:** whole integrated ES-WAM physical developmental forecast accuracy improves over its former checkpoint and persistence on one previously exposed real robot source, including true movement-rich outcomes. **Not demonstrated:** joint motion/stationary success, better action behavior than repeat-last, useful causal counterfactuals, reliable long event retrieval, independent out-of-source transfer or safe operation. Original model parameter-size and compliance gates pass, but these are not proof of intelligence.

No real customer deployment, observed operator value or actual ROI. Future work needs a separately preregistered real-data study focused on simultaneous moving/stationary quality and strong native/GRU/attention controls, plus authentic multi-entity long histories and verified action-outcome data. No further fitting authorized in completed ESWAM003; leave draft PR unmerged.
