# TAC-OSM language prediction and inference comparison

This is a proposed comparison and a threshold checker. It is not a registered study or measured result. The user objective is to beat nanochat and a strong GRU on language prediction, match or beat official Mamba-3, and do so with cheaper inference.

## What counts as a win

Use held-out bits per UTF-8 byte, calculated as summed next-token negative log likelihood in natural units divided by ln(2) times the number of scored bytes. The four arms must score the same byte ranges. Use a shared tokenizer and the same treatment of boundary tokens in the matched study. Proposed minimum advantage is 0.01 bits per byte over nanochat and the GRU. Proposed noninferiority margin versus Mamba-3 is 0.01 bits per byte. Apply all three thresholds on each of at least three preregistered paired seeds. Report paired uncertainty by document and seed, rather than using a nonsignificant difference to claim a match.

Use synchronized end-to-end wall time on identical hardware, precision, batch, context and generated token count. On the same device at the same rental rate, device time per completed request is a cost proxy. Proposed minimum saving is 20% against each baseline in each registered deployment profile. Report time to first token, median and p95 decode latency, peak device memory, and energy if power is measured. A model that drops long prompts, generates fewer tokens, uses a different device, or changes precision cannot claim the cost saving from that run.

Freeze at least two context lengths and a fixed output length before timing. Include one short prompt and one long prompt from the workload, with a batch-1 interactive profile and a fixed-throughput profile if the deployment needs batching. Keep model/kernel loading outside warmed timing for all arms, and report cold start separately. Repeat sufficiently to characterize p95. Include tokenization if measuring a text-to-text request. Keep scorer and input byte identities with the logs.

## Baselines and next qualifying experiment

- Use the native TAC-OSM CSL lineage. Start with the existing 10M memory-scaled checkpoints to test held-out quality and carry/reset behavior. The completed campaign covers FIT only. Mean late-training loss was 1.463 for memory-scaled TAC, 1.512 for original TAC and 1.247 for GRU. The new four-way study requires newly registered scoring and matched trained models. [Completed TAC-OSM audit](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41407899927)
- Use nanochat at a825e63 as the source for the matched Transformer. Also report the published nanochat checkpoint separately as an absolute capability reference. The direct external comparison needs a pinned checkpoint and evaluator and must document tokenizer and exposure differences. [nanochat](https://github.com/karpathy/nanochat/tree/a825e63)
- Use an optimized, competently tuned stacked GRU with the same context access, training exposure and comparable parameter budget. The existing GRU is a necessary regression control.
- Use official Mamba-3 at e9594ce1c732d97440f0332fdc43170a2294dbfa, with the upstream language model configured with ssm_cfg.layer set to Mamba3. Register SISO or MIMO and kernel versions. Run a prefill and cached-step smoke check on the target GPU and independently compare results before collecting timings. The official SISO prefill is Triton and the optimized decode is CuTe. Select the baseline mode prospectively or report both. [Upstream model factory](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/models/mixer_seq_simple.py) · [Mamba-3 implementation](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py)

Before starting, pin the dataset, rights, splits, tokenizer, source commits and model geometries. Use the declared-target-v2 gate, ≤1% parameter parity, identical data and tuning budgets, and measured training compute. Freeze the exact inference workload, hardware identity and measurement rules. Record runs and checkpoint replay through TrainLab and Research Lab. The available CPU workspace cannot exercise the official accelerated Mamba-3 kernels. A GPU run, model fits and independent review are still needed for the four-way comparison.

## Check a completed report

Run python assess.py measurements.json. It takes heldout_bits_per_byte as a list of rows with an integer seed and a scores object, and inference_profiles as a list of rows with a unique id and a median_end_to_end_ms object. In both score objects use keys tac_osm, nanochat, gru and mamba3. The checker verifies the arithmetic only. Preserve the full receipts and independently verify matched inputs, run conditions, complete profiles, checkpoint replay, and reported measurements before making a claim.

## Executed native-model research

The first larger prototype, original measurements, initialization diagnosis and fresh confirmation are available in the [10M CPU study](cpu10m). The [WikiText-103 FIT schedule](wikitext103) prepares 33.55 million distinct next-byte targets per seed. The four-way quality and deployment-cost target above remains open.
