# Native 10M byte model and CPU inference results

The selected candidate has one native CSL role-state layer, width 896, state width 128, feed-forward width 2304, one bounded grouped-query attention reader and 10,103,943 trainable parameters. The byte embedding and next-byte head share weights. Initializing that matrix at 1/896 of its original scale relieved a severe bias toward predicting the current byte again. The candidate remains untrained.

## Measured cost of the selected model

Median wall time in milliseconds for a batch of one to generate 16 bytes. Each cell contains 24 warmed requests on one Intel Xeon Platinum 8573C CPU thread, with state creation, prefill and all greedy decoding steps included. Tokenization and model loading were outside timing. The three models used the same raw byte vocabulary and shared prompt fixtures. Parameter counts were 10,092,770 for upstream nanochat and 10,103,552 for the GRU. We used FP32 projections and the native TAC FP64 memory recurrence. The upstream nanochat code ran with a reduced geometry and its CPU attention fallback.

| Prompt length | Selected TAC | nanochat | GRU | TAC time saving vs nanochat | TAC time saving vs GRU |
|---|---:|---:|---:|---:|---:|
| 32 bytes | 90.35 | 93.33 | 80.44 | 3.2% | −12.3% |
| 256 bytes | 146.88 | 131.35 | 287.26 | −11.8% | 48.9% |

Only the longer-prompt comparison with the GRU cleared the 20% savings threshold. In the preceding configuration screen the wide one-layer model appeared to save 30% over nanochat at 32 bytes. Nanochat's control timing varied across runs on this shared host. Use the confirmation table for the selected model. Raw latency samples, time to first token, p95 decode times and cache consistency checks are in the confirmation receipt. No device-energy or dollar-cost measurement was made. [Confirmation data](evidence/20261119T142816-315717633cd0/output/timings.json)

## Configuration and initialization search

The first screen compared two- and four-layer native configurations. At 32 and 256 input bytes respectively, the four-layer medians were 184.94 and 298.77 ms, and the two-layer medians were 127.18 and 218.80 ms. The one-layer follow-up measured 132.54 and 165.86 ms for width 512, and 106.19 and 169.19 ms for width 896. Each screen repeated its own GRU and nanochat controls. [First screen](evidence/20261119T142200-55b8a2e2e7c1/output/timings.json) · [One-layer screen](evidence/20261119T142448-40c1ad8ad34a/output/timings.json)

Three native configurations copied the current byte in all 256 predictions per model on two authentic CPython FIT excerpts at their initial weights. Only 77 next-byte targets actually repeated. We compared three initial scales in the width-896 candidate, using six source excerpts and three paired random seeds. All 18 cases cleared the gradient, confidence and copying thresholds at inverse-width initialization. Both other scale choices failed all 18. Total copies fell from 2,304 of 2,304 to 323 of 2,304. Initial loss ranged from 5.500 to 5.551 nats per byte with inverse-width initialization, near the 5.545 uniform reference. This establishes a better-conditioned starting point. Learning curves and held-out prediction still require training. [All 54 cases](evidence/20261119T142710-afd97533975e/output/initialization.json)

## Reproduce or continue

Use the implementation in model.py with TAC-OSM a7ec101ee67fce09aaf42b5c8e880586f77be0e9 on the Python path. The constructor selected_native_byte() builds the language model without importing nanochat. Call model(byte_ids) with a batch-by-time integer tensor to obtain byte logits and state. For batched streaming, pass explicit stream_ids in the same order from prefill through decode.

The commands below assume the current directory is this directory in the research branch. Use CPython 3.12.14 and PyTorch 2.15.0+cpu. The CPU benchmarks also imported tiktoken 0.14.0, tokenizers 0.23.3 and rustbpe 0.1.0 through nanochat.

```sh
base=$(mktemp -d)
git clone --filter=blob:none https://github.com/FPC-effortless/TAC-OSM "$base/tac"
git -C "$base/tac" fetch origin refs/pull/732/head
git -C "$base/tac" checkout a7ec101ee67fce09aaf42b5c8e880586f77be0e9
git clone https://github.com/karpathy/nanochat "$base/nanochat"
git -C "$base/nanochat" checkout a825e63f81e62e2e9fd38655e9b2e39417620545
git clone https://github.com/FPC-effortless/veritas "$base/veritas"
git -C "$base/veritas" checkout eec439492407b0aa26922c23b5ae40d95289f4eb
export PYTHONPATH="$base/tac/src:$base/nanochat:$base/veritas/integrations/trainlab/src:$base/veritas/integrations/research_lab/src"
export CSL_FIT_ROOT="$base/tac/research/source-bundles/LANGUAGE-SOURCE023"
python -m trainlab run confirm-recipe.json --runs "$base/runs"
python -m trainlab verify evidence/20261119T142816-315717633cd0
```

The four original TrainLab records pin their exact implementation commits and input hashes. The later extraction of model.py preserved the selected weights, prefill logits and cached-step logits on the checked seed. Run the corresponding recipe at the commit recorded in run.json for exact historical source identity. A new run has its own latency distribution. Research Lab imports and file-hash verification receipts accompany each run. [Archived receipts](evidence)

A longer schedule now contains 33,554,432 distinct next-byte target positions per seed from WikiText-103 FIT. Source hashes, data preparation and regeneration commands are available in the adjacent directory. A comparative fit still requires the signed prefit evidence specified by the repository, a frozen quality evaluator and matched compute receipts. This machine has no CUDA device for official Mamba-3. All four experiments above ran locally with zero optimizer updates. [Larger training schedule](../wikitext103)
