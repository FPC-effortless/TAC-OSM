# CSL-125-R2 independent model sizes — prefit shape/runtime feasibility only

Three separately instantiated models under the **hard 125 million learned parameter maximum** are source-checked. The smaller versions are NOT physical binary slices of the largest; only 2/4/8 etc exits share weights *within* each independently trained configuration. No 32K, 8K or 4K byte-fallback BPE tokenizer has been trained, and vocab sizes are declared model capacity only.

| Research scale | Actual learned parameters | Vocabulary capacity | Layers/exits | Width / FF / state | Local attention |
|---|---:|---:|---|---|---|
| CSL125R2-1M | 1,037,060 | 4,096 | 8, exits 2/4/8 | 112 / 128 / 16 | layers 4/8, GQA 4:1 |
| CSL125R2-10M | 9,739,421 | 8,192 | 12, exits 2/6/12 | 256 / 576 / 80 | layers 4/8/12, GQA 8:2 |
| CSL125R2-125M | 123,822,574 | 32,768 | 20, exits 2/8/20 | 768 / 1,408 / 256 | layers 4/8/14/20, GQA 12:3 |

All parameters include tied token embedding/output, every role/head/attention/FFN layer, and the new write-gate projections. The 1M and 10M models are within ±5% of optional v2 scientific scale targets if subsequently preregistered, but this no-fit screen is not a FAIR_COMPARISON registration or training authority.

Measured CPU PyTorch FP32 reference only, one thread, batch1, 32 tokens, three timing samples, **no token-vocabulary projection**:

| Model size | First exit tokens/s | Middle exit tokens/s | Full exit tokens/s |
|---|---:|---:|---:|
| ~1M | 12,140 | 4,290 | 2,698 |
| ~10M | 5,325 | 1,641 | 720 |
| 123.8M | 1,317 | 325 | 121 |

These runs use random weights, differing vocabularies/representations and no quality evaluation. Their absolute timings cannot establish cross-scale intelligence or mobile energy. Larger models must justify their extra compute through preregistered matched data/exposure/quality and credible Transformer+SSM controls. Before scaling trained experiments, the small mechanism must show *learned* identity binding and nontrivial write protection, including negative controls, under source-approved conditions.
