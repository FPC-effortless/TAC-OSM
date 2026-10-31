# LANGBASE107 — matched Transformer control engineering

LANGDEPTH106 showed that the four-layer, width-120 LANGCACHE104 allocation is
the stronger approximately one-million-parameter TAC configuration on Tiny
Shakespeare. LANGBASE107 prepares the next necessary control: a conventional
causal Transformer with a real incremental KV cache.

| Candidate | Layers | Width | FFN | Parameters |
| --- | ---: | ---: | ---: | ---: |
| LANGCACHE104 reference | 4 | 120 | 224 | 998,024 |
| Causal Transformer control | 4 | 128 | 680 | 998,816 |

The gap is 792 parameters, about 0.079%. The Transformer uses eight attention
heads, RoPE, pre-normalization, dense causal attention, GELU feed-forward
blocks and a tied 256-byte embedding/readout. During incremental inference it
stores the most recent 128 key/value positions per layer. It has no TAC
fast/slow bank, rotating signed field, associative lookup or recurrent summary.

The no-fit tests require causal future noninterference, full-sequence versus
token-at-a-time numerical agreement, finite connected gradients, bounded KV
storage and the exact parameter geometry. These are engineering properties,
not learned capability evidence.

The qualified scientific lane remains blocked because it requires authentic
primary outcomes, complete matched training-compute accounting and an
independently signed prefit source/reward/goal/access receipt. The user's
persisting authorization permits a separately labeled exploratory CPU lane:
three Transformer fits use exactly the LANGDEPTH106 seeds, sampled TRAIN
windows and 4,096-update exposure. Tiny Shakespeare remains a diagnostic, the
earlier TEST partition has already been opened, and the resulting post-hoc
comparison cannot be promoted as qualified architecture-superiority evidence.

