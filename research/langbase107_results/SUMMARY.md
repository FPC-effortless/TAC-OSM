# LANGBASE107 — exploratory matched Transformer result

Three 998,816-parameter causal Transformers completed 4,096 AdamW updates on
the exact LANGDEPTH106 Tiny Shakespeare TRAIN split. Seeds, byte-window
offsets, batch size, sequence length and training presentations match the
three frozen 998,024-parameter four-layer TAC runs.

| Mean across three seeds | Four-layer TAC | Causal Transformer |
| --- | ---: | ---: |
| Parameters | 998,024 | 998,816 |
| TEST reset NLL, nats/byte (lower is better) | **1.6448** | 1.7247 |
| TEST reset next-byte accuracy | **51.53%** | 48.77% |
| TEST continuous-carry NLL | **1.6234** | 1.7113 |
| CPU recurrent decode, bytes/s | 206.5 | **663.0** |
| Carried FP32 tensor bytes | **295,936** | 524,288 |
| Training wall seconds per fit | 736.9 | **325.0** |

TAC had lower reset-window TEST NLL in all three paired seeds. Its mean NLL
advantage was 0.07993 nats/byte and its accuracy advantage was 2.76 percentage
points. The Transformer trained about 2.27 times faster and decoded about 3.21
times faster on the respective two-thread GitHub CPU fixtures. The Transformer
used about 1.77 times as many carried tensor bytes at a 128-position KV window.

Continuous carry improved both systems. TAC improved by 0.02141 nats/byte;
the Transformer improved by 0.01347. This does not isolate a TAC memory-bank
effect because TAC carries its associative banks, rotating field and local KV
state together.

The comparison is exploratory and post-hoc. The TAC TEST result was known
before the Transformer control was registered, complete training FLOPs are not
certified matched, and Tiny Shakespeare is a small public language benchmark.
These results support a quality/speed tradeoff under this recipe; they do not
establish general architecture superiority.

