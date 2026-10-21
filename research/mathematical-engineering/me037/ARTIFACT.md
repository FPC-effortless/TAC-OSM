# ME037 original trained models and evidence

Local research deliveries; neither is a GitHub Actions training artifact.
Models are not promoted or deployed. No refitting was used for restoration.

| Delivery | Bytes | Payload |
|---|---:|---|
| `/workspace/me037-native-inference-models.tar.gz` | 24,200,882 | 48 physical-input native models, probes, channel metadata and standalone verifier |
| `/workspace/me037-models-and-evidence.tar.gz` | 864,239,756 | All 864 original parent/adaptation checkpoints and complete evidence |

Compact SHA-256:

`217d15bfe4d5135c21f57b5287114e51dc31c071487904deb7c7ca906569b756`

Complete SHA-256:

`c4014fbde767960933a7ef8fd4191525c88b3e1304ae4ea08869749ce1aa363b`

Complete source snapshot: `03116e4d33e3b7765e291bebe084349b1a5e2101`. All 3,477
indexed payloads verified before and after isolated restoration at
`/workspace/me037-restored`; all 864 checkpoints replayed 37,062,144 saved
validation scalars bit exactly, without training. See restoration.json/log.
The index excludes circular archive/restoration receipts and this delivery page.
EXECUTION.md was added after packaging and describes the completed run; source
and prediction files in the archive are the frozen, executed versions.

Compact restoration at `/workspace/me037-inference-restored` independently
hashes its payloads and loads all 48 original selected models with Torch only;
2,059,008 physical forecast values replay bit exactly without the TAC package.
See inference-restoration.json/log. The complete archive also includes the
standardized exports and optimizer/RNG states for every adaptation endpoint.
Raw source CSVs remain unmodified, with publisher attribution and provenance;
model checkpoints and derived reports are separate research artifacts.

[Execution and replay commands](EXECUTION.md) · [Measured outcomes and limits](RESULTS.md)
