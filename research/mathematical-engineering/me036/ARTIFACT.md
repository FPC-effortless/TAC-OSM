# Original ME036 model artifacts

Both archives exist in this workspace and were independently extracted and verified.
They are local research deliveries, not GitHub Actions download artifacts. No models
were refitted to reconstruct these original trained weights.

## Compact native inference bundle

- Path: `/workspace/me036-native-inference-models.tar.gz`
- Bytes: **18,647,499** (18.6 MB)
- SHA256: `2968f492a53478b0362b46adc27b162e91dda153cf15a96edb00dfe905918a47`
- Forty selected native models: five benchmark files × four horizons × two seeds.
- Physical-input TorchScript models include train normalization buffers; input
  B×96×C, output B×H×C in publisher units. Channel order is in bundled data.json.
- No TAC package or raw CSV download needed for inference. Torch required;
  NumPy needed only for the bundled verification probes.
- Extracted separately to `/workspace/me036-inference-restored`; its verification
  script hashed every payload and replayed **1,951,488 physical forecast values**
  bit exactly across all 40 models. See inference-restoration.json.

## Complete original research archive

- Path: `/workspace/me036-full-horizon-models-and-evidence.tar.gz`
- Bytes: **1,651,798,573** (1.65 GB)
- SHA256: `2d6f8ad97a4d7710dfbe3e8ad288d890a996ff1262d2fadc26bcf754484438df`
- Source snapshot: `25382a8fd36e618e26d1f6feb50faddab0b60642`
- Indexed payloads: **5,822** plus ARCHIVE-INDEX.json.
- All **1,680** original checkpoint endpoints, Adam/optimizer and RNG states,
  learning traces, saved validation forecast probes and model selections.
- **80** TorchScript exports: standardized and physical-input variants for each
  of the 40 selected native models. Export audit replayed **3,902,976** scalars
  exactly, including package-independent fresh Torch processes.
- Twenty original ridge coefficient files and twenty separately recorded post hoc
  objective-corrected ridge files. Original incorrect-objective results remain.
- Unmodified publisher/mirror CSVs, ETT license, publisher and mirror attribution,
  source hashes, protocols, baseline source, test records and execution logs.

SHA256 was verified before extraction. Independent offline restoration to
`/workspace/me036-restored` verified every indexed payload, replayed all **1,680**
checkpoints and **81,962,496** saved validation scalars, then rehashed every payload
unchanged. No refitting, original-worktree imports or network access were needed.
The initial 840-endpoint subset was also replayed separately. These are same-CPU
restoration checks, not independent statistical or hardware replication.

Final test means cover 34,774,790,400 neural scalar predictions. Those entire
forecast arrays were not stored or independently replayed; the audited arrays
are the saved validation probes. Test scoring had an explicitly recorded scheduling
resumption that preserved completed metrics, with no model/selection changes.

See [restoration receipt](restoration.json), [archive index](archive-index.json),
[reproduction commands](REPRODUCE.md), and [results/limitations](RESULTS.md).
The registered practical gate failed. No architectural superiority, paper-table
parity, zero-shot transfer, constant-time streaming or model promotion is claimed.
The historical ME035B initial-equivalence precision failure remains unchanged.
