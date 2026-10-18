# DEV-061 evidence contract

The protocol was registered before fitting. `protocol.json` defines all seven
arms, ten fresh seeds, data namespaces, tuning, resource caps, primary metrics,
minimum practical improvements and promotion rules. `casm-claim-policy.json`
names the later CASM attribution deadline rather than asserting a result now.

Fitting source was committed at `1e27956`. Later commits add auditing,
documentation and an equivalent inference-only metadata-allocation optimization;
the fitting/evaluation source is held unchanged. The restoration audit checks
those source files against every fit's recorded Git commit.

Raw fits retain every pilot, selected learning rate, fresh seed, checkpoint,
validation curve and held-out trajectory. No favorable seed replaces another.
The evidence index identifies archive hashes and matrix status. See
[the results](../../docs/DEV-061-RESULTS.md), paired seed tables and curves.
No candidate passes promotion. A separately registered numerical correction
preserves the original flawed retention fits; neither corrected arm passes the
physical no-regression guard.
Statistics and release gates are derived from these records, not hand-edited.

The archived frozen-reference diagnostic is generated separately with no new
training. It may execute while fresh models train, but cannot influence their
fixed data, recipes, selected learning rates or weights. Learned-model test
rollouts open only after all 70 fresh fits have closed.

Native Veritas replays execute recorded synthetic effects and check physical
state, action accounting and completion. They do not independently discover the
simulator's laws or validate a real industrial outcome verifier. Failed rollouts
from every arm/split remain part of the replay evidence.
