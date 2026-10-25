# Freeze-order exception: full study gate BLOCKED

All five trained checkpoints existed before scoring and have now been preserved
in original bytes with optimizer/RNG/schedules and live receipts. However, the
attempted `git add -f` failed because the directory was outside sparse checkout;
the shell did not stop. Scoring ran before the checkpoint commit. The eventual
commit adce8823 contains originals AND original scores, not a pre-score freeze.
Numerical thresholds passed, but `registered_gate_passed` in raw RESULTS refers
only to those arithmetic conditions. **Overall scientific/promotion gate BLOCKED**.
No retroactive preregistration or clean freeze is claimed. No model was tuned or
retrained inside ME096 after the scores. All five attempts count in the ledger.

ME097 is an explicitly registered unchanged-protocol replication with five NEW
seeds and separate budget, not a hidden retry. Its scoring checks committed Git
bytes against every original and refuses to run on uncommitted files. It retains
the exact data, model, optimizer,256-update budget and numerical thresholds. Its
result remains exposed-DEV descriptive evidence, never independent qualification.
