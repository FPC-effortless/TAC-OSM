# LANGDECODE064 — cached cap_all positional features

Status: implementation + preregistration; original checkpoint CPU original-vs-candidate acceptance PENDING.

Original cap_all positional encoding for clock >=128 returns the same 7D features at t127, but computes them per step. This original-source candidate reuses one reference 7D tensor and leaves original weights, graph, causal masking, snapshots and 3904B FP32 state unchanged. Follow `research/mathematical-engineering/langdecode064/PREREGISTRATION.md`; preserve PASS/FAIL honestly, including original 062/063 latency negatives. No training, Kaggle or TEST.
