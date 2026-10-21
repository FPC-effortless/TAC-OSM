# Final scoring execution resumption

All training, selections, source hashes and the full stride1 metric contract
were frozen before scoring. The initial evaluator allocated one complete dataset
per worker. After four datasets completed, that left one CPU idle while Electricity
continued. Resume remaining model tasks on the same two CPU quota units.

Preserve every completed metric from the original per-dataset/progress receipts
bit for bit. Stop the original evaluator, recording its CPU use and the number of
completed records. One in-progress arithmetic score may restart; this is numerical
execution recovery, not another model fit, selection opportunity or test-driven
hyperparameter change. No completed model is re-scored. Architecture, checkpoint
hashes, original scoring function, split, stride, targets and float64 aggregation
are unchanged and verified before continuation. The original evaluator and its
source hash remain intact; this supplemental executor only changes task scheduling.
All attempts and partial execution overhead remain recorded.
