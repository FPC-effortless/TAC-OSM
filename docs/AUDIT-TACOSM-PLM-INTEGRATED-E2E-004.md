# Audit — TACOSM-PLM-INTEGRATED-E2E-004

Status: READY FOR CONFIRMATORY EXECUTION

Benchmark: independently corrected E2E-003 generator; q1 and q2 target distinct
entities and every evaluation composition is registered held-out and absent
from TRAIN_COMBOS.

Information flow: typed_learned uses only current observations and the
supervised representation bit projection before action. typed_oracle is an
execution ceiling only. No answer, correctness, or future outcome enters
before action.

Addressing: both arms use explicit entity addresses, fixing the address variable
after E2E-003.

Statistics: five seeds; 300 training steps for typed_learned; 400 evaluation
episodes per seed; 5000 bootstrap draws.

Interpretation: oracle q2 must equal 1.00. Otherwise the exact execution
ceiling is invalid. If it equals 1.00, typed_learned is evaluated against the
0.80 capability criterion and 0.05 materiality threshold versus frozen E2E-003.
