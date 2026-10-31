The first local receipt, DIAGNOSTIC_LOCAL.json, is preserved unchanged: 48 PASS,
6 FAIL. The six failures required bitwise equality across duplicate rows in a
batched FP32 kernel. Maximum differences were 5.0e-8 to 6.7e-8, far below the
preregistered 2e-5 FP32 correspondence tolerance. This was a test-design error,
not evidence of hidden history access.

The corrected implementation retains the original tolerances, checks identical
argmax predictions, and adds an exact repeated singleton-call control. Exact
snapshot/determinism tests remain exact. No optimizer ran, no model/source was
changed, no capability threshold was tuned, and the original failure is retained.
