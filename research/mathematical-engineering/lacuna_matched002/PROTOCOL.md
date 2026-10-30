# LACUNA-MATCHED002: prospectively matched past contexts

Registered before first002model run.001 remains immutable: its fixed-age old
events were not required to match the current context, so their mean loss
influence does not by itself test task-relevant historical retrieval.

Use same3original640checkpoints and same2originalFITowners as001, observe first
2048bytes, queryevery8clocks t128..2047. Bin historical ages [8,32),[32,128),
[128,256),[256,513). For each bin choose the MOST RECENT observed historical
continuation j such that raw[j-4:j] equals current observed raw[t-3:t+1]. This
uses no future byte. No match => record unavailable. Source future raw[t+1]
is revealed only after every control prediction. Never select by continuation
being correct. Same source/query/j cells across seeds; counts are shared.

Native normal prediction still accepts ONLY its own state/current observed byte.
Evaluator-only transport table and supplied old keys are diagnostics. Decompose
slow writes as001 and verify source sum error<1e-5. Preserve directcurrentwrite,
query/gates/otherpriorstate when removing matching event. Wrong event chooses
index distance5..64, older than current, different observed4bytecontext, target
position excluded from wrongcontext, normmatched; missing controls reported
INCONCLUSIVE rather than dropped. Actual current writer vs matching pastwriter
key parity is structural; do not call it learned factual retrieval. Native
query alignment and fullmatrix bitcode past-byte read are separate probes.

Record all availability, context-matched historical continuation correctness,
eventremoval NLL versus normmatched wrong event, writerparity and normsurvival.
Three checkpoints, owner/seed strata, literal source positions only. Per-bin
>=20distinct available source/query cells with valid wrongcontrols required
before provisional utility gate: removaldelta>=.001, greaterthanwrongcontrol,
andpositive in eachseed/eachowner. Insufficientsupport=>INCONCLUSIVE. No iid
seed/token significance, no semantic/causal hallucination certification.

Old001 gate and all its results remain asregistered. This additional diagnostic
can inform a NEW protocol; it does not retroactively pass001 or authorize a
trained architecture. ExistingPNDS087/88 conditionaltransport formulation is
unchanged; full neuralIEEE/AD staysOPEN. Original model source/weights unchanged.
CPUbound15minutes, explicitsource-commit launch only; no GPU/DEV/TEST/newfit.
