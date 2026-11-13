# Request-only literal-template audit — 13 November 2026

**Internal source measurement, not independently qualified or model capability.**
Continues the explicit unfinished template audit in draft723, not729's completed
snapshot fixtures. No existing model/adapter/corpus/split/label/result is changed.

Top3 priorities: source and split validity before capability fitting; exact replay
and numerical contracts; useful learned memory on authentic data and eventual
physical offline qualification. Refreshed master planaf307318, current/default
AGENTS/CHARTER/CODEX, FPC Universal, real-world/capability/anti-cheat/reward/PNDS/
phone-lab rules, two execution levels, Issue472, drafts721–729 and PNDS118/117.
The user's100k rules prevail over default scale-v2 text. Internal research is not
blocked on external signers; qualification gates remain strict. Synthetic labels
are the relevant source limitation here, not an excuse to request private keys.

## Registration and method

Protocol and router were committed before tests/source audit in ccdc7d7f. First
router receipt rejected an unknown execution_level field in this historical
router's exact schema. The initial redirected stdout file was empty (errors went to stderr).
INITIAL_PLAN_BLOCKED.json explicitly records a reproduction of that exact
rejection, not an invented original transcript. Commit33cf2a24 uses
the existing schema unchanged; execution level remains explicit in PROTOCOL.json.
No research execution preceded the successful READY_NO_FIT plan. CPU5min/GPU0.

Source: original publisher mllmTeam/DroidCall TRAIN at immutable revision
42563ae614280d2891d57f1e7057c4bc50dd27bd, publisher-declared Apache-2.0. TRAIN SHA256
4cb2d5691c1b95b0908c59efcb361c8a9c9b12f0a4d182acfc2df0ccc92e6d3b. All three original
publisher-file hashes and license declaration checked. No TEST file opened.

Reuse [Lee et al. (ACL2022), Deduplicating Training Data Makes Language Models
Better](https://aclanthology.org/2022.acl-long.577/) as prior art for contamination
checks. These bespoke conservative lexical rules are not that paper's algorithm,
semantic embeddings, exhaustive paraphrase detection or a new scientific method.

Pure request grouping accepts strings only. It computes exact existing normalized
keys, then two preregistered typed-token fingerprints: punctuation-dropped word
sequences with standalone numbers redacted, and the same with balanced ASCII
quoted spans redacted first. Typed markers avoid collisions with literal words.
No tool/answer/catalogue dictionaries or answer values inform redaction/grouping.
Full-TRAIN fingerprints are calculated before joining the immutable original
lineage ledger by raw-line hash/index/span. Cohort caveat: historical accepted
membership itself depended on labels/schema; the fixed accepted cohort is not
misrepresented as independently label-blind selection.

All methods are reported; no new heuristics are added after seeing results.
Original FIT/DEV assignments stay frozen. Complete hash-only row ledger and
cross-partition group indices support replay without publishing request text.
No tokenizer/neural runtime is imported, fitted or executed. Output refuses reuse.

## Local measured result

10,071 original requests; unchanged accepted7,837 =FIT7,065 +DEV772; other2,234
remain quarantined under the original ledger. The old exact split has zero
normalized-request intersections, as required.

| Frozen method | All-TRAIN groups | Largest group | DEV with a FIT partner |
|---|---:|---:|---:|
| Exact normalized request |10,071|1|0/772|
| Numeric literals redacted |10,070|2|0/772|
| Quoted + numeric literals redacted |10,069|2|0/772|

These rules detect very few lexical collisions. **Zero flagged overlap does not
establish semantic disjointness or absence of contamination.** Paraphrases,
unquoted variable values, tool/template families and other shortcuts remain
OPEN. False-positive lexical groups are also possible. No model was evaluated;
there is no measured effect on language NLL, memory utility or tool accuracy.
Missing no-call coverage and absent observed operational tool outcomes persist.
Do not reinterpret source preparation as real-world capability readiness.

Constructed software checks:41selected tests PASS locally (24new +17lineage),
separate from the original-source counts above. Request-only label mutation,
order invariance, exact-split contradiction, quotes/numbers/typed-marker cases,
invalid source/cohort and immutable join controls. Hosted tests/replay are separate
receipts; workflow also runs required model-independent probe checks.

## Reproduce and remaining work

```sh
python -B scripts/phone_lab.py plan research/phone_queue/tool_template_20261113.json
PYTHONPATH=src:. python -m pytest -q tests/test_request_template_audit.py tests/test_jsonl_byte_lineage.py
python -B scripts/audit_tool_request_templates.py --publisher-dir ORIGINAL/publisher --output NEW_OUTPUT
```

Reuse original acquisition42262433533, or fetch only the three named publisher
files at the immutable revision with the original preparation script; exact pins
are mandatory. Frozen lineage ZIP/sidecar are already permanent in Git under
research/evidence/tool_lineage_20261107. This audit never recomputes adapter
eligibility from labels or substitutes a newly chosen split for the old one.

Next: qualify an authentic observed language/action corpus with meaningful source
and template-family separation plus no-call/outcome support, before a new100k
integrated fit. A prospective stronger paraphrase audit needs a new protocol;
this result does not justify tuning heuristics on exposed DEV or opening TEST.

No neural graph/variant change. PNDS118c8744f69 conditional scalar Lean/Python
scope unchanged; fresh Lean NOT_RUN, full source IEEE/AD/ownership/provenance
OPEN. No new mathematical certificate follows from regex or hash checks.
H64/#483 and11/84FP32 negatives and every historical checkpoint remain intact.
95k–105k active/deployed,<=1%matched/equal originaldata/train+tune unchanged;
500khistory separate. No fit/GPU/TEST/reward/safety certification/promotion/merge/
deploy/product work. Actual device target/operators/numerical/p50p95/RAM/energy/
airplane-mode NOT_RUN. Full inference/preprocessing/embedding/state/speech must
remain offline on user-controlled hardware. No remote fallback is introduced.
Six families share evidence vocabulary; no cross-domain skill transfer is claimed.

## First hosted runtime failure

Run43386367881 failed collecting the inherited model-independent probe tests:
CPU Torch was missing from the clean CI environment. No source audit ran there.
INITIAL_HOSTED_COLLECTION_ERROR.log preserves the exact relevant log excerpt;
failed artifact14019101594 retains the original test-error XML. Repair adds
torch2.10.0 from the official CPU index for inherited tests only. The source
audit remains a separate process that rejects any Torch/tokenizer import.
No method, source bytes, split, labels or local result was changed.
