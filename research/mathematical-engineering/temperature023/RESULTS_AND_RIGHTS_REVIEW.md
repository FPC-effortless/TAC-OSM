# TEMP023 — recovered original artifacts: transfer-byte audit and public IP gate

**28 October 2026. Draft System-1 language engineering.** Extends TEMP022 PR #551 and SOURCE023 PR #552 without changing their original checks, corpus splits, negative findings, checkpoints or status. No new neural fit, GPU, protected TEST, publisher-source redistribution or merge.

## What was *actually* executed locally
1. Located original user-owned 84,037,332-byte \`tac_language_long003_replay.zip\` in the research environment, plus its previously staged TEMP022 8 MiB numbered archive shards.
2. Verified all **11/11 original shard SHA256s**, reassembled the archive and matched whole-parent SHA256 \`cde9a91be40a4e613975c0bd8caa4c19ac24e3d1d0e993db5cacfbd002fecc32\`.
3. Verified original \`SOURCE_PARTITION_MANIFEST.json\` SHA256 \`4c7703dca3c5920055086687e07c24098766a2301b773cfb99c34f0a394a4b32\` and **36/36** original Vim 9.1 published chapter byte SHA256s.
4. Verified recovered \`ORIGINAL_RESULTS.json\` SHA256 \`4c197b5623b6dca74f7c8a25ba6931b7d83b25b1198a66898458860c672c34cd\`.
5. **Executed the original TEMP021 Torch2.10+CPU+NumPy2.3.5 calibrated inference in a separate Python process**, using source files exclusively from the staged handoff, and compared output *bytes* to the original receipt. **Byte-identical replay PASS for all five checkpoint seeds.** Score NLL unchanged from TEMP021: identity 2.5970073958, calibrated 2.5883772475 on three already exposed DEV chapters. Top1/top5 unchanged.
6. Independently ran a **strict offline no-unpickle/no-model** transfer audit on the full handoff ZIP: archive-shard structure, original parent hash, internal original protocol, 36 source hashes, original receipt, exact files, source-license/notice metadata and withheld publication flag. PASS. Four constructed negative tests passed: malicious path, duplicate ZIP member, wrong manifest and self-approved publication flag. They are engineering checks, **not new model results**.

## Source/publication review (NOT yet approved)

The installed original Vim \`runtime/doc/*\` is listed as **OPL-1+** in Debian \`/usr/share/doc/vim-runtime/copyright\`. Vim \`uganda.txt\` points to the **Open Publication License** for the manual, separate from Vim software license. OPL-1+ requires preserving publication notice/license terms and attribution; the handoff includes the unchanged upstream \`uganda.txt\` notice and Debian licensing record. This is a **provenance review, not an IP/legal sign-off**. The original 84 MB user-trained archive includes 15 trained model/optimizer files, executable Python model/training code and historical research material. Whether the original *model weights and all archive source code* may be published in a public repository is NOT authorized by publisher document licensing alone. The staged \`SOURCE_RIGHTS.json\` intentionally says \`redistribution_reviewed=false\` and the new script refuses to treat it as true. The original artifacts **must not be uploaded to the public Git repository yet**; user/owner permission and exact source/release rights review remain separate.

## Repository code and reproducibility

\`scripts/audit_temp023_original_transfer.py\` takes \`--transfer ... --output ...\`; it validates the downloaded original TEMP022 handoff archive **offline**, refuses tampering/traversal or extra members, checks publisher documents and reports **\`public_git_upload=BLOCKED_PENDING_MODEL_IP_AND_SOURCE_REDISTRIBUTION_REVIEW\`**, without opening pickle payloads or executing model code. The accompanying \`tests/test_temp023_original_transfer_intake.py\` provides synthetic *software-only* safety negative controls. GitHub CI runs source-only tests and explicitly states no original binary replay is performed in CI.

Example from an owner-controlled machine, **without publication**:

\`\`\`bash
python scripts/audit_temp023_original_transfer.py \
  --transfer /path/to/TEMP022_original_checkpoint_source_transfer.zip \
  --output /path/to/TEMP023_ORIGINAL_TRANSFER_AUDIT.json
\`\`\`

Actual original checkpoint/source bytes are preserved in a user-downloadable transfer archive, not committed to Git/LFS. **\`REPO_COLD_REPLAY=BLOCKED\` still**, despite separate-process local original replay passing. The future step is a documented, approved private-or-public archival destination and an independently reproducible runner; do not automatically equate private user-owned artifacts with authorized public redistribution.

## Scientific relationship to PR #552

The separate genuine CPython3.13 \`SOURCE023\` 18-document byte corpus had **ASCII-only original FIT**, then a separately registered Unicode FIT supplement. It is not the original Vim9.1 archive used to train the 500K model. This study does **not** train the untrained 256-symbol/499978 parameter candidate, merge the datasets, infer 256-byte ability, retune temperature on exposed documents, use new TEST data or claim more capable language intelligence. PNDS #61 scalar proofs unchanged; full end-to-end numerical/AD proof still OPEN.

**Gates:** original local artifacts authenticated PASS; original separate-process replay PASS; 11/11 archive shard recovery PASS; source rights/publication authorization PENDING; public Git binary upload NOT_PERFORMED; Git cold replay BLOCKED; independent TEST NOT_RUN; new predictive skill NOT_ESTABLISHED.
