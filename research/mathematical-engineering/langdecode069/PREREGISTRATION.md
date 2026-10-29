# LANGDECODE069 — native ARM64 original-checkpoint decoder portability

**Real-source inference-only, no new model fit; separate host ISA.** Prior [LANGDECODE068](https://github.com/FPC-effortless/TAC-OSM/pull/607) demonstrated three clean **x64 GitHub CPU** original trained source speed passes for the guarded 067 decoder. Those do not establish portable performance on a different ISA or Android.

GitHub's standard label ubuntu-24.04-arm uses a native ARM64 Linux VM for public repositories; the pinned torch 2.10.0+cpu CPython3.13 Linux aarch64 wheel exists. This is GitHub-hosted server-class ARM64, NOT actual Android/Termux/NNAPI mobile CPU/GPU energy, memory, temperature or SoC inference. Record platform.machine, platform.processor and torch.__version__ in evidence. Require aarch64/arm64 architecture or fail before benchmark. Do NOT use QEMU emulation masquerading as native.

Registered question: Does the **identical** 067 guarded-cached positional candidate, compared with unchanged original validated 057 decoder and original fixed checkpoint, preserve exact logits/state and pass same original unprofiled speed thresholds on one clean native ARM64 CPU runner?

### Freeze before run
- No neural training, no GPU/Kaggle, no sealed TEST, original licensed CPython3.13 FIT owner prefixes128/512, seed1701 FIT320 499623-param SHA-locked checkpoint, optimizer/RNG and model source identities unchanged.
- Constructed 067 cache mutation/replacement checks and 064 boundaries on native ARM; original source state/schema code must execute as on x64. Same original 16 generated next bytes, no new source/outcomes.
- Original state 3904 bytes/FP32 stream, all nine persistent fields and every incremental logit **bitwise identical original-vs-guarded**, snapshots bidirectional, original growing-prefix logit max abs <=3e-5 and identical argmax at both prompts.
- CPU FP32 eager batch1, Torch intraop2/interop1 threads, two warmup pairs and seven measured alternate-order pairs, original per-step validation included. Require both 128B and512B median reduction >=5%, p95 candidate/original <=1.10. Otherwise archive result as ARM speed **FAIL** or **INCONCLUSIVE**; no post-hoc retuning. ARM vs x64 raw latency numbers are different hardware conditions and cannot justify an absolute architecture win.

Even with pass, status is internally measured **NATIVE_ARM64_GITHUB_CPU_ONLY**, not actual smartphone inference. Tensor._version metadata guard is not adversarial raw-storage protection. Source IEEE/AD formalization, actual phone/Android execution, throughput vs matched GRU/SSM/Transformer, and independent cross-ISA repeatability remain OPEN. Draft/unmerged.
