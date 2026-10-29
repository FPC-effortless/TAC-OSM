# LANGMEM077 preregistration — prove/measure native old-state contribution decay

**Science question:** In LANGSTREAM076 the source native-memory one-step lesion caused a ~0.033–0.040 NLL penalty, but erasing native matrix once at clock128 caused only ~0.00026–0.00061 later NLL penalty. The leading explanation may be source-native update mathematics, not necessarily learned address collisions. This study is a **no-fit** falsification of that specific source law, not another model-size fit, a novel memory architecture or independent qualification.

## Exact source-derived equation and prediction

For each genuine observed byte (with write admission true), original `NativeStreamingTAC076` computes
`N_next = N P_t + Q_t`, `P_t = α_t (I - β_t w_t w_t^T)`,
`α_t = 0.995 sigmoid(g_t[0])`, `β_t = sigmoid(g_t[1])`, and `w_t=previous_key`, an L2-normalized k-dimensional write address. The original g_t is computed from current lexical/position input and does not consume native N_t.

For two runs with **identical true publisher bytes and owner clock**, differing only in native matrix at clock128, `Δ_(t+1)=Δ_t P_t`. Q cancels and the P_t factors are identical. Because β∈(0,1), ||w||<=1 and α∈(0,.995), `||Δ_(128+d)||_F <= ||Δ_128||_F ∏_(t=128)^(127+d) α_t` in exact real algebra. For directions orthogonal to w, retention α; for direction w, α(1−β||w||²). This is the **upper bound for the contribution of one old native-only state**, not a full-history memory theorem: other aligned/upper/concept/wave/basis streams still carry information and can be updated through native output. FLOAT32 cancellation imposes a measurable noise floor.

## Original, pre-committed numerical protocol

- The nine original 070 m8 /075 m12,m16 500K trained source model/seed checkpoints; restore original model/optimizer and all RNG/shas using already verified LANGSCALE075 read-only receipt #40881916911 and same original source owner+score identities. No fitting, no protected TEST, no Kaggle GPU or checkpoint modifications.
- Four authentic licensed publisher original CPython3.13 **already exposed DEV** owners with actual original bytes0..127 prefilled then original bytes128..319 ingested in order, no state reset within owners. Each clock t in128..319 predicts actual source `raw[t+1]` only after t input is supplied. **Exactly 192 genuine next-byte targets per owner = 768 per trained checkpoint**, all owner/clock/label exact SHA identical across nine.
- Freeze measured ages **1,2,4,8,16,32,64,128,160** after clock128 native-only erase (t128,129,131,135,143,159,191,255,287). At all 192 steps/owner trace α,β, previous_key norm, current query/write cosine, exact source P, observed Δ native and predicted Δprev P, max FP32 source-error <=1e-4, per-step accumulated α product upper bound allowed +1e-4 numerical floor, direct native-read projected delta, original vs same-source erase NLL/Brier/Top1. Keep full unrounded per-owner/per-time original rows and all negative results.
- `mean-log α` defines **a source-native contribution upper-bound half-life** in tokens, not a true empirical memory half-life; report `log(.5)/mean(log α)`. Distinguish observed matrix difference that falls to FP32 cancellation floor from exact proof.
- Query/write angular cos similarity measures geometry only, not fact recovery. Distinguish whole native-memory effect from aligned memory and upper state and from one-step newest-source write. No independent holdout or new trained performance claims from prior exposed DEV.
- Fail if sources, checkpoint hashes, sample schedules, owner names, exact observed target bytes, state shapes/clock, no-fits, masks, original prefix parity or any selected age differ. Report original k8, k12,k16 state sizes 3904/4752/5728B. Source reference theorem limited Nat/real affine and scalar retention; full PyTorch/source IEEE/AD and global graph proofs remain OPEN.

**Conclusions can only be scoped to:** in this original trained source family, the *initial clock128 native-only state perturbation* is contracted by the actual source retention gate and directional erase, at measurable rate; whether other channels store useful long-term facts and whether memory has trained address collision/interference requires separate source-grounded event-recovery evaluation. Do not promote a new architecture, claim a safer model or generalize to held-out unseen source.
