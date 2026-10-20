# ME005 native TAC mathematical contracts and unresolved proofs

TAC-OSM remains a native predictive-executable world model, **not an SSM**. The SSM literature supplies mathematical engineering discipline and competitive baselines rather than a mandated architecture.

## Proved local properties under explicit assumptions

1. **Convex state invariance.** For \(h,p\in[-1,1]^d\), \(g\in[0,1]^d\) and \(h'=(1-g)\odot h+g\odot p\), each coordinate of \(h'\) remains in [-1,1]. This applies to gated native fast/slow states and similarly gated assembly states starting in range, but does not prove useful memory.

2. **Forward invariance is not gradient stability.** For \(p=\tanh(Uh+Vx)\) and \(g=\sigma(W_h h+W_xx)\), the state Jacobian is \(J_{h'}=\mathrm{diag}(1-g)+\mathrm{diag}(g)J_p+\mathrm{diag}(p-h)J_g\). Using \(\|J_p\|_2\le\|U\|_2\) and \(\|J_g\|_2\le\frac14\|W_h\|_2\), we obtain a conservative bound \(\|J_{h'}\|_2\le 1-g_{\min}+g_{\max}\|U\|_2+\frac12\|W_h\|_2\). It can exceed 1, and products of Jacobians can still vanish. ME001C/ME002 previously diagnosed this limitation.

3. **Fast associative norm.** \(A_{t+1}=\Pi_{\|\cdot\|_F\le 2}((1-\lambda)A_t+\eta g_t (u_t-\hat u_t)p_t^\top/d)\), where \(\Pi\) is projection onto a Frobenius ball. This guarantees \(\|A_t\|_F\le2\), not freedom from interference. Predictive innovation is NOT the same as independently verified decision advantage.

4. **Sparse graph simplex.** Sparsemax returns nonnegative row weights summing to one and implements Euclidean simplex projection. Identical exchangeable assemblies receiving indistinguishable inputs may remain identical; specialization and semantic entity identity must be measured, not asserted.

5. **Bipolar role binding.** For \(r_i\in\{-1,+1\}\), the VSA operation \(B_r(x)=r\odot x\) obeys \(B_r(B_r(x))=x\). This proves algebraic unbinding only, not held-out learned compositional reasoning.

6. **Fixed-target energy descent.** For \(E(z,c)=\|z-c\|^2/2\) and \(z'=(1-\alpha)z+\alpha c\), \(0\le\alpha\le1\), \(E(z',c)=(1-\alpha)^2E(z,c)\). The learned target c may be incorrect or unsafe; energy descent alone does not establish reasoning quality or external physical safety.

7. **Original trained forecast preserved at initialization.** Define \(\hat y_{new}=\hat y_{ME004}+W_Rh_\eta+b_R\), freeze original real-trained ME004 native TAC weights, and initialize \(W_R=b_R=0\). Then initial outputs equal the original model. After adapter learning, quality must be rechecked; original ME005 full adapter gets worse than the frozen base in one seed.

8. **Action/evidence trust boundary.** Continuous critics may rank risk but must never authorize real tools. The revised provisional/persistent evidence interface refuses promotion without an external authenticated receipt validator and disallows reusing the same receipt. This is a fail-closed software contract, not proof the external validator is truthful.

## Computational engineering

With T observation updates, F feature tokens, N latent assemblies, d assembly width, K learned operator families, m recurrent width, and R refinement iterations: native matrix recurrence O(Tm²), feature-to-assembly binding O(TNFd), inter-assembly graph scoring/mixing O(TN²d) plus sparsemax sorting O(TN²log N), fast associative updates O(Td²) with O(d²) state, operator executions O(TKd²) for dense widths, and refinement O(Rm²) depending on head design. Inference state is finite, but neither constant-cost scanning nor Mamba-equivalent GPU performance is demonstrated. State-dependent plasticity complicates associativity.

## What the current science actually shows

ME005 mathematical reference: 9/9 original checks pass, plus GitHub independent unit suite. Authentic paired 4-fit exploratory MetroPT adapter study: **original backbones unmodified**, four SHA256-verified checkpoints, new modules connected and trainable. Full adapter gives negligible MAE benefit over a residual-only control at **about 1.57x training CPU time**, and does not consistently outperform original pretrained native TAC. No architecture promotion.

Frozen post-training five-pathway ablations reproduce original real source results with no refits. Graph has opposite-sign impacts across seeds, operator/refinement minor benefits, plasticity inconsistent tiny effects, and oscillatory modulation is negligible. They do not prove emergent reasoning, causal attribution, entity discovery, or commercial value. Preserve all historic negative findings and alternatives.

**Next evidence:** calibrated real per-horizon and action-relevant prediction, strong compute-matched GRU/boosted-tree controls, actual entity-correspondence outcomes, held-out compositional operations, authenticated action receipts and defensible costs. Do not add further mechanisms merely for novelty.
