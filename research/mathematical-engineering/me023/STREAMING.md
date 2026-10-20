# Exact native streaming transition

After fitting the fixed experiment, add a deployment operator without changing
any training equation or parameter: `stream_step(item, metadata, state)` updates
the 64-coordinate fast/slow state with the original native recurrence. Fitting
source remains `4b17f08`; the new operator is an engineering export, not a new
fit or a selected model. Original training-source standalone files are retained.

The caller owns state and can carry it across arbitrary chunk boundaries.
Computational working state for a single step is constant in history length,
with no replay of an input window. A forecast uses the existing `readout` plus
a caller-maintained 32-event response FIFO/count for the exact analytic prior 32.
This API does not itself implement entity routing, persistence storage, or
verification of external action outcomes.

Provided finite parameters/inputs and an initial state in[-1,1], the original
convex-state invariant applies to every step. Unit tests compare irregular
chunks with the original full 128-step encoder and final readout. A fixed
1 e-6 float 32 guard is declared for this new engineering check before real probe
execution; it is not a predictive acceptance threshold.

The full-window `encode` method still materializes embedded inputs using O(T)
input working memory. The streaming API avoids that allocation for incremental
execution; this does not imply constant training memory or an associative
parallel scan. Real probe/export replay verifies original 128-event behavior.
Predictive performance beyond 128 steps and across new streams is untested.
