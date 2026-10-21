# ME051 recurrence and factual retention objective

This preserves the ME038 native fast/slow/associative dynamics, not an SSM conversion.

From strictly historical user events `e_i = (movie_i,rating_i,timestamp_i)`, the last H entries (H∈{16,64,256}) are left-padded to width256 with an **explicit zero validity mask**. Masked slots are never interpreted as evidence. For j∈{1..32}, aggregate eight chronological slots to one token:
```
q_j = sum(mask_i * Emb(movie_i))/max(1,sum(mask_i))
r_j = sum(mask_i * (rating_i-3)/2)/max(1,sum(mask_i))
d_j = sum(mask_i * clip(log(1+elapsed_hours_i)/12,0,2))/max(1,sum(mask_i))
t_j = valid_j * tanh(W [q_j,r_j,d_j,valid_j]+b).
```
This retains the order of chronological blocks but loses exact within-block item order for long H, a declared information bottleneck. It evaluates 256 **observed past events** with **32 recurrent updates**, NOT 256 recurrent updates.

Native state at each compressed token is unchanged original ME038 `(f,s,M) -> step(t_j,f,s,M)`, with fast32, slow32 and 8×8 error-corrective associative memory. GRU64 and pooled-context controls observe the *identical* tokens. All use common 8-dimensional movie embedding (padding-index0), input adaptor and candidate-conditioned rating/retrieval heads (source ratings 1..5):
```
z=Memory(t_1,...,t_32)
rating_hat = 3+2*tanh(head([z, Emb(current_logged_movie)]))
retrieval_hat = 3+2*tanh(retrieval(z)).
```
The target movie was chosen and rated in the observed real system; conditioning on its ID is **not causal intervention**, bid selection or a learned reward-value policy.

Let `O_H` be the observed factual mean rating for the earliest quarter of the last H past events, `R_H` the most recent quarter, and `y` the current observed rating. The only within-native experimental difference is the historical retrieval target:
```
L_old     = mean(((rating_hat-y)/2)^2 + 0.20*((retrieval_hat-O_H)/2)^2)
L_recent  = mean(((rating_hat-y)/2)^2 + 0.20*((retrieval_hat-R_H)/2)^2).
```
Both heads have identical active trainable parameters, see identical real target windows under the same seeds, same number of recurrent steps and optimizer updates, and receive nonzero gradient from BOTH losses. GRU old/recent use exactly the same controlled objective contrast. Full training FLOPs and real optimizer CPU must be recorded; comparable training exposure and active capacity do not imply machine-time equality. The context model learns factual recent retrieval.

The important metric is heldout **next actual human rating MAE**, not reconstruction alone. Improved factual reconstruction without better rating predictions does not establish greater intelligence. Comparisons to GRU/context have different model structure and likely parameter/CPU budgets; report their numbers as diagnostic references, not as a fair matched-resource architectural win.

Prequential test histories may include **strictly earlier** observed DEV/TEST ratings but never future or same-timestamp outcomes, and their target predictions are not causal action effects. Original data usage is research only under GroupLens terms.