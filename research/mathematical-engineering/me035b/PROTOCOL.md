# ME035B — Performance-driven full-history backbone refinement

Registered before ME035B fitting, after observing all first-round ME035 outcomes.
This is a second explicitly disclosed development iteration, not a fresh test.
User goal: improve the model, using failed/working mechanisms to guide fixes.

ME035's chosen multiscale/residual TAC250 scores meanMSE.432308983 versus native
reference250 .497194032 (~13.1% gain), but full-history ridge .370001457 is better.
Replace the summary-only42->21 direct path with a trainable full96x7->21 path,
initialized from that existing training-only ridge. Keep the trained native
multiscale core and joint SwiGLU features from each matching seed's selected
ME035250 checkpoint. Reset only the nonlinear output projection to zero, so
the initial predictor preserves the stronger ridge forecast. Train the backbone
and memory together with conservative adaptation; never freeze a weak base and
force memory to repair its mistakes. This is a model construction/training change.

Same authentic ETTh1 source/license/splits/horizons/global statistics as ME035.
No new cohort, later source row, KuaiRec access or synthetic primary data. Use
original source and exact parent model/ridge coefficient hashes before fitting.
No data acquisitions beyond that source. No independent validation/superiority
claim: development results guide this second design. No further iteration here.

Three new fits, seeds35001/2/3. Each32154active weights, no inactive stored head.
All inherited memory/feature weights remain trainable. New250updates*batch64,
sampler seed+350; parent memory already trained250steps on the same FIT source.
Charge original ridge fitting and parent250training to each pipeline, but report
new physical CPU separately. The14,133-weight linear path replaces903weights;
capacity changes versus ME035, so no matched architecture-ranking claim.

AdamW memory.0002,mix/output.0005,direct backbone.00001,decay1e-4,clip1.
Loss same standardized21-target MSE plus25*mean((W_AR−W_AR_initial)^2).
Anchoring limits erosion of the working linear solution while allowing it to adapt.
No bias anchoring. The output is zero initially: step1 memory gradients may be
zero; they must become active subsequently. Save distinct checkpoints at
0,25,50,100,250 with Adam/sampler/modelRNG/loss/CPU.150CPU-second fit cap.
All3fits/15endpoints freeze before any new metric. Preserve all failures.

Select lowest mean development MSE across all3seeds at25/50/100/250. Step0 is
a baseline-equivalence check only, ineligible as a trained-memory candidate.
Development competitiveness target: selected meanMSE<=1.01*ridge MSE, and better
than parent ME035250 in all3seeds. Report exact ridge gap (even if unfavorable),
per-horizon/channel metrics, MAE, full active parameter count, actual costs and
unchanged-data/source provenance. No significance inference from selected scores.
Do not declare useful memory solely because this composite preserves ridge.

Export the selected three trained models with global normalization and inverse
scaling as raw physical-input/output TorchScript pipelines. Verify all saved
checkpoint forecasts and selection, original optimizer clocks, sampling RNG,
source/parent hashes, initial ridge equivalence within1e-6 standardized units,
raw-runtime agreement within1e-4 source units and fresh-process loading. Archive
both rounds with exact source, original weights/traces, license and reports;
restore and replay independently. No merge/promotion or commercial claim.
