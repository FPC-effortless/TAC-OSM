# Unified multimodal real-data phase

Status: design only; no confirmatory result.

## Purpose

After the synthetic unified benchmark passes its instrument and capability gates,
test the same shared structural/predictive core on real language, image and audio
data. These experiments are separate from the synthetic contract and cannot
change its thresholds retrospectively.

## Language

Primary: causal next-token prediction with a tokenizer fitted on training data
only. Secondary: long-range dependency prediction and held-out compositional
factor probes.

Initial dataset: WikiText-2 for a resource-bounded from-scratch test, followed
by WikiText-103 if the implementation passes. WikiText-103 has roughly 103M
training tokens and 60-article validation/test sets; its long document structure
is appropriate for testing persistent state and long-range prediction.

No pretrained language encoder. The tokenizer vocabulary, normalization rules
and training corpus are fitted only on the training split.

## Image

Primary: masked-patch prediction and future-augmentation prediction from raw
images. Secondary: class/object probes only as diagnostics.

Initial dataset: CIFAR-10, using its fixed 50,000-image training and 10,000-image
test split. The test set is never used for augmentation statistics, codebook
fitting, threshold tuning or model selection.

The image encoder is trained from scratch. Any normalization statistics are
training-only.

## Audio

Primary: future spectral-frame prediction from one-second recordings.
Secondary: keyword classification.

Initial dataset: Speech Commands v0.02. The dataset contains 105,829 one-second
waveforms and has speaker diversity. Speaker identifiers, labels and metadata
cannot enter the predictive state except when a registered supervised task
explicitly uses the label as the post-observation target.

Speaker-disjoint splits must be enforced for any speaker-generalization claim.

## Cross-modal

Initial paired benchmark: SpokenCOCO. It supplies spoken audio captions aligned
with MS-COCO images and written captions. The cross-modal task should test
whether a persistent structural latent can retain and retrieve information
across audio, image and text without receiving the item identity.

The primary cross-modal endpoint is held-out paired retrieval at fixed candidate
budget. Secondary endpoints include paired-vs-shuffled latent separation and
memory-conditioned retrieval after the original observation is removed.

## Leakage protocol

1. Train, validation and test items are disjoint by stable source identifier.
2. Speaker/author/image identity splits are disjoint wherever the benchmark permits.
3. Tokenizer, image normalization, audio normalization, codebooks and proposal
   indices are fit on training data only.
4. Test labels are never inputs to the model.
5. Future observations are used only as post-prediction targets.
6. Persistent writes occur only after a registered verifier accepts the transition.
7. Failed retrievals never fall back to exhaustive target lookup.
8. Checkpoint selection is completed before test evaluation.
9. Test metrics are computed once from the frozen checkpoint; no threshold tuning
   uses test results.
10. Any external pretrained component is disallowed in the primary from-scratch
    comparison; a pretrained control, if later used, is a separate arm.
11. Dataset contamination is reported as a limitation when the corpus is public;
    from-scratch training is used to avoid dependence on hidden foundation-model
    pretraining.

## Capability hierarchy

A modality progresses only through these gates:

synthetic predictive capability
-> real-data predictive capability
-> cross-domain generalization
-> multimodal cross-modal grounding
-> persistence benefit after raw observation removal.

A failure at an earlier gate stops promotion to the later claim.

## Interpretation

A positive language result means the model learned the registered language
prediction task. A positive image result means it learned the registered image
prediction task. A positive audio result means it learned the registered audio
prediction task. None of these alone means human-like understanding.

The strongest target is not reconstruction quality. It is whether verified
predictive structure is reusable across time, modalities and changing task
conditions.
