# Preserved recovery attempts and scope

- Initial no-fit manifest preflights failed required source metadata. Copied
  original086 publisher/license/split metadata into the no-fit registration;
  preflight passed before anchor generation or screen inference. No gate relaxed.
- Full-history clone from the original workspace partial clone failed when Git
  refused to lazily fetch an absent historical promisor blob. This was a source
  clone limitation, not checkpoint corruption. A direct shallow remote clone
  succeeded, and offline nine-checkpoint replay passed from that checkout.
- Independent GitHub checkout replay40996629169 passed. The original frozen FIT
  screen also replayed all nine in another checkout before source inference.
- New cross-host anchors matched within3e-5/3e-5 tolerance, maximum8.583069e-6;
  zero complete outputs/states were cross-host bitwise identical. Within-host
  cold reload was bitwise for allnine; direct clone on samehost also matched.
- Original086 saved global NumPy keys, not its complete tuple; the payload is
  preserved exactly. Its unused seeded global state can be reconstructed under
  inspected original code, but is not mislabeled as original complete metadata.
  Actual local sampler seed/order/window/count hashes were independently rebuilt.
- Original jobs did not save FIT logits/state anchors. Newly created anchors are
  labeled new. Equality to missing historical outputs and complete training-
  resumption certification remain INCONCLUSIVE, not fabricated PASS.

No original weights, model equations or archive bytes changed. Scientific FIT
screens were a separate registered phase. No research fit or DEV score was repeated.
