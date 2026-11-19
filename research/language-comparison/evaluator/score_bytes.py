"""Next-byte scorer for a frozen evaluation set. Context resets every 256 targets."""
import hashlib
import math

import torch

CONTEXT = 256


def score_documents(predict, documents, *, device='cpu'):
    """predict receives byte IDs only and returns [1,T,256] causal logits.

    documents yields (unique ID, precommitted SHA256, original UTF-8 bytes).
    The caller owns checkpoint loading, model.eval(), source access and split isolation.
    """
    rows = []
    seen = set()
    with torch.inference_mode():
        for doc_id, digest, raw in documents:
            if not isinstance(doc_id, str) or not doc_id or doc_id in seen:
                raise ValueError('Expected unique nonempty document IDs')
            seen.add(doc_id)
            if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(f'Byte identity differs for {doc_id}')
            raw.decode('utf-8', errors='strict')
            nats = []
            for start in range(0, len(raw) - 1, CONTEXT):
                # One preceding input byte overlaps adjacent windows. Targets never overlap.
                block = raw[start:start + CONTEXT + 1]
                inputs = list(block[:-1])
                if len(inputs) == 1:
                    # Upstream nanochat requires T>1 in its full-sequence path.
                    # All models receive the same fixed trailing pad and only logit 0 is scored.
                    inputs.append(0)
                ids = torch.tensor([inputs], dtype=torch.long, device=device)
                logits = predict(ids)
                if not isinstance(logits, torch.Tensor) or logits.shape != (1, len(inputs), 256):
                    raise ValueError('Expected one distribution over 256 bytes per input position')
                scored = logits[0, :len(block) - 1].double()
                if not torch.isfinite(scored).all():
                    raise ValueError('Nonfinite byte logits')
                targets = torch.tensor(list(block[1:]), dtype=torch.long, device=scored.device)
                # Independent arithmetic from the training loop's cross_entropy call.
                selected = scored.gather(1, targets.unsqueeze(1)).squeeze(1)
                nats.append((torch.logsumexp(scored, dim=-1) - selected).sum().item())
            n_targets = max(0, len(raw) - 1)
            rows.append(dict(document=doc_id, sha256=digest, targets=n_targets, nats=math.fsum(nats)))
    total = sum(row['targets'] for row in rows)
    if not total:
        raise ValueError('Evaluation set has no next-byte targets')
    nats = math.fsum(row['nats'] for row in rows)
    return dict(context=CONTEXT, documents=rows, target_bytes=total, summed_nats=nats,
                bits_per_byte=nats / (total * math.log(2)))
