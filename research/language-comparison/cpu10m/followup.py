"""Repeat the frozen timing screen and inspect initial gradients on original FIT bytes."""
import hashlib
import json
import os
from pathlib import Path

import torch
from tac_osm.byte100k_001 import byte_language_loss
from benchmark import make, measure

FILES = {
    'Doc/howto/unicode.rst': 'c5cf5081618e2b06a519d2ede00fb52a53922c5a8d636c4be337810fed2bd17c',
    'Doc/tutorial/appendix.rst': 'a932d7c8e7a180520ff7a83a12195be9d48cef684baf95045dd263d3f2e1fb0c',
}


def gradients():
    rows = []
    for arm in ('tac_two', 'tac_one_512', 'tac_one_896'):
        model = make(arm).train()
        for name, digest in FILES.items():
            data = (Path(os.environ['CSL_FIT_ROOT']) / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f'Changed FIT source {name}')
            tokens = torch.tensor([list(data[:129])])
            inputs = tokens[:, :-1]
            model.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            targets = tokens[:, 1:]
            loss = byte_language_loss(logits, targets)
            loss.backward()
            missing, nonfinite, zero = [], [], []
            for key, parameter in model.named_parameters():
                gradient = parameter.grad
                if gradient is None:
                    missing.append(key)
                elif not torch.isfinite(gradient).all():
                    nonfinite.append(key)
                elif gradient.abs().max().item() == 0:
                    zero.append(key)
            rows.append(dict(arm=arm, source=name, source_sha256=digest, target_bytes=128,
                             parameter_tensors=len(list(model.parameters())), missing=missing,
                             nonfinite=nonfinite, zero=zero, initial_nats_per_byte=loss.item(),
                             copies=int((logits.argmax(-1) == inputs).sum()),
                             true_repeats=int((inputs == targets).sum())))
    (Path(os.environ['TRAINLAB_OUTPUT_DIR']) / 'gradients.json').write_text(
        json.dumps(rows, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    measure(('tac_one_512', 'tac_one_896', 'nanochat', 'gru'))
    gradients()
