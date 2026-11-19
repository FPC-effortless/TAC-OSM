"""Preregistered initialization-only comparison for the wide one-layer native model."""
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from tac_osm.byte100k_001 import byte_language_loss
from benchmark import NativeByte, verify_sources
from followup import FILES

EXCERPTS = {name: (digest, 0) for name, digest in FILES.items()}
EXCERPTS.update({
    'Doc/tutorial/appetite.rst': ('c5766ddd864601b50b54a1882f2fbec67945d421339ddea67cd545d3b0ffad55', 512),
    'Doc/tutorial/classes.rst': ('fd28b82a81acda33c4997dc931ab63d5a4fea4ef65d820b579dcf2d4bf7304a4', 512),
    'Doc/tutorial/datastructures.rst': ('3c2b7a87519b3e7b1513f5fbb12c9d37ba715c402a6bf032f1b584d555eb979b', 512),
    'Doc/tutorial/errors.rst': ('c1d8deb184bce4aa005a03407d62312a774e38dd62ba73802e5e8ace030ae34e', 512),
})


def main():
    verify_sources()
    torch.set_num_threads(1)
    cases = {}
    for name, (digest, offset) in EXCERPTS.items():
        data = (Path(os.environ['CSL_FIT_ROOT']) / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest or offset + 129 > len(data):
            raise ValueError(f'FIT source mismatch for {name}')
        cases[name] = torch.tensor([list(data[offset:offset + 129])])
    rows = []
    metrics = {}
    for scale_name, scale in [('unit', 1), ('inverse_sqrt_width', 896**-0.5), ('inverse_width', 1/896)]:
        for seed in (1901, 1902, 1903):
            torch.manual_seed(seed)
            model = NativeByte(1, 896, 128, 2304)
            with torch.no_grad():
                model.core.embedding.weight.mul_(scale)
            for name, tokens in cases.items():
                model.zero_grad(set_to_none=True)
                inputs = tokens[:, :-1]
                logits, _ = model(inputs)
                targets = tokens[:, 1:]
                loss = byte_language_loss(logits, targets)
                loss.backward()
                norms = {key: float(p.grad.norm()) if p.grad is not None else None
                         for key, p in model.named_parameters()}
                missing = [key for key, value in norms.items() if value is None]
                zero = [key for key, value in norms.items() if value == 0]
                nonfinite = [key for key, value in norms.items() if value is not None and not math.isfinite(value)]
                finite = [v for v in norms.values() if v is not None and math.isfinite(v)]
                copies = int((logits.argmax(-1) == inputs).sum())
                nll = loss.item()
                passes = (not (missing or zero or nonfinite) and math.isfinite(nll)
                          and nll <= math.log(256)+0.5 and copies < 0.75 * inputs.numel())
                rows.append(dict(arm=scale_name, factor=scale, seed=seed, source=name, offset=EXCERPTS[name][1],
                                 copies=copies, true_repeats=int((inputs == targets).sum()), target_bytes=128,
                                 initial_nats_per_byte=nll if math.isfinite(nll) else None,
                                 missing=missing, zero=zero, nonfinite=nonfinite,
                                 grad_norm=math.hypot(*finite) if not (missing or nonfinite) else None,
                                 passes=passes))
        metrics[scale_name + '_passing_cases'] = sum(r['passes'] for r in rows if r['arm'] == scale_name)
    out = Path(os.environ['TRAINLAB_OUTPUT_DIR'])
    for name, value in [('initialization.json', rows), ('metrics.json', metrics)]:
        (out/name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
