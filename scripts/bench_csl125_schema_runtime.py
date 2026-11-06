"""Bounded same-weight CPU serving-path benchmark, NOT a capability result."""
from dataclasses import asdict
import json
import os
import platform
import statistics
import time

import torch
from profile_csl125_r2_scales import configs
from tac_osm.csl125_r3 import CSL125R3Config
from tac_osm.csl125_tool_model import CSL125R4
from tac_osm.csl125_tool_runtime import last_features, legal_byte_logits


def main():
    torch.set_num_threads(1)
    result = {'kind': 'same_weight_cpu_no_fit_serving_screen', 'optimizer_steps': 0,
              'quality': 'NOT_MEASURED', 'parent_scan_failures': 'UNCHANGED_11_OF_84',
              'torch': torch.__version__, 'python': platform.python_version(),
              'machine': platform.machine(), 'threads': 1, 'seed': 61,
              'batch': 1, 'prefill_tokens': 64, 'repetitions': 7, 'warmups': 2,
              'scope': 'prefill plus next-byte scores only; grammar compile/mask and autoregressive loop excluded',
              'scales': {}}
    for scale, base in configs().items():
        torch.manual_seed(61)
        cfg = CSL125R3Config(**asdict(base), correction_rank=16 if scale != '125m' else 32)
        model = CSL125R4(cfg).eval()
        x = torch.randint(0, 256, (1,64))
        ids = torch.arange(256, dtype=torch.long)
        def original():
            out, _ = model(x, depth=cfg.layers)
            return out[cfg.layers]['token_logits'][0,-1,:256]
        def restricted():
            z, _ = last_features(model, x, depth=cfg.layers)
            return legal_byte_logits(model, z, ids)
        arms = {'original_full_logits': original, 'restricted_last_byte_logits': restricted}
        with torch.inference_mode():
            ref, got = original(), restricted()
            error = float((ref-got).abs().max())
            torch.testing.assert_close(ref, got, atol=3e-4, rtol=3e-4)
            for _ in range(2):
                for fn in arms.values():
                    fn()
            samples = {k: [] for k in arms}
            for rep in range(7):
                order = list(arms) if rep % 2 == 0 else list(reversed(arms))
                for name in order:
                    begin = time.perf_counter_ns()
                    y = arms[name]()
                    samples[name].append((time.perf_counter_ns()-begin)/1e6)
                    assert y.shape == (256,)
        result['scales'][scale] = {
            'parameters': model.parameter_count(), 'vocab_size': cfg.vocab_size,
            'layers': cfg.layers, 'last_byte_max_abs_diff': error,
            'original_logit_elements': 64 * cfg.vocab_size,
            'restricted_logit_elements': 256,
            'samples_ms': samples,
            'median_ms': {k: statistics.median(v) for k,v in samples.items()},
        }
        del model
    text = json.dumps(result, indent=2, sort_keys=True)
    os.makedirs('ci-artifacts/csl125-schema', exist_ok=True)
    with open('ci-artifacts/csl125-schema/BENCHMARK.json','w') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
