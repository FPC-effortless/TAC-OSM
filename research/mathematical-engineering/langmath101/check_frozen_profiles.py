"""Recheck original real-FIT profiles without any optimizer or DEV loader."""
import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('langmath101_common', HERE/'common.py')
C = importlib.util.module_from_spec(spec); sys.modules[spec.name] = C; spec.loader.exec_module(C)

def check(fit, initials, output):
    profile = C.read(C.ROOT/C.read(HERE/'FAIR_COMPARISON.json')['operator_profile_path'])
    bundle = C.local('data').load_fit(fit)
    assert bundle.data_identity['source_manifest_sha256'] == profile['source_sha256']
    results = []
    for frozen in profile['models']:
        arm, seed = frozen['architecture'], frozen['seed']
        directory = HERE/'profiles'/str(seed)
        schedule = C.read(directory/'SCHEDULE.json')
        batch = C.padded_batch(bundle, schedule[frozen['source_schedule_index']])
        assert C.digest(batch['tokens'].tolist()) == frozen['source_input_sha256']
        assert C.digest(batch['targets'].tolist()) == frozen['source_targets_sha256']
        C.setup(seed); model = C.make_model(arm, C.read(directory/'TOKENIZER.json'))
        C.local('initials').restore(C, model, arm, seed, initials, profile)
        assert C.state_digest(model) == frozen['weights_before']
        actual = C.local('meter').profile_without_update(model, lambda m: C.loss(m, batch),
            architecture=arm, seed=seed, input_shape=[4, 272])
        for field in ('weights_before', 'weights_after', 'counted_dense_units',
                      'operator_sha256', 'trainable_parameters', 'deployed_parameters',
                      'parameter_tensor_count', 'finite_gradient_tensor_count', 'nonzero_gradient_tensor_count'):
            assert actual[field] == frozen[field], (arm, seed, field)
        results.append({'architecture': arm, 'seed': seed, 'status': 'PASS',
                        'counted_dense_units': actual['counted_dense_units'], 'optimizer_steps': 0})
        del model
        print({'architecture': arm, 'seed': seed, 'status': 'PASS_NO_UPDATE'}, flush=True)
    C.save(output, {'status': 'PASS_ALL_FROZEN_REAL_FIT_PROFILES', 'models': results,
                    'optimizer_steps': 0, 'DEV_access': False, 'protected_TEST_access': False})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--initials', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); check(args.fit, args.initials, args.output)
