"""SENSE-TOK training worker and source path. No authorization from this module.

The guarded entrypoint must qualify evidence before calling fit_worker. TEST
contents are never opened. A signed external bundle must bind all source and
execution contracts; hashes alone establish equality, not publisher authority.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random
import torch
from .adaptive_token_controls import TokenComparisonArm


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_corpus(root: Path, document: dict, history: int):
    """Verify local FIT/DEV files; inspect only metadata for sealed TEST."""
    if type(history) is not int or history < 1:
        raise ValueError('positive history required')
    root = root.resolve()
    rows = document.get('files')
    if not isinstance(rows, list) or not rows:
        raise ValueError('original corpus file records required')
    seen_ids, hashes, groups = set(), {}, {}
    data = {'FIT': [], 'DEV': []}
    tails = {}
    for row in rows:
        split, identity, group = row['split'], row['source_id'], row['group_id']
        if split not in ('FIT', 'DEV', 'TEST') or not identity or not group or identity in seen_ids:
            raise ValueError('invalid/duplicate source identity')
        seen_ids.add(identity)
        sha = row['sha256']
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('invalid source hash')
        if sha in hashes or group in groups and groups[group] != split:
            raise ValueError('duplicate content or source group crossing splits')
        hashes[sha], groups[group] = split, split
        if split == 'TEST':
            continue  # no path lookup, stat, existence check or byte read
        rel = Path(row['path'])
        path = root / rel
        if rel.is_absolute() or '..' in rel.parts or not path.resolve().is_relative_to(root):
            raise ValueError('escaped source path')
        if path.is_symlink():
            raise ValueError('source symlink denied')
        raw = path.read_bytes()
        if digest(raw) != sha or len(raw) != row['size_bytes']:
            raise ValueError('source bytes changed')
        tails[identity] = len(raw) % history
        chunks = [raw[i:i+history] for i in range(0, len(raw)-history+1, history)]
        data[split].extend((identity, i*history, chunk) for i, chunk in enumerate(chunks))
    if not all(data.values()) or 'TEST' not in groups.values():
        raise ValueError('FIT/DEV windows and sealed TEST metadata required')
    data['_discarded_tail_bytes'] = tails
    return data


def batch_at(windows, *, seed, step, batch_size, device):
    """Pure sample schedule; prefix replay does not depend on previous calls."""
    if not windows or min(step, seed) < 0 or batch_size < 1:
        raise ValueError('invalid batch schedule')
    rng = random.Random(f'SENSE-TOK-006:{seed}:{step}')
    selected = [windows[rng.randrange(len(windows))] for _ in range(batch_size)]
    targets = torch.tensor([list(x[2]) for x in selected], dtype=torch.long, device=device)
    ids = torch.cat((torch.full((batch_size, 1), 256, dtype=torch.long, device=device), targets[:, :-1]), 1)
    draws = torch.tensor([[rng.random() for _ in range(targets.shape[1]-1)] for _ in selected], device=device)
    receipt = [(identity, offset, digest(raw)) for identity, offset, raw in selected]
    return ids, targets, draws, receipt


def learning_rate(step, config):
    total = config['steps_per_arm_seed']
    warmup = config['learning_rate_schedule']['warmup_steps']
    peak = config['optimizer']['learning_rate']
    floor = config['learning_rate_schedule']['final_learning_rate']
    if not 0 <= step < total or not 0 < warmup < total:
        raise ValueError('invalid frozen schedule')
    if step < warmup:
        return peak * (step+1)/warmup
    progress = (step-warmup)/max(1, total-warmup-1)
    return floor + (peak-floor)*.5*(1+math.cos(math.pi*progress))


def dev_bpb(model, windows, *, seed, device):
    model.eval()
    nll, count = 0., 0
    with torch.no_grad():
        for i, window in enumerate(windows):
            ids, targets, draws, _ = batch_at([window], seed=seed, step=i, batch_size=1, device=device)
            logits = model(ids, random_draws=draws if model.name == 'random_gate' else None)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1,256).double(), targets.flatten(), reduction='sum')
            if not torch.isfinite(loss):
                raise ValueError('nonfinite DEV loss')
            nll += float(loss)
            count += targets.numel()
    return nll/(count*math.log(2))


def fit_worker(config, corpus, *, arm, seed, device, output: Path, random_probability=None):
    """Actual fitting core. Caller MUST pass all external launch gates first.

    Not called by no-fit tests. This function is not a security boundary or a
    standalone qualified launcher. Resource profiler/final auditor are external.
    """
    selection = config['checkpoint_steps']
    steps = config['steps_per_arm_seed']
    if selection[0] != 0 or selection[-1] != steps or selection != sorted(set(selection)):
        raise ValueError('frozen checkpoint schedule required')
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = TokenComparisonArm(arm, seed=seed, random_byte_probability=random_probability).to(device)
    opt = config['optimizer']
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=opt['learning_rate'],
                                 betas=tuple(opt['betas']), eps=opt['epsilon'], weight_decay=opt['weight_decay'])
    order_hash = hashlib.sha256()
    best, selected = float('inf'), None
    trajectory = []
    for step in range(steps+1):
        if step in selection:
            score = dev_bpb(model, corpus['DEV'], seed=seed, device=device)
            checkpoint = output / f'checkpoint-{step:06d}.pt'
            torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                        'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        'step':step, 'arm':arm, 'seed':seed, 'config':config,
                        'sample_order_sha256':order_hash.hexdigest(), 'dev_bpb':score}, checkpoint)
            trajectory.append({'step':step, 'dev_bpb':score, 'checkpoint_sha256':digest(checkpoint.read_bytes())})
            if score < best:
                best, selected = score, checkpoint.name
        if step == steps:
            break
        model.train()
        ids, targets, draws, receipt = batch_at(corpus['FIT'], seed=seed, step=step,
                                               batch_size=config['batch_size'], device=device)
        order_hash.update((json.dumps(receipt, separators=(',',':'))+'\n').encode())
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, config)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, mode='surrogate', random_draws=draws if arm == 'random_gate' else None)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1,256), targets.flatten())
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt['gradient_clip_l2'], error_if_nonfinite=True)
        optimizer.step()
    receipt = {'arm':arm, 'seed':seed, 'actual_steps':steps, 'actual_examples':steps*config['batch_size'],
               'actual_tokens':steps*config['batch_size']*config['history_bytes'],
               'actual_sample_order_sha256':order_hash.hexdigest(), 'selected_checkpoint':selected,
               'discarded_tail_bytes':corpus['_discarded_tail_bytes'],
               'checkpoints':trajectory, 'resource_parity':'INCONCLUSIVE_WITHOUT_EXTERNAL_PROFILER',
               'qualification':'UNQUALIFIED_UNTIL_INDEPENDENT_POSTFIT_AUDIT'}
    (output/'worker-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def sample_order_digest(windows, *, seed, steps, batch_size):
    result = hashlib.sha256()
    for step in range(steps):
        rng = random.Random(f'SENSE-TOK-006:{seed}:{step}')
        selected = [windows[rng.randrange(len(windows))] for _ in range(batch_size)]
        receipt = [(identity, offset, digest(raw)) for identity, offset, raw in selected]
        result.update((json.dumps(receipt,separators=(',',':'))+'\n').encode())
    return result.hexdigest()
