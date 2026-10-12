"""Development-only matched objective ablations; no confirmatory inference."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.development.predictive_world import (  # noqa: E402
    ARMS, PredictiveWorldModel, arm_loss, future_target, make_world_batch,
)
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from tac_osm.development.protection_benchmark import (  # noqa: E402
    fingerprint,make_pair,seed_for,tensors,
)

TRAIN_SEEDS = (43,59,71)
EVAL_SEEDS = (241,257,269)


def train(seed: int, *, smoke: bool = False):
    torch.manual_seed(seed)
    original = PredictiveWorldModel()
    models = {arm:copy.deepcopy(original) for arm in ARMS}
    initial_hashes = {arm:parameter_hash(model) for arm,model in models.items()}
    if len(set(initial_hashes.values()))!=1:
        raise AssertionError('unequal starting parameters')
    opts = {arm:torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.003)
            for arm,model in models.items()}
    records = {arm:{'initial_hash':initial_hashes[arm],'loss':[],'writer_grad':False}
               for arm in ARMS}
    batch_hashes=[]
    training_fingerprints=[]
    for step in range(2 if smoke else 120):
        batch = make_world_batch(seed,step,smoke=smoke)
        # Same exact model-visible training observations for every arm.
        batch_hashes.append(hashlib.sha256(batch.addresses.numpy().tobytes()+
                           batch.values.numpy().tobytes()+batch.query.numpy().tobytes()+
                           batch.current.numpy().tobytes()+batch.action.numpy().tobytes()).hexdigest())
        training_fingerprints.extend(batch.fingerprints)
        a,v,q,c,action = batch.inference()
        for arm,model in models.items():
            model.train()
            state = model.encode(a,v,trace=False)
            logits,world = model.predict_both(state,q,c,action)
            loss,_,_ = arm_loss(arm,logits,world,batch.answer,batch.next_latent)
            opts[arm].zero_grad(set_to_none=True)
            loss.backward()
            grad=model.memory.write.weight.grad
            records[arm]['writer_grad'] |= bool(grad is not None and torch.isfinite(grad).all()
                                                and grad.abs().sum()>0)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opts[arm].step()
            records[arm]['loss'].append(float(loss.detach()))
    for arm,model in models.items():
        model.eval()
        records[arm]['final_hash']=parameter_hash(model)
        records[arm]['parameters']=sum(p.numel() for p in model.parameters())
        if records[arm]['initial_hash']==records[arm]['final_hash'] or not records[arm]['writer_grad']:
            raise AssertionError(f'failed model training integrity: {arm}')
    return models, {'training_seed':seed,'training_batch_hashes':batch_hashes,
                    'training_fingerprints':training_fingerprints,'arms':records}


@torch.no_grad()
def measure_pair(model, seed: int, h: int, d: int, f: float,
                 noise: float, task: str, action: int, replicate: int = 0):
    pair=make_pair(seed,h,d,f,noise,task)
    episodes=[pair.left,pair.right]
    a,v,q,c=tensors(episodes)
    acts=torch.tensor([action,action])
    targets=torch.tensor([ep.target for ep in episodes])
    past=targets^c
    z=future_target(past,c,acts)
    model.eval()
    state=model.encode(a,v,trace=False)
    pred_task,pred_world=model.predict_both(state,q,c,acts)
    dispatch_calls=sum(x.operator_sample_calls for x in model.compute.last_trace)
    swapped_task,swapped_world=model.predict_both(state.flip(0),q,c,acts)
    reset_task,reset_world=model.predict_both(torch.zeros_like(state),q,c,acts)
    _,action_flip=model.predict_both(state,q,c,1-acts)
    return {
      'condition':{'H':h,'D':d,'hard_fraction':f,'noise':noise,'task':task,'action':action,'replicate':replicate},
      'visible_fingerprints':[fingerprint(ep) for ep in episodes],
      'target':targets.tolist(), 'future_target':z.tolist(),
      'task_prediction':pred_task.argmax(-1).tolist(),
      'world_prediction':pred_world.tolist(),
      'swapped_task_prediction':swapped_task.argmax(-1).tolist(),
      'swapped_world_prediction':swapped_world.tolist(),
      'reset_task_prediction':reset_task.argmax(-1).tolist(),
      'reset_world_prediction':reset_world.tolist(),
      'action_flip_world_prediction':action_flip.tolist(),
      'state_pair_distance':float((state[0]-state[1]).norm()),
      'world_pair_distance':float((pred_world[0]-pred_world[1]).norm()),
      'action_flip_distance':float((pred_world-action_flip).norm(dim=-1).mean()),
      'recurrent_state_scalars':state[0].numel(),
      'write_events':h+d,'operator_sample_calls':dispatch_calls,
    }


def evaluate(model, seed: int, *, smoke: bool = False):
    hs=(32,) if smoke else (32,128,512)
    ds=(4,) if smoke else (4,16,32)
    fs=(0.0,) if smoke else (0.0,1.0)
    noises=(.08,) if smoke else (.08,.25)
    tasks=('retention','acquisition')
    rows=[]
    for h in hs:
        for d in ds:
            for f in fs:
                for noise in noises:
                    for task in tasks:
                        for action in (0,1):
                            for replicate in range(1 if smoke else 2):
                                seed_i=seed_for('predworld-eval',seed,h,d,f,noise,task,action,replicate)
                                rows.append(measure_pair(model,seed_i,h,d,f,noise,task,action,replicate))
    return rows


def summarize(rows):
    n=sum(len(r['target']) for r in rows)
    good=sum(sum(x==y for x,y in zip(r['task_prediction'],r['target'])) for r in rows)
    reset=sum(sum(x==y for x,y in zip(r['reset_task_prediction'],r['target'])) for r in rows)
    sq=sum(sum(sum((x-y)**2 for x,y in zip(z,p))
               for z,p in zip(r['future_target'],r['world_prediction'])) for r in rows)
    return {'episode_count':n,'task_accuracy':good/n,'reset_accuracy':reset/n,
            'latent_mse':sq/(n*3),'public_conditional_mean_mse':1.01/3,
            'mean_world_pair_distance':sum(r['world_pair_distance'] for r in rows)/len(rows),
            'mean_action_flip_distance':sum(r['action_flip_distance'] for r in rows)/len(rows),
            'total_write_events':sum(r['write_events']*2 for r in rows),
            'total_operator_sample_calls':sum(r['operator_sample_calls'] for r in rows)}


def run(smoke: bool = False):
    torch.set_num_threads(2)
    output={'experiment_id':'PREDICTIVE-WORLD-DEV-006','status':'DEVELOPMENT_ONLY_NOT_CONFIRMATORY',
            'smoke':smoke,'records':[]}
    for seed,eval_seed in zip(TRAIN_SEEDS[:1] if smoke else TRAIN_SEEDS,
                              EVAL_SEEDS[:1] if smoke else EVAL_SEEDS):
        models,training=train(seed,smoke=smoke)
        rows={arm:evaluate(model,eval_seed,smoke=smoke) for arm,model in models.items()}
        report={'seed':seed,'eval_seed':eval_seed,'training':training,'rows':rows,
                'summaries':{arm:summarize(r) for arm,r in rows.items()}}
        # No identical model-visible episode is allowed in training and evaluation.
        train_set=set(training['training_fingerprints'])
        eval_set={fp for row in rows[ARMS[0]] for fp in row['visible_fingerprints']}
        if train_set & eval_set:
            raise AssertionError('training/evaluation example overlap')
        # Identical evaluation inputs (not predictions) across objective arms.
        fingerprints=[[r['visible_fingerprints'] for r in rows[arm]] for arm in ARMS]
        if any(x != fingerprints[0] for x in fingerprints[1:]):
            raise AssertionError('mismatched evaluation pairs')
        if any(abs(report['summaries'][arm]['reset_accuracy']-0.5)>1e-8 for arm in ARMS):
            raise AssertionError('no-memory counterfactual ceiling failed')
        output['records'].append(report)
        yield output, {arm:{k:v.cpu() for k,v in model.state_dict().items()}
                       for arm,model in models.items()}, seed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('artifacts/PREDICTIVE-WORLD-DEV-006'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    last=None
    for data,weights,seed in run(args.smoke):
        torch.save(weights,args.output/f'weights-{seed}.pt')
        last=data
    if last is None:
        raise AssertionError('no measured seed')
    (args.output/'report.json').write_text(json.dumps(last,indent=2)+'\n')
    print(json.dumps({'status':last['status'],'smoke':last['smoke'],
                      'summaries':[{str(r['seed']):r['summaries']} for r in last['records']]},indent=2))


if __name__=='__main__':
    main()
