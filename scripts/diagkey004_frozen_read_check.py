"""Cheap no-fit DIAGKEY003 retention/retrieval check; interventions are OOD."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from torch.nn import functional as F
from scripts.diagkey001_associative_recall import model_for
from scripts.diagkey002_no_fit_mechanism import original_key_query


@torch.no_grad()
def read_check(model,t,k,v,y):
    x=torch.tanh(model.encoder(model.kind_embed(t)+model.key(k)+model.value(v)))
    f=x.new_zeros(len(x),model.width);s=f.clone();m=x.new_zeros(len(x),8,8)
    stored_keys=[];stored_values=[]
    for j in range(x.shape[1]):
        if j==x.shape[1]-1:before=m.clone()
        e=x[:,j];q=F.normalize(f[:,:8]+s[:,:8],dim=-1,eps=1e-6)
        recall=torch.bmm(m,q.unsqueeze(-1)).squeeze(-1)
        proposal,gate=model.fast_map(torch.cat((e,f,recall),-1)).chunk(2,-1)
        f=torch.lerp(f,proposal.tanh(),gate.sigmoid())
        delta,gate=model.slow_map(torch.cat((f,s,recall),-1)).chunk(2,-1)
        s=torch.lerp(s,(s+delta.tanh()).tanh(),.25*gate.sigmoid())
        key,value,beta,relevance=model.write_map(torch.cat((e,f,s),-1)).split((8,8,1,1),-1)
        key=F.normalize(key,dim=-1,eps=1e-6)
        m=m+((beta.sigmoid()*relevance.sigmoid())*(value.tanh()-torch.bmm(m,key.unsqueeze(-1)).squeeze(-1))).unsqueeze(-1)*key.unsqueeze(1)
        m=m/m.square().sum((1,2),keepdim=True).sqrt().clamp_min(1.)
        if j<4:stored_keys.append(key);stored_values.append(value.tanh())
    slot=(k[:,:4]==k[:,-1,None]).long().argmax(-1)
    keys=torch.stack(stored_keys,1);values=torch.stack(stored_values,1)
    target=keys[torch.arange(len(x)),slot]
    query=F.normalize(model.query(x[:,-1]),dim=-1,eps=1e-6)
    def lookup(matrix,q):return torch.bmm(matrix,q.unsqueeze(-1)).squeeze(-1)
    def logits(matrix,q):return model.readout(torch.cat((f,s,lookup(matrix,q),x[:,-1]),-1))
    actual=logits(m,query);torch.testing.assert_close(actual,model(t,k,v),atol=2e-6,rtol=2e-6)
    variants={'actual_learned_query_post_query_write':actual,
              'oracle_key_post_query_write':logits(m,target),
              'oracle_key_before_query_write':logits(before,target)}
    out={name:{'correct32':int((z.argmax(-1)==y).sum()),'NLL_sum':float(F.cross_entropy(z,y,reduction='sum'))} for name,z in variants.items()}
    # This is four-way identification of originally emitted latent write values,
    # not the 32-way task and not a learned label decoder.
    for name,q in [('oracle',target),('learned',query)]:
        z=F.normalize(lookup(before,q),dim=-1,eps=1e-6)
        similarity=(F.normalize(values,dim=-1,eps=1e-6)*z[:,None]).sum(-1)
        out[name+'_latent4_correct']=int((similarity.argmax(-1)==slot).sum())
    out['query_target_cosines']=(query*target).sum(-1).tolist();out['cases']=len(y)
    return out


def main():
    torch.set_num_threads(1);rows=[]
    for seed in range(75001,75006):
        folder=Path(f'research/replay-bundles/DIAGKEY003/seed-{seed}');p=folder/'tac.pt'
        raw=p.read_bytes()
        if raw!=subprocess.check_output(['git','show','HEAD:'+str(p)]):raise RuntimeError('original DIAGKEY003 bytes changed')
        original=json.loads((folder/'tac.json').read_text());m=model_for('tac')
        m.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['weights']);m.eval()
        results=[read_check(m,*original_key_query(seed+30000,700000+i,0,batch_size=32,split='DEV_ZERO')) for i in range(4)]
        metrics={}
        for name in ('actual_learned_query_post_query_write','oracle_key_post_query_write','oracle_key_before_query_write'):
            metrics[name]={'accuracy32':sum(r[name]['correct32'] for r in results)/128,
                           'NLL32':sum(r[name]['NLL_sum'] for r in results)/128}
        metrics.update({name:sum(r[name] for r in results)/128 for name in ('oracle_latent4_correct','learned_latent4_correct')})
        metrics['mean_cosine']=float(np.mean([c for r in results for c in r['query_target_cosines']]))
        rows.append({'seed':seed,'checkpoint_sha256':hashlib.sha256(raw).hexdigest(),'cases':128,
                     'original_sealed_TEST_accuracy':original['final_original_sealed_TEST_512']['accuracy'],**metrics})
    report={'study':'DIAGKEY004_FROZEN_DIAGKEY003_CHECK','training_updates':0,'synthetic_only':True,
        'source':'Original DIAGKEY003 checkpoints and their already-exposed fixed128 DEV queries per seed',
        'limits':['Oracle-key substitutions are OOD interventions in an unchanged full readout, not causal proof.',
                  'Latent value identification has FOUR candidates/chance25%; output task has32/chance3.125%.',
                  'Only five seeds; cosine/accuracy correlation is descriptive and not a root-cause claim.'],
        'seed_cosine_vs_original_TEST_accuracy_correlation':float(np.corrcoef([r['mean_cosine'] for r in rows],[r['original_sealed_TEST_accuracy'] for r in rows])[0,1]),
        'per_seed':rows}
    target=Path('research/replay-bundles/DIAGKEY004/FROZEN_DIAGKEY003_READ_CHECK.json')
    if target.exists():
        if json.loads(target.read_text())!=report:raise RuntimeError('original diagnostic changed')
    else:target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
