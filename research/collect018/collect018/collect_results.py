"""Recompute retained predictions independently; never fit or select a model."""
import argparse, collections, hashlib, json, math, statistics
from pathlib import Path
from .common import ARMS, SEEDS, ROOT, canonical, digest, require, verify_data

def near(x,y,label,tol=2e-6):
    require(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=tol,label)

def audit_predictions(payload,spec,data):
    rows=payload['rows']; cohorts=payload['cohorts']; index={}
    cells=[(c['keys'],c['gap']) for c in spec['cells']]
    require(len(cohorts)==len(cells)*spec['histories'],'cohort count differs')
    for c in cohorts:
        key=(c['keys'],c['gap'],c['identity']); require(key not in index,'duplicate cohort')
        require(key[:2] in cells and 0<=key[2]<spec['histories'],'undeclared cohort')
        latest={}; wrong_latest={}; records=[]
        for group,dest in ((c['facts'],latest),(c['wrong_facts'],wrong_latest)):
            for f in group:
                quote=data[f['source_offset']:f['source_offset']+8]
                require(quote.decode('ascii')==f['quote'] and quote[:4].hex()==f['answer_hex'],'DEV quotation differs')
                dest[f['key']]=quote[:4]
                if dest is latest:records.append(b'@'+f['key'].encode()+b': '+quote+b'\n')
        require(list(latest)==list(wrong_latest) and len(latest)==c['keys'],'counterfactual keys differ')
        require(all((wrong_latest[k][0]-97)==((latest[k][0]-97+1)%8) for k in latest),'wrong-history class shift differs')
        l=c['leading_offset']; n=c['leading_bytes']; f=c['filler_offset']; gap=c['gap']
        prefix=data[l:l+n]+b''.join(records)+b'\n'+data[f:f+gap]
        require(hashlib.sha256(prefix).hexdigest()==c['source_prefix_sha256'],'original prefix provenance differs')
        require(len(prefix)+9-1==c['first_position'],'query timing differs')
        index[key]=(c,latest)
    seen=set(); histogram=collections.Counter()
    for r in rows:
        ck=(r['keys'],r['gap'],r['history']); c,latest=index[ck]
        identity=(*ck,r['condition'],r['key']); require(identity not in seen,'duplicate raw query');seen.add(identity)
        require(r['condition'] in spec['conditions'],'undeclared condition')
        target=bytes.fromhex(r['answer_hex']); pred=bytes.fromhex(r['prediction_hex'])
        require(target==(b'????' if r['absent'] else latest[r['key']]),'query target differs')
        require(r['absent']==(r['key'] not in latest),'presence flag differs')
        keys=list(latest); actual=keys[(keys.index(r['key'])+1)%len(keys)] if r['condition']=='query_swap' and not r['absent'] and len(keys)>1 else r['key']
        require(r['queried_key']==actual,'query intervention differs')
        z=r['first_logits'];require(len(z)==256 and all(math.isfinite(v) for v in z),'invalid raw vocabulary')
        top=max(range(256),key=z.__getitem__); candidates=list(b'abcdefgh?'); closed=max(candidates,key=z.__getitem__)
        require(len(pred)==4 and pred[0]==top==r['first_top1'] and closed==r['closed_first'],'raw argmax differs')
        require(r['first_correct']==int(top==target[0]) and r['closed_first_correct']==int(closed==target[0]) and r['exact4']==int(pred==target),'raw integer scores differ')
        high=max(z); exp=[math.exp(v-high) for v in z]; total=sum(exp)
        near(r['target_log_probability'],z[target[0]]-high-math.log(total),'target log probability differs',4e-6)
        near(r['target_alphabet_mass'],sum(exp[i] for i in candidates)/total,'alphabet mass differs')
        if not r['absent']:histogram[(*ck[:2],r['condition'],target[0])]+=1
    summary=[]
    mean=lambda xs,k:statistics.mean(r[k] for r in xs)
    for n,gap in cells:
      for condition in spec['conditions']:
        q=[r for r in rows if (r['keys'],r['gap'],r['condition'])==(n,gap,condition)]
        p=[r for r in q if not r['absent']];a=[r for r in q if r['absent']]
        require(len(p)==n*spec['histories'] and len(a)==2*spec['histories'],'raw query coverage differs')
        require(all(histogram[(n,gap,condition,k)]==len(p)//8 for k in b'abcdefgh'),'present classes not exactly balanced')
        s=dict(keys=n,gap=gap,condition=condition,present_queries=len(p),absent_queries=len(a),first_accuracy=mean(p,'first_correct'),closed_first_accuracy=mean(p,'closed_first_correct'),exact4=mean(p,'exact4'),
            complete_maps=sum(all(r['exact4'] for r in p if r['history']==i) for i in range(spec['histories']))/spec['histories'],absent_first=mean(a,'first_correct'),absent_exact4=mean(a,'exact4'),
            present_mean_log_probability=mean(p,'target_log_probability'),target_alphabet_mass=mean(p,'target_alphabet_mass'))
        original=next(s0 for s0 in payload['summary'] if (s0['keys'],s0['gap'],s0['condition'])==(n,gap,condition))
        require(set(s)==set(original),'summary schema differs')
        for k,v in s.items():
            if isinstance(v,(float,int)):near(v,original[k],'derived '+k+' differs',1e-12)
            else:require(v==original[k],'summary identity differs')
        s['present_question_mark_rate']=statistics.mean(r['first_top1']==63 for r in p)
        s['absent_question_mark_rate']=statistics.mean(r['first_top1']==63 for r in a)
        s['present_top1_histogram']=dict(sorted(collections.Counter(r['first_top1'] for r in p).items()))
        summary.append(s)
    return summary,hashlib.sha256(canonical(cohorts)).hexdigest()

def gates(summary,spec):
    result={}
    for name in ('positive_control','binding_gate'):
        g=spec[name];n,gap=g['keys'],g['gap'];s={r['condition']:r for r in summary if (r['keys'],r['gap'])==(n,gap)}; normal=s['normal']
        c=dict(present=normal['first_accuracy']>=g['min_present_first_accuracy'],absent=normal['absent_first']>=g['min_absent_first_accuracy'])
        for target,field in (('reset_all','normal_minus_reset_min'),('wrong_history','normal_minus_wrong_min'),('query_swap','normal_minus_swap_min')):
            if field in g:c[field]=normal['first_accuracy']-s[target]['first_accuracy']>=g[field]
        result[name]=dict(pass_all=all(c.values()),criteria=c)
    return result

def collect(dest,output,allow_partial=False):
    data,_=verify_data();spec=json.loads((ROOT/'EVALUATION_PROTOCOL.json').read_text()); results=[];missing=[]
    for arm in ARMS:
      for seed in SEEDS:
        folder=dest/f'{arm}_{seed}'
        if not (folder/'EVALUATION.json').is_file():missing.append(folder.name);continue
        receipt=json.loads((folder/'FIT_RECEIPT.json').read_text()); ev=json.loads((folder/'EVALUATION.json').read_text())
        require(receipt['status']=='FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED' and receipt['optimizer_updates']==1024,'incomplete fit')
        require(receipt['checkpoint_sha256']==digest(folder/'checkpoint_1024.pt')==ev['checkpoint_sha256'],'fixed-final differs')
        require(ev['optimizer_updates']==0 and ev['weights_unchanged'] and not ev['protected_test_read'],'evaluation scope differs')
        raw=json.loads((folder/'RETRIEVAL.json').read_text()); summary,cohort_hash=audit_predictions(raw,spec,data['dev'])
        stages=[]
        for step in (256,512):
            sub=dict(spec,cells=[dict(keys=1,gap=0)],conditions=['normal','wrong_history','reset_all'])
            ss,hh=audit_predictions(json.loads((folder/f'STAGE_{step:04}.json').read_text()),sub,data['dev']);stages.append(dict(step=step,summary=ss,cohort_hash=hh,selection=False))
        generation=json.loads((folder/'GENERATION.json').read_text());require(generation['weights_unchanged'] and generation['optimizer_updates']==0 and generation['checkpoint_sha256']==receipt['checkpoint_sha256'],'generation identity differs')
        for g,temp in zip(generation['rows'],(.8,1.2)):
            require(g['temperature']==temp and g['status']=='COMPLETE' and len(bytes.fromhex(g['hex']))==g['generated_bytes']==512,'incomplete generation')
            require([z['generated_bytes'] for z in g['payload']]==[256,512],'generation state coverage differs')
        j=[json.loads(z) for z in (folder/'journal.jsonl').read_text().splitlines()];require(len(j)==1024,'journal incomplete')
        windows=[]
        for lo,hi in ((1,64),(193,256),(449,512),(961,1024)):
            s=j[lo-1:hi];windows.append(dict(lo=lo,hi=hi,**{k:statistics.mean(r[k] for r in s) for k in ('first_accuracy','first_nll','natural_nll','answer_nll','objective','balance_loss')}))
        result=dict(arm=arm,seed=seed,checkpoint_sha256=receipt['checkpoint_sha256'],initial_parameter_sha256=receipt['initial_parameter_sha256'],parameters=receipt['trainable_parameters'],updates=1024,
            exposure_sequence_sha256=hashlib.sha256(canonical([r['exposure']['batch_sha256'] for r in j])).hexdigest(),cohort_sha256=cohort_hash,summary=summary,gates=gates(summary,spec),ordinary_dev=ev['ordinary_dev'],stage_diagnostics=stages,
            training_windows=windows,optimizer_seconds=receipt['total_recorded_optimizer_seconds'],fit_elapsed_seconds=receipt['elapsed_seconds'],mean_step_seconds=receipt['mean_step_seconds'],generation_state=generation['rows'][0]['payload'])
        audit=ROOT/'results'/'audits'/f'{folder.name}.json'
        result['trained_source_audit']=json.loads(audit.read_text()) if audit.is_file() else dict(status='PENDING')
        results.append(result)
    require(allow_partial or not missing,'missing original models: '+str(missing))
    for seed in SEEDS:
        group=[r for r in results if r['seed']==seed]
        require(len(set(r['exposure_sequence_sha256'] for r in group))<=1,'paired source exposure differs')
        for fam in ('phase','context','gru'):
            require(len(set(r['initial_parameter_sha256'] for r in group if r['arm'].startswith(fam+'_')))<=1,'objective-paired initialization differs')
        require(len(set(r['initial_parameter_sha256'] for r in group if not r['arm'].startswith('gru_')))<=1,'phase/context initial parameters differ')
    require(len(set(r['cohort_sha256'] for r in results))<=1,'evaluation cohorts differ across arms/seeds')
    out=dict(study='COLLECT018',status='PARTIAL_NOT_FINAL' if missing else 'ALL_ORIGINAL_FULL_FITS_AND_RAW_SCORES_VERIFIED',missing=missing,models=results,full_updates=sum(r['updates'] for r in results),
        raw_queries_verified=sum(len(spec['conditions'])*(c['keys']+2)*spec['histories'] for c in spec['cells'])*len(results),independently_qualified=False,protected_test_read=False,optimizer_updates_in_analysis=0)
    output.write_text(json.dumps(out,indent=2));print(json.dumps(dict(status=out['status'],models=len(results),missing=missing,raw_queries=out['raw_queries_verified'])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--downloads',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();collect(a.downloads,a.output,a.allow_partial)
if __name__=='__main__':main()
