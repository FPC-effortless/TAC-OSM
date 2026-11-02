"""Independent source/label audit and unmatched exact/local parsers; no model fit."""
import hashlib,json,random,re
from collections import Counter
from .common import ROOT,verify_data,require
from .data import pools,history,query
def parse(text):
    result={}
    for line in text.splitlines():
        m=re.fullmatch(rb'@([a-z]{2}): ([ -~]{8})',line)
        if m:result[m[1].decode('ascii')]=m[2][:4]
    return result
def main():
    data,_=verify_data();d=data['dev'];pool=pools(d);protocol=json.loads((ROOT/'EVALUATION_PROTOCOL.json').read_text());rows=[];cells=[]
    require(b'@' not in d,'source contains potential record marker; inspect parser baseline')
    for n in protocol['key_counts']:
      for gap in protocol['gaps']:
        answers=[];full=[];local=[];same_filler=[]
        for i in range(protocol['histories']):
            h=history(d,pool,random.Random(160002+i),n=n,partition='dev',identity=i,gap=gap,overwrite=True)
            w=history(d,pool,random.Random(160002+i),n=n,partition='dev',identity=i,gap=gap,overwrite=True,class_shift=1)
            require(h['keys']==w['keys'],'wrong history changes query key families')
            require(all(h['latest'][k][0]!=w['latest'][k][0] for k in h['keys']),'shifted label did not change')
            require(len(set(v[0] for v in h['latest'].values()))==n,'within-map labels not distinct')
            for fact in h['facts']:
                start=fact['source_offset'];require(d[start:start+8].decode('ascii')==fact['quote'],'source quote mismatch')
            count=n+n//2;end=count*14
            same_filler.append(h['prefix'][end:]==w['prefix'][end:])
            for k in h['keys']+h['absent']:
                target=h['latest'].get(k,b'????');observed=h['prefix']+query(k)
                a=parse(observed).get(k,b'????');b=parse(observed[-128:]).get(k,b'????')
                present=k in h['latest']
                if present:answers.append(target);full.append(a==target);local.append(b==target)
                rows.append(dict(history=i,keys=n,gap=gap,key=k,absent=not present,answer_hex=target.hex(),
                    exact_parser=a.hex(),last128_parser=b.hex(),source_prefix_sha256=hashlib.sha256(h['prefix']).hexdigest()))
        counts=Counter(z[0] for z in answers);require(len(counts)==8 and len(set(counts.values()))==1,'first-letter cohort is not exactly balanced')
        require(all(full),'full parser failed source-derived latest records')
        cells.append(dict(keys=n,gap=gap,present_queries=len(answers),first_class_counts=dict(counts),
            best_constant_first=max(counts.values())/len(answers),best_constant_exact4=max(Counter(answers).values())/len(answers),
            full_history_exact_parser=sum(full)/len(full),last128_exact_parser=sum(local)/len(local),
            frozen_wrong_history_same_filler=sum(same_filler),histories=len(same_filler)))
    out=dict(status='PASS_SOURCE_AND_LABEL_FIXTURES',summary=cells,rows=rows,optimizer_updates=0,neural_baseline=False,
        caution='Parser has supplied syntax and exact history storage; unmatched engineering upper bound, not a trained competitor. Prefix cases across delays reuse base record identities.')
    (ROOT/'results/TASK_INTEGRITY.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status=out['status'],summary=cells),indent=2))
if __name__=='__main__':main()
