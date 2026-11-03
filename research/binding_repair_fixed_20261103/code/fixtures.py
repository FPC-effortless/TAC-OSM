"""Constructed optimization fixtures only. No corpus, Torch or model import."""
import hashlib,json,random,re
from collections import Counter
from pathlib import Path
ALPHABET=b'abcdefgh'
TRAIN_KEYS=('aa','bb','cc','dd')
NEW_KEYS=('ee','ff','gg','hh')
NATURAL=(b'synthetic training-compatible filler; '*4)[:129]
QUERY=re.compile(rb'Copy ([a-z]{2}): \Z')
RECORD=re.compile(rb'@([a-z]{2}): ([a-h]{4})xxxx\n')

def factual_answer(prefix,query):
    q=QUERY.fullmatch(query)
    assert q is not None
    latest={key:value for key,value in RECORD.findall(prefix)}
    return latest.get(q.group(1),b'????')

def value(rng,first,parity):
    while True:
        v=bytes([ALPHABET[first]]+[rng.choice(ALPHABET) for _ in range(3)])
        if sum(ALPHABET.index(x) for x in v)%2==parity:return v

def prefix_bytes(records,position,gap,rng):
    history=b''.join(b'@'+key.encode()+b': '+val+b'xxxx\n' for key,val in records)+b'\n'
    lead=position+1-9-len(history)-gap
    assert lead>=0
    alphabet=b' .;_:0123456789'
    left=bytes(rng.choice(alphabet) for _ in range(lead));middle=bytes(rng.choice(alphabet) for _ in range(gap))
    return left+history+middle

def case(identity,records,key,position,gap,rng):
    prefix=prefix_bytes(records,position,gap,rng);query=b'Copy '+key.encode()+b': '
    # Generator semantics and an independent parse of the actual bytes agree.
    latest={k:v for k,v in records};target=latest.get(key,b'????')
    assert factual_answer(prefix,query)==target
    slab=prefix+query+target+b'.'*(129-len(prefix)-len(query)-4)
    assert len(slab)==129 and len(prefix+query)-1==position
    return dict(id=identity,records=[[k,v.hex()] for k,v in records],queried_key=key,first_position=position,gap=gap,prefix_hex=prefix.hex(),query_hex=query.hex(),target_hex=target.hex(),slab_hex=slab.hex())

def group(identity,records,keys,position,gap,seed,cohort,pair_id=None):
    prefix=prefix_bytes(records,position,gap,random.Random(seed)) if records else b''
    queries=[]
    for key in keys:
        q=b'Copy '+key.encode()+b': ';queries.append({'key':key,'query_hex':q.hex(),'target_hex':factual_answer(prefix,q).hex()})
    return {'id':identity,'cohort':cohort,'pair_id':pair_id,'prefix_hex':prefix.hex(),'queries':queries,'records':[[k,v.hex()] for k,v in records],'first_position':len(prefix)+8,'gap':gap}

def generate():
    fixed=[]
    for i in range(8):
        v=bytes(ALPHABET[(i+k)%8] for k in (0,1,3,4))
        assert sum(ALPHABET.index(x) for x in v)%2==0
        fixed.append(case(f'fixed-{i}', [('aa',v)],'aa',112,0,random.Random(70001)))
    fixed_batches=[]
    for step in range(128):
        fixed_batches.append([fixed[(step+2*r)%8]['id'] for r in range(3)])
    varying=[]
    for step in range(144):
        raw=[]
        for j in range(3):
            label=(step+2*j)%9;rng=random.Random(710000+step*101+j*100003)
            keys=rng.sample(TRAIN_KEYS,3);p=rng.choice((96,112,123));gap=rng.choice((0,8,16))
            if label==8:
                records=[(keys[0],value(rng,rng.randrange(8),0)),(keys[1],value(rng,rng.randrange(8),0))];key=keys[2]
            else:
                target=value(rng,label,0);other=value(rng,(label+1+rng.randrange(6))%8,0)
                records=[(keys[0],target),(keys[1],other)];key=keys[0]
            rng.shuffle(records)
            if rng.randrange(2)==0:
                oldkey=key if label!=8 else rng.choice(keys[:2])
                current=dict(records)[oldkey];oldfirst=(ALPHABET.index(current[0])+1)%8
                records.insert(0,(oldkey,value(rng,oldfirst,0)))
            raw.append(case(f'vary-{step:03}-{j}',records,key,p,gap,rng))
        varying.append([raw[(r-step)%3] for r in range(3)])
    fixed_test=[]
    for i in range(16):
        rng=random.Random(720000+i);v=value(rng,i%8,1)
        fixed_test.append(group(f'fixed-fresh-{i}',[('aa',v)],['aa'],112,0,70001,'fresh_combinations',i%8))
    held=[]
    for i in range(8):
        rng=random.Random(730000+i);keys=rng.sample(TRAIN_KEYS,3);new=rng.sample(NEW_KEYS,3)
        a,b=value(rng,i,1),value(rng,(i+3)%8,1);p=(96,112,123)[i%3];gap=(0,8,16)[(i//3)%3]
        records=[(keys[0],a),(keys[1],b)]
        if i%2:records.reverse()
        variants=[('base',records,keys),('swapped_values',[(k,b if k==keys[0] else a) for k,_ in records],keys),('record_order',list(reversed(records)),keys),('query_order',records,list(reversed(keys))),('overwrite',[(keys[0],value(rng,(i+1)%8,1))]+records,keys),('unseen_keys',[(new[0] if k==keys[0] else new[1],v) for k,v in records],new)]
        for cohort,rr,qq in variants:held.append(group(f'{cohort}-{i}',rr,qq,p,gap,740000+i,cohort,i))
        empty=group(f'empty-{i}',[],keys[:2],p,0,740000+i,'empty_history',i)
        empty['original_targets']={keys[0]:a.hex(),keys[1]:b.hex()};held.append(empty)
    return {'fixed_train_cases.json':fixed,'fixed_schedule.json':fixed_batches,'varying_train_batches.json':varying,'fixed_heldout_groups.json':fixed_test,'varying_heldout_groups.json':held}

def audit(data):
    fixed=data['fixed_train_cases.json'];byid={c['id']:c for c in fixed}
    schedules={'fixed':[[byid[x] for x in row] for row in data['fixed_schedule.json']],'varying':data['varying_train_batches.json']}
    result={};training_values=set()
    for stage,batches in schedules.items():
        counts=Counter();slots=[Counter() for _ in range(3)];labels=Counter();positions=Counter();queries=Counter();slabs=[]
        for batch in batches:
            assert len(batch)==3
            for slot,c in enumerate(batch):
                slab=bytes.fromhex(c['slab_hex']);prefix=bytes.fromhex(c['prefix_hex']);query=bytes.fromhex(c['query_hex']);target=bytes.fromhex(c['target_hex']);p=c['first_position']
                assert len(slab)==129 and slab[:p+1]==prefix+query and slab[p+1:p+5]==target
                assert factual_answer(prefix,query)==target
                counts[c['id']]+=1;slots[slot][target[0]]+=1;labels[target[0]]+=1;positions[p]+=1;queries[c['queried_key']]+=1;slabs.append(slab)
                for _,v in RECORD.findall(prefix):
                    assert sum(ALPHABET.index(x) for x in v)%2==0;training_values.add(v)
        assert len(set(labels.values()))==1
        assert all(len(set(x.values()))==1 for x in slots)
        result[stage]={'updates_per_arm':len(batches),'task_presentations_per_arm':len(batches)*3,'presented_bytes_per_arm':len(batches)*512,'first_target_counts':dict(labels),'slot_first_target_counts':[dict(x) for x in slots],'positions':dict(positions),'query_keys':dict(queries),'case_presentations':dict(counts),'ordered_task_slab_sha256':hashlib.sha256(b''.join(slabs)).hexdigest()}
    for name in ('fixed_heldout_groups.json','varying_heldout_groups.json'):
        groups=data[name]
        for g in groups:
            prefix=bytes.fromhex(g['prefix_hex'])
            for q in g['queries']:assert factual_answer(prefix,bytes.fromhex(q['query_hex']))==bytes.fromhex(q['target_hex'])
            for _,v in RECORD.findall(prefix):
                assert sum(ALPHABET.index(x) for x in v)%2==1 and v not in training_values
        result[name]={'prefix_groups':len(groups),'queries':sum(len(g['queries']) for g in groups),'model_forward_calls_per_arm':sum((1 if g['prefix_hex'] else 0)+5*len(g['queries']) for g in groups),'cohorts':dict(Counter(g['cohort'] for g in groups))}
    return result

if __name__=='__main__':
    out=Path(__file__).resolve().parents[1]/'fixtures';out.mkdir(exist_ok=True)
    data=generate();report=audit(data)
    for name,rows in data.items():(out/name).write_text(json.dumps(rows,indent=2)+'\n')
    (out/'INPUT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in row.items() if n not in ('case_presentations',)} for k,row in report.items()},indent=2))
