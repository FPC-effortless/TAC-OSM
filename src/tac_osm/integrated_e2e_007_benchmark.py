"""Fresh E2E-007 benchmark with residual-linear state write selected on development data."""
from __future__ import annotations
import hashlib,io
from typing import Iterable
import torch
from .integrated_e2e_005_benchmark import ALL_COMBOS,BITS,ENTITY_COUNT,make_modalities

SEALED_E2E005=(
 ("xor",0,4),("xor",5,8),("and",1,9),("and",2,6),
 ("or",7,10),("or",3,11),("xnor",4,9),("xnor",6,11),
)
DEV_COMBOS=(
 ("xor",0,5),("xor",0,6),("xor",0,7),("xor",0,8),("xor",0,9),("xor",0,10),("xor",0,11),("xor",1,4),
 ("and",0,4),("and",0,5),("and",0,6),("and",0,7),("and",0,8),("and",0,9),("and",0,10),("and",0,11),
 ("or",0,4),("or",0,5),("or",0,6),("or",0,7),("or",0,8),("or",0,9),("or",0,10),("or",0,11),
 ("xnor",0,4),("xnor",0,5),("xnor",0,6),("xnor",0,7),("xnor",0,8),("xnor",0,9),("xnor",0,10),("xnor",0,11),
)
E2E006_HELDOUT=(
 ("xor",1,5),("xor",2,9),("and",4,10),("and",3,8),
 ("or",1,7),("or",5,10),("xnor",2,7),("xnor",6,9),
)
HELDOUT=(
 ("xor",1,6),("xor",2,10),
 ("and",4,11),("and",5,9),
 ("or",2,8),("or",6,10),
 ("xnor",1,10),("xnor",3,9),
)
EXCLUDED=set(SEALED_E2E005)|set(DEV_COMBOS)|set(E2E006_HELDOUT)|set(HELDOUT)
TRAIN_COMBOS=tuple(c for c in ALL_COMBOS if c not in EXCLUDED)
GENERATOR_VERSION="integrated-e2e-v6-residual-linear-fresh-heldout"

assert len(HELDOUT)==8 and set(HELDOUT).isdisjoint(EXCLUDED-set(HELDOUT))
assert len(TRAIN_COMBOS)==len(ALL_COMBOS)-56

def make_query(payload,combo,entity):
    op,i,j=combo;a,b=payload[entity][i],payload[entity][j]
    answer={"xor":a^b,"and":a&b,"or":a|b,"xnor":1-(a^b)}[op]
    return entity,i,j,op,answer

def sample_episode(rng,*,combo1=None,combo2=None):
    entities=rng.sample(range(ENTITY_COUNT),3)
    payload={e:[rng.randrange(2) for _ in range(BITS)] for e in entities}
    combo1=combo1 or rng.choice(TRAIN_COMBOS);combo2=combo2 or rng.choice(TRAIN_COMBOS)
    q1=make_query(payload,combo1,entities[0]);q2=make_query(payload,combo2,entities[1])
    obs=[(e,*make_modalities(e,payload[e],rng)) for e in entities]
    return obs,q1,q2,payload

def validate_episode(ep):
    obs,q1,q2,payload=ep;ents=[r[0] for r in obs]
    assert len(set(ents))==3 and q1[0]==ents[0] and q2[0]==ents[1] and q1[0]!=q2[0]
    for q in (q1,q2):
        combo=(q[3],q[1],q[2])
        assert combo in HELDOUT
        a,b=payload[q[0]][q[1]],payload[q[0]][q[2]]
        assert q[4]=={"xor":a^b,"and":a&b,"or":a|b,"xnor":1-(a^b)}[q[3]]

def sample_evaluation_episodes(rng,count):
    out=[]
    for n in range(count):
        a=HELDOUT[n%len(HELDOUT)];b=HELDOUT[(n*5+2)%len(HELDOUT)]
        if a==b:b=HELDOUT[(n*5+3)%len(HELDOUT)]
        ep=sample_episode(rng,combo1=a,combo2=b);validate_episode(ep);out.append(ep)
    return out

def episode_key(ep):
    obs,q1,q2,p=ep;ents=tuple(r[0] for r in obs)
    return ents,tuple((int(e),tuple(map(int,p[e]))) for e in ents),(q1[3],q1[1],q1[2]),(q2[3],q2[1],q2[2])

def episode_fingerprint(episodes:Iterable[tuple])->str:
    buf=io.BytesIO();serial=[]
    for ep in episodes:
        obs,q1,q2,p=ep
        serial.append({"entities":[int(r[0]) for r in obs],"payload":{str(k):list(map(int,v)) for k,v in sorted(p.items())},
                       "q1":list(q1),"q2":list(q2),"text":[r[1].tolist() for r in obs],
                       "image":[r[2].tolist() for r in obs],"audio":[r[3].tolist() for r in obs]})
    torch.save(serial,buf);return hashlib.sha256(buf.getvalue()).hexdigest()

def generator_hash():
    return hashlib.sha256(open(__file__,"r",encoding="utf-8").read().encode()).hexdigest()

def benchmark_manifest():
    base=__import__("pathlib").Path(__file__).resolve().parent/"integrated_e2e_005_benchmark.py"
    return {"generator_version":GENERATOR_VERSION,"generator_sha256":generator_hash(),
            "base_generator_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),
            "heldout":[list(x) for x in HELDOUT],"dev_combos":[list(x) for x in DEV_COMBOS],
            "sealed_e2e005":[list(x) for x in SEALED_E2E005],"e2e006_heldout":[list(x) for x in E2E006_HELDOUT],
            "train_combos_count":len(TRAIN_COMBOS)}
