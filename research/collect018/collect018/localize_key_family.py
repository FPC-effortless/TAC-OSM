"""Post-partial-null key-family intervention; immutable original fixed-final models."""
import argparse,json
from pathlib import Path
import torch
from .common import ROOT,build,verify_data,digest,require,fingerprint
from .data import pools
from .evaluate import episode,retrieval
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    r=json.loads((a.run/'FIT_RECEIPT.json').read_text());ck=a.run/r['checkpoint'];require(digest(ck)==r['checkpoint_sha256'],'checkpoint differs')
    data,_=verify_data();pool=pools(data['dev'])
    for n in (1,4):
      for i in range(24):
        for shift in (0,1):
            aa=episode(data['dev'],pool,i,n,0,family='train',shift=shift);bb=episode(data['dev'],pool,i,n,0,family='dev',shift=shift)
            transform=lambda k:chr(ord(k[0])+13)+k[1]
            require([transform(k) for k in aa['keys']]==bb['keys'] and [transform(k) for k in aa['absent']]==bb['absent'],'family transform differs')
            for x,y in zip(aa['facts'],bb['facts']):require(transform(x['key'])==y['key'] and {k:v for k,v in x.items() if k!='key'}=={k:v for k,v in y.items() if k!='key'},'source quote intervention differs')
            for field in ('leading_offset','leading_bytes','filler_offset','filler_bytes','first_position'):require(aa[field]==bb[field],'non-key source span differs')
            transformed=bytearray(aa['observed_prefix']);position=aa['leading_bytes']
            for fact in aa['facts']:
                transformed[position+1]+=13
                position+=len(b'@'+fact['key'].encode()+b': '+fact['quote'].encode()+b'\n')
            require(bytes(transformed)==bb['observed_prefix'],'non-key prefix bytes changed')
    # Use the unmodified frozen evaluator equations; only its episode factory's
    # family default changes after asserting source/RNG pairing above.
    from . import evaluate
    original=evaluate.episode
    def train_family(*args,**kwargs):
        kwargs['family']='train';return original(*args,**kwargs)
    evaluate.episode=train_family
    m=build(r['arm'],r['seed']);m.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu')['model']);m.eval();before=fingerprint(m)
    spec=json.loads((ROOT/'EVALUATION_PROTOCOL.json').read_text());spec.update(cells=[dict(keys=1,gap=0),dict(keys=4,gap=0)],conditions=['normal','wrong_history','reset_all'])
    try:out=retrieval(m,data['dev'],spec)
    finally:evaluate.episode=original
    require(before==fingerprint(m),'diagnostic changed weights')
    out.update(arm=r['arm'],seed=r['seed'],checkpoint_sha256=digest(ck),weights_unchanged=True,optimizer_updates=0,primary_replacement=False,protocol='POST_PARTIAL_NULL_TRAIN_KEY_FAMILY_ON_DEV_QUOTES')
    a.output.write_text(json.dumps(out))
if __name__=='__main__':main()
