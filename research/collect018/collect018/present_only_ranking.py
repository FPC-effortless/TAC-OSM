"""Explicitly post-result ranking diagnostic on original logits, no new inference."""
import collections,json,statistics
from .common import ROOT
def main():
    out=[]
    for p in sorted((ROOT.parent/'collect018_downloads').iterdir()):
        if not p.is_dir() or not (p/'RETRIEVAL.json').exists():continue
        rows=json.loads((p/'RETRIEVAL.json').read_text())['rows'];metrics={}
        for n in (1,4):
          for condition in ('normal','wrong_history','query_swap','reset_all'):
            rr=[r for r in rows if (r['keys'],r['gap'],r['condition'],r['absent'])==(n,0,condition,False)]
            predictions=[max(range(97,105),key=r['first_logits'].__getitem__) for r in rr]
            metrics[f'{n}:{condition}']=dict(first8=statistics.mean(q==bytes.fromhex(r['answer_hex'])[0] for q,r in zip(predictions,rr)),h=dict(collections.Counter(predictions)))
        out.append(dict(model=p.name,metrics=metrics))
    (ROOT/'results/POST_NULL_PRESENT_ONLY_RANKING.json').write_text(json.dumps(dict(timing='Postall18fullscores; raw-logit-only8class ranking, no new fit or inference',primary_replacement=False,optimizer_updates=0,rows=out),indent=2))
if __name__=='__main__':main()
