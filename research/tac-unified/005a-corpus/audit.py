"""Bounded original TRAIN source audit; no content emitted, no training."""
import os,json,hashlib,collections,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    random.seed(int(os.environ['TRAINLAB_SEED']))
    out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    m=json.loads((ROOT/'SOURCE_PINS.json').read_text());pin=m['files'][10]
    data=(ROOT/'train_10.json').read_bytes()
    assert len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256']
    rows=json.loads(data);assert isinstance(rows,list)
    ids=set();actions=set();sites=set();ops=collections.Counter();orig=collections.Counter();counts=collections.Counter()
    for ep in rows:
        counts['episodes']+=1
        aid=ep.get('annotation_id');counts['duplicate_or_missing_episode_id']+=int(not isinstance(aid,str) or not aid or aid in ids);ids.add(aid)
        site=ep.get('website');sites.add(site)
        counts['goal_present']+=int(bool(ep.get('confirmed_task')))
        acts=ep.get('actions',[]);reps=ep.get('action_reprs',[])
        counts['aligned_reprs']+=int(len(acts)==len(reps))
        counts['adjacent_prestate_pairs']+=sum(bool(x.get('raw_html')) and bool(y.get('raw_html')) for x,y in zip(acts,acts[1:]))
        counts['terminal_without_explicit_poststate']+=int(bool(acts) and not any(k in acts[-1] for k in ('post_html','next_html','post_state','outcome')))
        for a in acts:
            counts['actions']+=1;uid=a.get('action_uid')
            counts['duplicate_or_missing_action_id']+=int(not isinstance(uid,str) or not uid or uid in actions);actions.add(uid)
            op=a.get('operation',{});v=op.get('op');o=op.get('original_op')
            ops[v if v in ('CLICK','TYPE','SELECT') else 'OTHER_OR_MISSING']+=1
            orig[o if o in ('CLICK','TYPE','SELECT','HOVER','ENTER') else 'OTHER_OR_MISSING']+=1
            counts['raw_html_present']+=int(bool(a.get('raw_html')))
            counts['clean_html_present']+=int(bool(a.get('cleaned_html')))
            counts['positive_candidate_present']+=int(bool(a.get('pos_candidates')))
            counts['original_candidate_present']+=int(any(c.get('is_original_target') for c in a.get('pos_candidates',[])))
            counts['explicit_post_action_field']+=int(any(k in a for k in ('post_html','next_html','post_state','outcome')))
            counts['explicit_time_field']+=int(any(k in a for k in ('timestamp','performed_at','observed_at','time_seconds')))
    result=dict(source_sha256=pin['sha256'],bytes=len(data),revision=m['revision'],scope='ONE_ORIGINAL_PUBLISHER_TRAIN_SHARD',counts=dict(counts),distinct_sites=len(sites),normalized_operations=dict(ops),original_operations=dict(orig),independent_identity_verified=False,causal_outcome_verified=False,forge_release='BLOCKED_RIGHTS_AND_TEMPORAL_PROVENANCE',splits='NOT_FROZEN',protected_test='NOT_ACCESSED',qualification='NOT_QUALIFIED')
    (out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'metrics.json').write_text(json.dumps({'episodes':counts['episodes'],'actions':counts['actions'],'distinct_sites':len(sites),'duplicate_ids':counts['duplicate_or_missing_action_id']+counts['duplicate_or_missing_episode_id']})+'\n')
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
