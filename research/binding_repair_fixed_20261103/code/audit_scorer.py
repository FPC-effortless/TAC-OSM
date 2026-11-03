"""Oracle/shortcut evaluator unit tests. No Torch, model, training or result claim."""
import json,sys
from pathlib import Path
import fixtures,runner

def test():
    assert 'torch' not in sys.modules
    root=Path(__file__).resolve().parents[1];data=fixtures.generate();fixtures.audit(data)
    groups=data['varying_heldout_groups.json'];bygroup={g['id']:g for g in groups};fixed=data['fixed_train_cases.json'];dictionary={bytes.fromhex(c['target_hex'])[0]:bytes.fromhex(c['target_hex']) for c in fixed}
    query_only={}
    for g in groups:
        if g['cohort']=='base':
            for q in g['queries']:query_only.setdefault(q['key'],bytes.fromhex(q['target_hex']))
    results={}
    for algorithm in ('oracle','query_only','last_record','first_byte_dictionary'):
        rows=[]
        for g in groups:
            prefix=bytes.fromhex(g['prefix_hex']);records=fixtures.RECORD.findall(prefix)
            for q in g['queries']:
                target=fixtures.factual_answer(prefix,bytes.fromhex(q['query_hex']))
                if algorithm=='oracle':pred=target
                elif algorithm=='query_only':pred=query_only.get(q['key'],b'????')
                elif algorithm=='last_record':pred=records[-1][1] if records else b'????'
                else:pred=dictionary.get(target[0],b'????')
                rows.append({'group':g['id'],'pair_id':g['pair_id'],'cohort':g['cohort'],'key':q['key'],'target_hex':target.hex(),'prediction_hex':pred.hex(),'first_correct':pred[0]==target[0],'exact4':pred==target})
        evaluations=[{'family':'test','label':'training_probe','step':s,'count':1,'first_correct':1,'exact4':1} for s in (128,144)]
        summary=runner.behavioral_summary('varying',{'late_optimization_gate_steps':[128,144],'probe_scope':'Synthetic scorer unit-test stub, not a learned training probe.'},evaluations,{'test':rows},['test'])['test'];results[algorithm]=summary
    assert results['oracle']['complete_present_maps']==48
    assert all(v['all_pairs'] for v in results['oracle']['counterfactual_pairs'].values())
    assert results['query_only']['counterfactual_pairs']['swapped_values']['both_factual_answers_correct_and_expected_change']==0
    assert results['last_record']['complete_present_maps']==0
    assert results['first_byte_dictionary']['complete_present_maps']==0
    assert all(v['present_first_correct']==v['present_queries'] for v in results['first_byte_dictionary']['cohorts'].values())
    assert 'torch' not in sys.modules
    return {'status':'SCORER_UNIT_TESTS_PASS_NOT_MODEL_RESULTS','torch_imported':False,'model_executions':0,'optimizer_updates':0,'findings':['Actual-input oracle passes all factual and paired gates.','Query-only answers cannot pass value-swap pairs.','Always copying the latest record cannot pass two-key maps.','Memorized first-byte continuations pass first-byte checks but fail novel full-answer combinations.'],'algorithm_summaries':results}
if __name__=='__main__':
    result=test();path=Path(__file__).resolve().parents[1]/'fixtures/SCORER_UNIT_TESTS.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='algorithm_summaries'},indent=2))
