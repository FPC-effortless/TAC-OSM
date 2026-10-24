"""Actual restored complete-graph operator census; no fit or forecast scoring."""
import torch,collections,argparse,json
from pathlib import Path
from scripts.me076_train_system1_real import write,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);root=Path('research/mathematical-engineering/me093');records={}
    for label in ['PHYSICAL_PREFILL32','PHYSICAL_INCREMENTAL_STEP']:
        p=root/(label+'_EXECUTABLE.pt');graph=torch.jit.load(str(p)).graph
        counts=collections.Counter(n.kind() for n in graph.nodes());folded=[]
        for node in graph.nodes():
            if node.kind()!='prim::Constant':continue
            value=node.output().toIValue()
            if isinstance(value,torch.Tensor) and tuple(value.shape) in [(141,108),(141,)]:
                uses=[use.user.kind() for use in node.output().uses()]
                folded.append({'shape':list(value.shape),'dtype':str(value.dtype),'consumers':uses})
        projected=[r for r in folded if r['shape']==[141,108]]
        if len(projected)!=1 or projected[0]['consumers']!=['aten::linear']:raise RuntimeError('Combined projection is not folded toone shared GEMM')
        records[label]={'executable_sha256':sha(p),'nodes':sum(counts.values()),'operator_counts':dict(sorted(counts.items())),'single_combined_distribution_projection_constants':folded}
    result={'status':'PASS','no_fit':True,'profiles':records,'scope':'Actual first-original restored fixed-profile physical input-to-output graphs. Folded141x108constant has one linear consumer. Restored graph canonicalization changes node representation: counts differ from pre-save frozen graph; do not interpret the difference as arithmetic work. Native serial32step operators remain. Node count is not FLOPs, latency, SRAM residency or learned skill.'}
    path=root/'GRAPH_OPERATION_CENSUS.json'
    if args.verify:
        original=json.loads(path.read_text())
        for name,row in records.items():
            if row['executable_sha256']!=original['profiles'][name]['executable_sha256']:raise RuntimeError('Graph executable changed')
    else:
        if path.exists():raise RuntimeError('Do not overwrite original graph census; use --verify')
        write(path,result)
    print({k:{'nodes':v['nodes'],'combined_projection':v['single_combined_distribution_projection_constants']} for k,v in records.items()})
if __name__=='__main__':main()
