"""Compress generated research records, keeping repository code in its repository."""
import hashlib,json,os,zipfile
from pathlib import Path
from .common import ROOT,require
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    dest=ROOT.parent/'collect018_deliverables';path=dest/'COLLECT018_RESEARCH_EVIDENCE.zip';require(not path.exists(),'final evidence exists; inspect before replacing')
    files=[]
    for p in sorted((ROOT/'results').rglob('*')):
        if not p.is_file() or p.suffix not in ('.json','.jsonl','.log'):continue
        relative=p.relative_to(ROOT/'results')
        if any(x.startswith('smoke_') or x in ('prefreeze_chunk32','prelaunch_workflow','__pycache__') for x in relative.parts):continue
        files.append(('generated_results/'+str(relative),p))
    for n in ('COLLECT018_results.png','COLLECT018_results.pdf','COLLECT018_training.png','MODEL_BUNDLES.json','MODEL_BUNDLE_VERIFICATION.json'):files.append(('figures_and_inventory/'+n,dest/n))
    downloads=ROOT.parent/'collect018_downloads';files.append(('original_prefit/collect018-prefit-41561294525.zip',downloads/'collect018-prefit-41561294525.zip'))
    guide='''COLLECT018 original evidence and read-only analysis

Source draft: https://github.com/FPC-effortless/TAC-OSM/pull/702
Mathematical draft: https://github.com/FPC-effortless/pnds-lean/pull/109
Original run: https://github.com/FPC-effortless/TAC-OSM/actions/runs/41561294525
Graph SHA: 7b9af0d5dccc1f52c5055db850f8f8c28c3ae70c
Full-execution SHA: 19f69b6ced3fa8baa4320f53e2b6dc2914b73308

The three companion checkpoint ZIPs each retain six unmodified original GitHub
artifact ZIPs, plus two smoke runs. Each full original has initialization and
256/512/1024 model/AdamW/RNG snapshots, hashes, full exposure journal, original
REPLAY tensors and EVALUATION/RETRIEVAL/GENERATION/STAGE files. Across the three
bundles: 72 full and 12 smoke snapshots. No original is replaced by a refit.

This archive contains generated analysis, scientific figures and failure records;
repository Python/Lean/workflow code stays in the pinned source drafts. Large
raw source-fit predictions stay in the original artifact ZIPs in the model
bundles. Inner and outer SHA256/member hashes and CRC checks are retained.

Original fits used CPU PyTorch2.5.1/Python3.12. Local trained-state analysis used
PyTorch2.15 CPU. A fresh source checkout needs its pinned older research folders
and original licensed TRAIN/DEV restore instructions. TEST evaluation is not
authorized. From research/collect018, a source-state audit is:

python -m collect018.audit_exhaustive --run /absolute/original/folder --output /new/path/audit.json

This performs zero optimizer updates and retains strict numerical failures.
Use torch.load(..., weights_only=True, map_location='cpu') for saved tensors.
Do not run training/launch scripts, overwrite originals, or use a later runtime's
newly computed score as the original score. Evaluation writes files into its run
folder, so any re-evaluation must use a separate copied directory.

Methods/protocols: EXPERIMENT_PROTOCOL, EVALUATION_PROTOCOL, AUDIT_PROTOCOL and
PHASE_ABLATION_PROTOCOL were frozen/published before full-score inspection.
LOCALIZATION_PROTOCOL is post-partial-null; NORMALIZED_GRADIENT_PROTOCOL and the
eight-present-letter ranking are post-failure/post-result diagnostics. They do
not replace the original full256 primary or clear original numerical misses.

All18 fits failed the registered retrieval gates. Fifteen strict FP32 sum-loss
gradient fixtures retain failures; forward/state, FP64, exposure/causality checks
are recorded separately. Mean-loss rescaling passes18 new probes but changes
the absolute-error scale. Full mathematical/source/IEEE/AD equivalence and
independent qualification remain OPEN. Bounded state does not establish global
attention equivalence or a quality/efficiency advantage.
'''
    inventory=[dict(member=n,size=p.stat().st_size,sha256=sha(p)) for n,p in files];temporary=path.with_suffix('.zip.building')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for n,p in files:z.write(p,n)
        z.writestr('READ_ME.txt',guide);z.writestr('CONTENTS.json',json.dumps(dict(study='COLLECT018',files=inventory,source_code_in_repository=True,optimizer_updates_in_analysis=0),indent=2))
    with temporary.open('rb') as f:os.fsync(f.fileno())
    os.replace(temporary,path)
    (dest/'EVIDENCE_BUNDLE.json').write_text(json.dumps(dict(filename=path.name,size=path.stat().st_size,sha256=sha(path),members=len(files)+2),indent=2));print(path.name,path.stat().st_size)
if __name__=='__main__':main()
