#!/usr/bin/env python3
"""Offline original bytes + FIT-only cold replay; never train or score DEV."""
from __future__ import annotations
import argparse,hashlib,importlib.util,io,json,random,stat,subprocess,sys,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
REG='research/replication/langmem086087'

def digest(x):return hashlib.sha256(x).hexdigest()
def safe(root,name):
 p=PurePosixPath(name)
 if p.is_absolute() or p.as_posix()!=name or '..' in p.parts or '\\' in name:raise ValueError('noncanonical asset path')
 q=root
 for part in p.parts:
  q=q/part
  if q.is_symlink():raise ValueError('symlink asset')
 if not q.resolve().is_relative_to(root.resolve()):raise ValueError('escaped asset')
 return q

def archive_bytes(root,entry):
 raw=safe(root,entry['path']).read_bytes()
 if digest(raw)!=entry['sha256'] or len(raw)!=entry['bytes']:raise ValueError('original archive checksum '+entry['path'])
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  names=z.namelist()
  if len(names)!=len(set(names)):raise ValueError('duplicate archive member')
  for i in z.infolist():
   safe(root,i.filename.rstrip('/'))
   if stat.S_IFMT(i.external_attr>>16)==stat.S_IFLNK or i.file_size>20_000_000:raise ValueError('unsafe archive member')
 return raw

def fit_sources(root):
 receipt=json.loads((root/'research/mathematical-engineering/language_source023/SOURCE_RECEIPT.json').read_text())
 supp=json.loads((root/'research/mathematical-engineering/language_source023/UTF8_SUPPLEMENT_RECEIPT.json').read_text())
 entries=[e for e in receipt['files'] if e['role']=='FIT']+[supp];docs={};h=hashlib.sha256()
 if len(entries)!=14 or supp['role']!='FIT_SUPPLEMENT':raise ValueError('only original14 FIT owners')
 for e in entries:
  raw=safe(root,'research/source-bundles/LANGUAGE-SOURCE023/'+e['path']).read_bytes()
  if digest(raw)!=e['sha256'] or len(raw)!=e['bytes'] or hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=e['git_blob']:raise ValueError('publisher bytes changed')
  docs[e['path']]=raw
 for name,raw in sorted(docs.items()):h.update(name.encode()+b'\0'+bytes.fromhex(digest(raw)))
 return docs,h.hexdigest()

def sampler(docs,seed,steps):
 rng=np.random.default_rng(seed);names=sorted(docs);order=hashlib.sha256();data=hashlib.sha256();counts=np.zeros(256,dtype=np.int64)
 for _ in range(steps):
  for _ in range(4):
   name=names[int(rng.integers(len(names)))];start=int(rng.integers(len(docs[name])-256));raw=docs[name][start:start+257]
   order.update(f'{name}:{start};'.encode());data.update(raw);counts+=np.bincount(np.frombuffer(raw[1:],dtype=np.uint8),minlength=256)
 return {'seed':seed,'steps':steps,'batch':4,'context':256,'targets':steps*1024,'sample_order_sha256':order.hexdigest(),'sample_bytes_sha256':data.hexdigest()},counts

def load_source(root,m):
 p=safe(root,m['model_source']);raw=p.read_bytes()
 if digest(raw)!=m['model_source_sha256']:raise ValueError('original model source changed')
 spec=importlib.util.spec_from_file_location('original_compact086_replay',p);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 return mod.CompactDualClippedTAC086

def weights(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode()+b'\0'+v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def original_checkpoints(root,m):
 archives={a['path']:archive_bytes(root,a) for a in m['archives']};rows=[]
 expected={(s,n) for s in (1701,1702,1703) for n in (160,640,1280)}
 if {(r['seed'],r['steps']) for r in m['checkpoints']}!=expected or len(m['checkpoints'])!=9:raise ValueError('incomplete original cohort')
 for r in m['checkpoints']:
  with zipfile.ZipFile(io.BytesIO(archives[r['archive']])) as z:raw=z.read(r['checkpoint_member']);fr=z.read(r['fit_receipt_member'])
  if digest(raw)!=r['checkpoint_sha256'] or digest(fr)!=r['fit_receipt_sha256']:raise ValueError('original checkpoint/receipt checksum')
  cp=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);fit=json.loads(fr)
  if cp['seed']!=r['seed'] or cp['steps']!=r['steps'] or fit['checkpoint_sha256']!=r['checkpoint_sha256'] or cp['model_source_sha256']!=m['model_source_sha256'] or cp['fit_source_sha256']!=m['fit_source_sha256']:raise ValueError('checkpoint source identity')
  if cp.get('parent_checkpoint_sha256')!=r['parent_checkpoint_sha256'] or fit.get('parent_checkpoint_sha256')!=r['parent_checkpoint_sha256']:raise ValueError('parent receipt mismatch')
  rows.append((r,cp,fit))
 for seed in (1701,1702,1703):
  chain={r['steps']:r for r in m['checkpoints'] if r['seed']==seed}
  if chain[640]['parent_checkpoint_sha256']!=chain[160]['checkpoint_sha256'] or chain[1280]['parent_checkpoint_sha256']!=chain[640]['checkpoint_sha256']:raise ValueError('parent chain broken')
 return rows

def run(root=ROOT,create=False):
 torch.set_num_threads(2);manifest_path=root/REG/'MANIFEST.json';m=json.loads(manifest_path.read_text())
 if m['new_fits'] or m['DEV_scoring'] or m['protected_TEST']:raise ValueError('no-fit FIT-only contract')
 if not create:
  tracked=set(subprocess.check_output(['git','-C',str(root),'ls-tree','-r','--name-only','HEAD'],text=True).splitlines())
  required={a['path'] for a in m['archives']}|set(m['source_assets'])|{x['path'] for x in m['anchors'].values()}|{m['model_source']}
  if required-tracked:raise ValueError('assets absent from committed Git tree')
 for name,sha in m['source_assets'].items():
  if digest(safe(root,name).read_bytes())!=sha:raise ValueError('source snapshot changed '+name)
 docs,sha=fit_sources(root)
 if sha!=m['fit_source_sha256']:raise ValueError('FIT source bundle hash')
 cls=load_source(root,m);rows=[];owner=next(iter(docs));fixture=torch.tensor([list(docs[owner][:16])]);prefix=torch.tensor([list(docs[owner][:128])])
 for row,cp,fit in original_checkpoints(root,m):
  key=f"{row['seed']}-{row['steps']}";schedule,counts=sampler(docs,row['seed'],row['steps'])
  if schedule!=cp['sample_receipt'] or schedule!=fit['sample_receipt'] or counts.tolist()!=cp['fit_byte_counts'] or len(cp['fit_loss_trace'])!=row['steps']:raise ValueError('actual original sampler/counts mismatch')
  model=cls().eval();model.load_state_dict(cp['model'],strict=True)
  if model.counts()!={'trainable':100521,'deployed':100521,'fixed_buffer_bytes':8192} or weights(model)!=fit['trained_model_hash']:raise ValueError('live model count/hash')
  optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);optimizer.load_state_dict(cp['optimizer'])
  if len(optimizer.state)!=len(list(model.parameters())) or any(int(s['step'])!=row['steps'] or not torch.isfinite(s['exp_avg']).all() or not torch.isfinite(s['exp_avg_sq']).all() for s in optimizer.state.values()):raise ValueError('optimizer restoration')
  if cp['torch_rng'].dtype!=torch.uint8 or not torch.equal(torch.Generator().set_state(cp['torch_rng']).get_state(),cp['torch_rng']):raise ValueError('Torch RNG')
  pr=random.Random();pr.setstate(cp['python_rng'])
  if pr.getstate()!=cp['python_rng']:raise ValueError('Python RNG')
  if row['steps']==160:
   if np.random.RandomState(row['seed']).get_state()[1].tolist()!=cp['numpy_rng']:raise ValueError('original NumPy key array')
   np_scope='ORIGINAL_KEY_ARRAY_ONLY;unused_global_seed_reconstructed_NOT_original_full_tuple'
  else:
   ns=cp['numpy_rng_full'];nr=np.random.RandomState();nr.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
   if nr.get_state()[1].tolist()!=ns[1] or nr.get_state()[2:]!=tuple(ns[2:]):raise ValueError('full NumPy RNG')
   np_scope='ORIGINAL_FULL_TUPLE_RESTORED'
  before=weights(model)
  with torch.inference_mode():
   logits,state=model(fixture,owners=('cold-replay',));cold=cls().eval();cold.load_state_dict(cp['model'],strict=True);again,cs=cold(fixture,owners=('cold-replay',))
   if not torch.equal(logits,again) or any(not torch.equal(state.tensors[k],cs.tensors[k]) for k in state.tensors):raise ValueError('cold reload not bitwise')
   full,fs=model(prefix,owners=('FIT-stream',));one=model.start(('FIT-stream',));parts=[]
   for t in range(128):
    part,one=model(prefix[:,t:t+1],state=one,owners=('FIT-stream',));parts.append(part)
   stream_error=float((full-torch.cat(parts,1)).abs().max())
   if stream_error>3e-5 or any(not torch.allclose(fs.tensors[k],one.tensors[k],atol=3e-5,rtol=3e-5) for k in fs.tensors):raise ValueError('stream/chunk mismatch')
   poisoned=fixture.clone();poisoned[:,8:]=(poisoned[:,8:]+113)%256;pl,_=model(poisoned,owners=('cold-replay',))
   if not torch.equal(logits[:,:8],pl[:,:8]) or state.tensor_bytes()!=1920:raise ValueError('causal/state check')
  anchor={'origin':m['anchors_origin'],'checkpoint_sha256':row['checkpoint_sha256'],'fixture_owner':owner,'fixture_sha256':digest(docs[owner][:16]),'logits':logits.tolist(),'state':{k:v.tolist() for k,v in state.tensors.items()},'clock':state.clock}
  if create:
   p=root/REG/f'anchors/{key}.json'
   if p.exists():raise FileExistsError('no replay anchor overwrite')
   p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(anchor,sort_keys=True,separators=(',',':'))+'\n');m['anchors'][key]={'path':p.relative_to(root).as_posix(),'sha256':digest(p.read_bytes())}
   ref=anchor
  else:
   a=m['anchors'][key];raw=safe(root,a['path']).read_bytes()
   if digest(raw)!=a['sha256']:raise ValueError('frozen NEW anchor checksum')
   ref=json.loads(raw)
   for k in ('origin','checkpoint_sha256','fixture_owner','fixture_sha256','clock'):
    if anchor[k]!=ref[k]:raise ValueError('anchor identity')
  pairs=[(logits,torch.tensor(ref['logits']))]+[(v,torch.tensor(ref['state'][k])) for k,v in state.tensors.items()]
  anchor_error=max(float((a-b).abs().max()) for a,b in pairs)
  if any(not torch.allclose(a,b,atol=3e-5,rtol=3e-5) for a,b in pairs) or weights(model)!=before:raise ValueError('new anchor parity/frozen weights')
  rows.append({'seed':row['seed'],'steps':row['steps'],'checkpoint_sha256':row['checkpoint_sha256'],'model_parameters':100521,'state_bytes':1920,'cold_reload_bitwise':True,'NEW_anchor_bitwise':all(torch.equal(a,b) for a,b in pairs),'NEW_anchor_max_error':anchor_error,'stream_max_error':stream_error,'optimizer_step_count':row['steps'],'numpy_rng_scope':np_scope,'sampler_sha_verified':True,'future_poison':'PASS','weights_unchanged':True})
 if create:manifest_path.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
 return {'status':'PASS_ARCHIVE_AND_FIT_COLD_REPLAY','checkpoints':rows,'archive_count':len(m['archives']),'original_historical_output_comparison':'INCONCLUSIVE_ORIGINAL_OUTPUTS_NOT_SAVED','anchor_origin':m['anchors_origin'],'training_steps_executed':0,'DEV_scoring':False,'GPU_minutes':0,'protected_TEST':False,'independently_qualified':False,'torch_version':str(torch.__version__),'numpy_version':np.__version__}

def main():
 p=argparse.ArgumentParser();p.add_argument('--create-anchors',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('receipt exists')
 r=run(create=a.create_anchors);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':r['status'],'checkpoints':len(r['checkpoints']),'anchor_origin':r['anchor_origin']}))
if __name__=='__main__':main()
