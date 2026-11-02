"""Authenticate historical FIT bytes/order only; no model, labels scored or fit."""
import argparse,json,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research/langcap105'))
import train_owner as owner

def h(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def run(inputs,out):
 protocol=json.loads((ROOT/'research/langcap105/OWNER_PROTOCOL.json').read_text())
 receipt_path=next(inputs.rglob('SOURCE_RECEIPT.json'));archive=next(inputs.rglob('FIT_ONLY.tar'))
 receipt=json.loads(receipt_path.read_text());data=owner.verify_source(receipt_path,archive,protocol)
 files=sorted([r for r in receipt['identity']['files'] if r['split']=='FIT'],key=lambda r:r['path'])
 blocks=owner.blocks_for(data,128)
 orders={}
 for seed in (10301,10302,10303):
  stream=owner.training_stream(blocks,seed);coords=[next(stream) for _ in range(2560*4)]
  orders[str(seed)]={'coordinates_sha256':h(coords),'windows':len(coords),'target_presentations':len(coords)*128}
 result={'status':'HISTORICAL_FIT_BYTES_AND_COORDINATES_AUTHENTICATED_NOT_INDEPENDENT_SIGNOFF',
   'source_sha256':receipt['source_sha256'],'source_receipt_sha256':owner.digest(receipt_path),'fit_archive_sha256':owner.digest(archive),
   'fit_files':len(files),'fit_file_identity_sha256':h(files),'source_metadata_scope':'Original receipt includes historical DEV metadata; no DEV bytes present in FIT-only archive and no DEV scoring',
   'new_model_data_access':False,'optimizer_updates':0,'orders':orders}
 out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'files_authenticated':len(files),'optimizer_updates':0,'new_model_data_access':False}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.inputs,a.out)
