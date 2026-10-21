import subprocess,json,re
from pathlib import Path
refs=subprocess.check_output(['git','for-each-ref','--format=%(objectname) %(refname)','refs/remotes/origin'],text=True).splitlines()
seen={}; hits=[]
for line in refs:
 sha,ref=line.split(' ',1)
 tree=subprocess.check_output(['git','ls-tree','-r',sha,'scripts/', 'research/'],text=True)
 for row in tree.splitlines():
  mode,typ,blobpath=row.split(' ',2);blob,path=blobpath.split('\t',1)
  if not path.endswith('.py') or blob in seen:continue
  seen[blob]=path
  data=subprocess.check_output(['git','cat-file','blob',blob],text=True,errors='replace')
  if re.search('KuaiRec|big_matrix.csv|kuai.?rec|publisher_bytes|FUTURE_MOD|source_future_user|read_original_user',data,re.I):
   lines=[{'line':n,'text':s.strip()} for n,s in enumerate(data.splitlines(),1) if re.search(r'%\s*16|FUTURE_MOD|user_id|read_csv|read_early|publisher_bytes|cohort',s)]
   hits.append({'blob':blob,'path':path,'first_ref':ref,'reader_lines':lines})
Path('/workspace/me034-kuai-reader-inventory.json').write_text(json.dumps({'remote_heads':refs,'unique_python_script_blobs':len(seen),'kuai_script_versions':hits},indent=2))
print('refs',len(refs),'unique scripts',len(seen),'kuai versions',len(hits))
for h in hits:
 if any(re.search(r'%\s*16.*[6789]|FUTURE_MOD.*[6789]|read_csv|DictReader',x['text']) for x in h['reader_lines']):print(h['path'],h['first_ref'],h['reader_lines'])
