#!/usr/bin/env python3
"""SYS1-SYS2-REAL-002 authentic KuaiRand-Pure source and observed-reward gate.
No trained TAC checkpoint is evaluated by this program. No multistep claim.
"""
import argparse, collections, csv, hashlib, json, math, pathlib, tarfile, tempfile, urllib.request
URL='https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz'
PUBLISHER_MD5='0820331067a3784d9691136f772b35a7'
STEMS={
 'train':'log_standard_4_08_to_4_21_pure.csv',
 'random':'log_random_4_22_to_5_08_pure.csv',
 'standard_later':'log_standard_4_22_to_5_08_pure.csv'
}
FIELDS={'user_id','video_id','date','time_ms','is_click','is_rand','duration_ms','tab'}

def checksum(p,alg):
 h=hashlib.new(alg)
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def fetch(p):
 req=urllib.request.Request(URL,headers={'User-Agent':'TAC-OSM-authentic-research/1.0'})
 with urllib.request.urlopen(req,timeout=120) as resp,open(p,'wb') as f:
  size=0
  for chunk in iter(lambda:resp.read(1<<20),b''):
   size+=len(chunk)
   if size>120_000_000:raise ValueError('archive above allowed bound')
   f.write(chunk)

def audit(path,out):
 path=pathlib.Path(path)
 md5=checksum(path,'md5')
 if md5!=PUBLISHER_MD5:raise ValueError('Publisher MD5 mismatch; not authenticated real source')
 result={'study':'SYS1-SYS2-REAL-002','stage':'source gate and one-step factual baselines',
 'real_source_verified':True,'planning_demonstrated':False,'tac_checkpoint_evaluated':False,
 'publisher_url':URL,'archive_md5':md5,'archive_sha256':checksum(path,'sha256'),
 'excluded_scoring_from':'2022-04-28','incomplete_interaction_histories':True,
 'prior_run_correction':'Run 39318654169 source statistics aggregated future-period clicks; do not treat 2022-04-28..05-08 as sealed in this study',
 'caveat':'Random-exposure logs are selected subsets; no counterfactual multistep policy value is identified.'}
 with tarfile.open(path,'r:gz') as archive:
  members={}
  for m in archive:
   if m.isfile() and pathlib.PurePosixPath(m.name).name in STEMS.values():
    name=pathlib.PurePosixPath(m.name).name
    if name in members:raise ValueError('Duplicate source member')
    if m.size<=0 or m.size>125_000_000:raise ValueError('Bad source file size')
    members[name]=m
  if set(members)!=set(STEMS.values()):raise ValueError('Missing expected CSV')
  user=collections.defaultdict(lambda:[0,0]);video=collections.defaultdict(lambda:[0,0]);prior=[0,0]
  stats={};val={'global':[0.,0.,0],'user':[0.,0.,0],'video':[0.,0.,0],'ensemble':[0.,0.,0]}
  for label,name in STEMS.items():
   stream=archive.extractfile(members[name]); rd=csv.DictReader(__import__('io').TextIOWrapper(stream,encoding='utf-8',newline=''))
   if not FIELDS.issubset(rd.fieldnames or []):raise ValueError('Missing publisher columns')
   n=clicks=rand=early=test=0;users=set();videos=set();min_date=99999999;max_date=0
   for row in rd:
    try:
     u=int(row['user_id']);v=int(row['video_id']);d=int(row['date']);t=int(row['time_ms']);y=int(row['is_click']);rr=int(row['is_rand']);tab=int(row['tab']);duration=float(row['duration_ms'])
    except (ValueError,TypeError,KeyError) as e:raise ValueError('Invalid log record') from e
    if u<0 or v<0 or t<=0 or y not in (0,1) or rr not in (0,1) or not 0<=tab<=14 or not math.isfinite(duration):raise ValueError('Schema bounds failure')
    lo,hi=(20220408,20220421) if label=='train' else (20220422,20220508)
    if not lo<=d<=hi:raise ValueError('Temporal split violation')
    n+=1;clicks+=y if (label=='train' or (label=='random' and d<20220428)) else 0;rand+=rr;users.add(u);videos.add(v);min_date=min(min_date,d);max_date=max(max_date,d)
    if label=='train':
     prior[0]+=y;prior[1]+=1
     for store,key in ((user,u),(video,v)):
      store[key][0]+=y;store[key][1]+=1
    elif label=='random' and d>=20220428:
     test+=1  # schema checked; labels excluded from all metrics and training
    elif label=='random' and rr==1:
     early+=1
     base=prior[0]/max(1,prior[1])
     a,b=user.get(u,(0,0));pu=(a+20*base)/(b+20)
     a,b=video.get(v,(0,0));pv=(a+20*base)/(b+20)
     for key,p in (('global',base),('user',pu),('video',pv),('ensemble',(pu+pv)/2)):
      p=max(1e-6,min(1-1e-6,p));r=val[key]
      r[0]+=(p-y)**2;r[1]-=y*math.log(p)+(1-y)*math.log1p(-p);r[2]+=1
   stats[label]={'rows':n,'distinct_users':len(users),'distinct_videos':len(videos),'first_date':min_date,'last_date':max_date,'click_rate':clicks/max(1,(n if label=='train' else (early if label=='random' else 0))) if label!='standard_later' else None,'is_rand_count':rand,'validation_randomized_rows':early,'later_rows_not_scored':test}
 result['sources']=stats
 if prior[1]<100 or stats['random']['validation_randomized_rows']<100:raise ValueError('Insufficient authentic training/validation')
 result['train_click_rate']=prior[0]/prior[1]
 result['one_step_validation']={k:{'n':v[2],'brier':v[0]/v[2],'logloss':v[1]/v[2]} for k,v in val.items()}
 pathlib.Path(out).parent.mkdir(parents=True,exist_ok=True)
 pathlib.Path(out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'real_verified':True,'train':stats['train']['rows'],'random_validation':stats['random']['validation_randomized_rows'],'one_step_validation':result['one_step_validation'],'planning_demonstrated':False},sort_keys=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive');p.add_argument('--download',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
 if a.download:
  with tempfile.TemporaryDirectory() as td:
   path=pathlib.Path(td)/'KuaiRand-Pure.tar.gz';fetch(path);audit(path,a.output)
 elif a.archive:audit(a.archive,a.output)
 else:p.error('Specify --archive or --download')
