"""Qualify pinned overhead video for exactly the existing ME076 FIT/REPORT roles.

No fitting, no target scoring, no new episodes. One verified source MP4 at a
time, deterministic ffmpeg gray32 extraction, full decoded/source digests.
"""
import argparse, hashlib, json, subprocess, tempfile, time, urllib.request
from pathlib import Path
import numpy as np
from scripts.me064_mixed_real_fit import task_registration
from scripts.me076_train_system1_real import classify_members, digest, write

REV='9290f5a96dc91ea58256e81d9ce2dd9621890b51'
CAMERA='observation.images.cam_high'
API='https://huggingface.co/api/datasets/BodenAI/RW-RL-HIL-Dataset/tree/'+REV+'/'
BASE='https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset/resolve/'+REV+'/'

def fetch(url,destination=None):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=60) as response:
                if destination is None:return json.load(response)
                with Path(destination).open('wb') as f:
                    while block:=response.read(1024*1024):f.write(block)
            return
        except Exception:
            if attempt==2:raise
            time.sleep(1)

def acquire(cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    fit,report=classify_members(task_registration());records=[]
    tables={}
    for task in sorted({t for t,_,_ in fit+report}):
        folder=f'{task}/videos/chunk-000/{CAMERA}'
        table=fetch(API+folder+'?limit=1000')
        tables[task]={v['path']:v for v in table if v.get('type')=='file'}
    for role,members in [('FIT',fit),('REPORT',report)]:
        for task,episode,frames in members:
            key=f'{task}/videos/chunk-{episode//1000:03d}/{CAMERA}/episode_{episode:06d}.mp4'
            metadata=tables[task][key];expected=metadata['lfs']['oid']
            target=cache/task/f'{episode:06d}.npz';receipt=target.with_suffix('.json')
            if receipt.exists() and target.exists():
                record=json.loads(receipt.read_text())
                if record['source_sha256']!=expected or record['cache_sha256']!=hashlib.sha256(target.read_bytes()).hexdigest() or record['role']!=role:raise RuntimeError('Prior source receipt differs')
                records.append(record);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='me091-video-') as work:
                video=Path(work)/'source.mp4';pixels=Path(work)/'gray32.raw'
                fetch(BASE+key,video)
                actual=hashlib.sha256(video.read_bytes()).hexdigest()
                if actual!=expected or video.stat().st_size!=metadata['size']:raise RuntimeError('Pinned publisher video checksum failed')
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,r_frame_rate,width,height','-of','json',str(video)],text=True))['streams'][0]
                if int(probe['nb_frames'])!=frames or probe['r_frame_rate']!='15/1':raise RuntimeError('Publisher retained-frame alignment failed')
                subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(video),'-vf','scale=32:32:flags=area,format=gray','-vsync','0','-f','rawvideo',str(pixels)],check=True,timeout=180)
                images=np.fromfile(pixels,dtype=np.uint8).reshape(-1,1,32,32)
                if len(images)!=frames:raise RuntimeError('Decoded frame count differs from qpos')
                pixel_sha=hashlib.sha256(images.tobytes()).hexdigest()
                np.savez_compressed(target,images=images)
                record={'task':task,'episode':episode,'role':role,'frames':frames,'source_path':key,'source_revision':REV,'source_sha256':actual,'source_bytes':metadata['size'],'retained_frame_probe':probe,'pixel_sha256':pixel_sha,'cache_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'cache_bytes':target.stat().st_size,'pixels':'uint8 gray 32x32, area resize, no augmentation','alignment':'frame_index i to retained video frame i; 15Hz retained grid; no future frame in causal query'}
                write(receipt,record);records.append(record)
            print(json.dumps({'verified':len(records),'role':role,'episode':f'{task}:{episode}','frames':frames,'source_MB':round(metadata['size']/1e6,2)}),flush=True)
    result={'source_revision':REV,'license':'CC-BY-4.0','credit':'BodenAI / RW-RL','camera':CAMERA,'original_roles_only':True,'FIT_episodes':len(fit),'REPORT_episodes':len(report),'sealed_CAL_TEST_opened':False,'future_image_inputs':False,'records':records,'records_sha256':digest(records),'source_bytes':sum(r['source_bytes'] for r in records),'cache_bytes':sum(r['cache_bytes'] for r in records),'ffmpeg_version':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]}
    write(cache/'SOURCE_INDEX.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path('/tmp/me091-visual-source'));a=p.parse_args();r=acquire(a.cache);print(json.dumps({k:v for k,v in r.items() if k!='records'},sort_keys=True),flush=True)
