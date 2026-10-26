#!/usr/bin/env python3
"""S1TEMP007 source eligibility discovery: metadata ONLY, ZERO source rows."""
from __future__ import annotations
import json,urllib.request
SOURCES=(
 "lerobot/aloha_static_thread_velcro",
 "lerobot/aloha_static_battery",
 "lerobot/aloha_static_fork_pick_up",
 "lerobot/aloha_static_tape",
 "lerobot/aloha_static_candy",
 "lerobot/aloha_static_pro_pencil",
 "lerobot/aloha_static_knife",
 "lerobot/aloha_static_box",
)
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={
      "User-Agent":"S1TEMP007-original-metadata-only/1.0"}),timeout=35) as r:
        return json.load(r)
def main():
    for name in SOURCES:
        try:
            latest=get(f"https://huggingface.co/api/datasets/{name}")
            commits=get(f"https://huggingface.co/api/datasets/{name}/commits/main")
            candidate=[]
            for c in commits[:30]:
                rev=c.get("id")
                if not rev or len(rev)!=40:continue
                try:
                    d=get(f"https://huggingface.co/datasets/{name}/resolve/{rev}/meta/info.json")
                except Exception:
                    continue
                n=d.get("total_episodes");F=d.get("total_frames")
                fps=d.get("fps");format=d.get("data_path","")
                min_frames=(F/n) if isinstance(n,int) and n else 0
                if (d.get("codebase_version")=="v2.0" and
                    n>=40 and min_frames>=550 and fps==50 and
                    "episode_{episode_index:06d}" in format):
                    candidate.append(dict(revision=rev,episodes=n,
                                          frames=F,mean_episode_frames=min_frames,
                                          publisher_license=(latest.get("cardData") or {}).get("license"),
                                          schema=d["codebase_version"],path=format))
                    break
            print("S1TEMP007_METADATA_SOURCE",json.dumps(dict(source=name,
                       compatible=candidate or None),sort_keys=True),flush=True)
        except Exception as e:
            print("S1TEMP007_METADATA_SOURCE",json.dumps(dict(source=name,
                       error=type(e).__name__,detail=str(e)[:120])),flush=True)
if __name__=="__main__":main()
