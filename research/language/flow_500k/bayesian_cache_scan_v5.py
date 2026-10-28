"""GRAPH009: amortized O(T * orders) owner-bound, causal count scan.

Exact integer counts of source-observed continuations, with an expiry queue.
No targets enter the cache. No novelty or neural superiority claimed.
"""
from collections import deque
import torch

class ScannedCountCache:
    orders=(2,3,4)
    def __init__(self,capacity=512):
        if type(capacity) is not int or capacity<2:raise ValueError('integer capacity >=2 required')
        self.capacity=capacity;self.maxk=max(self.orders)
    def empty(self,owners):
        if type(owners) is not tuple or not owners or any(type(o) is not str or not o for o in owners):
            raise ValueError('nonempty source owners required')
        return {'count':0,'source_ids':owners,'history':tuple(() for _ in owners),
                'events':tuple(tuple(() for _ in self.orders) for _ in owners)}
    def scan_counts(self,observed,*,start=0,source_ids=None,initial_state=None):
        if observed.ndim!=2 or observed.dtype!=torch.long:raise ValueError('int64[B,T] source required')
        b,t=observed.shape
        if t<1 or type(start) is not int or start<0 or type(source_ids) is not tuple or len(source_ids)!=b or any(type(o) is not str or not o for o in source_ids):
            raise ValueError('invalid count/source owner')
        if not bool(((observed>=0)&(observed<128)).all()):raise ValueError('byte outside vocabulary')
        st=self.empty(source_ids) if initial_state is None else initial_state
        if st.get('source_ids')!=source_ids or st.get('count')!=start or len(st.get('history',()))!=b or len(st.get('events',()))!=b:
            raise ValueError('owner/count mismatch')
        counts=torch.zeros((b,t,3,128),dtype=torch.int32,device=observed.device)
        totals=torch.zeros((b,t,3),dtype=torch.int32,device=observed.device)
        newhist=[];newqueues=[]
        for row in range(b):
            history=deque(st['history'][row],maxlen=self.maxk+1)
            if len(history)>min(start,self.maxk+1) or any(type(v) is not int or not 0<=v<128 for v in history):
                raise ValueError('invalid carried history')
            qs=[];dicts=[]
            for oi,k in enumerate(self.orders):
                q=deque(st['events'][row][oi]);d={};last=-1
                for ordinal,key,val in q:
                    if not (type(ordinal) is int and max(0,start-self.capacity-1)<=ordinal<=start-2 and ordinal>last
                            and type(key) is tuple and len(key)==k and all(type(v) is int and 0<=v<128 for v in key)
                            and type(val) is int and 0<=val<128):
                        raise ValueError('invalid causal source event')
                    last=ordinal
                    bucket=d.setdefault(key,{})
                    bucket[val]=bucket.get(val,0)+1
                qs.append(q);dicts.append(d)
            for col,byte in enumerate(observed[row].tolist()):
                now=start+col
                for oi,k in enumerate(self.orders):
                    q=qs[oi];d=dicts[oi]
                    while q and q[0][0]<now-self.capacity:
                        _,key,val=q.popleft()
                        bucket=d[key];bucket[val]-=1
                        if not bucket[val]:del bucket[val]
                        if not bucket:del d[key]
                    if len(history)>=k+1:
                        key=tuple(list(history)[-k-1:-1]);val=history[-1]
                        q.append((now-1,key,val))
                        bucket=d.setdefault(key,{})
                        bucket[val]=bucket.get(val,0)+1
                history.append(byte)
                for oi,k in enumerate(self.orders):
                    if len(history)<k:continue
                    key=tuple(list(history)[-k:])
                    found=dicts[oi].get(key)
                    if found is not None:
                        totals[row,col,oi]=sum(found.values())
                        for val,num in found.items():counts[row,col,oi,val]=num
            newhist.append(tuple(history));newqueues.append(tuple(tuple(q) for q in qs))
        return {'counts':counts,'totals':totals},{'count':start+t,'source_ids':source_ids,'history':tuple(newhist),'events':tuple(newqueues)}
