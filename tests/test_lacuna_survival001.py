from scripts.lacuna_survival001 import PackedContexts
import numpy as np

def test_packed_cache_physical_budget_and_prefix_timing():
 prior=np.ones(256)/256
 for policy in ('fifo','reservoir','diversity'):
  c=PackedContexts(policy,1701)
  for x in [1,2,3,4,9,1,2,3,4]:c.append(x)
  p,n,ages=c.predict(prior)
  assert n>=1 and p[9]>prior[9] and len(c.data)==256 and max(ages)<=8

def test_retention_handles_long_stream_without_unbounded_state():
 for policy in ('fifo','reservoir','diversity'):
  c=PackedContexts(policy,1701)
  for i in range(4096):c.append(i%256)
  assert len(c.data)==256 and len(c.slots(25))==25
