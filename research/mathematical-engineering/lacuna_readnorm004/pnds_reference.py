"""Independent list reference for conditional double-norm projection, not full NN."""
import math

EPS=1e-12
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def norm(v): return math.sqrt(dot(v,v))

def dual_norm(candidate, history, current):
    if any(len(v)!=8 or any(not math.isfinite(x) for x in v) for v in (candidate,history,current)):
        raise ValueError('finite eight-vectors required')
    cn=norm(current)
    if cn==0:
        vn=norm(candidate)
        if vn<=EPS: return list(history),'INCONCLUSIVE_ZERO_CANDIDATE'
        return [x*norm(history)/vn for x in candidate],'REGULAR_ZERO_CURRENT'
    if cn<=EPS: return list(history),'INCONCLUSIVE_NEAR_ZERO_CURRENT'
    u=[x/cn for x in current];a=dot(history,u)
    hp=[x-a*y for x,y in zip(history,u)];r=norm(hp)
    if r<=EPS: return list(history),'NO_ORTHOGONAL_FREEDOM'
    b=dot(candidate,u);vp=[x-b*y for x,y in zip(candidate,u)];pn=norm(vp)
    if pn<=EPS: return list(history),'INCONCLUSIVE_CANDIDATE_PARALLEL'
    return [a*x+r*y/pn for x,y in zip(u,vp)],'REGULAR'
