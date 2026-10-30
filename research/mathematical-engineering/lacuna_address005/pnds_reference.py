"""Independent list reference for fixed-span read projection; full NN proof OPEN."""
import math
EPS=1e-12
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def norm(v):return math.sqrt(dot(v,v))
def subtract_projection(v,basis):
    out=list(v)
    for _ in range(2):
        for q in basis:
            a=dot(out,q);out=[x-a*y for x,y in zip(out,q)]
    return out
def project(candidate,history,current,fast):
    if any(len(v)!=8 or any(not math.isfinite(x) for x in v) for v in (candidate,history,current,fast)):raise ValueError('finite eight-vectors')
    basis=[]
    for axis in (current,fast):
        n=norm(axis)
        if n==0:continue
        if n<=EPS:return list(history),'INCONCLUSIVE_NEAR_ZERO_AXIS',len(basis)
        residual=subtract_projection(axis,basis);rn=norm(residual)
        if rn>EPS:basis.append([x/rn for x in residual])
    hp=subtract_projection(history,basis);r=norm(hp)
    if r<=EPS:return list(history),'NO_DIRECTION_RADIUS',len(basis)
    vp=subtract_projection(candidate,basis);pn=norm(vp)
    if pn<=EPS:return list(history),'NO_DIRECTION_CANDIDATE',len(basis)
    b=[x-y for x,y in zip(history,hp)]
    return [x+r*y/pn for x,y in zip(b,vp)],'REGULAR',len(basis)
