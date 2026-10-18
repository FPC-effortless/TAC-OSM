"""Same-weight allocation optimization for the explicitly programmed MLP control."""
try:
    from .structured_memory import PublicFeatureMLP
except ImportError:
    from memory_model import PublicFeatureMLP


class InferencePublicFeatureMLP(PublicFeatureMLP):
    def observe(self,state,event,present=None,force_write=False):
        raw,occupied=state;kind=event[:,:4].argmax(-1);keep=kind!=3
        if present is not None:keep=keep&present.bool()
        rows=keep.nonzero(as_tuple=True)[0];slot=event[:,4:12].argmax(-1)+8*(kind==1).long()
        trace={'gate':keep,'writes':len(rows),'address':slot}
        if not len(rows):return state,trace
        raw=raw.clone();occupied=occupied.clone();raw[rows,slot[rows]]=event[rows];occupied[rows,slot[rows]]=1
        return (raw,occupied),trace
