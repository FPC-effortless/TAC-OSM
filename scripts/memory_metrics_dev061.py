"""Public-history read metrics. Evaluator only; never a learned-policy input."""


def visibility(raw,occupied,events,present,queries,query_valid,chosen):
    b,q,_=queries.shape;slots=raw.shape[1];keys=queries.argmax(-1)
    statuses=(events[:,:,0]>0)|(events[:,:,2]>0)
    eligible=(events[:,:,4:12].argmax(-1)[:,None,:]==keys[:,:,None])&statuses[:,None,:]&present[:,None,:]
    latest=events[:,:,21][:,None,:].expand(-1,q,-1).masked_fill(~eligible,-1e9).amax(-1)
    status=(raw[:,:,0]>0)|(raw[:,:,2]>0)
    matching=(raw[:,:,4:12].argmax(-1)[:,None,:]==keys[:,:,None])&status[:,None,:]&(occupied[:,:slots]>0)[:,None,:]
    available=(matching&(raw[:,:,21][:,None,:]>=latest[:,:,None]-1e-7)).any(-1)&eligible.any(-1)
    safe=chosen.clamp_max(slots-1)
    correct_kind_entity=matching.gather(-1,safe[:,:,None]).squeeze(-1)&(chosen<slots)
    dates=raw[:,:,21].gather(-1,safe)
    stale=correct_kind_entity&(dates<latest-1e-7)
    current=correct_kind_entity&(dates>=latest-1e-7)&eligible.any(-1)
    return [{'queries':int(query_valid[i].sum()),'stale':int((stale[i]&query_valid[i]).sum()),
             'latest_hits':int((current[i]&query_valid[i]).sum()),'wrong':int((~correct_kind_entity[i]&query_valid[i]).sum()),
             'latest_available':int((available[i]&query_valid[i]).sum())} for i in range(b)]
