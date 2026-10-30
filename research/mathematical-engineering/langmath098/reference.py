"""Independent NumPy sequential FULL forward, not a replacement Torch kernel.

Consumes an explicit parameter dictionary; never imports the source model.
Reference computes each observation and state update independently in time.
"""
import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def softmax(x):
    e = np.exp(x-x.max(axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)


def forward(p, tokens, state=None, clock=0, write_mask=None):
    tokens = np.asarray(tokens)
    batch, length = tokens.shape
    if state is None:
        state = {n: np.zeros((batch, *shape), dtype=p['embedding.weight'].dtype)
                 for n, shape in {'context':(48,), 'lexical':(3,32),
                     'previous_key':(2,8), 'memory':(2,16,8), 'mass':(2,8)}.items()}
    s = {n:v.copy() for n,v in state.items()}
    if write_mask is None:
        write_mask = np.ones(tokens.shape, dtype=bool)
    delta = sigmoid(-p['raw_context_decay'])
    linear = lambda x,n: x@p[n+'.weight'].T+p[n+'.bias']
    outputs=[]
    for t in range(length):
        e=np.tanh(p['embedding.weight'][tokens[:,t]])
        local=e.copy()
        for lag in range(3):
            local += np.tanh(p['context_lag'][lag])*s['lexical'][:,-1-lag]
        local=np.tanh(local)
        c_next=delta*s['context']+(1-delta)*np.tanh(linear(local,'context_input'))
        features=np.concatenate((e,local,s['context']),-1)
        raw=linear(features,'address_value')
        k=softmax(raw[:,:16].reshape(batch,2,8))
        q=softmax(raw[:,16:32].reshape(batch,2,8))
        v=np.tanh(raw[:,32:64].reshape(batch,2,16))
        alpha=p['bank_min']+(p['bank_max']-p['bank_min'])*sigmoid(raw[:,64:])
        legal=write_mask[:,t] & (clock+t>0)
        a=np.where(legal[:,None],alpha,1.)
        m=a[:,:,None,None]*s['memory']+(1-a[:,:,None,None])*v[:,:,:,None]*s['previous_key'][:,:,None,:]
        z=a[:,:,None]*s['mass']+(1-a[:,:,None])*s['previous_key']
        evidence=np.sum(z*q,-1)
        reads=np.sum(m*q[:,:,None,:],-1)/(evidence[:,:,None]+1e-4)
        fast,slow=reads[:,0],reads[:,1]
        g=sigmoid(linear(np.concatenate((local,s['context'],fast,slow,np.abs(fast-slow),evidence),-1),'compare'))
        mixed=g*fast+(1-g)*slow
        features=np.concatenate((e,local,s['context'],fast,slow,mixed,evidence),-1)
        centered=features-features.mean(-1,keepdims=True)
        norm=centered/np.sqrt(np.square(centered).mean(-1,keepdims=True)+1e-5)
        norm=norm*p['norm.weight']+p['norm.bias']
        up=linear(norm,'ffn_up');gate=linear(norm,'ffn_gate')
        outputs.append(linear(up*gate*sigmoid(gate),'ffn_out'))
        s={'context':c_next,'lexical':np.concatenate((s['lexical'][:,1:],e[:,None]),1),
           'previous_key':k,'memory':m,'mass':z}
    return np.stack(outputs,1),s


def controlled_loss(logits, targets, future, valid, domains, *, prefix_weight):
    """Equal domain/case means; Y coefficient fixed as prefix weight varies.

    Independent mathematical objective reference. No mask or target goes to
    the model. Both domains must be present; each case has prefix AND Y terms.
    This controls explicit coefficients, not nonlinear Adam/clipping shares.
    """
    if not np.isfinite(prefix_weight) or prefix_weight<0:
        raise ValueError('finite nonnegative prefix coefficient required')
    future=np.asarray(future);valid=np.asarray(valid);domains=np.asarray(domains)
    if future.dtype!=bool or valid.dtype!=bool or future.shape!=targets.shape or valid.shape!=targets.shape:
        raise ValueError('aligned Boolean target masks required')
    prefix=valid & ~future
    if (np.any(future & ~valid) or not np.all(future.sum(1)>0)
            or not np.all(prefix.sum(1)>0) or domains.shape!=(len(targets),)
            or set(domains.tolist())!={'code','arc'}):
        raise ValueError('both domains, nonempty cases and legal masks required')
    shifted=logits-logits.max(-1,keepdims=True)
    ce=np.log(np.exp(shifted).sum(-1))-np.take_along_axis(shifted,targets[...,None],-1)[...,0]
    per_case=(ce*future).sum(1)/future.sum(1)+prefix_weight*(ce*prefix).sum(1)/prefix.sum(1)
    return .5*sum(per_case[domains==d].mean() for d in ('code','arc'))
