"""DEV-035 diagnostics, not modified MTSK transitions or stability guarantees."""
import torch
import torch.nn.functional as F


def components(kernel,packed,x):
    """Reconstruct frozen transition; conditional row gain holds routing fixed."""
    c=kernel.config;j=c.experts;d=c.rank
    memory,previous=kernel.unpack(packed);key=kernel.key(x);value=kernel.value(x)
    prediction=torch.einsum('...jde,...e->...jd',memory,key)
    delta=value.unsqueeze(-2)-prediction
    prior=kernel.key_prior(x).reshape(*x.shape[:-1],j,d).softmax(-2)
    posterior=(prior.clamp_min(1e-8).log()-.5*delta.square()).softmax(-2)
    responsibility=(1-c.temporal_prior)*posterior+c.temporal_prior*previous
    responsibility=responsibility/responsibility.sum(-2,keepdim=True).clamp_min(1e-8)
    base=kernel.decay_logits.sigmoid().reshape(-1,1,1)
    conditioned=kernel.input_decay(x).reshape(*x.shape[:-1],j if kernel.multiple else 1,d,1).sigmoid()
    decay=(.5*base+.5*conditioned).clamp(1e-4,.9999)
    beta=F.softplus(kernel.beta_logits).reshape(-1,1,1)
    updated=decay*memory+beta*responsibility.unsqueeze(-1)*delta.unsqueeze(-1)*key.unsqueeze(-2).unsqueeze(-2)
    # Compute diagnostic gains in FP64 from the FP32 transition's coefficients.
    alpha=decay.squeeze(-1).double().expand_as(responsibility)
    coefficient=(beta.squeeze(-1).double()*responsibility.double())
    feedback=coefficient*key.double().square().sum(-1).unsqueeze(-1).unsqueeze(-1)
    row_gain=torch.maximum(alpha.abs(),(alpha-feedback).abs())
    return dict(next=torch.cat((updated.flatten(-3),responsibility.flatten(-2)),-1),
        row_gain=row_gain,alpha=alpha,feedback=feedback,responsibility=responsibility,
        key=key,beta=beta,memory=memory)


def responsibility_tangent_basis(experts,rank,device):
    """Orthonormal zero-sum expert directions for each responsibility coordinate."""
    spanning=torch.eye(experts,dtype=torch.float64,device=device)[:,:experts-1]
    spanning[-1]=-1
    q=torch.linalg.qr(spanning,mode='reduced').Q
    return torch.kron(q.contiguous(),torch.eye(rank,dtype=torch.float64,device=device))


def jacobian_metrics(kernel,packed,x,index=None):
    """One bank, actual FP32 Jacobian; SVD in FP64. No long-product claim."""
    if index is None:
        if packed.ndim!=1 or x.ndim!=1:raise ValueError('one bank and one public input required')
        base=packed
        function=lambda z:kernel.proposal(z[None],x[None])[0]
    else:
        base=packed[index]
        def function(z):
            edited=packed.clone();edited[index]=z
            return kernel.proposal(edited,x)[index]
    c=kernel.config;cut=c.experts*c.rank*c.rank
    jacobian=torch.autograd.functional.jacobian(function,base,vectorize=True).double()
    if not torch.isfinite(jacobian).all():raise FloatingPointError('nonfinite augmented-state Jacobian')
    tangent=torch.block_diag(torch.eye(cut,dtype=torch.float64,device=packed.device),responsibility_tangent_basis(c.experts,c.rank,packed.device))
    norm=lambda a:float(torch.linalg.svdvals(a)[0])
    return dict(ambient_gain=norm(jacobian),simplex_tangent_gain=norm(jacobian@tangent),
        memory_to_memory_gain=norm(jacobian[:cut,:cut]),responsibility_to_memory_gain=norm(jacobian[:cut,cut:]),
        memory_to_responsibility_gain=norm(jacobian[cut:,:cut]),responsibility_to_responsibility_gain=norm(jacobian[cut:,cut:]),
        coordinate_metric='Euclidean FP32 state coordinates; not invariant to rescaling',
        jacobian=jacobian)
