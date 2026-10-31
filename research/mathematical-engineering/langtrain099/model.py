"""LANGTRAIN099 contextual normalized associative TAC, untrained candidate.

Two native key/value banks, distinct learned addresses and time ranges, contextual
writes and independently exposed reads. Two work-efficient affine scans replace
the old nonlinear feedback and graph solve. No labels or loss masks enter forward.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


def affine_scan(a, b, initial, *, parallel=True):
    """Inclusive scan of s'=a*s+b; a may broadcast over trailing state axes.

    Pair reduction plus reconstruction: O(T) state operations, O(log T) span.
    This is not a Hillis--Steele O(T log T) implementation. Never divides by a
    prefix product, so long horizons/underflow and a=0 are legitimate.
    """
    if b.ndim < 3 or b.shape[1] < 1 or a.shape[:2] != b.shape[:2] or initial.shape != b[:, 0].shape:
        raise ValueError('aligned nonempty batch/time/state required')
    length = b.shape[1]
    if length < 1:
        raise ValueError('nonempty scan required')
    if not parallel:
        state = initial; result = []
        for i in range(length):
            state = a[:, i] * state + b[:, i]
            result.append(state)
        return torch.stack(result, 1)
    if length == 1:
        return a * initial[:, None] + b
    pairs = length // 2
    left_a, right_a = a[:, :2*pairs:2], a[:, 1:2*pairs:2]
    left_b, right_b = b[:, :2*pairs:2], b[:, 1:2*pairs:2]
    odds = affine_scan(right_a * left_a, right_a * left_b + right_b,
                       initial, parallel=True)
    before_even = torch.cat((initial[:, None], odds), 1)[:, :(length+1)//2]
    evens = a[:, ::2] * before_even + b[:, ::2]
    paired = torch.stack((evens[:, :pairs], odds), 2).flatten(1, 2)
    return torch.cat((paired, evens[:, -1:]), 1) if length % 2 else paired


@dataclass(frozen=True)
class Geometry099:
    embed: int = 256
    context: int = 376
    key: int = 24
    value: int = 32
    ffn: int = 4335

    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in (self.embed,self.context,self.key,self.value,self.ffn)):
            raise ValueError('positive integer geometry required')

    @property
    def features(self):return 2*self.embed+self.context
    @property
    def projection(self):return 4*self.key+2*self.value+2
    @property
    def comparison(self):return self.embed+self.context+3*self.value+2
    @property
    def decoder(self):return 2*self.embed+self.context+3*self.value+2

    def parameter_count(self):
        d,c,k,v,w=self.embed,self.context,self.key,self.value,self.ffn
        return 256*d+3*d+c*(d+1)+c+(self.features+1)*self.projection+(self.comparison+1)*v+2*self.decoder+2*(self.decoder+1)*w+256*(w+1)


@dataclass(frozen=True)
class State099:
    owners: tuple[str, ...]
    clock: int
    tensors: dict[str, torch.Tensor]

    @property
    def arm(self):
        return "tac099"

    def tensor_bytes(self):
        return sum(v.numel()*v.element_size() for v in self.tensors.values())


class ScaledContextualAssociativeTAC099(nn.Module):
    """Exactly10M active parameters at registered geometry; no frozen/dummy weights."""
    arm = 'tac099'

    def __init__(self, geometry=Geometry099()):
        super().__init__()
        self.geometry=geometry
        d,c,k,v,w=geometry.embed,geometry.context,geometry.key,geometry.value,geometry.ffn
        self.embedding = nn.Embedding(256, d)
        self.context_lag = nn.Parameter(torch.full((3, d), .1))
        self.context_input = nn.Linear(d, c)
        horizons = torch.logspace(math.log10(8.), math.log10(4096.), c)
        self.raw_context_decay = nn.Parameter(torch.log(torch.expm1(1./horizons)))
        self.address_value = nn.Linear(geometry.features, geometry.projection)
        self.compare = nn.Linear(geometry.comparison, v)
        self.norm = nn.LayerNorm(geometry.decoder)
        self.ffn_up = nn.Linear(geometry.decoder, w)
        self.ffn_gate = nn.Linear(geometry.decoder, w)
        self.ffn_out = nn.Linear(w, 256)
        self.register_buffer('bank_min', torch.tensor([.97, .995]))
        self.register_buffer('bank_max', torch.tensor([.995, .99995]))
        self.read_epsilon = 1e-4

    def counts(self):
        return {'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'deployed': sum(p.numel() for p in self.parameters()),
                'fixed_buffer_bytes': sum(p.numel()*p.element_size() for p in self.buffers())}

    def start(self, owners):
        owners = tuple(owners)
        if not owners or len(set(owners)) != len(owners) or any(type(o) is not str or not o for o in owners):
            raise ValueError('explicit unique nonempty stream owners required')
        shapes = {'context': (self.geometry.context,), 'lexical': (3, self.geometry.embed), 'previous_key': (2, self.geometry.key),
                  'memory': (2, self.geometry.value, self.geometry.key), 'mass': (2, self.geometry.key)}
        return State099(owners, 0, {n: self.embedding.weight.new_zeros((len(owners), *shape))
                                    for n, shape in shapes.items()})

    def validate(self, tokens, state, owners):
        if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long or tokens.ndim != 2
                or min(tokens.shape) < 1 or bool((tokens < 0).any()) or bool((tokens > 255).any())):
            raise ValueError('observed nonempty Long[B,T] byte IDs required')
        if self.embedding.weight.dtype not in (torch.float32, torch.float64):
            raise ValueError('FP32/FP64 only until mixed-precision correspondence is qualified')
        if tokens.device != self.embedding.weight.device:
            raise ValueError('tokens and model must share a device')
        if not owners or len(set(owners)) != len(owners) or any(type(o) is not str or not o for o in owners):
            raise ValueError('explicit unique nonempty stream owners required')
        # Validate metadata without allocating five throwaway zero tensors on
        # every streaming step. No learned-value or state cache is maintained.
        shapes = {'context': (self.geometry.context,), 'lexical': (3, self.geometry.embed), 'previous_key': (2, self.geometry.key),
                  'memory': (2, self.geometry.value, self.geometry.key), 'mass': (2, self.geometry.key)}
        if (state.owners != owners or type(state.clock) is not int or state.clock < 0
                or tokens.shape[0] != len(owners) or set(state.tensors) != set(shapes)):
            raise ValueError('complete same-owner state and legitimate clock required')
        for name, tensor in state.tensors.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != (len(owners), *shapes[name])
                    or tensor.dtype != self.embedding.weight.dtype or tensor.device != self.embedding.weight.device
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError('state tensor shape/precision/device/finite mismatch: '+name)

    def intervene(self, state, condition):
        if condition=='carry':return State099(state.owners,state.clock,{n:v.clone() for n,v in state.tensors.items()})
        if condition=='reset':return State099(state.owners,state.clock,self.start(state.owners).tensors)
        if condition=='corrupt':return State099(state.owners,state.clock,{n:v.roll(1,dims=-1) for n,v in state.tensors.items()})
        raise ValueError('registered carry/reset/corrupt condition required')

    def forward(self, tokens, *, state=None, owners=None, write_mask=None, parallel=True,
                return_trace=False):
        if owners is None:
            owners = tuple(f'train-{i}' for i in range(tokens.shape[0]))
        owners = tuple(owners)
        if state is None:
            state = self.start(owners)
        self.validate(tokens, state, owners)
        batch, length = tokens.shape; s = state.tensors
        k,v=self.geometry.key,self.geometry.value
        if write_mask is None:
            write_mask = torch.ones_like(tokens, dtype=torch.bool)
        if (not isinstance(write_mask, torch.Tensor) or write_mask.shape != tokens.shape or write_mask.dtype != torch.bool
                or write_mask.device != tokens.device):
            raise ValueError('legal write mask must be Bool[B,T] on source device')
        e = self.embedding(tokens).tanh()
        history = torch.cat((s['lexical'], e), 1)
        local = e
        for lag in range(3):
            local = local + self.context_lag[lag].tanh()*history[:, 2-lag:2-lag+length]
        local = local.tanh()
        delta = torch.sigmoid(-self.raw_context_decay)
        context_offer = self.context_input(local).tanh()
        contexts = affine_scan(delta[None, None].expand(batch, length, -1),
                               (1-delta)*context_offer, s['context'], parallel=parallel)
        past_context = torch.cat((s['context'][:, None], contexts[:, :-1]), 1)
        features = torch.cat((e, local, past_context), -1)
        raw = self.address_value(features)
        keys = raw[..., :2*k].reshape(batch, length, 2, k).softmax(-1)
        queries = raw[..., 2*k:4*k].reshape(batch, length, 2, k).softmax(-1)
        values = raw[..., 4*k:4*k+2*v].reshape(batch, length, 2, v).tanh()
        alpha = self.bank_min + (self.bank_max-self.bank_min)*raw[..., 4*k+2*v:].sigmoid()
        previous = torch.cat((s['previous_key'][:, None], keys[:, :-1]), 1)
        legal = write_mask
        if state.clock == 0:
            legal = torch.cat((torch.zeros_like(legal[:, :1]), legal[:, 1:]), 1)
        a = torch.where(legal[..., None], alpha, torch.ones_like(alpha))
        offer = torch.cat(((values[..., :, None]*previous[..., None, :]).flatten(-2), previous), -1)
        b = (1-a[..., None])*offer
        initial = torch.cat((s['memory'].flatten(-2), s['mass']), -1)
        banks = affine_scan(a[..., None], b, initial, parallel=parallel)
        memory = banks[..., :v*k].reshape(batch, length, 2, v, k)
        mass = banks[..., v*k:]
        numerator = (memory*queries[..., None, :]).sum(-1)
        evidence = (mass*queries).sum(-1)
        reads = numerator/(evidence[..., None]+self.read_epsilon)
        fast, slow = reads.unbind(-2)
        compare_input = torch.cat((local, past_context, fast, slow, (fast-slow).abs(), evidence), -1)
        gate = self.compare(compare_input).sigmoid()
        mixed = gate*fast+(1-gate)*slow
        z = self.norm(torch.cat((e, local, past_context, fast, slow, mixed, evidence), -1))
        logits = self.ffn_out(self.ffn_up(z)*F.silu(self.ffn_gate(z)))
        final = State099(owners, state.clock+length, {
            'context': contexts[:, -1], 'lexical': history[:, -3:],
            'previous_key': keys[:, -1], 'memory': memory[:, -1], 'mass': mass[:, -1]})
        if return_trace:
            return logits, final, {'keys': keys, 'queries': queries, 'values': values,
                'alpha': a, 'memory': memory, 'mass': mass, 'reads': reads,
                'evidence': evidence, 'compare_gate': gate, 'context': contexts}
        return logits, final
