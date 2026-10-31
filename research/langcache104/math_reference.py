"""LANGCACHE104 independent scalar/list mathematical reference.

No torch, numpy, source kernels, fitting, or reward evaluation.  It implements
the serial exact-form equations in ordinary Python float arithmetic.  This
reference is for small graph-correspondence fixtures, not a deployment kernel.
The explicit byte-key store is an external API and does not enter these logits.
"""
from __future__ import annotations

import math
from copy import deepcopy


def dot(x, y):
    if len(x) != len(y):
        raise ValueError("aligned vectors required")
    return sum(a*b for a, b in zip(x, y))


def sigmoid(x):
    if x >= 0:
        return 1/(1+math.exp(-x))
    e = math.exp(x)
    return e/(1+e)


def softmax(x):
    peak = max(x)
    weights = [math.exp(v-peak) for v in x]
    total = sum(weights)
    return [v/total for v in weights]


def rms_norm(x, weight, epsilon=1e-6):
    denominator = math.sqrt(sum(v*v for v in x)/len(x)+epsilon)
    return [v*w/denominator for v, w in zip(x, weight)]


def linear(x, weight, bias=None):
    if bias is None:
        bias = [0.]*len(weight)
    return [dot(row, x)+b for row, b in zip(weight, bias)]


def bank_step(q, keys, values, retention, write, memory, mass, epsilon=1e-4):
    """One head: q[K], keys[R,K], values[R,V], M[V,K], z[K].

    This is an inclusive write-then-read, using the current observed byte.
    It deliberately does not reuse chunk/prefix arithmetic from TAC source.
    """
    rank = len(keys)
    if not rank or len(values) != rank:
        raise ValueError("nonzero aligned rank required")
    kdim, vdim = len(q), len(memory)
    new_memory = [[retention*memory[i][j] + write/rank*sum(
        values[p][i]*keys[p][j] for p in range(rank))
        for j in range(kdim)] for i in range(vdim)]
    new_mass = [retention*mass[j]+write/rank*sum(keys[p][j] for p in range(rank))
                for j in range(kdim)]
    evidence = dot(new_mass, q)
    read = [dot(row, q)/(evidence+epsilon) for row in new_memory]
    return read, new_memory, new_mass, evidence


def rotate_pairs(vector, theta):
    if len(vector) != 2*len(theta):
        raise ValueError("complete pair angles required")
    output = []
    for p, angle in enumerate(theta):
        x, y = vector[2*p:2*p+2]
        c, s = math.cos(angle), math.sin(angle)
        output.extend([c*x-s*y, s*x+c*y])
    return output


def rotary_step(retention, theta, offer, previous):
    rotated = rotate_pairs(previous, theta)
    return [retention*x+(1-retention)*b for x, b in zip(rotated, offer)]


def causal_attention(query, keys, values, window):
    """Current query with already causally selected retained keys/values."""
    keys, values = keys[-window:], values[-window:]
    weights = softmax([dot(query, key)/math.sqrt(len(query)) for key in keys])
    return [sum(w*v[j] for w, v in zip(weights, values))
            for j in range(len(values[0]))]


def _reshape(flat, *shape):
    if len(shape) == 1:
        if len(flat) != shape[0]:
            raise ValueError("shape mismatch")
        return list(flat)
    stride = math.prod(shape[1:])
    if len(flat) != math.prod(shape):
        raise ValueError("shape mismatch")
    return [_reshape(flat[i*stride:(i+1)*stride], *shape[1:]) for i in range(shape[0])]


def _zeros(*shape):
    return [0.]*shape[0] if len(shape) == 1 else [_zeros(*shape[1:]) for _ in range(shape[0])]


def _linear(parameters, prefix, x):
    return linear(x, parameters[prefix+".weight"], parameters.get(prefix+".bias"))


def initial_state(config):
    h, k, v = (config[n] for n in ("n_heads", "key_dim", "value_dim"))
    return {"clock": 0, "layers": [{
        "fast_memory": _zeros(h, v, k), "fast_mass": _zeros(h, k),
        "slow_memory": _zeros(h, v, k), "slow_mass": _zeros(h, k),
        "rotary": _zeros(h, v), "recent_key": [[] for _ in range(h)],
        "recent_value": [[] for _ in range(h)]} for _ in range(config["n_layers"])]}


def forward_reference(tokens, parameters, config, *, state=None, write_mask=None):
    """Single-stream full neural graph from exported source state_dict lists.

    Parameter names match DeepCacheTAC104 at the source pin in SOURCE_MAP.json.
    State has exactly the source layer tensor names with the batch axis removed.
    This serial implementation supports tiny parity fixtures; it does not make
    owner authorization decisions or certify source finite-precision behavior.
    """
    if not tokens or any(type(x) is not int or x < 0 or x > 255 for x in tokens):
        raise ValueError("nonempty observed byte IDs required")
    if write_mask is None:
        write_mask = [True]*len(tokens)
    if len(write_mask) != len(tokens) or any(type(x) is not bool for x in write_mask):
        raise ValueError("aligned boolean write mask required")
    state = deepcopy(initial_state(config) if state is None else state)
    h, k, v, rank = (config[n] for n in ("n_heads", "key_dim", "value_dim", "write_rank"))
    scale = 1/math.sqrt(2*config["n_layers"])
    x = [list(parameters["embedding.weight"][token]) for token in tokens]
    for layer_index, layer in enumerate(state["layers"]):
        prefix = "blocks."+str(layer_index)+"."
        normalized = [rms_norm(row, parameters[prefix+"memory_norm.weight"]) for row in x]
        next_x = []
        for position, u in enumerate(normalized):
            reads, evidences = {}, {}
            for name in ("fast", "slow"):
                bank = prefix+name+"."
                query = [softmax(row) for row in _reshape(_linear(parameters, bank+"query", u), h, k)]
                keys = _reshape(_linear(parameters, bank+"key", u), h, rank, k)
                keys = [[softmax(row) for row in head] for head in keys]
                values = _reshape([math.tanh(y) for y in _linear(parameters, bank+"value", u)], h, rank, v)
                alpha = [sigmoid(y) for y in _linear(parameters, bank+"retention", u)]
                eligibility = [sigmoid(y) for y in _linear(parameters, bank+"eligibility", u)]
                reads[name], evidences[name] = [], []
                for head in range(h):
                    a = alpha[head] if write_mask[position] else 1.
                    w = (1-a)*eligibility[head]
                    read, m, z, evidence = bank_step(query[head], keys[head], values[head], a, w,
                        layer[name+"_memory"][head], layer[name+"_mass"][head])
                    layer[name+"_memory"][head], layer[name+"_mass"][head] = m, z
                    reads[name].append(read)
                    evidences[name].append(evidence)
            query = _reshape(_linear(parameters, prefix+"local_query", u), h, k)
            keys = _reshape(_linear(parameters, prefix+"local_key", u), h, k)
            values = _reshape(_linear(parameters, prefix+"local_value", u), h, v)
            if config.get("local_rope", True):
                frequencies = parameters[prefix+"local_frequencies"]
                angles = [(state["clock"]+position)*f for f in frequencies]
                query = [rotate_pairs(row, angles) for row in query]
                keys = [rotate_pairs(row, angles) for row in keys]
            recent = []
            for head in range(h):
                layer["recent_key"][head].append(keys[head])
                layer["recent_value"][head].append(values[head])
                recent.append(causal_attention(query[head], layer["recent_key"][head],
                                               layer["recent_value"][head], config["window"]))
                keep = config["window"]-1
                layer["recent_key"][head] = layer["recent_key"][head][-keep:] if keep else []
                layer["recent_value"][head] = layer["recent_value"][head][-keep:] if keep else []
            if config["rotary"]:
                alpha = [sigmoid(y) for y in _linear(parameters, prefix+"rotary_retention", u)]
                theta = _reshape([math.pi*math.tanh(y) for y in _linear(parameters, prefix+"rotary_angle", u)], h, v//2)
                offers = _reshape([math.tanh(y) for y in _linear(parameters, prefix+"rotary_offer", u)], h, v)
                for head in range(h):
                    a = alpha[head] if write_mask[position] else 1.
                    angle = theta[head] if write_mask[position] else [0.]*(v//2)
                    layer["rotary"][head] = rotary_step(a, angle, offers[head], layer["rotary"][head])
            mixed = []
            for head in range(h):
                fast, slow = reads["fast"][head], reads["slow"][head]
                mixed.extend(fast+slow+[a-b for a, b in zip(fast, slow)]+[a*b for a, b in zip(fast, slow)])
                if config["rotary"]:
                    mixed.extend(layer["rotary"][head])
                mixed.extend(recent[head]+[evidences["fast"][head], evidences["slow"][head]])
            delta = _linear(parameters, prefix+"perspective_out", mixed)
            residual = [a+scale*b for a, b in zip(x[position], delta)]
            n = rms_norm(residual, parameters[prefix+"ffn_norm.weight"])
            value, gate = (_linear(parameters, prefix+name, n) for name in ("ffn_value", "ffn_gate"))
            update = _linear(parameters, prefix+"ffn_out", [a*b*sigmoid(b) for a, b in zip(value, gate)])
            next_x.append([a+scale*b for a, b in zip(residual, update)])
        x = next_x
    state["clock"] += len(tokens)
    logits = [_linear(parameters, "output", rms_norm(row, parameters["output_norm.weight"])) for row in x]
    return logits, state
