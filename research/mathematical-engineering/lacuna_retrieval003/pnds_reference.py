"""Independent scalar-list reference; no trained-source/full IEEE equivalence."""
import math


def attention(keys, values, query):
    if not keys or len(keys) != len(values) or any(len(k) != len(query) for k in keys):
        raise ValueError('nonempty matching records and query')
    scores = [8*sum(a*b for a,b in zip(k,query)) for k in keys]
    weights = [math.exp(s-max(scores)) for s in scores]
    total = sum(weights)
    weights = [w/total for w in weights]
    return [sum(w*v[i] for w,v in zip(weights,values)) for i in range(len(query))], weights


def match_norm(vector, reference):
    norm = math.sqrt(sum(v*v for v in vector))
    if norm < 1e-12:
        return [0.]*len(vector)
    scale = math.sqrt(sum(v*v for v in reference))/norm
    return [scale*v for v in vector]


def transported_read(values, coefficients, query):
    return [sum(v[i]*sum(a*b for a,b in zip(c,query)) for v,c in zip(values,coefficients))
            for i in range(len(query))]
