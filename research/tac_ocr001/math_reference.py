"""Independent scalar cross-attention reference, PARTIAL not full TAC proof."""
import math


def attention_reference(query, keys, values):
    """One head, lists[T,D], lists[K,D], lists[K,V]; no torch/NumPy."""
    if not query or not keys or len(keys)!=len(values):
        raise ValueError("nonempty matching attention operands required")
    output=[]
    for q in query:
        scores=[sum(a*b for a,b in zip(q,k))/math.sqrt(len(q)) for k in keys]
        top=max(scores)
        weights=[math.exp(s-top) for s in scores]
        total=sum(weights)
        output.append([sum(w*v[j] for w,v in zip(weights,values))/total for j in range(len(values[0]))])
    return output
