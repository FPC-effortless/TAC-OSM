"""Same translated multimodal distribution, with its exact explicit median."""
from .multimode_distribution import log_density,mixture_crps,cdf
from .multimode_distribution import mixture_quantiles as inverted_quantiles


def implicit_median(prediction):
    return prediction['median'].double()


def mixture_quantiles(prediction,probabilities=(.05,.5,.95)):
    roots=inverted_quantiles(prediction,probabilities)
    return tuple(prediction['median'].double() if p==.5 else q for p,q in zip(probabilities,roots))
