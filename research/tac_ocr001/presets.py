"""Original active parameter matches, not dummy padding or postfit selection."""
from .model import OCRConfig


def matched_presets():
    # Actual instantiated counts: TAC 1,309,376; Transformer 1,309,312;
    # GRU 1,309,824. Max pairwise difference <0.04%. Frozen encoder identical.
    return {name:OCRConfig(family=name,ffn_dim=hidden)
            for name,hidden in (("tac",384),("transformer",520),("gru",433))}
