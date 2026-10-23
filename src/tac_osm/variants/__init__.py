"""Research-lane adapters around historical TAC-OSM implementations.

Importing this package never fits a model, initializes torch, merges branches,
touches held-out records, or grants authority for a real-world action.
"""
from .registry import VARIANTS, Variant, VariantSpec, describe_variant
from .contracts import ObservationStamp, ForecastEvidence, PredictionEnvelope

__all__ = [
    "VARIANTS", "Variant", "VariantSpec", "describe_variant",
    "ObservationStamp", "ForecastEvidence", "PredictionEnvelope",
]
