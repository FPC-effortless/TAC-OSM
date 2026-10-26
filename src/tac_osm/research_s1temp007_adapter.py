"""Version-agnostic Research Observatory adapter for ACTUAL multiscale16 state.

Engine untouched: the adapter describes source authorization and 72 fast + 16
true recurrent temporal floats. Four physically different retention rates
occupy four disjoint groups of four channels. No fictitious action candidates.
"""
from __future__ import annotations
from dataclasses import replace
from .research_s1temp006_adapter import persistent_slow16_adapter
from .research_system1_adapter import validate_system1_source
from .variants.s1temp006_persistent_slow import VelocityAnchoredSlowSystem1

def validate(model,sample):
    if not isinstance(model,VelocityAnchoredSlowSystem1) or model.memory_mode!="multiscale16":
        raise ValueError("not an actual multiscale16 native System1 model")
    validate_system1_source(model,sample)

multiscale16_observatory_adapter=replace(
    persistent_slow16_adapter,
    name="S1TEMP007-native-four-timescale-source-cut-v1",
    validate=validate,
)
