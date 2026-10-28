"""GRAPH011: separately versioned conservative source-query fusion.

The weight 0.1 was selected post hoc on GRAPH010 genuine Vim FIT14 after
the original .6 failed. Eight additional Python3.13 source file SHA256
digests were then preregistered BEFORE any external scoring. No source
target, future label, or external score enters the inference route.
"""
import math
from source_query_hybrid_v6 import SourceQueryHybridV6

class CalibratedSourceQueryV7(SourceQueryHybridV6):
    def __init__(self,graph009_hybrid,query_encoder,*,approx_weight=0.1):
        if (type(approx_weight) not in (float,int) or
            not math.isfinite(approx_weight) or not 0<=approx_weight<=1):
            raise ValueError("finite predeclared query fusion in [0,1] required")
        super().__init__(graph009_hybrid,query_encoder,fusion=approx_weight)
        self.calibration_source="GRAPH010 exposed Vim FIT14; never retune on external GRAPH011 source"
        self.evidence_status="component-level; no matched SSM or full 500K language evaluation"
