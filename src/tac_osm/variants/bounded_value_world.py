"""Native value retrieval with a FIT-noise-bounded quiet innovation."""
import torch
from .clock_value_world import ClockValueWorld


class BoundedValueWorld(ClockValueWorld):
    architecture='native_bounded_clock_value_world'

    def decode(self,x,observed,representation,feature,velocity,command,lag_bank,lag_age):
        out=super().decode(x,observed,representation,feature,velocity,command,lag_bank,lag_age)
        old=out['current'];moving_absolute=old+out['moving_delta']
        # Quiet-component innovation cannot borrow a large obsolete movement
        # from the raw-value buffer. All coordinates use the same FIT-only rule.
        raw=old-x[:,None]
        unit=self.quiet_noise*self.horizons[None,:,None].sqrt()
        bounded=raw/(1+(raw/unit).abs())
        anchor=x[:,None]+bounded
        return {**out,'current':anchor,'moving_delta':moving_absolute-anchor}
