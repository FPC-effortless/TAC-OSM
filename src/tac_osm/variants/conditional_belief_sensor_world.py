"""ME111 positive conditional scales over the unchanged native TAC state graph.

Only the forecast scale contract changes. The old empirical floor remains a
provenance buffer, not a claimed statistical bound. The split distribution is
valid; its final variance is not generally the base component sum.
"""
from .residual_role_sensor_world import ResidualRoleSensorWorld
from .median_split_sensor_world import MedianSplitForecast


def conditional_scales(future_variance,current_variance,response,missing,width_factors):
    base=(1e-6+future_variance+response.square()*current_variance*missing).sqrt()
    return base*width_factors[:,:6],base*width_factors[:,6:],base


class ConditionalBeliefSensorWorld(ResidualRoleSensorWorld):
    def forward(self,values,observed,ages,clock,owners,hours,*,state=None,return_trace=False):
        inherited,final,trace=super().forward(values,observed,ages,clock,owners,hours,
            state=state,return_trace=True)
        left,right,base=conditional_scales(trace['future_variance'],trace['current_variance'],
            trace['affine_response'],(~observed[:,-1,:6]).to(values.dtype),trace['width_factors'])
        forecast=MedianSplitForecast(inherited.log_location,left,right)
        if return_trace:
            return forecast,final,dict(trace,legacy_floor_scale=trace['protected_floor_scale'],
                base_log_scale=base,total_variance=base.square(),left_scale=left,right_scale=right,
                fixed_empirical_floor_applied=False)
        return forecast,final
