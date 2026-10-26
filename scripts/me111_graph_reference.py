"""Independent serial native state plus ME111 conditional scale equations."""
from scripts.me109_graph_reference import serial as inherited_serial


def serial(model,values,observed,ages,clock,owners):
    mu,_,_,ss,sm,es,memory,trace=inherited_serial(model,values,observed,ages,clock,owners)
    missing=(~observed[:,-1,:6]).to(values.dtype)
    variance=1e-6+trace['future_variance']+trace['affine_response']**2*trace['current_variance']*missing
    b=variance.sqrt();factors=trace['width_factors'];left=b*factors[:,:6];right=b*factors[:,6:]
    trace=dict(trace,total_variance=variance,base_log_scale=b,left_scale=left,right_scale=right,
        fixed_empirical_floor_applied=False)
    return mu,left,right,ss,sm,es,memory,trace
