"""Additional mathematical gate check; no learned model or optimizer."""
from pathlib import Path
import json,numpy as np

def main():
    s=np.array([[.2,.4],[.5,.1]],dtype=np.float64);v=np.array([1.,0.]);k=np.array([1.,0.]);decoder=np.diag([2.,3.]);scale=.1
    error=float(np.linalg.norm(v-s@k));scaled_error=float(np.linalg.norm(scale*v-scale*s@k))
    prediction=decoder@(s@k);scaled_prediction=(decoder/scale)@((s*scale)@k)
    difference=float(np.max(np.abs(prediction-scaled_prediction)))
    assert difference<1e-12 and (error>.75)!=(scaled_error>.75)
    relative=error/np.linalg.norm(v);scaled_relative=scaled_error/np.linalg.norm(scale*v)
    assert abs(relative-scaled_relative)<1e-12
    written=s+np.outer(v-s@k,k);post_error=float(np.linalg.norm(v-written@k));assert post_error<1e-12
    result=dict(status='PASS_CONSTRUCTED_SCALE_AND_POSTWRITE_COUNTEREXAMPLES',optimizer_updates=0,
        original_error=error,scaled_error=scaled_error,gate_before=bool(error>.75),gate_after_scaling=bool(scaled_error>.75),
        decoded_prediction_max_change=difference,relative_error=relative,scaled_relative_error=scaled_relative,
        unit_gain_delta_immediate_postwrite_error=post_error,
        interpretation='An absolute residual gate depends on representation scale even when a compensating readout preserves decoded output. Relative error fixes this constructed scale invariance only. Postwrite current-key error can vanish while older keys remain vulnerable; neither gate guarantees future relevance.')
    out=Path(__file__).with_name('SCALE_CHECK.json');out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
