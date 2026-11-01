"""Reproducible no-fit CPU timings. Not trained OCR or GPU measurements."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import statistics
import time

import torch

from .model import GroundedOCRDecoder
from .presets import matched_presets


def profile_candidates(output, *, steps=32,threads=1):
    if not 8<=steps<=256 or not 1<=threads<=8:
        raise ValueError("bounded no-fit profiling steps/threads required")
    torch.set_num_threads(threads)
    rows=[]
    for family,config in matched_presets().items():
        torch.manual_seed(701)
        model=GroundedOCRDecoder(config).eval()
        features=torch.randn(1,576,config.feature_dim)
        boxes=torch.rand(1,576,4)
        tokens=torch.randint(0,256,(1,128))
        preparation,prefill,latency=[],[],[]
        with torch.inference_mode():
            for repetition in range(4):  # first full round warm-up, then three
                start=time.perf_counter()
                refs=model.prepare_visual(features,boxes,owners=("profile",),page_ids=("p1",))
                visual_ms=1000*(time.perf_counter()-start)
                start=time.perf_counter()
                _,state=model(tokens,refs,owners=("profile",))
                prefill_ms=1000*(time.perf_counter()-start)
                before=state.tensor_bytes()
                local=[]
                for step in range(steps):
                    start=time.perf_counter()
                    _,state=model.step(tokens[:,step%128:step%128+1],refs,owners=("profile",),state=state)
                    local.append(1000*(time.perf_counter()-start))
                assert state.tensor_bytes()==before
                if repetition:
                    preparation.append(visual_ms)
                    prefill.append(prefill_ms)
                    latency.extend(local)
        ordered=sorted(latency)
        row={"family":family,"config":asdict(config),"counts":model.counts(),
            "visual_projection_median_ms":statistics.median(preparation),
            "prefill_128_bytes_median_ms":statistics.median(prefill),
            "decode_median_ms_per_byte":statistics.median(latency),
            "decode_p90_ms_per_byte":ordered[int(.9*(len(ordered)-1))],
            "decode_bytes_per_second_from_median":1000/statistics.median(latency),
            "carried_tensor_bytes":state.tensor_bytes(),"cached_visual_kv_bytes":refs.tensor_bytes(),
            "physical_compact_storage_verified":all(v.untyped_storage().nbytes()==v.numel()*v.element_size() for v in state.tensors().values()),
            "new_optimizer_updates":0,"initialized_randomly":True}
        if family=="tac":
            from .steady import SteadyTACSession
            compiled_latency=[]
            with torch.inference_mode():
                session=SteadyTACSession(model,refs,state)
                eager_state,compiled_state=state,state
                max_error=0.
                for i in range(12):
                    token=tokens[:,i:i+1]
                    expected,eager_state=model.step(token,refs,owners=("profile",),state=eager_state)
                    actual,compiled_state=session.step(token,refs,owners=("profile",),state=compiled_state)
                    max_error=max(max_error,float((actual-expected).abs().max()))
                for i in range(steps*3):
                    start=time.perf_counter()
                    _,compiled_state=session.step(tokens[:,i%128:i%128+1],refs,owners=("profile",),state=compiled_state)
                    compiled_latency.append(1000*(time.perf_counter()-start))
            assert max_error<1e-5
            median=statistics.median(compiled_latency)
            row["steady_torchscript"]={"median_ms_per_byte":median,"bytes_per_second":1000/median,
                "speedup_over_same_eager_weights":row["decode_median_ms_per_byte"]/median,
                "compile_seconds":session.compile_seconds,"eager_max_logit_error_12_steps":max_error,
                "shape_specialized":True,"warning_count":len(session.trace_warnings),
                "warnings":session.trace_warnings,"cold_start_included_in_decode_latency":False,
                "gpu_verified":False}
        rows.append(row)
    report={"schema":"tac-ocr001.cpu-profile.v1","torch_version":torch.__version__,
            "python_version":platform.python_version(),"platform":platform.platform(),
            "threads":threads,"timed_rounds":3,"warmup_rounds":1,"decode_steps_per_round":steps,
            "encoder_execution_included":False,"timing_scope":"random-weight decoder and cached visual projection only",
            "qualification":"engineering_execution_only_no_OCR_quality_or_GPU_claim","models":rows,
            "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (Path(__file__),Path(__file__).with_name("model.py"),Path(__file__).with_name("presets.py"),
                              Path(__file__).with_name("steady.py"))}}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    return report
