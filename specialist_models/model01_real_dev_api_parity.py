#!/usr/bin/env python3
"""No-refit API parity on the authentic, already-exposed UCI374 DEV cohort.

This is a mechanical reference-equivalence audit, NOT independent test,
new predictive skill, service qualification or any TAC-OSM neural result.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import uci_appliances_real_dev as uci
import uci_appliances_ridge_dev as ridge
import model01_reference_api as api

def main():
    data=uci.fetch_official_zip()
    if uci.digest(data)!=api.ORIGINAL_SOURCE_SHA256:
        raise RuntimeError("authentic source SHA changed")
    values=uci.original_rows(data) # only first 80%; TEST original targets unparsed
    m,q=api.load_original()
    reference=json.loads(api.RESULTS.read_bytes())
    n=0
    total_error=0.
    ninside=0
    for i in uci.eligible_indices(uci.CAL_END,uci.DEV_END):
        request={"schema":"model01-energy-source-aligned-v1",
                 "source_aligned_step_index":i,
                 "last_145_energy_wh":values[i-144:i+1]}
        result=api.predict(m,q,request)
        p=result["prediction_wh"]
        ref=ridge.forecast(m,values,i)
        if abs(p-ref)>1e-12:
            raise RuntimeError("original forecast graph != API prediction on authentic DEV row")
        y=values[i+uci.H]
        total_error+=abs(p-y)
        ninside+=(p-q<=y<=p+q)
        n+=1
    mae=total_error/n
    coverage=ninside/n
    if (n!=reference["dev"]["n"] or
        abs(mae-reference["dev"]["mae_wh"])>1e-10 or
        abs(coverage-reference["dev"]["nominal_90_coverage"])>1e-12):
        raise RuntimeError("API scored values differ from original frozen DEV receipt")
    print(api.canonical({
        "schema":"model01-real-exposed-dev-api-parity-v1",
        "source_zip_sha256":api.ORIGINAL_SOURCE_SHA256,
        "checkpoint_sha256":api.ORIGINAL_MODEL_SHA256,
        "authentic_exposed_dev_requests":n,
        "max_allowed_abs_point_error":1e-12,
        "all_predictions_reference_equivalent":True,
        "original_dev_mae_wh":mae,
        "original_dev_observed_90_coverage":coverage,
        "protected_TEST_target_rows_parsed":False,
        "fitted_this_run":False,
        "independent_evaluation":"NOT_PERFORMED",
        "cloud_device_qualification":"NOT_PERFORMED"}))

if __name__=="__main__":
    main()
