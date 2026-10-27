#!/usr/bin/env python3
"""Original UCI374 *timestamped* DEV adapter parity, no refit or TEST opening.

Parses only the first 80% original rows; explicitly validates publisher time
grid and compares all 1968 already-exposed DEV predictions with Git-owned
original trained numerical ridge graph. This is engineering, not independent
predictive qualification or genuine customer streaming data availability.
"""
from __future__ import annotations
import csv
import io
import json
import zipfile
from datetime import timedelta

import model01_source_timestamp_adapter as adapter
import model01_reference_api as api
import uci_appliances_real_dev as uci
import uci_appliances_ridge_dev as ridge


def actual_source_rows(zip_bytes):
    if uci.digest(zip_bytes) != api.ORIGINAL_SOURCE_SHA256:
        raise api.InputError("original UCI publisher archive SHA mismatch")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        chosen=[m for m in z.infolist() if m.filename.rsplit("/",1)[-1]=="energydata_complete.csv"]
        if len(chosen)!=1 or chosen[0].file_size>uci.MAX_CSV_BYTES:
            raise api.InputError("source CSV missing or duplicated")
        items=[]
        with z.open(chosen[0]) as stream:
            for j,row in enumerate(csv.DictReader(io.TextIOWrapper(stream,encoding="utf-8-sig",newline=""))):
                if j >= uci.DEV_END:break
                if set(("date","Appliances")).difference(row):
                    raise api.InputError("source required timestamp and outcome missing")
                dt=adapter.parse_source_time(row["date"])
                if dt != adapter.SOURCE_ORIGIN+adapter.STEP*j:
                    raise api.InputError("original source grid continuity/phase mismatch")
                try:
                    y=float(row["Appliances"])
                except ValueError as exc:
                    raise api.InputError("source numeric invalid") from exc
                items.append((dt,y))
        if len(items)!=uci.DEV_END:
            raise api.InputError("fewer than exposed original source rows")
    return items


def main():
    entries=actual_source_rows(uci.fetch_official_zip())
    values=[v for _,v in entries]
    model,q=api.load_original()
    original=json.loads(api.RESULTS.read_bytes())
    count=0
    max_gap=0.
    for i in uci.eligible_indices(uci.CAL_END,uci.DEV_END):
        window=entries[i-144:i+1]
        payload={"schema":adapter.SOURCE_SCHEMA,"source_series_id":adapter.SOURCE_ID,
                 "observations":[{"observed_at":d.strftime(adapter.FMT),"energy_wh":v} for d,v in window]}
        request=adapter.adapt(payload)
        if request["source_aligned_step_index"] != i:
            raise api.InputError("publisher timestamp converted to wrong original source index")
        out=api.predict(model,q,request)
        reference=ridge.forecast(model,values,i)
        max_gap=max(max_gap,abs(out["prediction_wh"]-reference))
        if max_gap>1e-12:
            raise api.InputError("real original timestamp wrapper changes original forecast")
        count+=1
    if count!=original["dev"]["n"]:
        raise api.InputError("real exposed DEV denominator mismatch")
    print(api.canonical({
        "schema":"model01-uci374-original-timestamp-source-dev-parity-v1",
        "source_zip_sha256":api.ORIGINAL_SOURCE_SHA256,
        "original_checkpoint_sha256":api.ORIGINAL_MODEL_SHA256,
        "original_local_source_clock":"2016-01-11 17:00:00 naive source timestamps",
        "source_dev_real_cases":count,"max_source_timestamp_adapter_numerical_error":max_gap,
        "original_checkpoint_refit":False,
        "original_protected_TEST_target_rows_parsed":False,
        "independent_model_qualification":"NOT_ESTABLISHED",
        "new_customer_source_evaluated":False
    }))

if __name__=="__main__":
    main()
