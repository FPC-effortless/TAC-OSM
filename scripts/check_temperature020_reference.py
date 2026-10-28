#!/usr/bin/env python3
"""Compare actual scalar CE derivatives to source-pinned PNDS Python reference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.scalar_calibration import LogitBatch,loss_derivatives


def main():
    link=json.loads((ROOT/'research/mathematical-engineering/temperature020/PNDS_MATH_LINK.json').read_text())
    url=f"https://raw.githubusercontent.com/FPC-effortless/pnds-lean/{link['pnds_commit']}/foundation/v1/temperature020.py"
    with urllib.request.urlopen(url,timeout=30) as response:raw=response.read()
    if hashlib.sha256(raw).hexdigest()!=link['python_sha256']:
        raise ValueError('PNDS reference source differs')
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'reference.py';path.write_bytes(raw)
        spec=importlib.util.spec_from_file_location('pnds_temperature_ref',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        comparisons=0
        for row in ((2.,-.1,.4),(-5.,0.,2.),(0.,0.,0.)):
            for y in range(3):
                for a in (.25,.7,1.,4.):
                    actual=loss_derivatives(LogitBatch((row,),(y,),('fixture',),'calibration'),a)
                    expected=module.categorical_terms(row,y,a)
                    if any(abs(x-y)>1e-12 for x,y in zip(actual,expected)):
                        raise ValueError('source-to-reference numerical difference')
                    comparisons+=3
    print(json.dumps(dict(status='PASS_CONSTRUCTED_REFERENCE_ONLY',scalar_comparisons=comparisons,
        pnds_commit=link['pnds_commit'],whole_model_IEEE_AD='OPEN',original_data_calibration='BLOCKED_NOT_RUN')))


if __name__=='__main__':main()
