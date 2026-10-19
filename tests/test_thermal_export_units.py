"""Independent semantic unit contract; numeric replay alone misses wrong labels."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from tac_osm.development.thermal_forecast import ThermalForecast


def test_portable_thermal_output_is_explicit_fahrenheit(tmp_path):
    root=Path(__file__).resolve().parents[1]
    shutil.copy(root/'src/tac_osm/development/thermal_forecast.py',tmp_path/'model.py')
    shutil.copy(root/'scripts/runtime_thermal_forecast_003.py',tmp_path/'runtime.py')
    model=ThermalForecast().double();torch.save(model.state_dict(),tmp_path/'model.pt')
    sha=hashlib.sha256((tmp_path/'model.pt').read_bytes()).hexdigest()
    (tmp_path/'metadata.json').write_text(json.dumps(dict(spec={},channels=list(range(10)),checkpoint_sha256=sha,temperature_unit='degF')))
    # Publisher tables specifyF. A constant68Fcontext must not be returned as68C.
    values=[[68.]*11+[1000.] for _ in range(8)]
    request=json.dumps(dict(values=values,minute_of_day=[i*15 for i in range(8)]))+'\n'
    guard="import sys,runpy;sys.path.insert(0,'.');runpy.run_path('runtime.py',run_name='__main__')"
    result=subprocess.run([sys.executable,'-I','-B','-c',guard],cwd=tmp_path,input=request,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    response=json.loads(result.stdout)
    assert 'temperature_C' not in response
    assert response['temperature_F']==[[68.]*10,[68.]*10]
    source=json.loads((root/'research/real-maint-003/unit-erratum.json').read_text())
    assert source['actual_temperature_unit']=='degF' and source['input_units']==['degF']*11+['W']
