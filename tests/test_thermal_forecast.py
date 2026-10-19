"""Physical forecast residual and test-exposure boundaries."""
from datetime import datetime,timedelta
import io
import csv
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import pytest
import torch
from tac_osm.development.thermal_forecast import ThermalForecast
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import data_real_maintenance_003 as data


@pytest.mark.parametrize('kind,revin',[('linear',False),('linear',True),('mlp',False),('mlp',True)])
def test_zero_head_is_exact_persistence_and_can_learn_a_change(kind,revin):
    torch.set_num_threads(1)
    model=ThermalForecast(kind=kind,revin=revin,seed=73).double()
    values=torch.arange(8,dtype=torch.float64)[None,:,None].expand(2,8,12)/10+20
    minutes=torch.arange(8,dtype=torch.float64)[None].expand(2,8)*15
    expected=values[:,-1,None,:10].expand(-1,2,-1)
    torch.testing.assert_close(model(values,minutes),expected,atol=0,rtol=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    loss=(model(values,minutes)-(expected+.5)).square().mean()
    optimizer.zero_grad();loss.backward();optimizer.step()
    assert not torch.equal(model(values,minutes),expected)
    assert torch.isfinite(model(torch.full_like(values,20),minutes)).all()


def test_future_rows_are_not_accepted_as_past_context():
    model=ThermalForecast();x=torch.zeros(2,9,12);minutes=torch.zeros(2,9)
    with pytest.raises(ValueError,match='context'):model(x,minutes)


def test_preparation_never_reads_future_sensor_records(tmp_path,monkeypatch):
    names=[f'1701{i:02d}_baseline.csv' for i in range(1,13)]+[f'1808{i:02d}_baseline.csv' for i in range(1,5)]+['190809_future.csv']
    splits={'train':names[:12],'validation':names[12:16],'test':names[16:]}
    source=tmp_path/'source.zip'
    with ZipFile(source,'w') as archive:
        for name in names[:-1]:
            stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['TIMESTAMP',*data.ROOMS,*data.EXOGENOUS])
            start=datetime.strptime(name[:6],'%y%m%d')
            for tick in range(96):writer.writerow([(start+timedelta(minutes=15*tick)).isoformat(),*[20+tick/100]*12])
            archive.writestr(name,stream.getvalue())
        # An accidental future parse fails, rather than giving a plausible score.
        archive.writestr(names[-1],'FORBIDDEN_FUTURE_SENSOR_VALUES')
    monkeypatch.setattr(data,'inventory',lambda path:(names,splits))
    destination=tmp_path/'prepared';metadata=data.prepare(source,destination)
    assert metadata['test_sensor_values_parsed'] is False
    assert not (destination/'test.pt').exists()
    assert len(torch.load(destination/'train.pt',weights_only=True)['x'])==12*85
    assert len(torch.load(destination/'validation.pt',weights_only=True)['x'])==4*85


def test_test_cannot_open_before_all_fits_and_frozen_selection(tmp_path):
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='INCOMPLETE',fits=79)))
    (tmp_path/'selection.json').write_text('{}')
    with pytest.raises(AssertionError):data.open_test(tmp_path/'nonexistent.zip',tmp_path)
    assert not (tmp_path/'TEST-EXPOSURE.json').exists()
