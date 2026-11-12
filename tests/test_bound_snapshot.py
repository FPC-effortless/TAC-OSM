"""Untrained persistence/replay controls, never trained model evidence."""
import hashlib
import json
from copy import deepcopy
import pytest
import torch
from safetensors.torch import load,save
from tac_osm.bound_snapshot import save_bound_snapshot,restore_bound_snapshot,BoundSnapshotError
from tac_osm.stream_binding import capture_stream_binding
from tac_osm.compact_stream_state import stream_state_storage
from research.bound_snapshot_20261112.fixture import model,prefix,context,continuation,values


@pytest.mark.parametrize('seed',[11,22,33])
@pytest.mark.parametrize('precise',[False,True])
@pytest.mark.parametrize('depth',[2,4])
@pytest.mark.parametrize('length',[0,17,65])
def test_roundtrip_actual_source_and_every_next_head(tmp_path,seed,precise,depth,length):
    original=model(seed,precise);ctx=context(depth,length)
    with torch.no_grad():
        state=prefix(original,depth,length)
        binding=capture_stream_binding(original,state,**ctx)
        rng=torch.get_rng_state().clone()
        pin=save_bound_snapshot(tmp_path/'snapshot',original,state,**ctx)
        assert torch.equal(rng,torch.get_rng_state())
        assert capture_stream_binding(original,state,**ctx)==binding
        proto=model(seed+100,precise)
        proto.attention['2'].inv_freq.add_(.125)
        before=capture_stream_binding(proto,prefix(proto,depth,0),**context(depth,0))
        rng=torch.get_rng_state().clone()
        restored,carry=restore_bound_snapshot(tmp_path/'snapshot',proto,expected_manifest_sha256=pin,**ctx)
        assert restored is not proto
        assert capture_stream_binding(restored,carry,**ctx)==binding
        assert capture_stream_binding(proto,prefix(proto,depth,0),**context(depth,0))==before
        assert torch.equal(rng,torch.get_rng_state())
        report=stream_state_storage(carry)
        assert report['retained_storage_bytes']==report['logical_bytes']
        assert not any(t.requires_grad or t.grad_fn is not None for t in values(carry).values())
        a=values(continuation(original,state,depth));b=values(continuation(restored,carry,depth))
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)


def fixture(tmp_path):
    m=model(11,True)
    with torch.no_grad():s=prefix(m);pin=save_bound_snapshot(tmp_path/'snapshot',m,s,**context())
    return tmp_path/'snapshot',pin


@pytest.mark.parametrize('fault',['bad_pin','missing_pin','manifest_bytes','tensor_bytes','missing_file',
                                  'tensor_symlink','root_symlink','schema','extra_path','duplicate_key'])
def test_bad_packet_rejected_before_return(tmp_path,fault):
    folder,pin=fixture(tmp_path);mf=folder/'manifest.json';tf=folder/'tensors.safetensors'
    if fault=='bad_pin':pin='0'*64
    elif fault=='missing_pin':pin=None
    elif fault=='manifest_bytes':mf.write_bytes(mf.read_bytes()+b' ')
    elif fault=='tensor_bytes':tf.write_bytes(tf.read_bytes()+b'x')
    elif fault=='missing_file':tf.rename(folder/'kept-original.safetensors')
    elif fault=='tensor_symlink':
        tf.rename(folder/'kept-original.safetensors');tf.symlink_to(folder/'kept-original.safetensors')
    elif fault=='root_symlink':
        p=tmp_path/'alias';p.symlink_to(folder,target_is_directory=True);folder=p
    else:
        obj=json.loads(mf.read_bytes())
        if fault=='schema':obj['schema']='qualified-training-checkpoint'
        elif fault=='extra_path':obj['path']='../../forbidden'
        raw=json.dumps(obj).encode()
        if fault=='duplicate_key':raw=raw[:-1]+b',"schema":"duplicate"}'
        mf.write_bytes(raw);pin=hashlib.sha256(raw).hexdigest()
    with pytest.raises(BoundSnapshotError):
        restore_bound_snapshot(folder,model(101,True),expected_manifest_sha256=pin,**context())


@pytest.mark.parametrize('fault',['key','shape','dtype','invalid_mass','garbage'])
def test_repinned_bad_layout_or_state_cannot_mutate_caller(tmp_path,fault):
    folder,pin=fixture(tmp_path)
    data=load((folder/'tensors.safetensors').read_bytes())
    if fault=='key':data['wrong']=data.pop('parameter.embedding.weight')
    elif fault=='shape':data['parameter.embedding.weight']=data['parameter.embedding.weight'][:1]
    elif fault=='dtype':data['parameter.embedding.weight']=data['parameter.embedding.weight'].double()
    elif fault=='invalid_mass':data['state.layer.0.fast_mass'].fill_(-1)
    payload=b'not a safetensors file' if fault=='garbage' else save(data)
    (folder/'tensors.safetensors').write_bytes(payload)
    obj=json.loads((folder/'manifest.json').read_bytes());obj['tensor_sha256']=hashlib.sha256(payload).hexdigest();obj['tensor_bytes']=len(payload)
    raw=json.dumps(obj).encode();(folder/'manifest.json').write_bytes(raw);pin=hashlib.sha256(raw).hexdigest()
    proto=model(101,True)
    with torch.no_grad():before=capture_stream_binding(proto,prefix(proto,4,0),**context(4,0))
    with pytest.raises(ValueError):restore_bound_snapshot(folder,proto,expected_manifest_sha256=pin,**context())
    with torch.no_grad():assert capture_stream_binding(proto,prefix(proto,4,0),**context(4,0))==before


@pytest.mark.parametrize('fault',['config','mode','training','owner','depth','cutoff'])
def test_wrong_prototype_or_context_remains_unmodified(tmp_path,fault):
    folder,pin=fixture(tmp_path);proto=model(101,fault!='mode');ctx=context()
    if fault=='config':object.__setattr__(proto.config,'max_correction_gain',.125)
    elif fault=='training':proto.train()
    elif fault=='owner':ctx['stream_ids']=('other',)
    elif fault=='depth':ctx['depth']=2
    elif fault=='cutoff':ctx['observed_tokens']=18
    before={n:t.detach().clone() for n,t in list(proto.named_parameters())+list(proto.named_buffers())}
    with pytest.raises(ValueError):restore_bound_snapshot(folder,proto,expected_manifest_sha256=pin,**ctx)
    assert all(torch.equal(before[n],t) for n,t in list(proto.named_parameters())+list(proto.named_buffers()))


def test_no_overwrite_and_budget(tmp_path,monkeypatch):
    folder,pin=fixture(tmp_path);original=(folder/'manifest.json').read_bytes()
    m=model(11,True)
    with torch.no_grad():s=prefix(m)
    with pytest.raises(BoundSnapshotError,match='snapshot_exists'):save_bound_snapshot(folder,m,s,**context())
    assert (folder/'manifest.json').read_bytes()==original
    monkeypatch.setattr('tac_osm.bound_snapshot.MAX_TENSOR_BYTES',16)
    with pytest.raises(BoundSnapshotError):save_bound_snapshot(tmp_path/'too-large',m,s,**context())
    assert not (tmp_path/'too-large').exists()
    with pytest.raises(BoundSnapshotError):restore_bound_snapshot(folder,m,expected_manifest_sha256=pin,**context())
