"""Source-pinned helpers; no model import until the released runner calls them."""
import hashlib,importlib.abc,importlib.util,json,sys
from pathlib import Path
from contextlib import contextmanager

def tensor_hash(torch,t):
    assert t.device.type=='cpu'
    value=t.detach().contiguous()
    h=hashlib.sha256(json.dumps({'shape':list(t.shape),'dtype':str(t.dtype)},sort_keys=True).encode())
    h.update(bytes(value.reshape(-1).view(torch.uint8).tolist()));return h.hexdigest()

def parameter_hash(torch,model):
    return hashlib.sha256(json.dumps({n:tensor_hash(torch,p) for n,p in model.named_parameters()},sort_keys=True).encode()).hexdigest()

class PinnedFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, source, files):
        self.source, self.entries, self.loaded = source, {}, {}
        prefixes = ['research/collect018', 'research/collect016', 'research/collect014',
                    'research/collect012', 'research/collect011', 'research/collect011/vendor']
        for path, row in files.items():
            if not path.endswith('.py'): continue
            prefix = next((p for p in reversed(prefixes) if path.startswith(p+'/')), None)
            if prefix is None: continue
            parts = path[len(prefix)+1:-3].split('/')
            package = parts[-1] == '__init__'
            name = '.'.join(parts[:-1] if package else parts)
            self.entries[name] = (path, row, package)
    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.entries:
            return importlib.util.spec_from_loader(fullname, self, is_package=self.entries[fullname][2])
        if fullname.split('.')[0] in {'collect011','collect012','collect014','collect016','collect018','tacopt109','tacround110'}:
            raise ImportError('UNPINNED_SOURCE_IMPORT '+fullname)
    def create_module(self, spec): return None
    def exec_module(self, module):
        path,row,package=self.entries[module.__name__]; file=self.source/path; raw=file.read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        module.__file__=str(file)
        if package: module.__path__=[str(file.parent)]
        self.loaded[module.__name__]=row['sha256']
        exec(compile(raw,str(file),'exec',dont_inherit=True),module.__dict__)

@contextmanager
def capture(torch,model):
    F=torch.nn.functional; old_linear,old_embedding=F.linear,F.embedding
    routers={id(b.phase_memory.router.weight):i for i,b in enumerate(getattr(model.core,'blocks',[]))}
    traces={};embeddings=[]
    def linear(value,weight,bias=None):
        result=old_linear(value,weight,bias)
        if id(weight) in routers:
            i=routers[id(weight)];assert i not in traces;traces[i]=result
        return result
    def embedding(input,weight,*args,**kw):
        result=old_embedding(input,weight,*args,**kw)
        if weight is model.core.embedding.weight:
            if result.requires_grad:result.retain_grad()
            embeddings.append(result)
        return result
    F.linear,F.embedding=linear,embedding
    try:yield traces,embeddings
    finally:F.linear,F.embedding=old_linear,old_embedding

def reference_balance(torch,traces):
    values=[]; counts=[]
    for i in range(4):
        p=traces[i].softmax(-1).reshape(-1,4)
        h=torch.zeros_like(p).scatter(-1,p.argmax(-1,keepdim=True),1.)
        values.append(4*(h.mean(0)*p.mean(0)).sum());counts.append(h.sum(0))
    return torch.stack(values).mean(),torch.stack(counts)

def reference_losses(torch,z,y,mask,first,balance):
    # Full 256-byte unmasked categorical loss, reconstructed independently.
    ce=-z.log_softmax(-1).gather(-1,y[...,None]).squeeze(-1)
    natural,task,answer,firstloss=ce[0].mean(),ce[1:].mean(),ce[mask].mean(),ce[first].mean()
    joint=natural+.25*task+answer+0*firstloss+.01*balance
    focused=natural+.25*task+answer+3*firstloss+.01*balance
    return {'first_ce':firstloss,'joint':joint,'first':focused}
