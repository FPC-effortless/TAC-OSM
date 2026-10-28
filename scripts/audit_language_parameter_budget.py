#!/usr/bin/env python3
"""Count frozen original untrained constructors; never run forward or optimize."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

COMMIT='2aeb631300f1465b5b62e7ffc966f8add8566f81'
SOURCE_SHA='acf80cb9051b7f2d2c3900d1676eed212890208cd5213088c6751f9cfd24392c'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    actual=subprocess.check_output(['git','-C',str(a.original_root),'rev-parse','HEAD'],text=True).strip()
    source=a.original_root/'research/language/flow_500k/tac_wave_ced_500k.py'
    if actual!=COMMIT or hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError('original implementation source differs')
    spec=importlib.util.spec_from_file_location('frozen_wave',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    for vocab,width in [(128,605),(256,605),(256,501),(4096,605)]:
        model=module.WaveCED500K(vocab=vocab,ffn_width=width)
        rows.append(dict(vocabulary=vocab,ffn_width=width,
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            deployed=sum(p.numel() for p in model.parameters()),
            embedding=model.embedding.weight.numel(),
            output=sum(p.numel() for p in model.ffn_out.parameters()),
            concept_output=model.concept_out.weight.numel()))
    report=dict(evidence='UNTRAINED_CONSTRUCTOR_COUNTS_ONLY_NOT_CAPABILITY',
        implementation_commit=COMMIT,source_sha256=SOURCE_SHA,model_executions=0,optimizer_steps=0,counts=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
