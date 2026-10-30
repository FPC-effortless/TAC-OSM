"""Constructed runner/boundary checks only; no optimizer steps or capability evidence."""
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'research/mathematical-engineering/langacquire097'


def load(name):
    spec=importlib.util.spec_from_file_location('langacquire097_runner_test_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

T=load('train');E=load('evaluate');RUN=load('run')


def row(i,domain='lexical',a=8,q=4,y=4):
    return {'id':f'fixture-{i}','domain':domain,'partition':'FIT','acquisition_bytes':[65+i%8]*a,
            'query_bytes':[66]*q,'target_bytes':[67]*y}


def artifact_fixture(base):
    """Mock artifact transport only: these are not trained checkpoints/results."""
    incoming=base/'incoming';incoming.mkdir();config=base/'register.json';config.write_text('{"fixture":true}')
    register={'training':{'steps':2,'batch_size':4,'context':272},'manifest_paths':{f:'fixture' for f in RUN.R.FAMILIES}}
    source={'fixture.py':'a'*64};commit='b'*40;register_sha=RUN.R.sha256(config);paths=[]
    for seed in RUN.R.SEEDS:
        for objective in RUN.R.OBJECTIVES:
            leg=incoming/f'leg-{seed}-{objective}';leg.mkdir();models=[]
            for fi,family in enumerate(RUN.R.FAMILIES):
                name=f'{family}-{objective}';folder=leg/name;folder.mkdir()
                (folder/'MODEL.pt').write_bytes(f'UNTRAINED ARTIFACT FIXTURE {family}-{objective}-{seed}'.encode())
                (folder/'INITIAL_MODEL.pt').write_bytes(b'UNTRAINED ARTIFACT FIXTURE')
                checkpoint_sha=RUN.R.sha256(folder/'MODEL.pt')
                receipt={'constructed_fixture':True,'study_id':'LANGACQUIRE097','status':'SEALED_BEFORE_DEV',
                    'family':family,'objective':objective,'seed':seed,'source_commit':commit,'source_hashes':source,
                    'register_sha256':register_sha,'optimizer_steps':2,'actual_steps':2,'actual_examples':8,
                    'actual_dense_compute_tokens':2176,'actual_tokens':1000,'schedule_sha256':f'{seed:064x}',
                    'actual_initial_model_sha256':f'{seed+fi:064x}',
                    'checkpoint_sha256':checkpoint_sha,'replay':{'cold_reload_bitwise_equal':True}}
                RUN.R.write_json(folder/'FIT_RECEIPT.json',receipt);paths.append(folder)
                models.append({'family':family,'objective':objective,'seed':seed,'checkpoint':name+'/MODEL.pt',
                    'checkpoint_sha256':checkpoint_sha,'receipt':name+'/FIT_RECEIPT.json',
                    'receipt_sha256':RUN.R.sha256(folder/'FIT_RECEIPT.json')})
            RUN.R.write_json(leg/'FIT_LEG_SEAL.json',{'schema':'langacquire097.objective-leg-fit-seal.v1',
                'seed':seed,'objective':objective,'source_commit':commit,'register_sha256':register_sha,
                'DEV_predictions_started':False,'models':models})
    return incoming,config,register,source,commit,paths


class RunnerBoundaryTests(unittest.TestCase):
    def test_dense_end_padding_and_next_byte_alignment(self):
        rows=[row(i,a=8+i,q=4,y=4) for i in range(4)]
        x,y,query,valid,cost=T.collate(rows,[0,1,2,3])
        self.assertEqual(x.shape,(4,272))
        for i,r in enumerate(rows):
            raw=r['acquisition_bytes']+r['query_bytes']+r['target_bytes'];n=len(raw)-1
            self.assertEqual(x[i,:n].tolist(),raw[:-1]);self.assertEqual(y[i,:n].tolist(),raw[1:])
            self.assertEqual(int(query[i].sum()),4)
            self.assertEqual(int(query[i].nonzero()[0]),len(r['acquisition_bytes'])+len(r['query_bytes'])-1)
            self.assertTrue(bool(valid[i,:n].all()));self.assertFalse(bool(valid[i,n:].any()))
            self.assertFalse(bool(query[i,n:].any()))
        self.assertEqual(cost['padded_compute_positions'],1088)
        self.assertEqual(cost['actual_observed_inputs']+cost['end_padding_positions'],1088)

    def test_future_target_poison_does_not_change_first_prediction_prefix(self):
        rows=[row(i) for i in range(4)];changed=json.loads(json.dumps(rows))
        for r in changed:r['target_bytes']=[v^31 for v in r['target_bytes']]
        a=T.collate(rows,[0,1,2,3]);b=T.collate(changed,[0,1,2,3])
        for i,r in enumerate(rows):
            boundary=len(r['acquisition_bytes'])+len(r['query_bytes'])
            self.assertTrue(torch.equal(a[0][i,:boundary],b[0][i,:boundary]))
        self.assertFalse(torch.equal(a[1],b[1]))

    def test_sampler_is_seed_deterministic_and_domain_balanced(self):
        rows=[row(i,'lexical') for i in range(128)]+[row(i+128,'arc') for i in range(24)]
        a=T.schedule(rows,1701,12);b=T.schedule(rows,1701,12)
        self.assertEqual(a,b);self.assertNotEqual(a,T.schedule(rows,1702,12))
        for indices in a:self.assertEqual([rows[i]['domain'] for i in indices],['lexical','lexical','arc','arc'])

    def test_disabled_fit_fails_before_optimizer_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            args=type('Args',(),{'output':Path(temp)/'new','config':HERE/'REGISTER.json'})()
            with mock.patch.dict(os.environ,{'GITHUB_ACTIONS':'true'}), \
                 mock.patch.object(T.R,'load_register',return_value={'training':{},'fit_enabled':False}), \
                 mock.patch.object(torch.optim,'AdamW',side_effect=AssertionError('no optimizer')) as optimizer:
                with self.assertRaisesRegex(ValueError,'disabled'):T.train(args)
                optimizer.assert_not_called()

    def test_pinned_zero_update_initialization_restores_exact_weights(self):
        # An untrained source fixture, with no optimizer or model forward.
        model=T.M.make_model('gru',1701);before=T.R.model_digest(model)
        payload={'schema':'langacquire097.initial-state.v1','family':'gru','seed':1701,
                 'optimizer_steps':0,'model':T.R.cpu_tree(model.state_dict()),'model_sha256':before}
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);path=directory/'INIT_gru_1701.pt'
            torch.save(payload,path)
            profile={'initial_state_file':path.name,'initial_state_sha256':T.R.sha256(path),'weights_before':before}
            with torch.no_grad():next(model.parameters()).add_(.125)
            self.assertNotEqual(T.R.model_digest(model),before)
            T.load_initial_state(model,profile,directory,'gru',1701)
            self.assertEqual(T.R.model_digest(model),before)
            with self.assertRaisesRegex(ValueError,'bytes differ'):
                T.load_initial_state(model,{**profile,'initial_state_sha256':'0'*64},directory,'gru',1701)
            with self.assertRaisesRegex(ValueError,'basename'):
                T.load_initial_state(model,{**profile,'initial_state_file':'../INIT_gru_1701.pt'},directory,'gru',1701)
            payload['optimizer_steps']=1;torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,'zero-update initialization identity'):
                T.load_initial_state(model,{**profile,'initial_state_sha256':T.R.sha256(path)},directory,'gru',1701)

    def test_actor_denies_answers_and_gold_dependent_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'actor.json'
            r={'id':'fixture','domain':'lexical','query_bytes':[65],'generation_limit':4}
            p.write_text(json.dumps([r]));self.assertEqual(E.load_actor(p,'query'),[r])
            for bad in ({**r,'target_bytes':[66]},{**r,'acquisition_bytes':[1]}, {**r,'generation_limit':3}):
                p.write_text(json.dumps([bad]))
                with self.assertRaises(ValueError):E.load_actor(p,'query')

    def test_all_forty_seals_are_required_before_dev(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'all five'):RUN.all_fits(temp,{},Path('config'), '0'*40)

    def test_assembly_missing_duplicate_or_changed_model_is_rejected(self):
        # Only the tensor-loader is mocked; matrix, identity, seal and byte checks run.
        for damage in ('missing','duplicate','changed'):
            with self.subTest(damage=damage),tempfile.TemporaryDirectory() as temp:
                incoming,config,register,source,commit,paths=artifact_fixture(Path(temp))
                if damage=='missing':(paths[-1]/'FIT_RECEIPT.json').unlink()
                elif damage=='duplicate':shutil.copyfile(paths[0]/'FIT_RECEIPT.json',paths[-1]/'FIT_RECEIPT.json')
                else:(paths[-1]/'MODEL.pt').write_bytes(b'changed original bytes')
                with mock.patch.object(RUN,'verify_model_checkpoint') as verify:
                    with self.assertRaises(ValueError):RUN.model_artifacts(incoming,register,config,commit,source)
                    verify.assert_not_called()

    def test_assembly_builds_five_complete_seed_seals_without_data_access(self):
        with tempfile.TemporaryDirectory() as temp:
            incoming,config,register,source,commit,paths=artifact_fixture(Path(temp))
            args=type('Args',(),{'fits':incoming,'config':config,'source_commit':commit,'output':Path(temp)/'assembled'})()
            with mock.patch.object(RUN.R,'load_register',return_value=register), \
                 mock.patch.object(RUN.R,'verify_source',return_value=source), \
                 mock.patch.object(RUN.R,'load_local',side_effect=AssertionError('no dataset import')), \
                 mock.patch.object(RUN,'verify_model_checkpoint') as verify:
                RUN.assemble(args)
            self.assertEqual(verify.call_count,40)
            self.assertEqual(len(list(args.output.rglob('FITS_SEAL.json'))),5)
            matrix=RUN.all_fits(args.output,register,config,commit)
            self.assertEqual(len(matrix),40)
            receipt=RUN.R.read_json(args.output/'ASSEMBLY.json')
            self.assertFalse(receipt['datasets_opened']);self.assertEqual(receipt['optimizer_steps'],0)

    def test_objective_pair_initial_weights_must_match_before_assembly(self):
        with tempfile.TemporaryDirectory() as temp:
            incoming,config,register,source,commit,paths=artifact_fixture(Path(temp))
            target=paths[4]/'FIT_RECEIPT.json';receipt=RUN.R.read_json(target)
            receipt['actual_initial_model_sha256']='f'*64;target.write_text(json.dumps(receipt))
            with mock.patch.object(RUN,'verify_model_checkpoint') as verify:
                with self.assertRaisesRegex(ValueError,'identical actual weights'):RUN.model_artifacts(incoming,register,config,commit,source)
                verify.assert_not_called()

    def test_unavailable_donor_and_binding_cases_are_preserved(self):
        actor=[{'id':'lex','domain':'lexical','query_bytes':[65],'generation_limit':4},
               {'id':'arc','domain':'arc','query_bytes':[65],'generation_limit':40}]
        judge={'lex':{'domain':'lexical','target_bytes':[65]*4,'cluster_id':'source','group':'positive','donor_available':True,'binding_shuffle_available':False},
               'arc':{'domain':'arc','target_bytes':list(b'111;'),'cluster_id':'task','group':'unknown_rule','donor_available':False,'binding_shuffle_available':True}}
        available=np.ones((5,2),dtype=bool);available[2,1]=False;available[4,0]=False
        payload={'log_probs':np.full((5,2,256),-math.log(256),dtype=np.float32),
                 'generated_bytes':np.zeros((5,2,40),dtype=np.uint8),'available':available,'lengths':np.array([4,40])}
        metadata={'schema':'langacquire097.predictions.v1','conditions':list(E.CONDITIONS),'case_ids':['lex','arc'],
                  'optimizer_steps':0,'checkpoint':{'family':'tac','objective':'full','seed':1701}}
        main,binding=RUN.prediction_rows(payload,metadata,actor,judge)
        self.assertEqual(len(main),8);self.assertEqual(len(binding),1)
        unavailable=[r for r in main if r['case_id']=='arc' and r['condition']=='donor'][0]
        self.assertEqual(unavailable['status'],'INCONCLUSIVE');self.assertIsNone(unavailable['nll'])
        self.assertEqual(binding[0]['condition'],'binding_shuffle')

    def test_exact_replay_checks_saved_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            a=Path(temp)/'a.npz';b=Path(temp)/'b.npz'
            values={key:np.zeros((2,3),dtype=np.float32) for key in ('log_probs','generated_bytes','lengths','available')}
            np.savez(a,**values,metadata=np.asarray('{}'));np.savez(b,**values,metadata=np.asarray('{}'))
            RUN.replay(a,b)
            values['log_probs'][0,0]=1
            np.savez(b,**values,metadata=np.asarray('{}'))
            with self.assertRaisesRegex(ValueError,'replay differs'):RUN.replay(a,b)

    def test_native_acquisition_roundtrip_and_query_do_not_mutate_state(self):
        torch.set_num_threads(1)
        rows=[{'id':f'fixture-{i}','domain':'lexical','acquisition_bytes':[65+i]*8} for i in range(2)]
        queries=[{'id':r['id'],'domain':'lexical','query_bytes':[66,67,68],'generation_limit':4} for r in rows]
        for family in ('tac','gru','ssm','transformer'):
            with self.subTest(family=family),tempfile.TemporaryDirectory() as temp,torch.inference_mode():
                model=E.M.make_model(family,1701).eval()
                states,aux,cost=E.acquire(model,rows);before=E.R.state_digest(states)
                path=Path(temp)/'state.pt';checkpoint={'family':family}
                E.save_states(path,states,aux,rows,checkpoint,cost,'a'*64,None)
                loaded,loaded_aux,info=E.load_states(path,checkpoint,queries)
                self.assertEqual(E.R.state_digest(loaded),before)
                output,cost=E.predict(model,loaded,loaded_aux,queries,[1,0])
                self.assertEqual(output['generated_bytes'].shape,(5,2,40))
                self.assertTrue(output['available'][:4].all());self.assertFalse(output['available'][4].any())
                self.assertEqual(E.R.state_digest(loaded),before)
                self.assertGreaterEqual(cost['carry']['query_seconds'],0.)

    def test_donor_clock_mismatch_is_rejected(self):
        torch.set_num_threads(1)
        model=E.M.make_model('transformer',1701)
        with torch.inference_mode():
            states,_,_=E.acquire(model,[{'id':'a','domain':'arc','acquisition_bytes':[65]*8},
                                      {'id':'b','domain':'arc','acquisition_bytes':[65]*9}])
        donors=[{'id':'a','donor_id':'b','available':True,'reason':'fixture'},
                {'id':'b','donor_id':'a','available':True,'reason':'fixture'}]
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'donors.json';p.write_text(json.dumps(donors))
            with self.assertRaisesRegex(ValueError,'same-clock'):E.donor_map(p,[{'id':'a','domain':'arc'},{'id':'b','domain':'arc'}],states)


if __name__=='__main__':unittest.main()
