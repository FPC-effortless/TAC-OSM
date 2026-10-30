"""Constructed boundary/security checks, never intelligence evidence."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import subprocess
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'research/mathematical-engineering/langpersist096'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


judge=load('langpersist096_general_judge_test',HERE/'general_judge.py')
try:
    import torch
except ImportError:
    torch=None
worker=load('langpersist096_general_worker_test',HERE/'general_worker.py') if torch is not None else None


def confinement_available():
    return bool(shutil.which('docker' if os.environ.get(judge.BACKEND_ENV,'bubblewrap') == 'docker' else 'bwrap'))


class GeneralJudgeTests(unittest.TestCase):
    def test_grid_and_numeric_outputs_only(self):
        self.assertTrue(judge.judge_case({'domain':'arc','expected':[[1,2],[3,4]]},b'[[1,2],[3,4]]')['correct'])
        self.assertFalse(judge.judge_case({'domain':'arc','expected':[[1]]},b'[[true]]')['correct'])
        self.assertTrue(judge.judge_case({'domain':'gsm8k','answer':'1200'},b'Answer: 1,200\nextra')['correct'])
        self.assertFalse(judge.judge_case({'domain':'gsm8k','answer':'42'},b'It could be 42')['correct'])

    def test_denied_syntax_reflection_and_io(self):
        sources=["import os\ndef f(x): return os.getcwd()", "def f(x): return open('a').read()",
                 "def f(x): return (i for i in [1]).gi_frame.f_back", "def f(x): return '{0.__class__}'.format(x)",
                 "import math\ndef f(x):\n    math.sqrt=lambda x:0\n    return 0", "class X:\n    pass"]
        for source in sources:
            with self.subTest(source=source):
                with self.assertRaises((judge.UnsupportedPython,SyntaxError)):
                    judge._compile_supported(source,'<fixture>')

    def test_no_host_execution_fallback(self):
        with mock.patch.object(judge.shutil,'which',return_value=None), mock.patch.object(judge.subprocess,'Popen',side_effect=AssertionError('no execution')):
            self.assertEqual(judge.execute_program('','', 'f')['status'],'INCONCLUSIVE')

    def test_docker_command_is_pinned_restricted_and_cleans_up(self):
        process=mock.Mock(returncode=0,pid=999999)
        process.communicate.return_value=(json.dumps({'status':'PASS','correct':True,'reason':'public assertions passed'}),'')
        variables={judge.BACKEND_ENV:'docker','DOCKER_HOST':'tcp://denied','DOCKER_CONTEXT':'denied',
                   'DOCKER_TLS_VERIFY':'1','DOCKER_CERT_PATH':'/denied','DOCKER_CONFIG':'/fixture/config'}
        with mock.patch.dict(os.environ,variables), mock.patch.object(judge.shutil,'which',return_value='/usr/bin/docker'), \
             mock.patch.object(judge.subprocess,'Popen',return_value=process) as launch, \
             mock.patch.object(judge.subprocess,'run') as cleanup:
            result=judge.execute_program('def f(x): return x','def check(candidate): pass','f')
        self.assertEqual(result['status'],'PASS')
        command=launch.call_args.args[0]
        self.assertEqual(command[:3],['/usr/bin/docker','--host='+judge.DOCKER_HOST,'run'])
        for flag,value in (('--network','none'),('--user','65534:65534'),('--memory','256m'),
                           ('--cpus','1'),('--pids-limit','16'),('--cap-drop','ALL'),
                           ('--security-opt','no-new-privileges')):
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertIn('--read-only',command); self.assertIn('--pull=never',command)
        self.assertEqual(command[-5:],[judge.DOCKER_IMAGE,'python','-I','/judge.py','--docker-child'])
        self.assertEqual(command.count('--mount'),1)
        self.assertEqual(command[command.index('--mount')+1],f'type=bind,source={judge.__file__},target=/judge.py,readonly')
        environment=launch.call_args.kwargs['env']
        self.assertEqual(environment['DOCKER_CONFIG'],'/fixture/config')
        for key in ('DOCKER_HOST','DOCKER_CONTEXT','DOCKER_TLS_VERIFY','DOCKER_CERT_PATH'):
            self.assertNotIn(key,environment)
        self.assertEqual(cleanup.call_args.args[0][:4],['/usr/bin/docker','--host='+judge.DOCKER_HOST,'rm','--force'])
        self.assertEqual(cleanup.call_args.kwargs['timeout'],3)
        self.assertEqual(cleanup.call_args.args[0][-1],command[command.index('--name')+1])
        with mock.patch.dict(os.environ,{judge.BACKEND_ENV:'unsafe-host'}), \
             mock.patch.object(judge.subprocess,'Popen',side_effect=AssertionError('no fallback')):
            self.assertEqual(judge.execute_program('','','f')['status'],'INCONCLUSIVE')

    def test_docker_child_requires_nonroot_no_new_privileges_and_no_caps(self):
        clean='NoNewPrivs:\t1\nCapEff:\t0000000000000000\nCapPrm:\t0000000000000000\nCapAmb:\t0000000000000000\n'
        with mock.patch.object(judge.os,'getresuid',return_value=(65534,)*3), \
             mock.patch.object(judge.os,'getresgid',return_value=(65534,)*3), \
             mock.patch.object(judge.Path,'read_text',return_value=clean):
            self.assertTrue(judge._docker_privileges_ok())
            with mock.patch.object(judge.Path,'read_text',return_value=clean.replace('NoNewPrivs:\t1','NoNewPrivs:\t0')):
                self.assertFalse(judge._docker_privileges_ok())
            with mock.patch.object(judge.Path,'read_text',return_value=clean.replace('CapEff:\t0000000000000000','CapEff:\t0000000000000001')):
                self.assertFalse(judge._docker_privileges_ok())
        with mock.patch.object(judge.os,'getresuid',return_value=(0,)*3), \
             mock.patch.object(judge,'_compile_supported',side_effect=AssertionError('no candidate execution')):
            self.assertEqual(judge._child({},docker_child=True)['status'],'INCONCLUSIVE')

    def test_docker_timeout_removes_container_even_when_client_is_killed(self):
        process=mock.Mock(pid=999999)
        process.communicate.side_effect=[subprocess.TimeoutExpired('docker',3),('','')]
        with mock.patch.dict(os.environ,{judge.BACKEND_ENV:'docker'}), \
             mock.patch.object(judge.shutil,'which',return_value='/usr/bin/docker'), \
             mock.patch.object(judge.subprocess,'Popen',return_value=process), \
             mock.patch.object(judge.subprocess,'run') as cleanup, \
             mock.patch.object(judge.os,'killpg') as kill:
            result=judge.execute_program('','','f')
        self.assertEqual(result['reason'],'bounded execution timeout')
        self.assertEqual(cleanup.call_count,2)
        kill.assert_called_once_with(process.pid,judge.signal.SIGKILL)

    def test_reference_cache_is_bound_to_selected_backend(self):
        judge._reference_cached.cache_clear()
        with mock.patch.object(judge,'_execute_program',return_value={'status':'PASS','correct':True,'reason':'fixture'}) as execute:
            with mock.patch.dict(os.environ,{judge.BACKEND_ENV:'bubblewrap'}):
                judge._reference('prompt','solution','test','f')
            with mock.patch.dict(os.environ,{judge.BACKEND_ENV:'docker'}):
                judge._reference('prompt','solution','test','f')
        self.assertEqual(execute.call_count,2)
        judge._reference_cached.cache_clear()

    @unittest.skipUnless(confinement_available(),'OS confinement unavailable')
    def test_supported_code_and_independent_checker_namespaces(self):
        program='all=lambda x:True\ndef f(x):\n    return 0\n'
        test='def check(candidate):\n    assert all([candidate(1)==2])\n'
        result=judge.execute_program(program,test,'f')
        if result['status']=='INCONCLUSIVE':
            self.skipTest('bubblewrap unavailable in this executor')
        self.assertEqual(result['status'],'FAIL')
        good=judge.execute_program('def f(x):\n    return x+1\n','def check(candidate):\n    assert candidate(2)==3\n','f')
        self.assertEqual(good['status'],'PASS')

    @unittest.skipUnless(confinement_available(),'OS confinement unavailable')
    def test_math_mutation_cannot_tamper_with_checker(self):
        result=judge.execute_program('import math\ndef f(x):\n    math.sqrt=lambda x:0\n    return 0\n',
            'import math\ndef check(candidate):\n    assert candidate(9)==math.sqrt(9)\n','f')
        if result['status']=='INCONCLUSIVE':
            self.skipTest('bubblewrap unavailable in this executor')
        self.assertNotEqual(result['status'],'PASS')

    def test_reference_failure_is_inconclusive_not_model_failure(self):
        row={'domain':'humaneval','prompt':'def f(x):\n','canonical_solution':'    return x\n',
             'test':'def check(candidate):\n    assert candidate(2)==3\n','entry_point':'f'}
        with mock.patch.object(judge,'_reference',return_value={'status':'FAIL','correct':False}):
            self.assertEqual(judge.judge_case(row,b'    return x+1\n')['status'],'INCONCLUSIVE')

    @unittest.skipUnless(confinement_available(),'OS confinement unavailable')
    def test_fixed_completion_stops_preserve_finished_entry_function(self):
        row={'domain':'humaneval','prompt':'def f(x):\n','canonical_solution':'    return x+1\n',
             'test':'def check(candidate):\n    assert candidate(2)==3\n','entry_point':'f'}
        completion=b'    return x+1\n\ndef truncated('
        result=judge.judge_case(row,completion)
        if result['status']=='INCONCLUSIVE':
            self.skipTest('bubblewrap unavailable in this executor')
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['raw_completion_bytes'],len(completion))
        self.assertLess(result['judged_completion_bytes'],len(completion))
        self.assertEqual(judge.normalize_completion('    return 1\n# next'), '    return 1')


@unittest.skipIf(torch is None,'optional torch unavailable')
class GeneralWorkerTests(unittest.TestCase):
    def test_actor_only_schemas_and_generation_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'actor.json'
            good={'task_id':'fixture','domain':'arc','query_bytes':[65,66],'generation_limit':4}
            path.write_text(json.dumps([good]))
            self.assertEqual(worker.load_cases(path,'query')[0]['generation_limit'],4)
            for changed in ({**good,'expected':[[1]]},{**good,'acquisition_bytes':[1]},
                            {**good,'generation_limit':129},{**good,'query_bytes':[256]}):
                path.write_text(json.dumps([changed]))
                with self.assertRaises(ValueError):
                    worker.load_cases(path,'query')

    def test_variable_clock_state_roundtrip_and_no_query_mutation(self):
        torch.set_num_threads(1)
        for arm in ('tac','gru','ssm'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as folder:
                torch.manual_seed(96002)
                model=worker.ADAPTERS.TRAIN.make(arm).eval()
                model._checkpoint_sha256_096='0'*64
                adapter=worker.ADAPTERS.FrozenAdapter096(model,arm)
                acquisition=[{'task_id':f'fixture-{i}','domain':'arc','acquisition_bytes':[65+i]*length}
                             for i,length in enumerate((8,12))]
                queries=[{'task_id':r['task_id'],'domain':'arc','query_bytes':[66,67,68], 'generation_limit':4} for r in acquisition]
                states=worker.acquire_cases(adapter,acquisition)
                self.assertEqual([s.clock for s in states],[8,12])
                path=Path(folder)/'state.pt'
                worker.save_states(path,adapter,states,acquisition,{'setup_seconds':0.},'a'*64)
                loaded,meta=worker.load_states(path,adapter,queries)
                self.assertEqual(worker._digest(states),worker._digest(loaded))
                before=worker._digest(loaded)
                output,cost=worker.query_cases(adapter,loaded,queries)
                self.assertEqual(output.shape,(2,2,128))
                self.assertEqual(worker._digest(loaded),before)
                self.assertGreaterEqual(cost['generation_seconds'],0.)
                self.assertAlmostEqual(cost['generation_seconds'],sum(v['generation_seconds'] for v in cost['per_condition_costs'].values()))
                bad=[{**queries[0],'task_id':'wrong'},queries[1]]
                with self.assertRaises(ValueError):
                    worker.load_states(path,adapter,bad)


if __name__=='__main__':
    unittest.main()
