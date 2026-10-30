"""Bounded public-development task judges, not a general intelligence certificate.

HumanEval code runs only in an isolated bubblewrap process with a supported
Python subset. Missing confinement or unsupported public reference code stays
INCONCLUSIVE. There is no unrestricted host-exec fallback.
"""
from __future__ import annotations

import argparse
import ast
from decimal import Decimal, InvalidOperation
import functools
import json
import math
import os
from pathlib import Path
import re
import random
import shutil
import signal
import subprocess
import sys


BUILTINS = ("abs all any bool dict divmod enumerate filter float frozenset int isinstance "
            "issubclass iter len list map max min next ord chr pow range repr reversed round "
            "set slice sorted str sum tuple zip ValueError TypeError IndexError KeyError "
            "ZeroDivisionError StopIteration Exception").split()
MATH_NAMES = {n for n in dir(math) if not n.startswith('_')}
RE_NAMES = {'split','search','findall','fullmatch','match','sub','subn','escape','IGNORECASE','MULTILINE','DOTALL','ASCII'}
RANDOM_NAMES = {'randint','randrange','random','choice','choices','sample','shuffle','uniform'}
MODULE_NAMES = {'math':MATH_NAMES,'re':RE_NAMES,'random':RANDOM_NAMES}
METHODS = set(("append extend insert pop remove clear copy count index sort reverse keys "
               "values items get setdefault update popitem add discard union intersection "
               "difference symmetric_difference issubset issuperset isdisjoint lower upper "
               "strip lstrip rstrip split rsplit splitlines join startswith endswith replace "
               "find rfind rindex isdigit isalpha isalnum isspace islower isupper istitle "
               "capitalize title swapcase casefold center ljust rjust zfill partition "
               "rpartition is_integer bit_length").split())
NODES = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
         ast.AnnAssign, ast.AugAssign, ast.Expr, ast.If, ast.For, ast.While, ast.Break,
         ast.Continue, ast.Pass, ast.Assert, ast.Raise, ast.Try, ast.ExceptHandler,
         ast.Import, ast.ImportFrom, ast.alias, ast.List, ast.Tuple, ast.Set, ast.Dict,
         ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
         ast.Lambda, ast.Call, ast.keyword, ast.Name, ast.Load, ast.Store, ast.Constant,
         ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Subscript,
         ast.Slice, ast.Attribute, ast.Starred, ast.Delete, ast.operator, ast.unaryop,
         ast.boolop, ast.cmpop, ast.expr_context, ast.JoinedStr, ast.FormattedValue)


class UnsupportedPython(ValueError):
    pass


class _MathView:
    __slots__ = ()
    def __getattr__(self, name):
        if name not in MATH_NAMES:
            raise AttributeError(name)
        return getattr(math, name)
    def __setattr__(self, name, value):
        raise TypeError('immutable math view')


class _ModuleView:
    __slots__ = ('_module','_names')
    def __init__(self, name):
        object.__setattr__(self,'_module',re if name == 're' else random.Random(96002))
        object.__setattr__(self,'_names',MODULE_NAMES[name])
    def __getattr__(self, name):
        if name not in object.__getattribute__(self,'_names'):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self,'_module'),name)
    def __setattr__(self,name,value):
        raise TypeError('immutable module view')


def _imports(descriptors):
    modules={'math':_MathView(),'re':_ModuleView('re'),'random':_ModuleView('random')}
    return {alias:modules[module] if attribute is None else getattr(modules[module],attribute)
            for alias,(module,attribute) in descriptors.items()}


def _compile_supported(source, filename):
    """Positive syntax/attribute allowlist; candidate/test namespaces are separate."""
    if not isinstance(source, str) or len(source.encode()) > 65536:
        raise UnsupportedPython('source size')
    tree = ast.parse(source, filename=filename)
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    parameters = {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    imported = {}
    for node in ast.walk(tree):
        if not isinstance(node, NODES):
            raise UnsupportedPython(type(node).__name__)
        if isinstance(node, (ast.Name, ast.arg, ast.FunctionDef)):
            name = node.id if isinstance(node, ast.Name) else node.arg if isinstance(node, ast.arg) else node.name
            if name.startswith('__'):
                raise UnsupportedPython('private name')
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise UnsupportedPython('decorators')
        if isinstance(node, ast.Attribute):
            if not isinstance(node.ctx, ast.Load) or node.attr not in METHODS | MATH_NAMES | RE_NAMES | RANDOM_NAMES:
                raise UnsupportedPython('attribute')
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in MODULE_NAMES or (alias.asname and alias.asname.startswith('_')):
                    raise UnsupportedPython('import')
                imported[alias.asname or alias.name] = (alias.name,None)
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in (*MODULE_NAMES,'typing'):
                raise UnsupportedPython('import')
            for alias in node.names:
                if alias.name == '*' or (alias.asname and alias.asname.startswith('_')):
                    raise UnsupportedPython('import alias')
                if node.module in MODULE_NAMES:
                    if alias.name not in MODULE_NAMES[node.module]:
                        raise UnsupportedPython('module name')
                    imported[alias.asname or alias.name] = (node.module,alias.name)
                elif alias.name not in ('List', 'Dict', 'Tuple', 'Set', 'Optional', 'Any', 'Union', 'Iterable'):
                    raise UnsupportedPython('typing name')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in set(BUILTINS) | functions | parameters | set(imported) | {'candidate'}:
                    # Safe aliases may be imported inside a function.
                    if not any(isinstance(n, ast.ImportFrom) and n.module in MODULE_NAMES and
                               any((a.asname or a.name) == node.func.id for a in n.names) for n in ast.walk(tree)):
                        raise UnsupportedPython('call')
            elif not isinstance(node.func, ast.Attribute):
                raise UnsupportedPython('dynamic call')

    class StripAnnotations(ast.NodeTransformer):
        def visit_Import(self, node):
            return None
        def visit_ImportFrom(self, node):
            return None
        def visit_arg(self, node):
            node.annotation = None
            return node
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            node.returns = None
            return node
        def visit_AnnAssign(self, node):
            return ast.copy_location(ast.Assign(targets=[node.target], value=node.value or ast.Constant(None)), node)
    tree = ast.fix_missing_locations(StripAnnotations().visit(tree))
    return compile(tree, filename, 'exec'), imported


def _child(payload):
    """Runs inside an empty-workspace/network namespace; emits fixed JSON only."""
    import builtins
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
    try:
        candidate_code, ci = _compile_supported(payload['program'], '<candidate>')
        test_code, ti = _compile_supported(payload['test'], '<checker>')
    except (UnsupportedPython, SyntaxError, ValueError, MemoryError) as error:
        return {'status':'UNSUPPORTED' if isinstance(error, UnsupportedPython) else 'FAIL',
                'correct':False, 'reason':type(error).__name__}
    safe = {name:getattr(builtins,name) for name in BUILTINS}
    candidate_env = {'__builtins__':safe, 'math':_MathView(), **_imports(ci)}
    test_env = {'__builtins__':safe.copy(), 'math':_MathView(), **_imports(ti)}
    allowed = {id(candidate_code),id(test_code)}
    def audit(event, args):
        if event == 'exec' and args and id(args[0]) in allowed:
            return
        if event in ('open','compile','exec') or event.startswith(('os.','socket.','subprocess.','ctypes.','sys._getframe','sys.settrace','sys.setprofile')):
            raise RuntimeError('denied runtime event')
    sys.addaudithook(audit)
    try:
        exec(candidate_code,candidate_env)
        candidate = candidate_env.get(payload['entry_point'])
        if not callable(candidate):
            return {'status':'FAIL','correct':False,'reason':'entry point unavailable'}
        exec(test_code,test_env)
        check = test_env.get('check')
        if not callable(check):
            return {'status':'UNSUPPORTED','correct':False,'reason':'checker unavailable'}
        check(candidate)
        return {'status':'PASS','correct':True,'reason':'public assertions passed'}
    except BaseException as error:
        return {'status':'FAIL','correct':False,'reason':type(error).__name__}


def execute_program(program, test, entry_point, timeout=3):
    """No host-exec fallback: unavailable OS confinement is INCONCLUSIVE."""
    bwrap = shutil.which('bwrap')
    if not bwrap or not Path('/usr/bin/python3').is_file():
        return {'status':'INCONCLUSIVE','correct':None,'reason':'bubblewrap unavailable'}
    command = [bwrap,'--unshare-all','--die-with-parent','--new-session','--clearenv',
               '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib',
               '--ro-bind','/lib64','/lib64','--ro-bind',str(Path(__file__).resolve()),'/judge.py',
               '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/tmp',
               '/usr/bin/python3','-I','/judge.py','--child']
    raw = json.dumps({'program':program,'test':test,'entry_point':entry_point})
    if len(raw.encode()) > 160000:
        return {'status':'INCONCLUSIVE','correct':None,'reason':'judge payload too large'}
    try:
        proc = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                text=True,close_fds=True,start_new_session=True,env={'PATH':'/usr/bin:/bin'})
    except OSError:
        return {'status':'INCONCLUSIVE','correct':None,'reason':'confinement launch unavailable'}
    try:
        out, error = proc.communicate(raw,timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        return {'status':'FAIL','correct':False,'reason':'bounded execution timeout'}
    if proc.returncode != 0:
        return {'status':'INCONCLUSIVE' if 'bwrap:' in error else 'FAIL', 'correct':None if 'bwrap:' in error else False,
                'reason':'confinement unavailable' if 'bwrap:' in error else 'bounded child terminated'}
    if len(out.encode()) > 4096:
        return {'status':'FAIL','correct':False,'reason':'judge output limit'}
    try:
        result = json.loads(out)
        if set(result) != {'status','correct','reason'} or result['status'] not in ('PASS','FAIL','UNSUPPORTED'):
            raise ValueError('schema')
    except (ValueError,TypeError):
        return {'status':'INCONCLUSIVE','correct':None,'reason':'judge output unavailable'}
    return {**result,'confinement':'bubblewrap_isolated_namespace_restricted_python'}


@functools.lru_cache(maxsize=128)
def _reference(prompt, canonical_solution, test, entry_point):
    return execute_program(prompt+canonical_solution,test,entry_point)


def _validate_reference(record):
    """Preflight public references before any model output is judged."""
    if record['domain'] == 'humaneval':
        return _reference(record['prompt'],record['canonical_solution'],record['test'],record['entry_point'])
    if record['domain'] == 'arc':
        grid=record['expected']
        ok=isinstance(grid,list) and 1 <= len(grid) <= 6 and all(isinstance(r,list) and len(r) == len(grid[0]) and 1 <= len(r) <= 6 and all(type(v) is int and 0 <= v <= 9 for v in r) for r in grid)
    elif record['domain'] == 'gsm8k':
        try:
            ok=Decimal(str(record['answer']).replace(',','')).is_finite()
        except InvalidOperation:
            ok=False
    else:
        ok=False
    return {'status':'PASS' if ok else 'INCONCLUSIVE','correct':None,'reason':'public gold structure validation only'}


def validate_reference(record):
    return _validate_reference(record)


def _text(generated):
    if isinstance(generated,str):
        return generated
    if hasattr(generated,'tobytes'):
        generated = generated.tobytes()
    return bytes(generated).decode('utf-8',errors='replace')


def normalize_completion(text):
    """Fixed code-completion delimiters, independent of gold and generated truth."""
    positions=[text.find(marker) for marker in ('\ndef ','\nclass ','\n#','\nif ','\nprint')]
    positions=[p for p in positions if p >= 0]
    return text[:min(positions)] if positions else text


def judge_case(record, generated, validate_reference=True):
    """Candidate correctness is separate from reference/harness validity."""
    domain, text = record['domain'], _text(generated)
    if domain == 'arc':
        expected = record['expected']
        grid = None
        for position,char in enumerate(text):
            if char == '[':
                try:
                    grid,_ = json.JSONDecoder().raw_decode(text[position:])
                    break
                except ValueError:
                    pass
        valid = isinstance(grid,list) and 1 <= len(grid) <= 6 and all(isinstance(row,list) and 1 <= len(row) <= 6 and all(type(x) is int and 0 <= x <= 9 for x in row) for row in grid)
        return {'status':'PASS' if valid and grid == expected else 'FAIL','correct':bool(valid and grid == expected),
                'reason':'exact grid' if valid else 'invalid grid output'}
    if domain == 'gsm8k':
        line = next((line.strip() for line in text.splitlines() if line.strip()),'')
        line = re.sub(r'^(?:Answer:|####)\s*','',line,flags=re.I).replace(',','')
        valid = re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',line)
        try:
            correct = bool(valid and Decimal(line).is_finite() and Decimal(line) == Decimal(str(record['answer']).replace(',','')))
        except InvalidOperation:
            correct = False
        return {'status':'PASS' if correct else 'FAIL','correct':correct,'reason':'numeric answer' if valid else 'invalid numeric output'}
    if domain != 'humaneval':
        return {'status':'INCONCLUSIVE','correct':None,'reason':'unsupported domain'}
    completion=normalize_completion(text)
    raw_length=len(generated.encode()) if isinstance(generated,str) else len(generated.tobytes()) if hasattr(generated,'tobytes') else len(bytes(generated))
    preprocessing={'raw_completion_bytes':raw_length,'judged_completion_bytes':len(completion.encode()),
                   'completion_preprocessing':'fixed unindented newline def/class/comment/if/print delimiters; no gold-dependent stop'}
    if validate_reference:
        reference = _validate_reference(record)
        if reference['status'] != 'PASS':
            return {'status':'INCONCLUSIVE','correct':None,'reason':'public reference unsupported or failed', 'reference':reference,**preprocessing}
    result = execute_program(record['prompt']+completion,record['test'],record['entry_point'])
    if result['status'] == 'UNSUPPORTED':
        result = {**result,'status':'FAIL','correct':False,'reason':'generated code outside supported subset'}
    return {**result,**preprocessing}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    if not args.child:
        parser.error('Use the trusted judge_case API')
    raw=sys.stdin.read(160001)
    try:
        result=_child(json.loads(raw)) if len(raw)<=160000 else {'status':'FAIL','correct':False,'reason':'payload limit'}
    except BaseException as error:
        result={'status':'FAIL','correct':False,'reason':type(error).__name__}
    print(json.dumps(result,separators=(',',':')))


if __name__=='__main__':
    main()
