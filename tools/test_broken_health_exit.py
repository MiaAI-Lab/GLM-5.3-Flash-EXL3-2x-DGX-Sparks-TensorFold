"""CPU-only patch 0118 broken latch, health/metrics and process-exit checks; no torch or GPU.

python3 -B tools/test_broken_health_exit.py --source-root /path/to/patched/src
Extracts the real decoder's control methods with AST, leaving GPU imports/decorators out.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def decoder_type(root):
    tree = ast.parse((Path(root) / 'tensorfold/families/glm5_next/cuda/multi.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiDecoder')
    # Audit the full class too: upstream must not add a latch that bypasses the exit helper.
    writes = []
    calls = []
    for method in cls.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and node.attr == 'broken':
                writes.append(method.name)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == '_break':
                calls.append(method.name)
    assert sorted(writes) == ['__init__', '_break'], writes
    assert sorted(calls) == ['admit', 'complete_load', 'drop', 'round'], calls
    names = {'_break', '_check', '_pump', 'round', 'drop', 'complete_load'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    for n in methods:
        n.decorator_list = []
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'exit_on_broken']
    nodes += [ast.ClassDef(name='Decoder', bases=[], keywords=[], body=methods, decorator_list=[])]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    ns = {'os': os, 'time': time, '_watch': lambda *a: None, 'LOADED': 1}
    exec(compile(module, str(root), 'exec'), ns)
    return ns['Decoder'], ns['exit_on_broken']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--child', action='store_true')
    args = ap.parse_args()
    Decoder, setting = decoder_type(args.source_root)
    d = Decoder()
    d.broken = None
    d.exit_on_broken = args.child
    if args.child:
        d._break(RuntimeError('synthetic GPU failure'))
        raise AssertionError('process did not exit')
    for value, expected in (('', False), ('0', False), ('1', True), (' 1 ', True)):
        assert setting(value) is expected
    try:
        setting('yes')
    except ValueError:
        pass
    else:
        raise AssertionError('invalid setting accepted')
    os.environ.pop('TF_GLM_EXIT_ON_BROKEN', None)
    os.environ['TF_GLM_MULTI_WATCHDOG_EXIT'] = '1'
    assert setting() is False, 'stall exit must not enable broken exit'
    cause = RuntimeError('synthetic GPU failure')
    d._break(cause)
    d._break(ValueError('secondary failure'))
    assert d.broken is cause
    try:
        d._check()
    except RuntimeError as exc:
        assert exc.__cause__ is cause and 'out of step' in str(exc)
    else:
        raise AssertionError('broken decoder accepted work')
    def fail(*a):
        raise cause
    for path in ('round', 'complete_load', 'drop'):
        d.broken = None
        d.watchdog = 0
        d.disk = None
        d._iterate = fail
        d._finish_load = fail
        d._emit = lambda *a: None
        d._flush = lambda: None
        d.lanes = {}
        d.partial = None
        try:
            getattr(d, path)(SimpleNamespace(lid=1, done=False, lj=None)) if path == 'complete_load' else getattr(d, path)()
        except RuntimeError:
            pass
        assert d.broken is not None, path
    sys.path.insert(0, args.source_root)
    from tensorfold.cuda.health import Health
    from tensorfold.server import metrics
    d.streams = {}
    d.filling = []
    d.health = fail  # a failing optional counters read cannot hide the latch
    app = SimpleNamespace(engine=SimpleNamespace(scheduler=SimpleNamespace(decoder=d, max_streams=4)))
    h = Health()
    d.broken = None
    assert h.snapshot(app)['ok'] is True
    assert 'tensorfold_health:ok 1' in metrics.render(app)
    d.broken = cause
    body = h.snapshot(app)
    assert body['ok'] is False and body['broken'] == 'RuntimeError: synthetic GPU failure'
    assert 'tensorfold_health:ok 0' in metrics.render(app)
    d.broken = RuntimeError('x' * 5000)        # an unauthenticated route: a long cause is cut, its type kept
    assert h.snapshot(app)['broken'] == 'RuntimeError: ' + 'x' * 200, len(h.snapshot(app)['broken'])
    d.broken = cause
    p = subprocess.run([sys.executable, '-B', __file__, '--source-root', args.source_root, '--child'],
                       capture_output=True, text=True, timeout=10)
    assert p.returncode == 1, p
    assert 'exiting with code 1' in p.stderr and 'synthetic GPU failure' in p.stderr, p.stderr
    print('OK: settings, first-cause latch, request refusal, round/load/drop, health, metrics, non-zero process exit')


if __name__ == '__main__':
    main()
