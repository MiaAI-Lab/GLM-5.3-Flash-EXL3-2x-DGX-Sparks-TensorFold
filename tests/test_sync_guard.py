#!/usr/bin/env python3
"""CPU control-flow tests only. No CUDA/allocator safety assertion.

The historical interleaving and assertions are preserved; only source selection
and evidence output are made portable. An explicit fixture adapter supplies the
new initialized fields/helpers. Run with IDLE_TRIM_SOURCE set to patched multi.py.
All thread waits have bounded FAILURE timeouts; a stuck thread never passes.
"""
import ast, contextlib, copy, importlib.util, io, os, pathlib, sys, threading, types, unittest
ROOT = pathlib.Path(__file__).resolve().parent
# Python stdlib only. Point to a recipe-patched source file, not site-packages.
# IDLE_TRIM_SOURCE=path/to/multi.py python3 tests/test_sync_guard.py
# IDLE_TRIM_SOURCE=path/to/multi.py python3 tests/test_sync_guard.py historical candidate
# For RED, select the original Patch 0130 source and use historical upstream.
# CPU control-flow only: CUDA, transport and clock are explicit doubles.


def methods(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiDecoder')
    funcs = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef)]
    for n in funcs:
        n.decorator_list = []
        n.returns = None
        for a in n.args.posonlyargs+n.args.args+n.args.kwonlyargs:
            a.annotation = None
    return compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), str(path), 'exec')


def namespace(source, trim=lambda: (10, 3), clock=lambda: 0):
    ns = dict(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(set_device=lambda _: None)),
              time=types.SimpleNamespace(monotonic=clock), trim_memory=trim,
              seal=lambda out, sent: out, unseal=lambda values, *args: values, SEAL_MOD=100,
              _watch=lambda *args: None, _unwatch=lambda *args: None, MAX_WINDOW=128)
    exec(methods(source), ns)
    return ns


def fixture(ns, rank=0):
    d = types.SimpleNamespace(idle_trim_s=120., idle=True, idle_lock=threading.Lock(), idle_epoch=0,
        trim_busy=False, rank=rank, trims=0, trimmed_bytes=0, device_index=0,
        trim_stop=threading.Event(), sent=0, outbox=[1], fill_budget=0, received=0, watchdog=0,
        tune=types.SimpleNamespace(async_msg=False), g=types.SimpleNamespace(_ring=lambda: None, _share=lambda _: []))
    for name in ('_set_idle', '_trim_work'):
        if name in ns:
            setattr(d, name, types.MethodType(ns[name], d))
    return d


def historical(source, label):
    spec = importlib.util.spec_from_file_location('historical_regression', ROOT/'test_busy_guard.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ns = namespace(source)
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiDecoder')
    mod.code = compile(ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef)
        and n.name in ('_trim_idle', '_flush')], type_ignores=[]), str(source), 'exec')
    real_simple = types.SimpleNamespace
    def adapted(**kw):
        d = real_simple(**kw)
        if 'idle_trim_s' in kw:
            d.idle_lock=threading.Lock(); d.idle_epoch=0; d.trim_busy=False
            for name in ('_set_idle', '_trim_work'):
                if name in ns: setattr(d, name, types.MethodType(ns[name], d))
        return d
    mod.types = real_simple(SimpleNamespace=adapted)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(mod))
    return result.wasSuccessful()

SOURCE = pathlib.Path(os.environ['IDLE_TRIM_SOURCE'])

class SyncTests(unittest.TestCase):
    def test_once_and_fast_new_phase_all_ranks(self):
        for rank in range(3):
            calls=[]; now=[0]; actions=iter([0,121,400,'cycle',401,522,800])
            ns=namespace(SOURCE,lambda: (calls.append(d.idle) or (10,3)),lambda:now[0]);d=fixture(ns,rank)
            def wait(_):
                try: a=next(actions)
                except StopIteration:return True
                if a=='cycle':
                    d._trim_work(True);d._set_idle(False);d._set_idle(True);d._trim_work(False)
                else:now[0]=a
                return False
            d.trim_stop=types.SimpleNamespace(wait=wait)
            with contextlib.redirect_stderr(io.StringIO()):ns['_trim_idle'](d)
            self.assertEqual(calls,[True,True]);self.assertEqual(d.trims,2);self.assertEqual(d.trimmed_bytes,14)

    def test_busy_transition_waits_for_trim_and_finishes_all_ranks(self):
        for rank in range(3):
            for method,args in [('_set_idle',(False,)),('_trim_work',(True,))]:
                entered=threading.Event();resume=threading.Event();attempt=threading.Event();finished=threading.Event();errors=[]
                ticks=iter([0,121]);waits=iter([False,False,True])
                def trim():
                    entered.set()
                    if not resume.wait(3):raise RuntimeError('test failed to release trim')
                    self.assertTrue(d.idle);self.assertFalse(d.trim_busy)
                    return 10,3
                ns=namespace(SOURCE,trim,lambda:next(ticks));d=fixture(ns,rank)
                d.trim_stop=types.SimpleNamespace(wait=lambda _:next(waits))
                def run():
                    try:ns['_trim_idle'](d)
                    except BaseException as e:errors.append(repr(e))
                def busy():
                    attempt.set()
                    try:getattr(d,method)(*args);finished.set()
                    except BaseException as e:errors.append(repr(e))
                t=threading.Thread(target=run,daemon=True);b=threading.Thread(target=busy,daemon=True)
                with contextlib.redirect_stderr(io.StringIO()):
                    t.start()
                    try:
                        self.assertTrue(entered.wait(3));b.start();self.assertTrue(attempt.wait(3))
                        self.assertFalse(finished.wait(.05),'busy transition overlapped trim')
                    finally:resume.set();t.join(3);b.join(3)
                self.assertFalse(t.is_alive(),'DEADLOCK trim');self.assertFalse(b.is_alive(),'DEADLOCK busy')
                self.assertTrue(finished.is_set());self.assertEqual(errors,[])
                self.assertTrue(d.idle_lock.acquire(timeout=.5));d.idle_lock.release()

    def test_trim_exception_releases_lock_once_per_phase(self):
        now=[0];actions=iter([0,121,300,'cycle',301,422]);calls=[]
        def trim():calls.append(1);raise ValueError('explicit CPU injected failure')
        ns=namespace(SOURCE,trim,lambda:now[0]);d=fixture(ns)
        def wait(_):
            try:a=next(actions)
            except StopIteration:return True
            if a=='cycle':d._set_idle(False);d._set_idle(True)
            else:now[0]=a
            return False
        d.trim_stop=types.SimpleNamespace(wait=wait)
        with contextlib.redirect_stderr(io.StringIO()):ns['_trim_idle'](d)
        self.assertEqual(len(calls),2);self.assertEqual(d.trims,0)
        self.assertTrue(d.idle_lock.acquire(timeout=.5));d.idle_lock.release()

    def test_follow_bell_wait_unlocked_work_and_pump_busy_even_after_idle(self):
        for rank in (1,2):
            ns=namespace(SOURCE);d=fixture(ns,rank);seen=[]
            def unlocked(tag):
                self.assertTrue(d.idle_lock.acquire(timeout=.5),tag+' holds trim lock')
                d.idle_lock.release();seen.append(tag)
            def bell():unlocked('bell');self.assertFalse(d.trim_busy)
            def share(_):unlocked('share');self.assertTrue(d.trim_busy);self.assertFalse(d.idle);return []
            def apply(_):d._set_idle(True)
            def pump():unlocked('pump');self.assertTrue(d.trim_busy)
            d.g=types.SimpleNamespace(_await_bell=bell,_share=share);d.apply=apply;d._pump=pump
            ns['follow'](d,once=True)
            self.assertEqual(seen,['bell','share','pump']);self.assertFalse(d.trim_busy);self.assertTrue(d.idle)
            d.g._share=lambda _:(_ for _ in ()).throw(ValueError('transport failure'))
            with self.assertRaises(ValueError):ns['follow'](d,once=True)
            self.assertTrue(d.trim_stop.is_set());self.assertFalse(d.trim_busy)
            self.assertTrue(d.idle_lock.acquire(timeout=.5));d.idle_lock.release()

    def test_scheduler_queue_wait_unlocked_admit_and_finish_busy_exception_release(self):
        tree=ast.parse(SOURCE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GlmScheduler')
        f=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_iteration'))
        ns=namespace(SOURCE);exec(compile(ast.Module(body=[f],type_ignores=[]),str(SOURCE),'exec'),ns)
        d=fixture(ns);d.live=lambda:0
        seen=[]
        def work(tag):
            self.assertTrue(d.trim_busy,tag);self.assertTrue(d.idle_lock.acquire(timeout=.5));d.idle_lock.release();seen.append(tag)
        def get():
            self.assertFalse(d.trim_busy);self.assertTrue(d.idle_lock.acquire(timeout=.5));d.idle_lock.release();return object()
        s=types.SimpleNamespace(decoder=d,held=None,loading=[],ready=[],prefix=[],gather_s=0,
            waiting=types.SimpleNamespace(get=get),_cancel_waiting=lambda:None,_yield=lambda:None,
            _requeue=lambda:work('requeue'),_admit=lambda *a:(work('admit') or []))
        d.round=lambda:(work('round') or [])
        d.finish=lambda _:(d._set_idle(True),work('finish'))
        ns['_iteration'](s)
        self.assertEqual(seen,['admit','round','requeue','finish']);self.assertFalse(d.trim_busy)
        s._admit=lambda *a:(_ for _ in ()).throw(ValueError('admission failure'))
        with self.assertRaises(ValueError):ns['_iteration'](s)
        self.assertFalse(d.trim_busy);self.assertTrue(d.idle_lock.acquire(timeout=.5));d.idle_lock.release()

    def test_protocol_write_inventory(self):
        tree=ast.parse(SOURCE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiDecoder')
        writes=[]
        for f in cls.body:
            if isinstance(f,ast.FunctionDef):
                for n in ast.walk(f):
                    if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store) and isinstance(n.value,ast.Name) and n.value.id=='self' and n.attr=='idle':writes.append(f.name)
        self.assertEqual(sorted(writes),['__init__','_set_idle'])

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='historical':
        label=sys.argv[2];src=SOURCE
        sys.exit(0 if historical(src,label) else 1)
    unittest.main(verbosity=2)
