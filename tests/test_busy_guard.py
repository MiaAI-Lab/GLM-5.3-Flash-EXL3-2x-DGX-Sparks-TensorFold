#!/usr/bin/env python3
"""CPU-only control-flow regression against exact patched methods.

CUDA/transport/clock are explicit test doubles. This is NOT GPU evidence.
No installed modules imported and no production process modified.
"""
import ast,contextlib,io,json,os,pathlib,sys,threading,types,unittest
ROOT=pathlib.Path(__file__).resolve().parent
# Set IDLE_TRIM_SOURCE to the recipe-patched multi.py; no TensorFold/torch import.
SOURCE=pathlib.Path(os.environ['IDLE_TRIM_SOURCE'])
tree=ast.parse(SOURCE.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiDecoder')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_trim_idle','_flush')]
code=compile(ast.Module(body=methods,type_ignores=[]),str(SOURCE),'exec')

class GuardTests(unittest.TestCase):
 def test_real_flush_can_make_busy_after_idle_check_before_trim(self):
  reached=threading.Event();resume=threading.Event();calls=[];ticks=iter([0.0,121.0]); waits=iter([False,False,True])
  d=types.SimpleNamespace(idle_trim_s=120.0,idle=True,rank=0,trims=0,trimmed_bytes=0,device_index=0,
   trim_stop=types.SimpleNamespace(wait=lambda poll:next(waits)),sent=0,outbox=[1],fill_budget=0,
   tune=types.SimpleNamespace(async_msg=False),g=types.SimpleNamespace(_ring=lambda:None,_share=lambda msg:None))
  clock_calls=0
  def monotonic():
   nonlocal clock_calls
   clock_calls+=1
   if clock_calls==2:
    reached.set()
    if not resume.wait(5):raise RuntimeError('main thread failed to resume')
   return next(ticks)
  def trim():
   calls.append({'idle_at_trim':d.idle,'kind':'CPU test double, not CUDA'})
   return 0,0
  ns={'torch':types.SimpleNamespace(cuda=types.SimpleNamespace(set_device=lambda x:None)),
      'time':types.SimpleNamespace(monotonic=monotonic),'trim_memory':trim,'seal':lambda out,sent:out,'SEAL_MOD':100}
  exec(code,ns)
  errors=[]
  def worker():
   try:ns['_trim_idle'](d)
   except BaseException as e:errors.append(repr(e))
  with contextlib.redirect_stderr(io.StringIO()):
   t=threading.Thread(target=worker);t.start()
   self.assertTrue(reached.wait(5),'did not reach post-idle-check scheduling point')
   # Execute the exact production transition while trim thread is descheduled.
   ns['_flush'](d)
   self.assertFalse(d.idle)
   resume.set();t.join(5)
  self.assertFalse(t.is_alive());self.assertEqual(errors,[])
  observed={'name':'race_busy_guard','expected_no_trim_while_busy':True,'observed_calls':calls,
            'trims':d.trims,'source_lines':{'idle_check':802,'trim_call':812,'busy_transition':890},
            'evidence_type':'CPU AST method extraction with explicit doubles; no GPU allocation'}
  print(json.dumps(observed),flush=True)
  self.assertEqual(calls,[],'REGRESSION: upstream trim runs after exact _flush has set idle=False')

if __name__=='__main__':unittest.main(verbosity=2)
