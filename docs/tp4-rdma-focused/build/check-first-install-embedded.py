"""CPU-only final image/source binding and exact embedded-fatbin proof."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
NATIVE='be94e919e35c628261bd38364b1743a2e4353a26'
ROOT=Path('/home/spark/lab/glm-tp4-focused-20261022/first-install-be94e91')
TAG='spark/glm-tp4:focused-rdma-be94e91'
def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
raw=(ROOT/'build-receipt-rdma-be94e91.json').read_bytes();receipt=json.loads(raw)
image=json.loads(subprocess.check_output(['docker','image','inspect',TAG]))[0]
fp=sha(canonical({'rootfs':image['RootFS'],'config':image['Config']}))
require(receipt['passed'] is True and receipt['native_commit']==NATIVE and receipt['image_id']==image['Id'] and receipt['image_fingerprint']==fp,'Build/image binding differs')
expected=json.loads((ROOT/'build-prep/expected-fatbins-be94e91.json').read_text())
require(len(expected)==len(set(expected))==56,'Invalid exact fatbin registry')
with tempfile.TemporaryDirectory(prefix='glm-first-install-embedded-') as tmp:
 out=Path(tmp);cid=subprocess.check_output(['docker','create','--runtime','runc','--network','none','--entrypoint','/bin/true',image['Id']],text=True).strip()
 try:
  for source,name in [('/opt/tensorfold/native/bin/tensorfold-native','engine'),('/opt/tensorfold/bin/tf-glm-transport-probe','probe'),('/opt/tensorfold/fatbin','fatbins'),('/opt/tensorfold/share/doc/focused-fatbins.json','exports.json')]:
   subprocess.run(['docker','cp',cid+':'+source,str(out/name)],check=True,stdout=subprocess.DEVNULL)
  files={p.name:p for p in (out/'fatbins').iterdir()}
  require(set(files)==set(expected) and all(p.is_file() and not p.is_symlink() and p.stat().st_size>0 for p in files.values()),'Fatbin export inventory differs')
  exported={name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for name,p in sorted(files.items())}
  producer=json.loads((out/'exports.json').read_text())
  require(producer['native_commit']==NATIVE and producer['sm']==121 and producer['count']==56 and producer['files']==exported,'Producer NVCC export manifest differs')
  binary_sha={name:sha((out/name).read_bytes()) for name in ('engine','probe')}
  require(all(binary_sha[name]==receipt[name] for name in binary_sha),'Build binary digest differs')
  embedded={name:{f:{**exported[f],'found_offset':(out/name).read_bytes().find(p.read_bytes())} for f,p in sorted(files.items())} for name in ('engine','probe')}
  require(all(embedded[name]['fn_roce.fatbin']['found_offset']>=0 for name in embedded),'Required RDMA kernel missing')
  engine_absent={f for f,v in embedded['engine'].items() if v['found_offset']<0}
  require(engine_absent=={'gdn.fatbin','probe.fatbin','glm_foundations.fatbin','glm_msa.fatbin','fp8_experts.fatbin'},'Unexpected engine embedding coverage')
  require({f for f,v in embedded['probe'].items() if v['found_offset']>=0}=={'fn_roce.fatbin'},'Unexpected weight-free probe embedding coverage')
 finally:subprocess.run(['docker','rm',cid],check=True,stdout=subprocess.DEVNULL)
result={'schema':'glm-focused-embedded-v1','passed':True,'native_commit':NATIVE,'build_receipt_sha256':sha(raw),'image_fingerprint':fp,**binary_sha,'sm':121,'nvcc_registry_names':expected,'exported':exported,'embedded':embedded,'no_gpu_execution':True,'linker_unused_engine_blobs':{'gdn.fatbin':'No caller in zig/src','probe.fatbin':'No caller in zig/src','glm_foundations.fatbin':'Only foundations branch; no caller selects it','glm_msa.fatbin':'Msa.load has no caller','fp8_experts.fatbin':'Fp8Experts has no external caller'},'probe_scope':'Only fn_roce is loaded by the weight-free transport probe'}
print(json.dumps(result,indent=2))
