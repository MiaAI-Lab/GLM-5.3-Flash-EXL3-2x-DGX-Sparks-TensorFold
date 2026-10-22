"""CPU-only immutable image hashes; no GPU request, model load or credentials."""
import concurrent.futures,hashlib,json,shlex,subprocess
from pathlib import Path
HOSTS=['spark','spark-peer','msi-b9dc','msi-c21f']
TAG='spark/glm-tp4:focused-rdma-be94e91'
NATIVE='be94e919e35c628261bd38364b1743a2e4353a26'
HERE=Path(__file__).resolve().parent
BUILD_RAW=(HERE/'build-receipt-rdma-be94e91.json').read_bytes();BUILD=json.loads(BUILD_RAW)
EMBED_RAW=(HERE/'embedded-proof-be94e91.json').read_bytes();EMBED=json.loads(EMBED_RAW)
BUILD_SHA=hashlib.sha256(BUILD_RAW).hexdigest();EMBED_SHA=hashlib.sha256(EMBED_RAW).hexdigest()
if not (BUILD['passed'] is True and EMBED['passed'] is True and BUILD['native_commit']==EMBED['native_commit']==NATIVE and EMBED['build_receipt_sha256']==BUILD_SHA and all(BUILD[k]==EMBED[k] for k in ['image_fingerprint','engine','probe'])):raise RuntimeError('Exact new build binding differs')
ENGINE='/opt/tensorfold/native/bin/tensorfold-native';PROBE='/opt/tensorfold/bin/tf-glm-transport-probe';AOT='/opt/tensorfold/share/tensorfold/cuda/sm121'
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
def remote(host,args):return subprocess.check_output(['ssh','-oBatchMode=yes','-oConnectTimeout=10','-oProxyJump=none','-oProxyCommand=none',host,shlex.join(args)],text=True,timeout=90)
def one(host):
 image=json.loads(remote(host,['sudo','-n','docker','image','inspect',TAG]))[0]
 common=['sudo','-n','docker','run','--rm','--runtime','runc','--network','none','--read-only','--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges']
 def run(entry,args):return remote(host,common+['--entrypoint',entry,image['Id']]+args)
 sums=run('/usr/bin/sha256sum',[ENGINE,PROBE]);files={v.split(None,1)[1]:v.split(None,1)[0] for v in sums.splitlines()}
 rows=run('/usr/bin/find',[AOT,'-type','f','-exec','sha256sum','{}','+']);aot={v.split(None,1)[1]:v.split(None,1)[0] for v in rows.splitlines()}
 require(len(aot)==583,'AOT file count differs')
 fat=json.loads(run('/bin/cat',['/opt/tensorfold/share/doc/focused-fatbins.json']))
 require(fat['native_commit']==NATIVE and fat['sm']==121 and fat['count']==56 and fat['files']==EMBED['exported'],'NVCC fatbin manifest differs')
 require(files[ENGINE]==BUILD['engine'] and files[PROBE]==BUILD['probe'] and hashlib.sha256(canon({'rootfs':image['RootFS'],'config':image['Config']})).hexdigest()==BUILD['image_fingerprint'],'Exact build/image artifact differs')
 return host,{'native_commit':NATIVE,'build_receipt_sha256':BUILD_SHA,'embedded_proof_sha256':EMBED_SHA,'image_id':image['Id'],'image_fingerprint':hashlib.sha256(canon({'rootfs':image['RootFS'],'config':image['Config']})).hexdigest(),'engine':files[ENGINE],'probe':files[PROBE],'aot':hashlib.sha256(canon(aot)).hexdigest(),'aot_files':len(aot),'nvcc_fatbins':fat,'no_gpu_execution':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=dict(pool.map(one,HOSTS))
for key in ['image_fingerprint','engine','probe','aot','nvcc_fatbins']:require(len({json.dumps(v[key],sort_keys=True) for v in result.values()})==1,'Cross-host image artifact differs: '+key)
Path(__file__).with_name('focused-image-artifacts-be94e91.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({h:{k:v for k,v in row.items() if k!='nvcc_fatbins'} for h,row in result.items()}))
