import os,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class PinTests(unittest.TestCase):
    def test_latest_standard_pin_and_variant_override(self):
        base={'PATH':os.environ['PATH'],'HOME':os.environ['HOME'],'MASTER_ADDR':'127.0.0.1'}
        script='source scripts/config.sh; printf "%s\\n" "$MODEL_REVISION"'
        r=subprocess.run(['bash','-c',script],cwd=ROOT,env=base,capture_output=True,text=True,check=True)
        self.assertEqual(r.stdout.strip(),'6c5b28260ab9e80c6608de8b419e624cbe71b7cf')
        base.update(MODEL_ID='bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit',MODEL_REVISION='14858211ed81d7fa773f8a0db02f38f36d230252')
        r=subprocess.run(['bash','-c',script],cwd=ROOT,env=base,capture_output=True,text=True,check=True)
        self.assertEqual(r.stdout.strip(),base['MODEL_REVISION'])
if __name__=='__main__':unittest.main()
