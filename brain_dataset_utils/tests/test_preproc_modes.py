import tempfile
import os, sys, tempfile, shutil, subprocess
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, REPO)
import numpy as np, nibabel as nib
W = os.path.join(tempfile.mkdtemp(prefix='ct2mri_test_'), 'preproc'); shutil.rmtree(W, ignore_errors=True)
IN, OUT = os.path.join(W,'in'), os.path.join(W,'out'); os.makedirs(IN)
rng = np.random.default_rng(5); shape=(50,60,40); aff=np.diag([0.9,0.9,1.5,1.0])
for pid in ['P1','P2']:
    d=os.path.join(IN,pid); os.makedirs(d)
    m=np.zeros(shape,bool); m[5:45,7:53,4:36]=True
    ct=(rng.random(shape)*100).astype(np.float32); mr=(rng.random(shape)*800).astype(np.float32)
    for n,v in [('ct.nii.gz',ct),('mr.nii.gz',mr),('mask.nii.gz',m.astype(np.uint8))]:
        nib.save(nib.Nifti1Image(v,aff), os.path.join(d,n))

def run(extra):
    r=subprocess.run([sys.executable,'-u','finetune_preprocess.py','--input_dir',IN,'--output_dir',OUT,'--workers','2']+extra,
                     cwd=REPO,capture_output=True,text=True)
    if r.returncode!=0: print(r.stdout[-2000:]); print(r.stderr[-2000:]); raise SystemExit('failed')
    return r.stdout

run(['--geometry_only'])
files = sorted(os.listdir(os.path.join(OUT,'P1')))
assert files == ['geometry.json'], f"geometry_only wrote {files}"
print("  geometry_only: wrote only geometry.json")

import json; g1 = json.load(open(os.path.join(OUT,'P1','geometry.json')))
run([])
files = sorted(os.listdir(os.path.join(OUT,'P1')))
assert files == ['ct.nii','geometry.json','mr.nii'], f"full mode wrote {files}"
g2 = json.load(open(os.path.join(OUT,'P1','geometry.json')))
assert g1 == g2, "geometry differs between geometry_only and full mode"
ct = np.asanyarray(nib.load(os.path.join(OUT,'P1','ct.nii')).dataobj)
mr = np.asanyarray(nib.load(os.path.join(OUT,'P1','mr.nii')).dataobj)
assert ct.shape == mr.shape and ct.shape[:2] == (256,256), f"shapes {ct.shape} {mr.shape}"
assert 0.0 <= mr.min() and mr.max() <= 1.0, f"mr range [{mr.min()},{mr.max()}]"
print(f"  full mode: ct/mr {ct.shape}, mr range [{mr.min():.3f},{mr.max():.3f}]")
print("  geometry.json identical in both modes")
print("preprocess modes verified")
