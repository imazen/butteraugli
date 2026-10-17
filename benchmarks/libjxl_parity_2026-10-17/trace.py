import os,pathlib,subprocess,csv,numpy as np
root=pathlib.Path.home()/'tmp/bu-libjxl-data'; env={**os.environ,'RAYON_NUM_THREADS':'1'}
a=str(root/'benchmark_reference.png');b=str(root/'benchmark_distorted.png')
for label,binary in [('cpp',str(pathlib.Path.home()/'tmp/libjxl-v012/build/tools/butteraugli_main')),('rust',str(pathlib.Path.home()/'tmp/bu-libjxl-trace-runner'))]:
    d=root/f'trace-{label}';d.mkdir(exist_ok=True)
    args=[binary,a,b]
    if label=='cpp':args+=['--intensity_target','80','--pnorm','3','--rawdistmap',str(root/'trace.cpp.pfm')]
    else:args+=[str(root/'trace.rust.f32')]
    t=subprocess.check_output(args,env={**env,'PARITY_DUMP':str(d)},text=True);print(label,t,flush=True);(root/f'trace.{label}.txt').write_text(t)
rows=[]
for f in sorted((root/'trace-rust').glob('*.f32')):
    other=root/'trace-cpp'/f.name
    if not other.exists():continue
    r=np.fromfile(f,dtype='<f4');c=np.fromfile(other,dtype='<f4')
    diff=np.abs(r.astype(float)-c.astype(float));i=int(diff.argmax())
    rows.append(dict(stage=f.name,max_abs=float(diff[i]),mean_abs=float(diff.mean()),index=i,rust=float(r[i]),cpp=float(c[i]),different=int(np.count_nonzero(diff))))
with open(root/'stages.tsv','w') as f:
    wr=csv.DictWriter(f,rows[0].keys(),delimiter='\t');wr.writeheader();wr.writerows(rows)
for r in rows:
    if '512x512' in r['stage']:print(r,flush=True)
# Re-run with each family of C++ intermediates substituted into Rust.
for family in ['opsin','ac_pre','ac','mf','lf']:
    rep=root/f'replace-{family}';rep.mkdir(exist_ok=True)
    for f in (root/'trace-cpp').glob(f'{family}[012]_*.f32'):
        dst=rep/f.name
        if not dst.exists():dst.symlink_to(f)
    d=root/f'trace-swap-{family}';d.mkdir(exist_ok=True)
    t=subprocess.check_output([str(pathlib.Path.home()/'tmp/bu-libjxl-trace-runner'),a,b,str(root/f'swap-{family}.f32')],env={**env,'PARITY_DUMP':str(d),'PARITY_REPLACE':str(rep)},text=True)
    (root/f'swap-{family}.txt').write_text(t);print('swap',family,t,flush=True)
# Two Google-old max-norm checks on the exact benchmark and one generated photo pair.
from PIL import Image
with open(root/'google-old.tsv','w') as f:
    wr=csv.writer(f,delimiter='\t');wr.writerow(['case','rust_max','cpp_max','abs_delta'])
    for case,ap,bp in [('benchmark',a,b),('1025469_512x512_q50',str(root/'1025469_512x512_q50_ref.png'),str(root/'1025469_512x512_q50_dist.png'))]:
        paths=[]
        for lab,path in [('ref',ap),('dist',bp)]:
            dst=root/f'{case}.old.{lab}.rgb';dst.write_bytes(Image.open(path).convert('RGB').tobytes());paths.append(str(dst))
        v=[float(subprocess.check_output([str(pathlib.Path.home()/f'tmp/bu-google-old/{binary}'),'512','512',*paths],text=True)) for binary in ['rust_old','cpp_old']]
        wr.writerow([case,*v,abs(v[0]-v[1])]);print('old',case,v,flush=True)
