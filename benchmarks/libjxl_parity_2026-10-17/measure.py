import csv, json, os, pathlib, struct, subprocess, sys, math
root=pathlib.Path(sys.argv[1]); label=sys.argv[2]; binary=sys.argv[3]
cpp=pathlib.Path.home()/'tmp/libjxl-v012/build/tools/butteraugli_main'
rows=[]
for c in json.loads((root/'manifest.json').read_text()):
    case=c['case']; stem=root/case
    rd=str(stem)+f'.{label}.f32'; cp=str(stem)+'.cpp.pfm'
    rust=list(map(float,subprocess.check_output([binary,c['reference'],c['distorted'],rd],text=True,env={**os.environ,'RAYON_NUM_THREADS':'1'}).split()))
    text=subprocess.check_output([str(cpp),c['reference'],c['distorted'],'--intensity_target','80','--pnorm','3','--rawdistmap',cp],text=True)
    (root/f'{case}.cpp.txt').write_text(text)
    with open(cp,'rb') as f:
        assert f.readline()==b'Pf\n'; w,h=map(int,f.readline().split()); assert f.readline()==b'-1.0\n'
        vals=struct.unpack('<'+'f'*(w*h),f.read())
    # Independent double-precision aggregation avoids butteraugli_main's 6-decimal printf.
    ns=[math.fsum(float(v)**p for v in vals)/(w*h) for p in (3,6,12)]
    cpp3=sum(v**(1/p) for v,p in zip(ns,(3,6,12)))/3
    cppmax=max(vals)
    rv=struct.unpack('<'+'f'*(w*h),pathlib.Path(rd).read_bytes())
    cv=[v for y in range(h-1,-1,-1) for v in vals[y*w:(y+1)*w]]
    row=dict(case=case,rust_revision=label,width=w,height=h,q=c['q'],rust_max=rust[0],cpp_max=cppmax,max_abs=abs(rust[0]-cppmax),max_rel=abs(rust[0]-cppmax)/cppmax,rust_3=rust[1],cpp_3=cpp3,p3_abs=abs(rust[1]-cpp3),p3_rel=abs(rust[1]-cpp3)/cpp3,diffmap_max_abs=max(abs(a-b) for a,b in zip(rv,cv)),cpp_cli_3=float(text.splitlines()[1].split()[-1]))
    rows.append(row); print(case, row['max_abs'], row['p3_abs'],flush=True)
with open(root/f'{label}.tsv','w') as f:
    wr=csv.DictWriter(f,fieldnames=rows[0].keys(),delimiter='\t'); wr.writeheader();wr.writerows(rows)
