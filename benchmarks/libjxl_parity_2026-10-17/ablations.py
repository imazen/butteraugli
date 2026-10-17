from pathlib import Path
import subprocess,os,json,numpy as np,csv
root=Path.home()/'tmp/bu-libjxl-data';home=Path.home()
a=str(root/'benchmark_reference.png');b=str(root/'benchmark_distorted.png');rows=[]
for variant,binary,rep in [('baseline','latest',None),('kernel_f64','kernel',None),('cpp_fir','strict',None),('baseline_cpp_opsin','latest','opsin'),('cpp_fir_cpp_opsin','strict','opsin'),('baseline_cpp_lf','latest','lf')]:
    env={**os.environ,'RAYON_NUM_THREADS':'1'}
    if rep:
        dump=root/f'ablation-{variant}';dump.mkdir(exist_ok=True);env.update(PARITY_DUMP=str(dump),PARITY_REPLACE=str(root/f'replace-{rep}'))
    t=subprocess.check_output([str(home/f'tmp/bu-libjxl-{binary}-runner'),a,b,str(root/f'ablation-{variant}.f32')],text=True,env=env)
    v=list(map(float,t.split()));rows.append(dict(variant=variant,max=v[0],pnorm3=v[1],max_abs=abs(v[0]-1.2085671424865723),p3_abs=abs(v[1]-0.49498148913202938)));print(rows[-1],flush=True)
with open(root/'ablations.tsv','w') as f:
    wr=csv.DictWriter(f,rows[0].keys(),delimiter='\t');wr.writeheader();wr.writerows(rows)
