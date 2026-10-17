"""Round 2: frozen-main bit comparisons, independent-oracle deltas, timings."""
import csv,hashlib,json,os,pathlib,statistics,struct,subprocess,sys
root=pathlib.Path.home()/'tmp/bu-libjxl-data';out=root/next((v.split('=',1)[1] for v in sys.argv[2:] if v.startswith('--output-dir=')),'round2');out.mkdir(exist_ok=True)
binary=sys.argv[1];baseline=next((v.split('=',1)[1] for v in sys.argv[2:] if v.startswith('--baseline=')),None)
main={r['case']:r for r in csv.DictReader((root/'latest-confirmed.tsv').open(),delimiter='\t')}
if baseline:
    assert hashlib.sha256(pathlib.Path(binary).read_bytes()).digest()!=hashlib.sha256(pathlib.Path(baseline).read_bytes()).digest(),'Baseline/current executable collision'
manifest=json.loads((root/'manifest.json').read_text());rows=[]
for case in manifest:
    name=case['case'];scores={}
    for mode in ['fast','exact']:
        path=out/f'{name}.{mode}.f32'
        text=subprocess.check_output([binary,case['reference'],case['distorted'],str(path),*(['--exact'] if mode=='exact' else [])],env={**os.environ,'RAYON_NUM_THREADS':'1'},text=True)
        scores[mode]=list(map(float,text.split()))
    expected=main[name]; fast=scores['fast']; exact=scores['exact']
    previous=next((v.split('=',1)[1] for v in sys.argv[2:] if v.startswith('--exact-baseline-dir=')),None)
    if previous:
        prior=next(r for r in csv.DictReader((pathlib.Path(previous)/'parity.tsv').open(),delimiter='\t') if r['case']==name)
        assert struct.pack('dd',*exact)==struct.pack('dd',float(prior['exact_max']),float(prior['exact_pnorm3'])),('Exact-score drift',name)
        assert (out/f'{name}.exact.f32').read_bytes()==(pathlib.Path(previous)/f'{name}.exact.f32').read_bytes(),('Exact-map drift',name)
    assert struct.pack('dd',*fast)==struct.pack('dd',float(expected['rust_max']),float(expected['rust_3'])),name
    assert (out/f'{name}.fast.f32').read_bytes()==(root/f'{name}.latest-confirmed.f32').read_bytes(),name
    cpp=[float(expected['cpp_max']),float(expected['cpp_cli_3'])]
    assert abs(exact[0]-cpp[0])<=1e-6 and abs(exact[1]-cpp[1])<=1e-6,('Exact-mode oracle gate',name,exact,cpp)
    r=dict(case=name,width=case['width'],height=case['height'],default_score_bits_identical=True,default_diffmap_bits_identical=True,exact_max=exact[0],cpp_max=cpp[0],max_abs=abs(exact[0]-cpp[0]),max_rel=abs(exact[0]-cpp[0])/cpp[0],exact_pnorm3=exact[1],cpp_pnorm3=cpp[1],p3_abs=abs(exact[1]-cpp[1]),p3_rel=abs(exact[1]-cpp[1])/cpp[1]);rows.append(r);print(r,flush=True)
with (out/'parity.tsv').open('w') as f:
    wr=csv.DictWriter(f,rows[0].keys(),delimiter='\t');wr.writeheader();wr.writerows(rows)
if '--parity-only' in sys.argv:sys.exit(0)
# Seven timed calls after one warm call; image decoding and map writing excluded.
cpus=sorted(os.sched_getaffinity(0))[:max(1,os.cpu_count()-4)]
rows=[]
for size in [512,2048]:
    c=next(c for c in manifest if c['case']==f'1025469_{size}x{size}_q50')
    for threads in ['1','default']:
        env={**os.environ};env.pop('RAYON_NUM_THREADS',None)
        if threads=='1':env['RAYON_NUM_THREADS']='1'
        prefix=['taskset','-c',','.join(map(str,cpus))]
        for label,exe,args in [('origin_main',baseline,[]),('fast',binary,[]),('exact',binary,['--exact'])]:
            if exe is None:continue
            t=subprocess.check_output(prefix+[exe,c['reference'],c['distorted'],str(out/f'timing-{size}-{threads}-{label}.f32'),'--bench=7',*args],text=True,env=env)
            line=next(line for line in t.splitlines() if line.startswith('BENCH\t'));_,median,samples=line.split('\t')
            r=dict(width=size,height=size,threads=threads,affinity_cpus=len(cpus),mode=label,median_ns=int(median),samples_ns=samples);rows.append(r);print(r,flush=True)
with (out/'speed.tsv').open('w') as f:
    wr=csv.DictWriter(f,rows[0].keys(),delimiter='\t');wr.writeheader();wr.writerows(rows)
