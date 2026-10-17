"""Summarize retained zenbench, parity, GNU time and perf results."""
import csv, hashlib, json, pathlib, re, shutil, struct
root=pathlib.Path.home()/'tmp/bu-libjxl-data'
out=pathlib.Path('benchmarks/libjxl_exact_review_2026-10-17');out.mkdir(exist_ok=True)
def tsv(name,rows):
    with (out/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),delimiter='\t');w.writeheader();w.writerows(rows)
speed=[]
for size in [512,2048]:
    for threads in ['1','default']:
        row=dict(width=size,height=size,threads=threads,affinity_cpus=28)
        for version in ['before','after']:
            d=json.loads((root/f'round3-{version}'/f'zen-{size}-{threads}.json').read_text())
            assert not d['unreliable'],(version,size,threads)
            for c in d['comparisons']:
                for b in c['benchmarks']:
                    stats=b['summary'];assert stats['n']>=7
                    row[f"{version}_{b['name']}_median_ms"]=stats['median']/1e6
                    row[f"{version}_{b['name']}_n"]=stats['n']
                    row[f"{version}_{b['name']}_mad_ms"]=stats['mad']/1e6
        speed.append(row)
tsv('speed.tsv',speed)
parity=list(csv.DictReader((root/'round3-after/parity.tsv').open(),delimiter='\t'))
assert len(parity)==19
assert all(r['default_score_bits_identical']=='True' and r['default_diffmap_bits_identical']=='True' for r in parity)
shutil.copyfile(root/'round3-after/parity.tsv',out/'parity.tsv')
print('maxima', {k:max(float(r[k]) for r in parity) for k in ['max_abs','max_rel','p3_abs','p3_rel']})
memory=[]
for mode in ['fast','exact']:
    for threads in ['1','default']:
        row=dict(width=2048,height=2048,mode=mode,threads=threads)
        scores=[]
        for version in ['before','after']:
            d=root/f'round3-{version}'
            rss=(d/f'rss-{mode}-{threads}.txt').read_text()
            row[f'{version}_peak_rss_kib']=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',rss)[1])
            text=(d/f'memory-{mode}-{threads}.txt').read_text()
            rows=re.findall(r'score=([\d.]+) p3=([\d.]+) retained=(\d+) precompute=(\d+)',text)
            assert len(rows)==3
            scores.append([struct.pack('dd',float(a),float(b)) for a,b,_,_ in rows])
            row[f'{version}_estimated_planar_peak_bytes']=int(re.search(r'estimate=(\d+)',text)[1])
            row[f'{version}_retained_bytes']=int(rows[-1][2])
            row[f'{version}_precompute_bytes']=int(rows[-1][3])
        assert scores[0]==scores[1],('memory workload score drift',mode,threads)
        row['scores_bits_identical']=True
        memory.append(row)
tsv('memory.tsv',memory)
perf=[]
for version in ['before','after']:
    text=(root/f'round3-{version}'/'perf-exact.txt').read_text()
    row={'version':version,'width':2048,'height':2048,'mode':'exact','threads':1}
    for counter in ['cycles','instructions','branches','branch-misses']:
        row[counter]=int(re.search(r'([\d,]+)\s+'+re.escape(counter)+r'\s',text)[1].replace(',',''))
    perf.append(row)
tsv('perf.tsv',perf)
shutil.copyfile(root/'round3-after/zen-provenance.json',out/'provenance.json')
# Large logs, raw maps and executable images stay off git; hash their pointers.
paths=[pathlib.Path.home()/'tmp/bu-r3-gates.log',pathlib.Path.home()/'tmp/bu-r3-measure.log']
for version in ['before','after']:
    directory=root/f'round3-{version}'
    paths+=list(directory.glob('zen-*.json'))+list(directory.glob('rss-*.txt'))+list(directory.glob('memory-*.txt'))+list(directory.glob('perf-*.txt'))
meta=[]
for path in paths:
    actual=path.resolve()
    meta.append(dict(host='r5900xt',path=str(actual),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
# Gate logs initially lived on the root filesystem; mirror to the data volume.
for entry in meta:
    if entry['path'].startswith(str(pathlib.Path.home())):
        src=pathlib.Path(entry['path']);dest=root/'round3-after'/src.name
        shutil.copyfile(src,dest);entry['path']=str(dest.resolve())
(out/'artifacts.json').write_text(json.dumps(meta,indent=2)+'\n')
print('speed',speed);print('memory',memory);print('perf',perf)
