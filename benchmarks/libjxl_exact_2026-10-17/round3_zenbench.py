"""Round 3 speed reporting via zenbench; fixed-count perf/RSS runs are separate."""
import csv, hashlib, json, os, pathlib, shutil, subprocess
work = pathlib.Path(__file__).resolve().parents[2]
root = pathlib.Path.home()/'tmp/bu-libjxl-data'
manifest = json.loads((root/'manifest.json').read_text())
before = root/'round3-before'
after = root/'round3-after'
after.mkdir(exist_ok=True)
source = before/'zen-source'
source.mkdir()
archive = subprocess.Popen(['git', 'archive', 'd29e066830ee41c6b4492274747aebb0310d459f'],cwd=work,stdout=subprocess.PIPE)
subprocess.run(['tar','-xf','-','-C',str(source)],stdin=archive.stdout,check=True)
archive.stdout.close()
assert archive.wait()==0
for relative in ['butteraugli-bench/examples/libjxl_exact_speed.rs','butteraugli-cli/examples/libjxl_memory.rs']:
    shutil.copyfile(work/relative,source/relative)
env = {**os.environ,'BUTTERAUGLI_BENCH_NO_CPP':'1','CARGO_INCREMENTAL':'0'}
executables = {}
for label,src,target in [('before',source,before/'zen-target'),('after',work,after/'zen-target')]:
    for package,example in [('butteraugli-bench','libjxl_exact_speed'),('butteraugli-cli','libjxl_memory')]:
        subprocess.run(['cargo','build','--release','--locked','--manifest-path',str(src/'Cargo.toml'),'--target-dir',str(target),'-p',package,'--example',example],env=env,check=True)
        binary=(before if label=='before' else after)/example
        shutil.copyfile(target/'release/examples'/example,binary)
        binary.chmod(0o755)
        executables[(label,example)]=binary
cpus=sorted(os.sched_getaffinity(0))[:max(1,os.cpu_count()-4)]
prefix=['taskset','-c',','.join(map(str,cpus))]
for size in [512,2048]:
    case=next(c for c in manifest if c['case']==f'1025469_{size}x{size}_q50')
    for threads in ['1','default']:
        runenv={**env};runenv.pop('RAYON_NUM_THREADS',None)
        if threads=='1':runenv['RAYON_NUM_THREADS']='1'
        for label in ['before','after']:
            out=(before if label=='before' else after)/f'zen-{size}-{threads}.json'
            subprocess.run(prefix+[str(executables[(label,'libjxl_exact_speed')])],env={**runenv,'BU_BENCH_REFERENCE':case['reference'],'BU_BENCH_DISTORTED':case['distorted'],'BU_BENCH_OUTPUT':str(out)},check=True)
# No timers in the workload: GNU time measures process peak RSS.
for label in ['before','after']:
    for mode in ['fast','exact']:
        for threads in ['1','default']:
            runenv={**env};runenv.pop('RAYON_NUM_THREADS',None)
            if threads=='1':runenv['RAYON_NUM_THREADS']='1'
            directory=before if label=='before' else after
            with (directory/f'memory-{mode}-{threads}.txt').open('w') as stdout:
                subprocess.run(['/usr/bin/time','-v','-o',str(directory/f'rss-{mode}-{threads}.txt'),*prefix,str(executables[(label,'libjxl_memory')]),'2048','2048',mode],env=runenv,stdout=stdout,check=True)
# Hardware counters for exactly the same reference + three comparison workload.
for label in ['before','after']:
    directory=before if label=='before' else after
    subprocess.run(['perf','stat','-e','cycles,instructions,branches,branch-misses','-o',str(directory/'perf-exact.txt'),*prefix,str(executables[(label,'libjxl_memory')]),'2048','2048','exact'],env={**env,'RAYON_NUM_THREADS':'1'},check=True)
metadata={'before_commit':'d29e066830ee41c6b4492274747aebb0310d459f','after_commit':subprocess.check_output(['git','log','-1','--format=%H'],cwd=work,text=True).strip(),'executables':{f'{label}/{example}':hashlib.sha256(binary.read_bytes()).hexdigest() for (label,example),binary in executables.items()},'affinity_cpus':cpus,'speed':'zenbench, seven rounds, warmup 500ms, one iteration per round; resident RGB8 including result drop; decoding excluded','memory':'2048x2048 synthetic planar input, one reference plus three comparisons per process, includes caller input/output; GNU time -v'}
(after/'zen-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
