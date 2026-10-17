"""Freeze libjxl v0.12 scores for the procedural CI fixture generator."""
import pathlib,subprocess,csv,sys,tempfile
from PIL import Image
base=pathlib.Path.home()/'tmp/bu-libjxl-data';base.mkdir(parents=True,exist_ok=True)
root=pathlib.Path(tempfile.mkdtemp(prefix='oracles-',dir=base))
libjxl=pathlib.Path(sys.argv[1])
source=libjxl/'tools/butteraugli_main.cc'
if 'printf("%g-norm: %.17g\\n", p, pnorm);' not in source.read_text():
    raise RuntimeError('Missing libjxl pnorm printf instrumentation: require %.17g, rebuild butteraugli_main before freezing oracles')
rows=[]
for w,h in [(16,16),(17,19),(65,49),(128,96),(257,193),(512,512)]:
    a=bytes((x*13+y*17+c*71+(x*y)%29)%256 for y in range(h) for x in range(w) for c in range(3));b=bytes(v//8*8 for v in a)
    paths=[]
    for label,data in [('ref',a),('dist',b)]:
        path=root/f'{w}x{h}-{label}.png';Image.frombytes('RGB',(w,h),data).save(path);paths.append(str(path))
    text=subprocess.check_output([str(libjxl/'build/tools/butteraugli_main'),*paths,'--intensity_target','80','--pnorm','3','--rawdistmap',str(root/f'{w}x{h}.pfm')],text=True)
    pnorm_text=text.splitlines()[1].split()[-1]
    if len(pnorm_text.replace('.', '').replace('-', '')) < 12:
        raise RuntimeError('butteraugli_main output lacks the %.17g pnorm instrumentation; rebuild the binary')
    rows.append(dict(width=w,height=h,max_norm=float(text.splitlines()[0]),pnorm3=float(text.splitlines()[1].split()[-1])));print(rows[-1],flush=True)
with open(sys.argv[2],'w') as f:
    wr=csv.DictWriter(f,rows[0].keys(),delimiter='\t');wr.writeheader();wr.writerows(rows)
