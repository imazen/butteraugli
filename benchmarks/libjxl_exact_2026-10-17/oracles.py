"""Freeze libjxl v0.12 scores for the procedural CI fixture generator."""
import pathlib,subprocess,csv,sys
from PIL import Image
root=pathlib.Path.home()/'tmp/bu-libjxl-data/round2/oracles';root.mkdir(parents=True,exist_ok=True)
source=pathlib.Path.home()/'tmp/libjxl-v012/tools/butteraugli_main.cc'
if 'printf("%g-norm: %.17g\\n", p, pnorm);' not in source.read_text():
    raise RuntimeError('Missing libjxl pnorm printf instrumentation: require %.17g, rebuild butteraugli_main before freezing oracles')
rows=[]
for w,h in [(16,16),(17,19),(65,49),(128,96),(257,193),(512,512)]:
    a=bytes((x*13+y*17+c*71+(x*y)%29)%256 for y in range(h) for x in range(w) for c in range(3));b=bytes(v//8*8 for v in a)
    paths=[]
    for label,data in [('ref',a),('dist',b)]:
        path=root/f'{w}x{h}-{label}.png';Image.frombytes('RGB',(w,h),data).save(path);paths.append(str(path))
    text=subprocess.check_output([str(pathlib.Path.home()/'tmp/libjxl-v012/build/tools/butteraugli_main'),*paths,'--intensity_target','80','--pnorm','3','--rawdistmap',str(root/f'{w}x{h}.pfm')],text=True)
    rows.append(dict(width=w,height=h,max_norm=float(text.splitlines()[0]),pnorm3=float(text.splitlines()[1].split()[-1])));print(rows[-1],flush=True)
with open(sys.argv[1],'w') as f:
    wr=csv.DictWriter(f,rows[0].keys(),delimiter='\t');wr.writeheader();wr.writerows(rows)
