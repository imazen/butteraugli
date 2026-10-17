"""Deterministic eval-only JPEG pairs. No metadata carried to scoring PNGs."""
import hashlib, io, json, pathlib, sys
from PIL import Image
out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
inputs=pathlib.Path(sys.argv[2])
rows=[]
for name in ['1025469','1044329','1189261']:
    src=Image.open(inputs/f'{name}.png').convert('RGB')
    sizes=[(512,512)]
    if name=='1025469': sizes += [(257,193),(1001,667),(2048,2048)]
    for w,h in sizes:
        # Tile for the larger edge/multiresolution probes; no invented high-res detail.
        ref=Image.new('RGB',(w,h))
        for y in range(0,h,src.height):
            for x in range(0,w,src.width): ref.paste(src,(x,y))
        for q in [20,50,80]:
            case=f'{name}_{w}x{h}_q{q}'
            rp=out/f'{case}_ref.png'; dp=out/f'{case}_dist.png'; jp=out/f'{case}.jpg'
            ref.save(rp)
            ref.save(jp,quality=q,subsampling=2,optimize=False,progressive=False)
            Image.open(jp).convert('RGB').save(dp)
            rows.append(dict(case=case,width=w,height=h,q=q,reference=str(rp),distorted=str(dp),jpeg=str(jp),source_sha256=hashlib.sha256((inputs/f'{name}.png').read_bytes()).hexdigest(),jpeg_sha256=hashlib.sha256(jp.read_bytes()).hexdigest()))
(out/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
print(f'{len(rows)} pairs in {out}',flush=True)
