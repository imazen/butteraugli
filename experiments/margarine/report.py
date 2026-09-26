#!/usr/bin/env python3
"""Build an offline, interactive view of committed Margarine measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--build-commit', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    files = {
        'quality': 'margarine_vector_row_primary_comparison_2026-09-26.tsv',
        'rgb8': 'margarine_latest_rgb8_resources_2026-09-26.tsv',
        'native16': 'margarine_native16_lut_resources_2026-09-26.tsv',
        'cid22': 'margarine_vector_row_cid22_point_choices_2026-09-26.tsv',
        'aic3': 'margarine_vector_row_aic3_point_choices_2026-09-26.tsv',
    }
    data = {}
    hashes = {}
    for key, name in files.items():
        source = root/'benchmarks'/name
        with source.open() as f:
            data[key] = list(csv.DictReader(f, delimiter='\t'))
        if not data[key]:
            raise ValueError(f'empty measurement table: {source}')
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'progress.log').open('x', buffering=1) as log:
        print('Loaded five committed measurement tables', file=log, flush=True)
        html = TEMPLATE.replace('DATA_PLACEHOLDER', json.dumps(data).replace('<', '\\u003c'))
        (args.output/'index.html').write_text(html)
        manifest = dict(build_commit=args.build_commit, inputs=hashes,
                        report_sha256=hashlib.sha256(html.encode()).hexdigest(),
                        qualification='incomplete; corpus point screens are not universal or uncertainty-based choice guarantees')
        (args.output/'_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print('Saved offline report and input hashes', file=log, flush=True)
    print(args.output/'index.html', flush=True)


TEMPLATE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Margarine — measured qualification</title>
<style>
:root{font-family:system-ui,sans-serif;color:#202d3a;background:#f5f7fa}body{max-width:1120px;margin:auto;padding:28px}h1{font-size:36px;margin-bottom:8px}h2{font-size:22px;margin-top:32px}p{line-height:1.55;max-width:90ch}.notice{border-left:4px solid #b17416;background:#fff8e8;padding:12px 18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}.card{background:white;border:1px solid #dbe2e9;border-radius:8px;padding:18px}.big{font-size:27px;font-weight:650}label{display:inline-block;margin:0 20px 12px 0}select{padding:6px;font:inherit}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}td,th{text-align:right;border-bottom:1px solid #e3e8ed;padding:8px}td:first-child,th:first-child{text-align:left}th{background:#eef2f6}.scroll{overflow:auto}.bad{color:#a23123}.good{color:#176b49}.muted{color:#566574;font-size:14px}svg{width:100%;height:auto}code{background:#e9eef4;padding:3px 5px}a{color:#14658c}
</style>
<h1>Margarine</h1><p>A streamed Butteraugli-lineage approximation. Primary score: <strong>max</strong>. Lower scores indicate less distortion.</p>
<p class="notice"><strong>Qualification remains incomplete.</strong> All completed corpus/cohort primary rank point screens meet the 0.01 limit. CID22 has no participant uncertainty for the harmful-choice gate; some narrower quality bands lose more than 0.01 SROCC. Decode-inclusive speed is below 4× on the measured large pairs.</p>
<div class="grid"><div class="card"><div class="big">43,506 pairs</div>Nine corpora/releases; 121 estimated AIC3 labels remain separate.</div><div class="card"><div class="big">4,292 exact replays</div>All CID22 scalar variants and native diffmap bytes match the frozen candidate after exact RGB16 conversion caching.</div><div class="card"><div class="big">1 / 328 LIVE budgets</div>One participant-supported harmful choice; 1 / 87 in its affected session. Dataset and session rates differ.</div></div>
<h2>Quality against Butteraugli’s default max</h2>
<label>Margarine pooling <select id="norm"><option>max</option><option>p1</option><option>p2</option><option>p3</option><option>p6</option></select></label>
<label>Statistic <select id="stat"><option value="srocc">SROCC</option><option value="krocc">KROCC</option><option value="plcc">PLCC</option><option value="pwrc">PWRC</option><option value="or">OR</option><option value="z_rmse">Z-RMSE</option><option value="geomean3">Geometric composite</option><option value="harmean3">Harmonic composite</option><option value="min3">Minimum composite</option></select></label>
<p id="orientation" class="muted"></p><div class="card" id="chart"></div><div class="scroll"><table id="quality"></table></div>
<h2>Resources of the latest measured command</h2>
<p>Zen 3 / AVX2, two threads. Interleaved zenbench means; peak RSS is the maximum of three fresh processes and includes inputs and decoding. Scoring includes encoded-to-linear conversion. Decode timing uses a warm filesystem cache. These are measured pairs, not unmeasured size or content guarantees.</p>
<label>Input workload <select id="input"><option value="rgb8">RGB8 photo crops</option><option value="native16">CID22 RGB8 reference / RGB16 distortion</option></select></label>
<div class="scroll"><table id="resources"></table></div>
<p class="muted">Large-image targets are 4× scoring speed and at most 25% process RAM. Smaller-image ratios may taper provided they remain below Butteraugli. The RGB16 workload ends at its native 512² size; no larger RGB16 claim is inferred.</p>
<h2>Observed byte-budget choices</h2>
<p>Point-label decreases are <strong>not</strong> statistically distinguishable harm. Neither CID22 nor the supplied AIC3 table includes the participant uncertainty needed for that classification. AIC4 lacks the original byte-budget table here. LIVE’s participant result above is separate.</p>
<div class="scroll"><table id="choices"></table></div>
<h2>Run the experimental command</h2>
<p><code>just margarine-build</code></p><p><code>experiments/margarine/target/release/margarine --all-scores reference.png distorted.png</code></p>
<p>Add <code>--diffmap NEW.f32le</code> to save the native row-major little-endian f32 map. Encoded-sRGB RGB/RGBA 8/16-bit inputs retain their sample precision; alpha must be opaque. This is an unpublished command, not a released library.</p>
<p class="muted">Metric build: b6c1e6b1. Frozen quality build: 6bb371fb. Each source table and this report are hashed in the adjacent _MANIFEST.json. Bootstrap intervals and subgroup panels are separate artifacts; bars here are point estimates.</p>
<script>
const data=DATA_PLACEHOLDER;
const $=id=>document.getElementById(id);
function table(id,heads,rows){const t=$(id);t.replaceChildren();const h=t.createTHead().insertRow();heads.forEach(s=>{const c=document.createElement('th');c.textContent=s;h.append(c)});const b=t.createTBody();rows.forEach(row=>{const r=b.insertRow();row.forEach(s=>{r.insertCell().textContent=s})})}
function fmt(x,n=6){return Number.isFinite(Number(x))?Number(x).toFixed(n):'unavailable'}
function quality(){const norm=$('norm').value,stat=$('stat').value,low=['or','z_rmse'].includes(stat),rank=['srocc','krocc'].includes(stat),rows=data.quality.filter(r=>r.candidate_norm===norm);$('orientation').textContent=`Candidate minus Butteraugli max. ${low?'Lower':'Higher'} is better. ${rank?'Dashed line: −0.01 rank-loss limit.':'No acceptance threshold is assigned to this statistic.'} Primary qualification uses Margarine max; other norms are diagnostics.`;
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg'),w=1000,h=rows.length*30+40,left=220,right=930;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);const values=rows.map(r=>Number(r[stat+'_delta'])).filter(Number.isFinite),lo=Math.min(-.012,...values),hi=Math.max(.003,...values),x=v=>left+(v-lo)/(hi-lo)*(right-left);
 function el(name,attrs,text){let n=document.createElementNS(ns,name);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=text;svg.append(n);return n}
 el('line',{x1:x(0),x2:x(0),y1:4,y2:h-24,stroke:'#8a99a7'});if(rank)el('line',{x1:x(-.01),x2:x(-.01),y1:4,y2:h-24,stroke:'#ad5128','stroke-dasharray':'4 3'});
 rows.forEach((r,i)=>{let y=i*30+20,v=Number(r[stat+'_delta']);el('text',{x:4,y:y+4,'font-size':13},r.dataset);if(Number.isFinite(v)){el('rect',{x:Math.min(x(0),x(v)),y:y-7,width:Math.max(1,Math.abs(x(v)-x(0))),height:14,fill:(low?v<=0:v>=0)?'#338467':'#b46546'});el('title',{},`${r.dataset}: ${v}`)}});
 el('text',{x:left,y:h-3,'font-size':12},fmt(lo));el('text',{x:right,y:h-3,'font-size':12,'text-anchor':'end'},fmt(hi));$('chart').replaceChildren(svg);
 table('quality',['Corpus / cohort','Pairs','Candidate − teacher'],rows.map(r=>[r.dataset,r.n,fmt(r[stat+'_delta'])]));
 table('choices',['Corpus / cohort','Budgets','Point decreases','Rate','Mean signed loss','Largest loss'],[...data.cid22,...data.aic3].filter(r=>r.norm===norm).map(r=>[r.dataset,r.budgets,r.exceeding_budgets,fmt(100*r.pooled_exceedance_rate,2)+'%',fmt(r.mean_signed_quality_loss),fmt(r.maximum_quality_loss)]));}
 function resources(){table('resources',['Dimensions','Scoring speedup','With decoding','Candidate peak MiB','Teacher peak MiB','RAM fraction'],data[$('input').value].filter(r=>r.arm==='simd-row-malta').map(r=>{const teacher=data[$('input').value].find(t=>t.arm==='teacher'&&t.width===r.width&&t.height===r.height);return [`${r.width} × ${r.height}`,fmt(r.mean_speedup,2)+'×',fmt(r.decode_included_mean_speedup,2)+'×',fmt(r.peak_rss_bytes/1048576,1),fmt(teacher.peak_rss_bytes/1048576,1),fmt(100*r.rss_fraction_of_teacher,1)+'%']}))}
 $('norm').onchange=quality;$('stat').onchange=quality;$('input').onchange=resources;quality();resources();
</script></html>'''


if __name__ == '__main__':
    main()
