#!/usr/bin/env python3
"""Build evaluation manifests from original human labels, preserving native units.

No legacy normalized feature-table targets are consumed. Optional --labels-only
writes metadata with image verification explicitly pending; scoring still audits
all image bytes. CSIQ needs openpyxl, LIVE needs scipy on the dataset host.
"""
import argparse
import csv
import json
import io
import math
from pathlib import Path
import re
import shutil

from score_manifest import FIELDS, digest

EXPECTED = {'kadid': 10125, 'tid': 3000, 'csiq': 866, 'live': 779,
            'aic3': 600, 'pipal': 23200}


def case_index(directory):
    result = {}
    for p in directory.iterdir():
        key = p.name.lower()
        if key in result:
            raise ValueError(f'case-ambiguous filenames in {directory}: {key}')
        result[key] = p
    return result


def records(dataset, root):
    """Return (reference, distortion, family, target, direction, sigma) rows."""
    labels = []
    rows = []
    if dataset == 'kadid':
        labels = [root / 'dmos.csv']
        for r in csv.DictReader(io.StringIO(labels[0].read_text())):
            # Despite its column name, dmos is quality-oriented DCR in [1,5].
            rows.append((root/'images'/r['ref_img'], root/'images'/r['dist_img'],
                         'distortion_'+r['dist_img'].split('_')[1], float(r['dmos']),
                         'quality', math.sqrt(float(r['var']))))
    elif dataset == 'tid':
        labels = [root/'mos_with_names.txt', root/'mos_std.txt']
        refs = case_index(root/'reference_images_png')
        distorted = case_index(root/'distorted_images_png')
        data = [line.split() for line in labels[0].read_text().splitlines() if line.strip()]
        std = [float(line) for line in labels[1].read_text().splitlines() if line.strip()]
        if len(data) != len(std):
            raise ValueError('TID MOS and standard deviations have different counts')
        for (target, name), sigma in zip(data, std):
            stem = Path(name).stem.lower()
            rows.append((refs[stem.split('_')[0]+'.png'], distorted[stem+'.png'],
                         'distortion_'+stem.split('_')[1], float(target), 'quality', sigma))
    elif dataset == 'aic3':
        labels = [root/'decoded/info.csv']
        for r in csv.DictReader(io.StringIO(labels[0].read_text())):
            if not any(r.values()):
                continue  # empty CSV separator, not an image row
            if r['method'] not in ('subjective', 'estimated'):
                raise ValueError('unknown AIC3 label provenance')
            source, codec, quality = r['img.name'], r['codec'], r['quality']
            rows.append((root/'original'/f'{source}.png',
                         root/'decoded'/source/f'{codec}_{source}_{quality}.png',
                         codec, float(r['score.jnd']), 'quality', None))
    elif dataset == 'csiq':
        import openpyxl
        labels = [root/'csiq.DMOS.xlsx']
        workbook = openpyxl.load_workbook(labels[0], read_only=True, data_only=True)
        try:
            data = list(workbook['all_by_image'].iter_rows(values_only=True))
        finally:
            workbook.close()
        i = next(i for i,r in enumerate(data) if 'image' in r and 'dmos' in r)
        fields = data[i]
        names = {'noise': ('awgn','AWGN'), 'blur': ('blur','BLUR'),
                 'contrast': ('contrast','contrast'), 'fnoise': ('fnoise','fnoise'),
                 'jpeg': ('jpeg','JPEG'), 'jpeg 2000': ('jpeg2000','jpeg2000')}
        for values in data[i+1:]:
            r = dict(zip(fields,values))
            if r['image'] is None:
                if any(r[k] is not None for k in ['dmos','dst_type','dst_lev']):
                    raise ValueError('partially populated CSIQ label row')
                continue  # spreadsheet padding, not a labeled stimulus
            name, level = str(r['image']), str(int(r['dst_lev']))
            folder, token = names[r['dst_type']]
            rows.append((root/f'{name}.png', root/folder/f'{name}.{token}.{level}.png',
                         folder, float(r['dmos']), 'distortion', float(r['dmos_std'])))
    elif dataset == 'live':
        import scipy.io
        labels = [root/'dmos_realigned.mat', root/'refnames_all.mat', root/'readme.txt']
        d = scipy.io.loadmat(labels[0])
        refs = scipy.io.loadmat(labels[1])['refnames_all'].flatten()
        values = {k:d[k].flatten() for k in ['dmos_new','dmos_std','orgs']}
        if any(len(v)!=982 for v in [refs,*values.values()]):
            raise ValueError('LIVE release-2 arrays must contain 982 entries')
        offset = identities = 0
        for folder,count in [('jp2k',227),('jpeg',233),('wn',174),('gblur',174),('fastfading',174)]:
            for k in range(count):
                idx = offset+k
                if values['orgs'][idx] == 1:
                    identities += 1
                    continue  # explicitly labeled reference copies, not distorted stimuli
                if values['orgs'][idx] != 0:
                    raise ValueError('invalid LIVE identity marker')
                rows.append((root/'refimgs'/str(refs[idx][0]), root/folder/f'img{k+1}.bmp',
                             folder, float(values['dmos_new'][idx]), 'distortion',
                             float(values['dmos_std'][idx])))
            offset += count
        if identities != 203:
            raise ValueError('LIVE identity exclusion count differs from release 2')
    elif dataset == 'pipal':
        labels = sorted((root/'Train_Label').glob('*.txt'))
        index = {}
        for i in range(1,5):
            for p in (root/f'Distortion_{i}').glob('*.bmp'):
                if p.name in index:
                    raise ValueError(f'duplicate PIPAL distortion {p.name}')
                index[p.name] = p
        for label in labels:
            for name, target in csv.reader(io.StringIO(label.read_text())):
                rows.append((root/'Train_Ref'/(label.stem+'.bmp'), index[name],
                             'distortion_group_'+Path(name).stem.split('_')[1],
                             float(target), 'quality', None))
    else:
        raise ValueError(f'unknown corpus {dataset}')
    return rows, labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=EXPECTED)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--build-commit', required=True)
    p.add_argument('--destination-root', type=Path, help='future staged image root in output paths')
    p.add_argument('--labels-only', action='store_true')
    args = p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.build_commit):
        p.error('build commit must be a full lowercase Git SHA')
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'progress.log').open('x',buffering=1) as log:
        def report(message):
            print(message,flush=True); print(message,file=log,flush=True)
        records_in, labels = records(args.dataset,root)
        if len(records_in)!=EXPECTED[args.dataset]:
            raise ValueError(f'expected {EXPECTED[args.dataset]} pairs, got {len(records_in)}')
        methods = {}
        if args.dataset == 'aic3':
            for r in csv.DictReader(io.StringIO(labels[0].read_text())):
                if any(r.values()):
                    name=f"{r['codec']}_{r['img.name']}_{r['quality']}.png"
                    methods[name]=r['method']
        rows, images, seen = [], {}, set()
        for i,(reference,distorted,family,target,direction,sigma) in enumerate(records_in):
            ref,dist = reference.relative_to(root), distorted.relative_to(root)
            if str(dist) in seen or not math.isfinite(target):
                raise ValueError('duplicate pair or nonfinite label')
            if sigma is not None and (not math.isfinite(sigma) or sigma<0):
                raise ValueError('invalid subjective standard deviation')
            seen.add(str(dist))
            paths=[]
            for rel in [ref,dist]:
                source=root/rel
                if str(rel) not in images:
                    if args.labels_only:
                        images[str(rel)]={'verification':'pending'}
                    else:
                        from cid22_manifest import audit
                        dimensions=audit(source)
                        images[str(rel)]={'sha256':digest(source),'bytes':source.stat().st_size,
                                          'dimensions':list(dimensions)}
                paths.append(str((args.destination_root or root)/rel))
            if not args.labels_only and images[str(ref)]['dimensions']!=images[str(dist)]['dimensions']:
                raise ValueError(f'pair dimensions differ: {dist}')
            method=methods[dist.name] if methods else 'published'
            dataset=args.dataset+'_'+method if methods else args.dataset
            rows.append(dict(zip(FIELDS,(dataset,str(ref),family,str(dist),target,direction,*paths)),
                             sigma='' if sigma is None else sigma, label_method=method))
            if (i+1)%100==0: report(f'Prepared {i+1}/{len(records_in)} rows')
        with (args.output/'pairs.tsv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS+['sigma','label_method'],delimiter='\t')
            writer.writeheader();writer.writerows(rows)
        (args.output/'labels').mkdir()
        label_provenance={}
        for label in labels:
            rel=label.relative_to(root);dst=args.output/'labels'/rel
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(label,dst)
            label_provenance[str(rel)]=digest(label)
        manifest=dict(build_commit=args.build_commit,dataset=args.dataset,n_pairs=len(rows),
                      n_sources=len({r['source'] for r in rows}),root=str(root),
                      destination_root=str(args.destination_root or root),labels=label_provenance,
                      target_units='native raw labels; sigma in the same units',
                      label_method_counts={k:sum(r['label_method']==k for r in rows) for k in sorted({r['label_method'] for r in rows})},
                      status='labels-only' if args.labels_only else 'images-audited',
                      images=images,pairs_sha256=digest(args.output/'pairs.tsv'))
        (args.output/'_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        report('Complete: '+manifest['status'])


if __name__=='__main__':
    main()
