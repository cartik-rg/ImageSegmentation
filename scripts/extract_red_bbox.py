from PIL import Image
import os
import json

img_dir = 'input/image'
ann_dir = 'input/annotations'

images = [f for f in os.listdir(img_dir) if f.startswith('damaged-propeller') and f.lower().endswith('.jpg')]

for img_name in images:
    img_path = os.path.join(img_dir, img_name)
    im = Image.open(img_path).convert('RGB')
    w,h = im.size
    pixels = im.load()
    red_pixels = []
    for y in range(h):
        for x in range(w):
            r,g,b = pixels[x,y]
            if r>200 and g<100 and b<100:
                red_pixels.append((x,y))
    if not red_pixels:
        print(f'No red pixels found in {img_name}, skipping')
        continue
    xs = [p[0] for p in red_pixels]
    ys = [p[1] for p in red_pixels]
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    bbox = {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}

    base = os.path.splitext(img_name)[0]
    ann_name = base + '.json'
    ann_path = os.path.join(ann_dir, ann_name)
    if os.path.exists(ann_path):
        with open(ann_path,'r',encoding='utf-8') as f:
            ann = json.load(f)
    else:
        ann = {
            'image_id': base,
            'file_name': img_name,
            'source': {
                'aircraft_ontology': 'ontology/aircraft-ontology.json',
                'damage_ontology': 'ontology/aircraft-damage-ontology.json'
            },
            'annotations': []
        }
    # Choose labels and damage types heuristically by filename
    if base.endswith('001-boxed'):
        label = 'propeller impact damage'
        damage = 'ImpactDamage'
        damage_sub = 'ForeignObjectDamage'
    elif base.endswith('002-boxed'):
        label = 'propeller broken blade'
        damage = 'ImpactDamage'
        damage_sub = 'ForeignObjectDamage'
    elif base.endswith('003-boxed'):
        label = 'bent propeller blade'
        damage = 'DeformationDamage'
        damage_sub = 'Bending'
    else:
        label = 'propeller gouge'
        damage = 'SurfaceDamage'
        damage_sub = 'Gouge'

    ann_entry = {
        'label': label,
        'object': {
            'id': 'Engine',
            'iri': 'http://www.robbins-gioia.com/aircraft-ontology#Engine',
            'source': 'aircraft-ontology'
        },
        'part_of': {
            'id': 'FanBlade',
            'iri': 'http://www.robbins-gioia.com/aircraft-ontology#FanBlade',
            'source': 'aircraft-ontology'
        },
        'damage': {
            'id': damage,
            'iri': f'http://www.robbins-gioia.com/aircraft-damage-ontology#{damage}',
            'source': 'aircraft-damage-ontology'
        },
        'damage_subtype': {
            'id': damage_sub,
            'iri': f'http://www.robbins-gioia.com/aircraft-damage-ontology#{damage_sub}',
            'source': 'aircraft-damage-ontology'
        },
        'bbox': bbox
    }

    ann['annotations'] = [ann_entry]
    with open(ann_path,'w',encoding='utf-8') as f:
        json.dump(ann,f,indent=2)
    print(f'Wrote {ann_path} with bbox {bbox}')
