import json

DATA_TYPES = ['trainval']#, 'val', 'train'

for t in DATA_TYPES:
    with open(t + '.json', 'r') as f:
        anns = json.load(f)
    
    for i, a in enumerate(anns['annotations']):
        anns['annotations'][i]['iscrowd'] = 0
        anns['annotations'][i]['segmentation'] = []
        anns['annotations'][i]['area'] = int(a['bbox'][2] * a['bbox'][3])
        anns['annotations'][i]["id"] = i
    
    with open(t + '_extra_key.json', 'w') as f1:
        json.dump(anns, f1)
    
