'''
prepare_img_meta.py

This .py file is to prepare four annotation files:
    1. 'img_meta' : /img_meta/<filename>.pth
    2. trainval_pos_dict.json：/trainval_pos_dict.json.json
    3. train_pos_dict.json：/train_pos_dict.json.json
    4. val_pos_dict.json：/val_pos_dict.json.json

'''

import os
import os.path as osp
import pickle
import shutil
import tempfile
from tqdm import tqdm
import json
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model, build_assigner, build_sampler, bbox2roi
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector, build_roi_extractor
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random

config_path = 'configs/baseline_config.py' 
checkpoint_path = 'mmdet/models/pretrained_models/faster_rcnn_r50_fpn_1x_cls_24.pth'

def init_config():
    '''Init configuration'''    
    # load configuration
    cfg = mmcv.Config.fromfile(config_path)
    return cfg

def save_pos_pair(save_path, ann_file_path):
    '''save file_name' and 'file_name's pair 
    {
        'i_123_1321.jpg':[i_124_131.jpg, i_213_213.jpg, v_213_1231,jpg ...],
        '1243412':[],
        ...
    }
    '''
    print("Prepare pos_pair_dict:")
    pos_pair_dict = {}
    
    # load trainval.json
    with open(ann_file_path, 'r') as f:
        anns = json.load(f)
    
    # init the keys in the dict
    for img_info in anns['images']:
        pos_pair_dict[img_info['file_name']] = []
    
    # update the dict
    for k in tqdm(list(pos_pair_dict.keys())):
        
        # get k image's image_id 
        for img_info in anns['images']:
            if img_info['file_name'] == k:
                img_id = img_info['id']
                break
        
        # get k image's instance_id
        for ann_info in anns['annotations']:
            if ann_info['image_id'] == img_id:
                instance_id = ann_info['instance_id']
                break
        
        # get postive image ids
        pos_img_ids = []
        for ann_info in anns['annotations']:
            # exclude the postive-pair image with same image id or unmatch instance_id
            if ann_info['instance_id'] == instance_id and ann_info['image_id'] != img_id and ann_info['instance_id'] != 0:
                pos_img_ids.append(ann_info['image_id'])
        
        # get postive image file_name
        pos_img_fns = []
        for pii in pos_img_ids:
            for img_info in anns['images']:
                if img_info['id'] == pii and img_info['file_name'] not in pos_img_fns:
                    pos_img_fns.append(img_info['file_name'])
        
        # update
        pos_pair_dict[k] = pos_img_fns
    

    with open(save_path, 'w') as save_f:
        json.dump(pos_pair_dict, save_f)
    print("pos_pair_dict is saved to ", save_path)
    
    return pos_pair_dict

def save_all_img_meta(cfg, ann_file_path):
    '''
    Extract all 'img_meta' from trainval.json and save them.
    
    'img_meta' is a [dict]:
        [{'filename': 'data/coco/images/v_033370_280.jpg',
         'ori_shape': (960, 540, 3),
         'img_shape': (1333, 750, 3),
         'pad_shape': (1344, 768, 3),
         'scale_factor': 1.3885416666666666,
         'flip': False,
         'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32),
         'to_rgb': True}}]
    '''
    cfg.data.test['ann_file'] = ann_file_path   
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    # fns_txt = open('./all_img_fns.txt', 'w') # save all image filenames
    
    # len(data_loaders) = batchsize
    # batchsize = 585112, since we specific imgs_per_gpu = 1 that is each batch input 1 image
    print("Starting extract and save all 'img_meta' and prepare all_img_fns.txt")
    for d in tqdm(data_loader):
        data = d
        # get image meta data
        img_meta = data['img_meta'][0].data[0]
   
        # save img_meta
        fn = img_meta[0]['filename'].split('/')[-1][:-4]
        torch.save(img_meta, './img_meta/' + str(fn) + '.pth')
    
    print("'img_meta' are saved to ./img_meta/<filename>.pth")

def pos_pair_statistic(pos_pair_dict_path):
    '''
    display the frequency of positive-paired images of each image
    '''
    with open(pos_pair_dict_path, 'r') as f:
        pos_pair_dict = json.load(f)

    freq_pos_dict = {}
    for k in list(pos_pair_dict.keys()):
        if len(pos_pair_dict[k]) not in list(freq_pos_dict.keys()):
            freq_pos_dict[len(pos_pair_dict[k])] = 1
        else:
            freq_pos_dict[len(pos_pair_dict[k])] += 1
            # if len(pos_pair_dict[k]) == 21:
            #     print(k,':', pos_pair_dict[k])

    fig = plt.figure(figsize=(12,12))
    x = list(freq_pos_dict.keys())
    y = list(freq_pos_dict.values())
    plt.bar(x, y)
    for a, b in zip(x,y):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

    plt.xticks(x)
    plt.xlabel("The number of positive-paired images")
    plt.ylabel("Frequency of images")
    plt.show()

def split_data(pos_pair_dict_path, train_ratio):
    ''' split data for training and validating matchrcnn model'''
    with open(pos_pair_dict_path, 'r') as f:
        pos_pair_dict = json.load(f)

    # get match-avaible images
    ma_images = []
    for k in list(pos_pair_dict.keys()):
        if pos_pair_dict[k] == []:
            pass
        else:
            ma_images.append(k)

    # get the training number
    train_num = int(len(ma_images) * 0.85)
    train_idxs = random.sample(range(len(ma_images)), train_num)

    # form the train.json
    train = {}
    for i in train_idxs:
        imgn = ma_images[i]
        train[imgn] = pos_pair_dict[imgn]

    with open('train_pos_pair_dict.json', 'w') as train_f:
        json.dump(train, train_f)

    print("The train annotation file for match rcnn is saved to '/train_pos_pair_dict.json'")

    # form the val.json
    val = {}
    for j, k in enumerate(ma_images):
        if k not in list(train.keys()):
            val[k] = pos_pair_dict[k]
        print(str(j+1),'/',str(len(ma_images)))
    with open('val_pos_pair_dict.json', 'w') as val_f:
        json.dump(val, val_f)
    print("The val annotation file for match rcnn is saved to '/val_pos_pair_dict.json'")

cfg = init_config()
pos_pair_dict = save_pos_pair('trainval_pos_pair_dict.json', 'data/coco/annotations/trainval.json')   
save_all_img_meta(cfg, 'data/coco/annotations/trainval.json')
pos_pair_statistic('trainval_pos_pair_dict.json')
split_data('trainval_pos_pair_dict.json', 0.85)

