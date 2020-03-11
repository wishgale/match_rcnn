import json
from tqdm import tqdm
import random
import os
import copy
import torch.optim as optim
import torch.nn as nn
import torch
import cv2
import numpy as np
import time

import mmcv
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector, build_roi_extractor
from mmcv.runner import load_checkpoint
from mmdet.core import bbox2roi

config_path = 'configs/baseline_config.py' # model cfg
checkpoint_path = 'mmdet/models/pretrained_models/epoch_3.pth' # clothing-pretrained object detection model
train_path = 'pos_pair_dict/train_pos_pair_dict.json' # 'train.json'-like file for match network
val_path = 'pos_pair_dict/val_pos_pair_dict.json' #  'val.json'-like  file for match network
img_meta_path = 'img_meta/' # meta data folder


def init_config():
    '''Init configuration'''    
    # load configuration
    cfg = mmcv.Config.fromfile(config_path)
    return cfg

def get_model(config, max_num=1000):
    '''build the object detection model and load checkpoint
    Args:
        1. config      : the cfg from init_config().
        2. max_num  : the maximum proposals of RPN, default = 1000 
                      ,which means the 1000 proposals are reseverd.
    Return: model and cfg
    '''
    # build model
    config.test_cfg['rpn']['max_num'] = max_num
    od_model = build_detector(config.model, train_cfg=None, test_cfg=config.test_cfg)
    # load checkpoint
    checkpoint = load_checkpoint(od_model, checkpoint_path,  map_location='cpu')
    return od_model, config

def get_data_loader(config, ann_file_path):
    '''get data loader'''
    config.data.test['ann_file'] = ann_file_path
    dataset = build_dataset(config.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=config.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    return data_loader

def get_roi_feat(config, img_meta, img, model):
    '''Get a 7x7x256 ROI feature from object detection model
    Args:
        1. config     : the cfg.
        2. img_meta   : the image meta data from pre-prepared img_meta database.
        3. img        : the image tensor.
        4. model      : the object detection model.
        
    Return: 
    '''
    # Input should be Tensor, output a tuple with 4 tensor.
    # feat[0-3].shape:
    # 1 256 336 192
    # 1 512 168 96
    # 1 1024 84 48
    # 1 2048 42 24
    feat = od_model.backbone(img) # ResNet -> tuple
    
    # Input is a tuple，output a tuple with 5 tensor
    # fpn_feat[0-4].shape:
    # 1 256 336 192
    # 1 256 168 96
    # 1 256 84 48
    # 1 256 42 24
    # 1 256 21 12 # the feature is usefulless
    fpn_feat = model.neck(feat) # FPN -> tuple
    
    # Input is a tuple, output is a tuple(loss_rpn_cls,loss_rpn_bbox)
    # loss_rpn_cls, loss_rpn_bbox
    # 1 3/12 336 192
    # 1 3/12 168 96
    # 1 3/12 84 48
    # 1 3/12 42 24
    # 1 3/12 21 12
    rpn_outs = model.rpn_head(fpn_feat) # RPN -> (loss_rpn_cls, loss_rpn_bbox)

    # get 'max_num' proposals for each feature maps with different scales. proposal_list[0].shape -> torch.Size(['max_num', 5])
    # eg. >>proposal_list[0][0]
    # tensor([559.7507,  90.6900, 579.0510, 120.9980, 0.6428]) which is (xmin, ymin, xmax, ymax) and score
    proposal_cfg =config.test_cfg.get('rpn_proposal', config.test_cfg.rpn)
    proposal_cfg['max_num'] = 16 # we only want the proposals with the top 16 score
    proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
    proposal_list = model.rpn_head.get_bboxes(*proposal_inputs)    

    # convert proposals to rois
    proposals = proposal_list[0].cuda()
    rois = bbox2roi([proposals])

    # the ROI features after ROIAlign. roi_feats.size() -> 'max_num'x256x7x7
    fpn_feat = [fpn_feat[0].cuda(),fpn_feat[1].cuda(),fpn_feat[2].cuda(),fpn_feat[3].cuda(),fpn_feat[4].cuda()]
    roi_feats = model.bbox_roi_extractor(
        fpn_feat[:len(model.bbox_roi_extractor.featmap_strides)],rois)

    # random select a roi_feats
    rand_idx = random.sample(range(proposal_cfg['max_num']), 1)
    roi_feat = roi_feats[rand_idx,:,:,:].expand(1,-1,-1,-1) # 1x256x7x7

    return roi_feat

def get_img(img_meta):
    img = cv2.imread(img_meta[0]['filename'])

    # Resize
    sf = img_meta[0]['scale_factor']
    img, scale_factor = mmcv.imrescale(img, sf, True)

    # Normalize
    m = img_meta[0]['img_norm_cfg']['mean']
    s = img_meta[0]['img_norm_cfg']['std']
    t = img_meta[0]['img_norm_cfg']['to_rgb']
    img = mmcv.imnormalize(img, m, s, t)

    # Pad
    sd = 32 # size_divisor
    img = mmcv.impad_to_multiple(img, 32, 0)

    # H x W x C -> C x H x W and expand an dim
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).expand(1,-1,-1,-1)
    
    return img

# define the match model
class MatchModel(nn.Module):
    def __init__(self):
        super(MatchModel, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 1024, 3, padding=1)
        self.fc1 = nn.Linear(1024*7*7, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)
        #self.sigmoid = nn.Sigmoid()
        #self.logsoftmax = nn.LogSoftmax 
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn_init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif m == self.fc3:
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) 
            else:
                nn_init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)            
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x1 = x1.view(x1.size(0),-1)
        x2 = x1.view(x1.size(0),-1)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x2 = self.fc1(x2)
        x2 = self.fc2(x2)
        x = x1 - x2
        x = x * x
        x = self.fc3(x)
        #x = self.sigmoid(x)
        #x = self.logsoftmax(x)
        return x

# instance a match model
m_model = MatchModel().cuda()
print("The Match Model Structure:")
print(m_model)


with open(train_path, 'r') as train_f:
    train_pos_pair_dict = json.load(train_f)


with open(val_path, 'r') as val_f:
    val_pos_pair_dict = json.load(val_f)


cfg = init_config()
od_model, cfg = get_model(cfg, 16)
data_loader = get_data_loader(cfg, 'data/coco/annotations/train.json')

# get all image names from taining dataset
all_img_names = list(train_pos_pair_dict.keys())

# Model Setting
EPOCH_NUM = 12
criterion = nn.CrossEntropyLoss()#nn.BCELoss() # binary cross-entropy loss
optimizer = optim.SGD(m_model.parameters(), lr=0.0025, momentum=0.9)

print("Start training...")
all_start_time = time.time()
for epoch in range(EPOCH_NUM):
    running_loss = 0.0
    for i, data in enumerate(data_loader):
        start_time = time.time()
        i += 1 # the ith image iteration
        img1 = data['img'][0].data
        img1_meta = data['img_meta'][0].data[0]
        img1_name = img1_meta[0]['filename'].split('/')[-1]

        # get a 1x256x7x7 ROI feature
        roi_feat1 = get_roi_feat(cfg, img1_meta, img1, od_model)

        # get images which can match with img1
        try:
            pos_pair_imgs = train_pos_pair_dict[img1_name]
        except KeyError:
            continue 
        pos_pair_num = len(pos_pair_imgs) # the number of positive-available images
        roi_feat1 = roi_feat1.detach() # cut down ob_model gradient backward
        
        # train the positve pairs
        for img2_name in pos_pair_imgs:
            img2_meta = torch.load(img_meta_path + img2_name[:-4] + '.pth')
            img2 = get_img(img2_meta)
            roi_feat2 = get_roi_feat(cfg, img2_meta, img2, od_model)
            
            roi_feat2 = roi_feat2.detach() # cut down ob_model gradient backward
            # zero the parameter gradients
            optimizer.zero_grad()  

            output = m_model(roi_feat1, roi_feat2).cuda()
            label = torch.tensor([1.0],dtype=torch.long).cuda()

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # train the negative pairs
        neg_pair_ratio = 1 # determine the ratio of negative pairs / positive pairs
        neg_pair_num = pos_pair_num * neg_pair_ratio 
        neg_pair_imgs = []
        neg_count = 0
        while neg_count <= neg_pair_num:
            neg_rand_name = random.sample(all_img_names,1)[0]
            if neg_rand_name not in pos_pair_imgs and neg_rand_name != img1_name:
                neg_count += 1
                neg_pair_imgs.append(neg_rand_name)
        for img2_name in neg_pair_imgs:
            img2_meta = torch.load(img_meta_path + img2_name[:-4] + '.pth')
            img2 = get_img(img2_meta)
            roi_feat2 = get_roi_feat(cfg, img2_meta, img2, od_model)
            roi_feat2 = roi_feat2.detach()
            # zero the parameter gradients
            optimizer.zero_grad()  
            output = m_model(roi_feat1, roi_feat2)
            label = torch.tensor([0.0],dtype=torch.long).cuda()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        end_time = time.time()
        print('[Epoch:%d, Iter(img):%d, Pos_pair_num:%d, Neg_pair_num:%d] loss: %.4f, time(s): %d' %
                 (epoch + 1, i, pos_pair_num, neg_pair_num, running_loss / (pos_pair_num + neg_pair_num), end_time - start_time))
        running_loss = 0.0
        torch.save(m_model, 'mm_outputs/checkpoint_' + str(epoch+1) + '.pth')
all_end_time = time.time()
print('Finished Training by using %d mins' % (all_end_time - all_start_time) / 3600 )

