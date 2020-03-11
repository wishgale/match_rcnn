import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
import random

@HEADS.register_module
class MatchNNBBoxHead(nn.Module):
    """BBox Head for Match Network, with shared conv and fc layers and other operations like below.
                                                               /-> feats \
    random roi selection -> shared convs * 2  -> shared fcs * 2           sub(-) -> square -> match fc -> match score
                                                               \-> feats /
    参数解释：
        - with_avg_pool:        是否在convs后接average pooling。
        #- with_instanceid:      是否有instance_id，默认True。
        - roi_feat_size:        输入的ROIAlign后的特征图尺寸，默认7x7。
        - in_channels:          输入的ROIAlign后的特征图维度，默认256
        #- num_classes:          类别数量（包括背景）。
        - target_means:         bbox的平均值。
        - targt_stds:           bbox的标准差。
        #- reg_class_agnostic:   回归类别不可知。默认False，则回归类别可知。
        #- num_shared_convs:     共享分支下convs数量，默认2。
        #- num_shared_fcs:       共享分支下fc数量，默认2。
        - conv_cfg & norm_cfg:  建立Conv模块用，默认None。
        # - conv_out_channels:    conv输出的维度。
        - fc_out_channels:      fc输出的维度。             
        - loss_match:           匹配损数 PairLoss。
        
    """

    def __init__(self,
                with_avg_pool=False,
                #with_instanceid=True,
                roi_feat_size=7,
                in_channels=256,
                #num_classes=24,
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2],
                #reg_class_agnostic=False,
                # num_shared_convs=2,
                # num_shared_fcs=2,
                conv_cfg=None,
                norm_cfg=None,
                # conv_out_channels=256,
                fc_out_channels=1024,
                loss_match=dict(
                    type="PairLoss")):
        super(MatchNNBBoxHead, self).__init__()
        assert with_instanceid
        self.with_avg_pool = with_avg_pool
        #self.with_instanceid = with_instanceid
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        #self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        #self.reg_class_agnostic = reg_class_agnostic\
        # self.num_shared_convs = num_shared_convs
        # self.num_shared_fcs = num_shared_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.loss_match = build_loss(loss_match)

        # in_channels输入维度为256
        in_channels = self.in_channels
        
        if self.with_avg_pool:
            # 7x7 kernel size的池化，in_channels为256
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        
        # 定义共享支路的各模块
        self.shared_conv_1 = ConvModule(in_channels, 256，3, padding=1,
                                     conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.shared_conv_2 = ConvModule(256, 1024，3, padding=1,
                                     conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.shared_fc_1 = nn.Linear(1024)
        self.shared_fc_2 = nn.Linear(256)
        
        # 定义平方特征差之后的fc（即match_fc）
        # 最后输出2个，即matching score和not matching score
        self.match_fc = nn.Linear(256, 2)

        
    def init_weights(self):
        # 初始化共享支路上fc模块们的权重
        nn_init.xavier_uniform_(self.shared_fc_1.weight)
        nn.init.constant_(self.shared_fc_1.bias, 0)
        nn_init.xavier_uniform_(self.shared_fc_2.weight)
        nn.init.constant_(self.shared_fc_2.bias, 0)
        # 初始化match_fc权重
        nn.init.normal_(self.match_fc.weight, 0, 0.01)
        nn.init.constant_(self.match_fc.bias, 0)

    def forward(self, x):
        # x输入维度：1024×256x7x7
        # 随机取一个proposal feature
        x = random.sample(range(x.size(0)), 1)

    