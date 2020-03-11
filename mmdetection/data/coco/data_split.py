'''
data_split.py
该文件为目标检测网络划分COCO-like数据集至tran.json和val.json。

cd <ROOT>
作者：Hongfeng Ai
创建时间：2020-02-26
'''
import json
import os
import collections

# # 验证集占比
# val_pert = 0.15

# data_prepare_1.py 得到的tranval.json路径
trainval_path = 'data/annotations/trainval.json'

with open(trainval_path, 'r') as json_f:
    trainval_json = json.load(json_f)

# categories = trainval_json['categories']

# 打印trainval.json内的数量信息
print("trainval.json:")
print("图片数量：%d" % len(trainval_json['images']))
print("标注数量：%d" % len(trainval_json['annotations']))
print("类别数量：%d" % len(trainval_json['categories']))
for cat in trainval_json['categories']:
    print(cat['name'], cat['id'])
print("="*50)

# 图像库图片数量和标注数量
img_num = 0
img_ann_num = 0
# 视频库切片数量和标注数量
frame_num = 0
frame_ann_num = 0

for img in trainval_json['images']:
    if img['file_name'].startswith('i'):
        img_num += 1
    else:
        frame_num +=1

for ann in trainval_json['annotations']:
    if int(ann['image_id']) <= 200323:
        img_ann_num += 1
    else:
        frame_ann_num += 1 

print("图像库图片数量:%d 标注数量:%d" % (img_num, img_ann_num))
print("视频库切片数量:%d 标注数量:%d" % (frame_num, frame_ann_num))
# # 统计tranval.json内各类的标注数量
# cls_num_dict = collections.OrderedDict()
# for c in categories:
#     cls_name = c['name']
#     cls_id = str(c['id'])

#     for ann in trainval['annotations']:
#         if str(ann['category_id']) == cls_id and cls_id not in list(cls_num_dict.keys()):
#             cls_num_dict[str(cls_id)] = 1
#         elif str(ann['category_id']) == cls_id and cls_id in list(cls_num_dict.keys()):
#             cls_num_dict[str(cls_id)] += 1

# # 记录验证集中各类别数量
# val_cls_num_dict = collections.OrderedDict()

# # 打印tranval.json内各类的标注数量信息
# print("tranval.json内各类的标注数量信息(category_id:num)：")
# for i in list(cls_num_dict.keys()):
#     print(str(i) + ': ', cls_num_dict[str(i)])

#     # 计算验证集中各类别数量
#     val_num = int(cls_num_dict[str(i)] * val_pert)
#     val_cls_num_dict[str(i)] = val_num

# print("="*30)
# print("val.json内各类的标注数量上限(category_id:num)：")
# for i in list(val_cls_num_dict.keys()):
#     print(str(i) + ': ', val_cls_num_dict[str(i)])


# # 开始划分数据集
# val_images = []
# train_images = []
# val_annotations = []
# train_annotations = []

# for cls_id in list(val_cls_num_dict.keys()):
    
