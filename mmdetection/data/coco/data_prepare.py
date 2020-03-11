'''
data_prepare_1.py
该文件为目标检测网络准备COCO-like数据集。

步骤：
 1. 做图片库图片标注文件。
 2. 做视频库直播切片标注文件。

cd <ROOT>
作者：Hongfeng Ai
创建时间：2020-02-26
'''

import json
import glob
import os
import cv2
import collections
from tqdm import tqdm

# 导入路径
data_rpath = 'data/'
data_paths = glob.glob(data_rpath + '*')

img_paths = [] # 所有图片的路径
for p in data_paths:
    img_paths.extend(glob.glob(p + '/image/*/*.jpg'))


# 设置保存路径
img_spath = 'data/images/'
anns_spath = 'data/annotations/'
video_img_spath = 'data/video_images/'
if not os.path.exists(img_spath):
    os.mkdir(img_spath)
if not os.path.exists(anns_spath):
    os.mkdir(anns_spath)
if not os.path.exists(video_img_spath):
    os.mkdir(video_img_spath)

# 23类类别信息（其中'古风'与'古装'同为第2类）
CLASS_DICT = collections.OrderedDict({
'短外套':1,
'古风':2, '古装':2,
'短裤':3,
'短袖上衣':4, 
'长半身裙':5, 
'背带裤':6, 
'长袖上衣':7, 
'长袖连衣裙':8, 
'短马甲':9, 
'短裙':10, 
'背心上衣':11, 
'短袖连衣裙':12, 
'长袖衬衫':13, 
'中等半身裙':14, 
'无袖上衣':15, 
'长外套':16, 
'无袖连衣裙':17, 
'连体衣':18, 
'长马甲':19, 
'长裤':20, 
'吊带上衣':21, 
'中裤':22, 
'短袖衬衫':23})

# =======1. 对图像库数据集进行标注文件的准备=======
#{
# "images":[
#       {"file_name": i_img_name_item_id,
#       "id": 1,
#       "height": h,
#       "width": w}, ...
#       ]
#  "annotations":[
#       {"image_id":1, "bbox":[xmin,ymin,w,h], "category_id":1}, ...   
#       ]
#  "categories":[
#       {"id":1, "name":'短外套'}, ...
#       ]
#}
# ============================================
print("开始对图像库数据集进行标注文件的准备:")
images = []
annotations = []
categories = []
img_id = 0

# 更新categories
for k in list(CLASS_DICT.keys()):
    categories.append({"id": CLASS_DICT[k], "name":k})

for ip in tqdm(img_paths):
    img = cv2.imread(ip)
    h, w, _ = img.shape
    del img
    # 获取图像路径ip对应的标注路径ap
    ap = ip.replace('image', 'image_annotation')
    ap = ap.replace('jpg', 'json')

    with open(ap, 'r') as json_f:
        img_ann = json.load(json_f)

    # 若标注为空
    if len(img_ann['annotations']) == 0:
        pass
    # 若存在标注
    else:
        # 更新images
        file_name = 'i_' + str(img_ann['img_name'][:-4]) + '_' + str(img_ann['item_id'] + '.jpg')
        
        # # 保存图片至images文件夹
        # cv2.imwrite(img_spath + file_name, img) 
        # del img
        
        img_id += 1
        images.append({'file_name':file_name,
                        'id':img_id,
                        'height':h,
                        'width':w})
        # 更新annotations
        for ann in img_ann['annotations']:
            xmin = float(ann['box'][0])
            ymin = float(ann['box'][1])
            box_w = float(ann['box'][2] - ann['box'][0] + 1)
            box_h = float(ann['box'][3] - ann['box'][1] + 1)
            cls_id = CLASS_DICT[ann['label']]
            annotations.append({'image_id':img_id,
                                'bbox':[xmin, ymin, box_w, box_h],
                                'category_id':cls_id,
                                'instance_id':ann['instance_id']})

print('Finish preparing item images!')
print("Frame image starts ‘id' from ", img_id)

del img_paths
# =======2. 对视频库直播切片进行标注文件的准备=======
# 在图像库标注基础上追加内容即可
# {
# "images":[
#       {"file_name": v_video_id_frame_index,
#       "id": 1,
#       "height": h,
#       "width": w}, ...
#       ]
#  "annotations":[
#       {"image_id":1, "bbox":[xmin,ymin,w,h], "category_id":1}, ...   
#       ]
#  "categories":[
#       {"id":1, "name":'短外套'}, ...
#       ]
#}
# ============================================
video_paths = [] # 所有视频的路径
# video_ann_paths = [] # 所有视频标注的路径
for p in data_paths:
    video_paths.extend(glob.glob(p + '/video/*.mp4'))

print("开始对视频库直播切片进行标注文件的准备：")
def get_frame_img(video_path, frame_index):
    cap =  cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    _, frame_img = cap.read()
    cap.release()
    return frame_img

for vp in tqdm(video_paths):
    # 获取视频路径p对应的标注路径vap
    vap = vp.replace('video', 'video_annotation')
    vap = vap.replace('mp4', 'json')

    with open(vap, 'r') as json_f2:
        video_ann = json.load(json_f2)

    for frame in video_ann['frames']:
        # 如果单个frame下没有标注：
        if len(frame['annotations']) == 0:
            pass
        # 如果单个frame下有标注：
        else:
            frame_index = frame['frame_index']
            frame_img = get_frame_img(vp, frame_index)

            vh, vw, _ = frame_img.shape
            del frame_img
            # 更新images
            img_id += 1
            vfile_name = 'v_' + str(video_ann['video_id']) + '_' + str(frame_index) + '.jpg'
            images.append({'file_name':vfile_name,
                            'id':img_id,
                            'height':vh,
                            'width':vw})

            # # 保存图片至images文件夹
            # cv2.imwrite(video_img_spath + vfile_name, frame_img)   
            # del frame_img

            # 更新annotations
            for fann in frame['annotations']:
                fxmin = float(fann['box'][0])
                fymin = float(fann['box'][1])
                fbox_w = float(fann['box'][2] - fann['box'][0] + 1)
                fbox_h = float(fann['box'][3] - fann['box'][1] + 1)
                fcls_id = CLASS_DICT[fann['label']]
                annotations.append({'image_id':img_id,
                                    'bbox':[fxmin, fymin, fbox_w, fbox_h],
                                    'category_id':fcls_id,
                                    'instance_id':fann['instance_id']})

print('Finish preparing frame images!')

# ‘古装’和‘古风’合为‘古风’
new_categories = [categories[i] for i, cat in enumerate(categories) if cat['name'] != '古装']

# 保存标注至annotations文件夹
all_anns = {"images": images, "annotations":annotations, "categories":new_categories}
with open(anns_spath + 'trainval.json', 'w') as json_f3:
    json.dump(all_anns, json_f3)

print('Finish saving trainval.json')