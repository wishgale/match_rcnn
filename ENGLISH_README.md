## Tianchi: Taobao live product identification
[READ.md 中文版](README.md)   
Live broadcast with products is an important way for Taobao to connect products and consumers. Buyers buy their favorite products in the process of watching live broadcast. In a single live broadcast, the host often shows, tries and introduces hundreds of products. If the buyer wants to buy the products being explained, he needs to select them manually in the list of commodities (including hundreds of commodities) associated with the live broadcast, which greatly affects the purchase efficiency and user experience of the user. This project can automatically identify the currently explained products according to the live video screen and the host's explanation through the algorithm of product identification, and recommend the corresponding purchase link to the user, which will greatly improve the user's purchase experience. This competition requires the contestants to identify the products being explained in the video through artificial intelligence algorithms such as computer vision and natural language processing, so as to improve the purchase experience of users in Taobao live broadcast.  

More details, please visit：[Competition Web](https://tianchi.aliyun.com/competition/entrance/231772/introduction) [Competition Article](https://mp.weixin.qq.com/s?__biz=MzUyNzA1OTcxNg==&mid=2247483692&idx=1&sn=34c1737ab81e8d75246ef8dde5549732&chksm=fa041f47cd7396516b2efbceafb6baf26a272847667671b1510106537617d01a2412d71e197c&token=266731819&lang=zh_CN#rd)
### 1. Environment Installment  
Please install [mmdetection](https://github.com/open-mmlab/mmdetection) environment by your own。  
### 2. Data Preparation
The dataset preparation for model training follow the above steps：  
- ``data_prepare.py``: gain ``images`` and ``video_images``image folders which save product images and video-frame images。Also, the annotation file ``trainval.json`` is obtained。  
- ``add_keys.py``: assign ``id`` key to ``annotaions``。  
- ``create_val.py``: stratified split the dataset to ``train.json`` and ``val.json``。  
- ``prepare_img_meta.py``：return two files ``./img_meta/<fn>.pth``和``/pos_pair_dict.json``），They save metadata of each image，and the file names of positive-paired images for each imsge。 
```
python data_prepare.py
python add_keys.py
python create_val.py
python prepare_img_meta.py
```
Finally, the file structure will be sorted as below：  
```
├──| mmdetection
├────| data
├───────| coco
├──────────| annotations （coco-like format）
├──────────────| trainval.json
├──────────────| train.json
├──────────────| val.json
|───────| images (copy the images in ’images‘ and ‘video_images’ into this folder)
```
The above annotation file will have the below format：
```
{
  "images":
	[
	 {"file_name":"i_'img_name'_'item_id'.jpg", "id":1, "height":1000, "width":1000},
	 {"file_name":"v_'video_id'_'frame_index'.jpg", "id":2, "height":1000, "width":1000},
	 ...
	]
  "annotations":
	[
	 {'image_id': 1,
	 'bbox': [148.0, 74.0, 505.0, 661.0],
	 'category_id': 7,
	 'instance_id': 20004101,
	 'iscrowd': 0,
	 'segmentation': [],
	 'area': 333805,
	 'id': 0}，
	 
	 {'image_id': 2,
	 'bbox': [139.0, 68.0, 512.0, 584.0],
	 'category_id': 7,
	 'instance_id': 20004101,
	 'iscrowd': 0,
	 'segmentation': [],
	 'area': 299008,
	 'id': 1}
	 ...
	]
  "categories":
	[
	 {"id":1, "name":"长袖"}
	 ...
	]
}
```
``./img_meta/<fn>.pth`` saves metadata of each image, which is the metadata defined by the data_loader of mmdetection：  
```
	# 'img_meta' is a [dict]。
	[{'filename': 'data/coco/images/v_033370_280.jpg',
	 'ori_shape': (960, 540, 3),
	 'img_shape': (1333, 750, 3),
	 'pad_shape': (1344, 768, 3),
	 'scale_factor': 1.3885416666666666,
	 'flip': False,
	 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32),
	 'to_rgb': True}]
```
``/pos_pair_dict.json`` saves the postive-paired information：
```
# the keys in this dict include all images including 'instance_id=0', but the [positive-paired images] only involves images with 'instance_id>0'
    {'i_123_1321.jpg':[i_124_131.jpg, i_213_213.jpg, v_213_1231,jpg ...],
     'i_234_1241.jpg':[],
        ...}
```
23 classes：
```
# 23 classes（the class '古风' and '古装' are belong to the second class）， they are counted as ‘古风’.
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
```
### 3. Object Dection Model Training 
Before training, you should do：  
- Download the pretrained model file from [Model Zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md). Here we use the R-50-FPN-1x model file which is the same as Match R-CNN, and then we put it under ``mmdetection/mmdet/models/pretrained_models``. Last, we do the below operations：  
- **Modify the pretrained model weight file**：run ``cocopth.py``，to get a new file - ``faster_rcnn_r50_fpn_1x_cls_24.pth``。
- **Modify the model configuration file**：``mmdetection/configs/faster_rcnn_r50_fpn_1x.py`` is changed and renamed to ``baseline_config.py``。  
Start training the object detection model：
```
cd mmdetection
CUDA_VISIBLE_DEVICES=4,5 PORT=29501 tools/dist_train.sh configs/baseline_config.py 2 --validate --seed 123456
```

### 4. Matching Model Training
Change the content of ``load_from`` in ``baseline_config.py`` to point to the path of object-detection pretrained model file.
```
python tools/train_mm.py
```
### Modification:
1. ``match_rcnn/mmdetection/data/coco``：the codes for data preparation。  
2. ``match_rcnn/mmdetection/mmdet/core/evaluation/class_name.py``：change class information。  
3. ``match_rcnn/mmdetection/mmdet/datasets/coco.py``：change class information。  
4. ``match_rcnn/mmdetection/configs/baseline_config.py``: modify ``faster_rcnn_r50_fpn_1x.py``。  
5. ``match_rcnn/mmdetection/tools``：add ``prepare_img_meta.py`` and ``train_mm.py`` to prepare data and train matching network。  
