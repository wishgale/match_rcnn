import mmcv
import os
import cv2
from tqdm import tqdm

imgdir = 'data/Live_demo_20200117/image'
anndir = 'data/Live_demo_20200117/image_annotation'

imglist = os.listdir(imgdir)
imglist.sort()
annlist = os.listdir(anndir)
annlist.sort()

for imgname_cls in tqdm(imglist):
    imgdir_cls = os.listdir(imgdir+'/'+imgname_cls)   # 每一类有几张图
    imgdir_cls.sort(key=lambda x: int(x.replace(".jpg", "")))
    for imgname in imgdir_cls:
        imgreaddir = imgdir+'/'+imgname_cls + '/' + imgname
        img = mmcv.imread(imgreaddir)
        annreaddir =imgreaddir.replace("image","image_annotation").replace(".jpg", ".json")
        ann = mmcv.load(annreaddir)
        for anns in ann["annotations"]:
            if anns["instance_id"]>=0:
                x,y,x2,y2 = anns["box"]
                cv2.rectangle(img, (x, y), (x2,y2), (0, 0, 255), 3)

        savedir = imgreaddir.replace("image","result")
        mmcv.imwrite(img,savedir)
