import cv2
import os
import json
def cutvideo(videoPath,imageDirPath):
    name=str(videoPath).split('/')[-1].split('.')[0]
    vc = cv2.VideoCapture(videoPath)

    if vc.isOpened():
        with open(r"/media/alvinai/Documents/alitianchi/data/Live_demo_20200117/video_annotation/{}.json".format(name), 'r') as load_f:
            f = json.load(load_f)
        anns = f["frames"]
        index=0
        timeF = 40  # 采样间隔，每隔timeF帧提取一张图片
        c = 0
        s=0
        success = True
        for frame1 in anns:
            frame_index = frame1["frame_index"]
            bbox = frame1["annotations"]
            while (success):
                success, frame = vc.read()
                if (c % timeF == 0):
                    if s == frame_index :
                        for ann in bbox:
                            x1 = int(ann['box'][0])
                            y1 = int(ann['box'][1])
                            x2= int(ann['box'][2])
                            y2= int(ann['box'][3])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 8)
                            # frame = cv2.resize(frame,(640,360))
                        pigname=name+str(index) + '.jpg'
                        index = index + 1
                        imagePath = os.path.join(imageDirPath, str(pigname))
                        print(imagePath)
                        cv2.imwrite(imagePath, frame)
                    s=s+1
                c = c + 1
                cv2.waitKey(1)
        vc.release()
        print("open ok")
    else:
        success = False
        print("open error")
        exit(0)

root_path = r"/media/alvinai/Documents/alitianchi/data/Live_demo_20200117/video"#视频存放路径
imageDirPath = r"/media/alvinai/Documents/alitianchi/data/Live_demo_20200117/images"# 切片存放路径
indexes = [f for f in os.listdir(os.path.join(root_path))]
for i in indexes:
    videoPath = os.path.join(root_path, i)  # 获取视频
    cutvideo(videoPath,imageDirPath)
