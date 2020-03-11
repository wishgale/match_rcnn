import torch
import numpy as np

num_classes = 25
model_coco = torch.load(r"/media/alvinai/Documents/model/faster_rcnn_r50_fpn_1x_20190610-bf0ea559.pth")
# print(model_coco)
# print(model_coco["state_dict"]["rpn_head.rpn_cls.weight"].shape)
# a = model_coco["state_dict"]["rpn_head.rpn_cls.weight"][0]
# model_coco["state_dict"]["rpn_head.rpn_cls.weight"]=np.insert(model_coco["state_dict"]["rpn_head.rpn_cls.weight"], 0, values=a, axis=0)
# print(model_coco["state_dict"]["rpn_head.rpn_cls.weight"].shape)
# b=model_coco["state_dict"]["rpn_head.rpn_cls.bias"][0]
# model_coco["state_dict"]["rpn_head.rpn_cls.bias"] = np.insert(model_coco["state_dict"]["rpn_head.rpn_cls.bias"], 0, values=b, axis=0)
# print(model_coco["state_dict"]["rpn_head.rpn_cls.bias"].shape)
# c= model_coco["state_dict"]["rpn_head.rpn_reg.weight"][0].repeat(4,1,1,1)
# model_coco["state_dict"]["rpn_head.rpn_reg.weight"]=np.insert(model_coco["state_dict"]["rpn_head.rpn_reg.weight"], 0, values=c, axis=0)
# # c= model_coco["state_dict"]["rpn_head.rpn_reg.weight"][1]
# # model_coco["state_dict"]["rpn_head.rpn_reg.weight"]=np.insert(model_coco["state_dict"]["rpn_head.rpn_reg.weight"], 0, values=c, axis=0)
# # c= model_coco["state_dict"]["rpn_head.rpn_reg.weight"][2]
# # model_coco["state_dict"]["rpn_head.rpn_reg.weight"]=np.insert(model_coco["state_dict"]["rpn_head.rpn_reg.weight"], 0, values=c, axis=0)
# # c= model_coco["state_dict"]["rpn_head.rpn_reg.weight"][3]
# # model_coco["state_dict"]["rpn_head.rpn_reg.weight"]=np.insert(model_coco["state_dict"]["rpn_head.rpn_reg.weight"], 0, values=c, axis=0)
# print(model_coco["state_dict"]["rpn_head.rpn_reg.weight"].shape)
# d=model_coco["state_dict"]["rpn_head.rpn_reg.bias"][0].repeat(4,)
# model_coco["state_dict"]["rpn_head.rpn_reg.bias"] = np.insert(model_coco["state_dict"]["rpn_head.rpn_reg.bias"], 0, values=d, axis=0)
# print(model_coco["state_dict"]["rpn_head.rpn_reg.bias"].shape)
# # model_coco["state_dict"]["rpn_head.rpn_reg.weight"] = model_coco["state_dict"]["rpn_head.rpn_reg.weight"].repeat(2,1,1,1)
# # model_coco["state_dict"]["rpn_head.rpn_reg.bias"] = model_coco["state_dict"]["rpn_head.rpn_reg.bias"].repeat(2,)
# weight
model_coco["state_dict"]["bbox_head.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.fc_cls.weight"][
                                                        :num_classes, :]
model_coco["state_dict"]["bbox_head.fc_reg.weight"] = model_coco["state_dict"]["bbox_head.fc_reg.weight"][
                                                        :num_classes*4, :]
# bias
model_coco["state_dict"]["bbox_head.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.fc_cls.bias"][:num_classes]

model_coco["state_dict"]["bbox_head.fc_reg.bias"] = model_coco["state_dict"]["bbox_head.fc_reg.bias"][:num_classes*4]
# save new model
torch.save(model_coco, r"/media/alvinai/Documents/underwater/model/libra_faster_rcnn_r50_fpn_1x_cls_%d.pth" % num_classes)
