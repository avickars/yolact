from yolact import Yolact
from config import set_cfg, cfg
import cv2
import torch
from augmentations import FastBaseTransform
from output_utils import postprocess

set_cfg('yolact_plus_base_config')

cfg.mask_proto_debug = False

net = Yolact()

net.load_weights('yolact_plus_base_39_25000.pth')

net.eval()
net.detect.use_fast_nms = True
net.detect.use_cross_class_nms = False

frame = torch.from_numpy(cv2.imread("3dogs.jpg")).cuda().float()
w,h = frame.shape[1], frame.shape[0]

frame = torch.stack([frame, frame])

batch = FastBaseTransform()(frame)

net = net.cuda()
preds = net(batch)

output = []
for imageIndex, pred in enumerate(preds):
    save = cfg.rescore_bbox
    cfg.rescore_bbox = True
    classes, scores, bboxes, masks = postprocess(preds, w, h, batch_idx=imageIndex, visualize_lincomb=False,
                                crop_masks=True,
                                score_threshold=0.3)
    cfg.rescore_bbox = save

    output.append([])

    for detectionIndex in range(0, classes.shape[0]):
        output[imageIndex].append(
            {
                'class': classes[detectionIndex].item(),
                'scores': scores[detectionIndex].item(),
                'bbox': bboxes[detectionIndex].tolist(),
                'mask': masks[detectionIndex].tolist()

            })


    # print(bboxes)
#
# cfg.rescore_bbox = save
#
# output = []
# for i in range(0, classes.shape[0]):
#     output.append({
#         'class': classes[i].item(),
#         'scores': scores[i]
#     })
#
#     break
#
# # print(output)
# # print(bboxes)
# # print(scores)


# net(img)
