import cv2
import torch
import glob
import os
from mmdet.apis import inference_detector, init_detector, show_result
import sys


def main():
    config_file = sys.argv[1]
    checkpoint_file = sys.argv[2]
    model = init_detector(config_file, checkpoint_file, device=torch.device('cuda:0'))
    results = open(sys.argv[3], 'w')
    imgs = glob.glob('{}/*.jpg'.format(sys.argv[4]))
    total = len(imgs)
    nums = 1
    for img in imgs:
        baseName = os.path.splitext(os.path.basename(img))[0]
        oriImgX = int(baseName.split('_')[0])
        oriImgY = int(baseName.split('_')[1])
        
        result = inference_detector(model, img)

        bbox, label = show_result(img, result, model.CLASSES, score_thr=0.5, wait_time=1)
        for k in range(len(bbox)):
            if bbox[k][4] >= 0.5:
                results.write('{} {} {} {} {} {}\n'.format(oriImgX + bbox[k][0], oriImgY + bbox[k][1], oriImgX + bbox[k][2], oriImgY + bbox[k][3], bbox[k][4], label[k]))
        # print('{}/{}'.format(nums, total))
        nums += 1

if __name__ == '__main__':
    main()
