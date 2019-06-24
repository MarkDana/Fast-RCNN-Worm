# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np
from PIL import Image
from PIL import ImageDraw

patch_size=16
half_patch_size=patch_size//2

frames = np.load('./data/img_stack.npy')  # (50, 1024, 1024, 23)
neuron_position = np.load('./data/ordered_neuron_position.npy')  # (50, 107, 3)

frame_number = neuron_position.shape[0]  # 50
neuron_number = neuron_position.shape[1]  # 107
channel_number = frames.shape[3]  # 23

data={}
data['imgs']=[]
data['boxes']=[]
data['gt_classes']=[]
data['num_objs']=[]
data['selective_search_boxes']=[]


for i in range(frame_number):  # traverse all 50 frames
    for j in range(channel_number):
        print('--------%d_%d--------'%(i,j))
        frame=frames[i,:,:,j]

        whole_pic = Image.fromarray(np.uint8(frame))
        whole_pic.save('./data/images/%d_%d.jpg' % (i, j))
        draw = ImageDraw.Draw(whole_pic)

        labels = []
        for k in range(neuron_number):
            point = neuron_position[i, k, :]
            if point[2] == j:
                ctrx,ctry=point[0], point[1]
                labels.append((ctrx,ctry, k))
                draw.point([(ctrx,ctry)], fill=(255))
                draw.line([(ctrx - half_patch_size, ctry - half_patch_size),
                          (ctrx + half_patch_size, ctry - half_patch_size)], fill=(255), width=2)
                draw.line([(ctrx - half_patch_size, ctry - half_patch_size),
                          (ctrx - half_patch_size, ctry + half_patch_size)], fill=(255), width=2)
                draw.line([(ctrx + half_patch_size, ctry + half_patch_size),
                          (ctrx - half_patch_size, ctry + half_patch_size)], fill=(255), width=2)
                draw.line([(ctrx + half_patch_size, ctry + half_patch_size),
                          (ctrx + half_patch_size, ctry - half_patch_size)], fill=(255), width=2)
                draw.text((ctrx,ctry), '%d' % (k+1), fill='white')

        whole_pic.save('./data/images/%d_%d_gt.jpg' % (i, j))
        img = cv2.imread('./data/images/%d_%d.jpg' % (i, j))

        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)

        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:continue
            x, y, w, h = r['rect']
            if h * w == 0 or w / h > 1.2 or h / w > 1.2:continue#排除狭条形状的矩形影响
            candidates.add(r['rect'])

        # draw rectangles on the original image
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        ax.imshow(img)
        for x, y, w, h in candidates:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
            ax.add_patch(rect)

        plt.savefig('./data/images/%d_%d_ss.jpg' % (i, j))
        os.remove('./data/images/%d_%d.jpg' % (i, j))

        print('%d ground-truths, %d ss' % (len(labels), len(candidates)))
        if labels and candidates:#当groundtruth和selectivesearch都不为空才加入训练集
            data['imgs'].append(frame)
            data['num_objs'].append(len(labels))
            datalabel=[]
            dataclasses=[]
            datass=[]

            for label in labels:
                ctrx, ctry, k=label

                xmin=ctrx-half_patch_size
                ymin=ctry-half_patch_size
                xmax=ctrx+half_patch_size
                ymax=ctry+half_patch_size

                datalabel.append([xmin, ymin, xmax, ymax])
                dataclasses.append(k+1)

            for x, y, w, h in candidates:
                datass.append([x,y,x+w,y+h])

            data['boxes'].append(datalabel)
            data['gt_classes'].append(dataclasses)
            data['selective_search_boxes'].append(datass)

            with open('./data/train_set.txt','a')as fin:
                fin.write('%d_%d\n'%(i,j))


data['imgs']=np.array(data['imgs'])
data['boxes']=np.array(data['boxes'])
data['gt_classes']=np.array(data['gt_classes'])
data['num_objs']=np.array(data['num_objs'])
data['selective_search_boxes']=np.array(data['selective_search_boxes'])


import pickle
train_pkl_path = 'data/train_data.pkl'
with open(train_pkl_path, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
