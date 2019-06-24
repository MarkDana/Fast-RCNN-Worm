import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from PIL import Image
from PIL import ImageDraw
from roi import *
from utils import *
import time
from tqdm import trange
import sys
import os,shutil

N_CLASS = 107
img_stack=np.load('data/img_stack.npy')
npz = np.load('data/test.npz')
test_imgs = npz['test_imgs']
test_img_info = npz['test_img_info']
test_roi = npz['test_roi']
test_orig_roi = npz['test_orig_roi']


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        rawnet = torchvision.models.vgg16_bn(pretrained=False)
        self.first_conv=nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        #out is batchsize*64*224*224
        self.seq = nn.Sequential(*list(rawnet.features.children())[1:-1])
        # self.roipool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        self.roipool = SlowROIPool(output_size=(7,7))
        self.feature = nn.Sequential(*list(rawnet.classifier.children())[:-1])

        _x = Variable(torch.Tensor(1, 1, 224, 224))
        _r = np.array([[0., 0., 1., 1.]])
        _ri = np.array([0])
        _x = self.feature(self.roipool(self.seq(self.first_conv(_x)), _r, _ri).view(1, -1))
        feature_dim = _x.size(1)  # 4096

        self.cls_score = nn.Linear(feature_dim, N_CLASS + 1)
        self.bbox = nn.Linear(feature_dim, 4 * (N_CLASS + 1))  # +1是为了处理空白

        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, inp, rois, ridx):
        res = inp
        res = self.first_conv(res)
        res = self.seq(res)#now in shape of batchsize*512*14*14
        res = self.roipool(res, rois, ridx)#now in shape of num_of_inds(ingeneral 128)*512*7*7

        res = res.detach()
        res = res.view(res.size(0), -1)#now in shape of num_of_inds(ingeneral 128)*25088
        feat = self.feature(res)

        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat).view(-1, N_CLASS + 1, 4)
        return cls_score, bbox

    def calc_loss(self, probs, bbox, labels, gt_bbox):
        loss_sc = self.cel(probs, labels)
        lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 4)
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 4)
        loss_loc = self.sl1(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox * mask)
        lmb = 1.0
        loss = loss_sc + lmb * loss_loc
        return loss, loss_sc, loss_loc


rcnn = RCNN()
rcnn.load_state_dict(torch.load('model/worm_0623.mdl'))


def test_image(im_index):
    '''
    :param im_index: int, 0~717 (717=len(train_set))
    :return:
    '''
    with open('data/train_set.txt','r')as f:
        train_set_name=[line.strip() for line in f.readlines()]
    assert im_index>=0 and im_index<len(train_set_name), "index out of range"
    name=train_set_name[im_index]
    print('forward for %s.jpg'%name)

    img=test_imgs[im_index,:,:,:]
    img=img.reshape((1,)+img.shape)
    img = torch.from_numpy(img)

    info_now = test_img_info[im_index]
    idxs = info_now['idxs']
    idxs = np.array(idxs)
    rois = test_roi[idxs]
    orig_rois = test_orig_roi[idxs]
    img_size=info_now['img_size']

    ridx = np.zeros(len(idxs)).astype(int)
    sc, tbbox = rcnn(img, rois, ridx)
    sc = nn.functional.softmax(sc)
    sc = sc.data.cpu().numpy()
    tbbox = tbbox.data.cpu().numpy()
    bboxs = reg_to_bbox(img_size, tbbox, orig_rois)

    res_bbox = []
    res_cls = []

    for c in range(1, N_CLASS + 1):
        c_sc = sc[:, c]
        c_bboxs = bboxs[:, c, :]

        boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=0.3, score_threshold=0.6)
        res_bbox.extend(boxes)
        res_cls.extend([c] * len(boxes))

    if len(res_cls) == 0:
        for c in range(1, N_CLASS+1):
            c_sc = sc[:,c]
            c_bboxs = bboxs[:,c,:]

            boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=0, score_threshold=0)
            res_bbox.extend(boxes)
            res_cls.extend([c] * len(boxes))
        res_bbox = res_bbox[:1]
        res_cls = res_cls[:1]

    i, j = tuple([int(x) for x in name.split('_')])
    frame = img_stack[i, :, :, j]
    whole_pic = Image.fromarray(np.uint8(frame))
    draw = ImageDraw.Draw(whole_pic)

    for k in range(len(res_bbox)):
        point=res_bbox[k]
        cls=res_cls[k]
        xmin,ymin,xmax,ymax=int(point[0]),int(point[1]),int(point[2]),int(point[3])

        draw.line([(xmin, ymin), (xmin, ymax)], fill=(255), width=2)
        draw.line([(xmin, ymax), (xmax, ymax)], fill=(255), width=2)
        draw.line([(xmax, ymax), (xmax, ymin)], fill=(255), width=2)
        draw.line([(xmax, ymin), (xmin, ymin)], fill=(255), width=2)
        draw.text(((xmin+xmax)//2, (ymin+ymax)//2), '%d'%cls,fill='white')

    pth='./model/forward/%d_%d'%(i,j)
    if not os.path.exists(pth):os.mkdir(pth)
    whole_pic.save(os.path.join(pth,'forward.jpg'))
    shutil.copyfile('./data/images/%d_%d_gt.jpg'%(i,j), os.path.join(pth,'ground_truth.jpg'))
    shutil.copyfile('./data/images/%d_%d_ss.jpg' % (i, j), os.path.join(pth, 'selective-search.jpg'))

    print('see '+pth)

if __name__ == '__main__':

    test_image(50)


