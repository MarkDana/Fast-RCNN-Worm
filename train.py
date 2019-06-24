#part of the network is reused from @gary1346aa
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from PIL import Image
from roi import *
from utils import *
import time
from tqdm import trange
import sys

N_CLASS = 107

# Transform = torchvision.transforms.Normalize(
#     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ?for one layer, is it necessary to normalize?

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


rcnn = RCNN().cuda()
# rcnn = RCNN()
print(rcnn)

npz = np.load('data/train.npz')
train_imgs = npz['train_imgs']#一共n张图，则为n*1*244*244
train_img_info = npz['train_img_info']#其中，pos_idx neg_idx均为 0~∑num_j-1 的值
train_roi = npz['train_roi']#假设n张图，每张图有num_j个iou>0.1的ss_box，for j in [1,n]，则train_roi是(∑num_j)*4，4dim is xmin,ymin,xmax,ymax, 0~1
train_cls = npz['train_cls']#∑num_j dim list， 值在 0、1-107
train_tbbox = npz['train_tbbox']#(∑num_j)*4，4dim是[targets_dx, targets_dy, targets_dw, targets_dh]

train_imgs = torch.from_numpy(train_imgs)
# train_imgs = Transform(train_imgs)


Ntotal = train_imgs.size(0)
Ntrain = int(Ntotal * 0.8)
pm = np.random.permutation(Ntotal)
train_set = pm[:Ntrain]
val_set = pm[Ntrain:]

optimizer = torch.optim.Adam(rcnn.parameters(), lr=1e-4)

def train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=False):
    '''
    :param img: ingeneral 2*1*224*224, for this two pics, n1 and n2 ss valid boxes, if both has pos and neg, then
    :param rois: ingeneral 128*4
    :param ridx: ingeneral 128dim 000..001...111, each 64
    :param gt_cls: ingeneral 128dim value between 0,107
    :param gt_tbbox: ingeneral 128*4
    :param is_val:
    :return:
    sometimes not general, maybe only one pic, or the pic may all POS or all NEG
    '''
    sc, r_bbox = rcnn(img, rois, ridx)#torch.Size([64, 108]) torch.Size([64, 108, 4])
    loss, loss_sc, loss_loc = rcnn.calc_loss(sc, r_bbox, gt_cls, gt_tbbox)

    # no need for [0]
    fl = loss.data.cpu().numpy()
    fl_sc = loss_sc.data.cpu().numpy()
    fl_loc = loss_loc.data.cpu().numpy()

    if not is_val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return fl, fl_sc, fl_loc

def train_epoch(epoch, run_set, is_val=False):
    #run_set为一组照片号，随机，如[1,3,5,2]
    I = 2#batchsize=2
    B = 64
    POS = int(B * 0.25)
    NEG = B - POS
    Nimg = len(run_set)
    perm = np.random.permutation(Nimg)
    perm = run_set[perm]#内部混顺序

    # if is_val:
        # rcnn.eval()
    # else:
        # rcnn.train()

    losses = []
    losses_sc = []
    losses_loc = []
    for i in trange(0, Nimg, I):
        lb = i
        rb = min(i+I, Nimg)#防止nimg为奇数
        torch_seg = torch.from_numpy(perm[lb:rb])#ingeneral 2-dim
        img = Variable(train_imgs[torch_seg], volatile=is_val).cuda()
        # img = Variable(train_imgs[torch_seg], volatile=is_val)#in general 2*1*224*224
        ridx = []
        glo_ids = []

        for j in range(lb, rb):
            info = train_img_info[perm[j]]
            pos_idx = info['pos_idx']
            neg_idx = info['neg_idx']
            ids = []

            if len(pos_idx) > 0:
                ids.append(np.random.choice(pos_idx, size=POS))#fulfill
            if len(neg_idx) > 0:
                ids.append(np.random.choice(neg_idx, size=NEG))
            if len(ids) == 0:
                continue
            ids = np.concatenate(ids, axis=0)
            glo_ids.append(ids)
            ridx += [j-lb] * ids.shape[0]
            #向前跳数

        if len(ridx) == 0:
            continue
        glo_ids = np.concatenate(glo_ids, axis=0)
        ridx = np.array(ridx)

        rois = train_roi[glo_ids]
        gt_cls = Variable(torch.from_numpy(train_cls[glo_ids]), volatile=is_val).cuda()
        gt_tbbox = Variable(torch.from_numpy(train_tbbox[glo_ids]), volatile=is_val).cuda()
        # gt_cls = Variable(torch.from_numpy(train_cls[glo_ids]), volatile=is_val)
        # gt_tbbox = Variable(torch.from_numpy(train_tbbox[glo_ids]), volatile=is_val)

        loss, loss_sc, loss_loc = train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=is_val)
        if is_val:
            with open('model/loss_val.txt', 'a')as fin:
                fin.write('%f\t%f\t%f\t%f\n' % (epoch + i / Nimg, loss, loss_sc, loss_loc))
        else:
            with open('model/loss_train.txt','a')as fin:
                fin.write('%f\t%f\t%f\t%f\n'%(epoch+i/Nimg, loss, loss_sc, loss_loc))

        losses.append(loss)
        losses_sc.append(loss_sc)
        losses_loc.append(loss_loc)

    avg_loss = np.mean(losses)
    avg_loss_sc = np.mean(losses_sc)
    avg_loss_loc = np.mean(losses_loc)
    print(f'Avg loss = {avg_loss:.4f}; loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}')

def start_training(n_epoch=100):
    for i in range(n_epoch):
        print(f'===========================================')
        print(f'[Training Epoch {i+1}]')
        train_epoch(i,train_set, False)
        print(f'[Validation Epoch {i+1}]')
        train_epoch(i,val_set, True)


# rcnn.load_state_dict(torch.load('model/worm_0623.mdl'))若继续训练，uncomment
start_training(n_epoch=100)
torch.save(rcnn.state_dict(), 'model/worm_0623.mdl')

