import pickle
from tqdm import trange
from utils import *
from skimage import transform
#注意img_size这里是shape，np和pil中的shape/size不同
### Train

train_pkl_path = 'data/train_data.pkl'
data = pickle.load(open(train_pkl_path, 'rb'))

train_imgs = []
train_img_info = []
train_roi = []
train_cls = []
train_tbbox = []

N_train = len(data['imgs'])#int，一共多少条数据
print('%d pictures for training\n'%N_train)

for i in trange(N_train):
    img = np.array(data['imgs'][i])
    gt_boxs = np.array(data['boxes'][i])#groundtruth
    gt_classes = np.array(data['gt_classes'][i])#num_of_gt-dim的int数组，表示每一个gtbox的类别，注意，从1开始编号，0代表背景
    nobj = data['num_objs'][i]
    bboxs = np.array(data['selective_search_boxes'][i])
    nroi = len(bboxs)

    img = np.array(img)
    img_size = img.shape #比如 1024，1024
    img=img.reshape((img_size[0],img_size[1],1))

    img=transform.resize(img, (224, 224))

    img = img.astype(np.float32)
    img = np.transpose(img, [2, 0, 1])

    rbboxs = rel_bbox(img_size, bboxs)#相应调整坐标,rbboxs为nroi*4, 4dim is xmin,ymin,xmax,ymax, 0~1

    ious = calc_ious(bboxs, gt_boxs)#ious是nroi*number_of_groundtruth的array
    max_ious = ious.max(axis=1)#对于每个ss_box，选出其iou最大的相应的gt_box，依次为最大iou
    max_idx = ious.argmax(axis=1)#选出其中iou最大的一个，1*nroi的idx，储存对应的gt_box的index
    tbbox = bbox_transform(bboxs, gt_boxs[max_idx])#tbbox是nroi*4的array，4dim是[targets_dx, targets_dy, targets_dw, targets_dh]

    pos_idx = []
    neg_idx = []

    for j in range(nroi):
        if max_ious[j] < 0.1:#iou太小了，不考虑
            continue

        gid = len(train_roi)#当前ss_box的标号
        train_roi.append(rbboxs[j])
        train_tbbox.append(tbbox[j])

        if max_ious[j] >= 0.5:
            pos_idx.append(gid)
            train_cls.append(gt_classes[max_idx[j]])
        else:
            neg_idx.append(gid)
            train_cls.append(0)

    #suppose选出nroi_alternative个box,则
    #train_roi，train_tbbox，train_cls都是nroi_alternative-dim

    pos_idx = np.array(pos_idx)
    neg_idx = np.array(neg_idx)
    train_imgs.append(img)
    train_img_info.append({
        'img_size': img_size,
        'pos_idx': pos_idx,
        'neg_idx': neg_idx,
    })
    # print(len(pos_idx), len(neg_idx))#相加应为nroi_alternative

train_imgs = np.array(train_imgs)
train_img_info = np.array(train_img_info)
train_roi = np.array(train_roi)
train_cls = np.array(train_cls)
train_tbbox = np.array(train_tbbox).astype(np.float32)

# print('train_imgs.shape=',end='')
# print(train_imgs.shape)
# print(train_roi.shape, train_cls.shape, train_tbbox.shape)

np.savez(open('data/train.npz', 'wb'), 
         train_imgs=train_imgs, train_img_info=train_img_info,
         train_roi=train_roi, train_cls=train_cls, train_tbbox=train_tbbox)

### Test

test_pkl_path = 'data/train_data.pkl'
data = pickle.load(open(test_pkl_path, 'rb'))

test_imgs = []
test_img_info = []
test_roi = []
test_orig_roi = []

N_test = len(data['imgs'])
print('%d pictures for testing\n'%N_test)
for i in trange(N_test):
    img = np.array(data['imgs'][i])

    img = np.array(img)
    img_size = img.shape  # 比如 1024，1024
    img = img.reshape((img_size[0], img_size[1], 1))

    img = transform.resize(img, (224, 224))
    # print(img_size)

    img = img.astype(np.float32)
    img = np.transpose(img, [2, 0, 1])

    bboxs = np.array(data['selective_search_boxes'][i])
    nroi = len(bboxs)


    rbboxs = rel_bbox(img_size, bboxs)
    idxs = []

    for j in range(nroi):
        gid = len(test_roi)
        test_roi.append(rbboxs[j])
        test_orig_roi.append(bboxs[j])
        idxs.append(gid)

    idxs = np.array(idxs)
    test_imgs.append(img)
    test_img_info.append({
        'img_size': img_size,
        'idxs': idxs
    })
    # print(len(idxs))

test_imgs = np.array(test_imgs)
test_img_info = np.array(test_img_info)
test_roi = np.array(test_roi)
test_orig_roi = np.array(test_orig_roi)

# print(test_imgs.shape)
# print(test_roi.shape)
# print(test_orig_roi.shape)

np.savez(open('data/test.npz', 'wb'),
         test_imgs=test_imgs, test_img_info=test_img_info, test_roi=test_roi, test_orig_roi=test_orig_roi)
