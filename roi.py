import numpy as np
import torch
import torch.nn as nn

class SlowROIPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.size = output_size

    def forward(self, images, rois, roi_idx):
        #why here h and w are inversed?
        n = rois.shape[0]#一共多少个box
        h = images.size(2)#14
        w = images.size(3)#14
        x1 = rois[:,0]
        y1 = rois[:,1]
        x2 = rois[:,2]
        y2 = rois[:,3]

        x1 = np.floor(x1 * w).astype(int)
        x2 = np.ceil(x2 * w).astype(int)
        y1 = np.floor(y1 * h).astype(int)
        y2 = np.ceil(y2 * h).astype(int)

        res = []
        for i in range(n):
            img = images[roi_idx[i]].unsqueeze(0)#now[1, 512, 14, 14]
            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]]
            # img = img[:, :, x1[i]:x2[i], y1[i]:y2[i]],why?
            img = self.maxpool(img)
            res.append(img)
        res = torch.cat(res, dim=0)
        return res


