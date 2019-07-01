import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import models

from nmf import NMF
from utils import imresize, save_mask2d

cuda = True
root = '~/dataset/CUB_200_2011/CUB_200_2011/images/'


def process_one_category(data_path):
    bird_category = int(data_path.split('/')[-1].split('.')[0])
    filenames = os.listdir(data_path)
    out_dir = 'output/bird_{0:03d}'.format(bird_category)
    os.mkdir(out_dir)

    # load images
    raw_images = [plt.imread(os.path.join(data_path, filename)) for filename in filenames]
    for i in range(len(raw_images)):
        img = raw_images[i]
        if np.array(img).shape[-1] > 3:
            raw_images[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(out_dir, 'raw_{0:03d}_{1}.png'.format(bird_category, i)), img)
    raw_images = [imresize(img, 224, 224) for img in raw_images]  # resize
    raw_images = np.stack(raw_images)

    # preprocess
    images = raw_images.transpose((0, 3, 1, 2)).astype('float32')  # to numpy, NxCxHxW, float32
    images -= np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))  # zero mean
    images /= np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))  # unit variance

    images = torch.from_numpy(images)  # convert to pytorch tensor
    if cuda:
        images = images.cuda()

    net = models.vgg19(pretrained=True)  # load pre-trained VGG-19
    if cuda:
        net = net.cuda()
    del net.features._modules['36']  # remove max-pooling after final conv layer

    with torch.no_grad():
        features = net.features(images)
        flat_features = features.permute(0, 2, 3, 1).contiguous().view((-1, features.size(1)))  # NxCxHxW -> (N*H*W)xC

    print('Reshaped features from {0}x{1}x{2}x{3} to ({0}*{2}*{3})x{1} = {4}x{1}'.format(*features.shape,
                                                                                         flat_features.size(0)))

    for K in [15]:
        with torch.no_grad():
            W, _ = NMF(flat_features, K, random_seed=0, cuda=cuda, max_iter=50)

        heatmaps = W.cpu().view(features.size(0), features.size(2), features.size(3), K).permute(0, 3, 1, 2)
        # (N*H*W)xK -> NxKxHxW
        heatmaps = torch.nn.functional.interpolate(heatmaps, size=(224, 224), mode='bilinear', align_corners=False)
        # 14x14 -> 224x224
        heatmaps /= heatmaps.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        # normalize by factor (i.e., 1 of K)
        heatmaps = heatmaps.cpu().numpy()
        # print(heatmaps.shape) # (60, K, 224, 224)
        save_mask2d(heatmaps, K, out_dir)


def test_img_size_error():
    data_path = '~/dataset/CUB_200_2011/CUB_200_2011/images/025.Pelagic_Cormorant'
    process_one_category(data_path)


for path in os.listdir(root):
    data_path = os.path.join(root, path)
    print(data_path)
    process_one_category(data_path)
