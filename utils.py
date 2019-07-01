import os

import PIL
import numpy as np
import skimage
import skimage.transform
from matplotlib import pyplot as plt
# Image resize
from scipy.ndimage import center_of_mass


def imresize(img, height=None, width=None):
    # load image
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]

    return skimage.transform.resize(img, (int(ny), int(nx)), mode='constant')


# Heat map visualization
def show_heatmaps(imgs, masks, K, enhance=1, title=None, cmap='gist_rainbow'):
    if K > 0:
        _cmap = plt.cm.get_cmap(cmap)
        colors = [np.array(_cmap(i)[:3]) for i in np.arange(0, 1, 1 / K)]
    plt.figure(figsize=(4 * len(imgs), 4))
    if title is not None:
        plt.suptitle(title + '\n', fontsize=24).set_y(1.05)
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)

        img = imgs[i]
        if img.max() <= 1:
            img *= 255
        img = np.array(PIL.ImageEnhance.Color(PIL.Image.fromarray(np.uint8(img))).enhance(enhance))
        plt.imshow(img)
        plt.axis('off')
        for k in range(K):
            layer = np.ones((*img.shape[:2], 4))
            for c in range(3):
                layer[:, :, c] *= colors[k][c]
            mask = masks[i][k]
            layer[:, :, 3] = mask
            plt.imshow(layer)
            plt.axis('off')

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()


# save heatmaps into png (3rows * 10columns)
def save_heatmaps(imgs, masks, K, enhance=1, title=None, cmap='gist_rainbow', birdid=100):
    if K > 0:
        _cmap = plt.cm.get_cmap(cmap)
        colors = [np.array(_cmap(i)[:3]) for i in np.arange(0, 1, 1 / K)]
    plt.figure(figsize=(4 * len(imgs) / 3, 4 * 3))
    if title is not None:
        plt.suptitle(title + '\n', fontsize=24).set_y(1.05)
    for i in range(len(imgs)):
        plt.subplot(3, len(imgs) // 3, i + 1)

        img = imgs[i]
        if img.max() <= 1:
            img *= 255
        img = np.array(PIL.ImageEnhance.Color(PIL.Image.fromarray(np.uint8(img))).enhance(enhance))
        plt.imshow(img)
        plt.axis('off')
        for k in range(K):
            layer = np.ones((*img.shape[:2], 4))
            for c in range(3):
                layer[:, :, c] *= colors[k][c]
            mask = masks[i][k]
            layer[:, :, 3] = mask
            plt.imshow(layer)
            plt.axis('off')

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig('output/birds_{0:03d}_{1}.png'.format(birdid, K))


# generate part-segmentation mask, save into png one by one
def generate_masks(imgs, masks, K, enhance=1, title=None, cmap='gist_rainbow', birdid=100):
    if K > 0:
        _cmap = plt.cm.get_cmap(cmap)
        colors = [np.array(_cmap(i)[:3]) for i in np.arange(0, 1, 1 / K)]
    plt.figure(figsize=(8, 8))
    if title is not None:
        plt.suptitle(title + '\n', fontsize=24).set_y(1.05)
    for i in range(len(imgs)):
        img = imgs[i]
        if img.max() <= 1:
            img *= 255
        img = np.array(PIL.ImageEnhance.Color(PIL.Image.fromarray(np.uint8(img))).enhance(enhance))
        if K == 0:
            plt.imshow(img)
        plt.axis('off')
        for k in range(K):
            layer = np.ones((*img.shape[:2], 4))
            for c in range(3):
                layer[:, :, c] *= colors[k][c]
            mask = masks[i][k]
            layer[:, :, 3] = mask
            plt.imshow(layer)
            plt.axis('off')

        plt.tight_layout(pad=0)
        plt.savefig('output/bird{0:03d}_{1}parts_{2}.png'.format(birdid, K, i))


# generate part-segmentation mask and key-points
def save_mask2d(masks, K, bird_dir):
    out_mask = np.zeros((masks.shape[2], masks.shape[3]))
    out_point = np.zeros((K, 2))
    for i in range(masks.shape[0]):
        for k in range(K):
            mask = masks[i][k]
            np.putmask(out_mask, mask > 0.9, k + 1)

            tmp = np.zeros((masks.shape[2], masks.shape[3]))
            np.putmask(tmp, mask > 0.9, 1)
            out_point[k] = center_of_mass(tmp)

        path = os.path.join(bird_dir, 'bird_point_mask_{0:03d}_{1}_{2}.npz'.format(bird_dir.split('_')[-1], K, i))
        np.savez_compressed(path, out_point, out_mask)
