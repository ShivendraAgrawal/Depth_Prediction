import numpy as np
import matplotlib.pyplot as plt
from time import time
from skimage.io import imread
from scipy import ndimage
from PIL import Image

plt.rcParams['image.cmap'] = 'gray'

if __name__ == '__main__':
    img_sample = imread("/Users/shivendra/Desktop/CU/ML Project/Portrait mode photos -Saumya/test1.jpeg")
    img_sample = imread("/Users/shivendra/Desktop/CU/ML Project/1136.png")

    img0 = img_sample[:,:,0]
    img1 = img_sample[:, :, 1]
    img2 = img_sample[:, :, 2]
    print(img_sample.shape)

    im = Image.open("/Users/shivendra/Desktop/CU/ML Project/1136.png")
    print(im)
    # plt.imshow(img0)
    # plt.imshow(img1)
    # plt.imshow(img2)
    # plt.show()

    # blurred_face = ndimage.gaussian_filter(img_sample, sigma=3)
    # very_blurred = ndimage.gaussian_filter(img_sample, sigma=5)
    # local_mean = ndimage.uniform_filter(img_sample, size=11)
    #
    # plt.figure(figsize=(9, 3))
    # plt.subplot(131)
    # plt.imshow(blurred_face, cmap=plt.cm.gray)
    # plt.axis('off')
    # plt.subplot(132)
    # plt.imshow(very_blurred, cmap=plt.cm.gray)
    # plt.axis('off')
    # plt.subplot(133)
    # plt.imshow(local_mean, cmap=plt.cm.gray)
    # plt.axis('off')
    #
    # plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01,
    #                     left=0.01, right=0.99)
    #
    # plt.show()

