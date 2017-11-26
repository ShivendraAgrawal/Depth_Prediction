import matplotlib.pyplot as plt
import os

import numpy as np
from PIL import Image

def format_pixel_data(pixel_array, h, w):
    list = []
    for i in range(h):
        for j in range(w):
            list.append(tuple(pixel_array[:, j, i]))
    return list

def save_RGB_images_to_disk(images, root_dir):
    # print len(images)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for i in range(len(images)):
        print("saving image: %s" % i)
        image = images[i]
        H, W, C = image.shape
        im = Image.new('RGB', (W, H))
        im.putdata(format_pixel_data(image, H, W))
        im.save(os.path.join(root_dir, "%s.png" % i))

def save_depth_images_to_disk(images, root_dir):
    cmap = plt.cm.jet
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for i in range(len(images)):
        print(os.path.join(root_dir, "%s.png" % i))
        plt.imsave(os.path.join(root_dir, "%s.png" % i), images[i], cmap=cmap)


if __name__ == "__main__":
    transformed_test_y = np.load('transformed_test_y.npy')
    predicted_y = np.load('predicted_y.npy')
    n_test = transformed_test_y.shape[0]
    save_depth_images_to_disk(transformed_test_y.reshape((n_test, 55, 74)), "Test_Y")
    save_depth_images_to_disk(predicted_y.reshape((n_test, 55, 74)), "Predicted_Y")
