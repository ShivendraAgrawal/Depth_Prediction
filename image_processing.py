import matplotlib.pyplot as plt
import os
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
