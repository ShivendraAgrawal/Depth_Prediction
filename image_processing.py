from random import randrange
from skimage.io import imread
import matplotlib.pyplot as plt
import os
import glob
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

def normalize_like_NYU(image):
    max_pix = 9.99547
    min_pix = 0.71329951
    result = image / image.max()
    return min_pix + result * (max_pix - min_pix)

def save_object_data_to_disk():
    base = "rgbd-dataset"
    sample = 10
    folders = os.listdir(base)
    folders_contents = {}
    for object in folders:
        if object not in [".DS_Store"]:
            folders_contents[object] = os.listdir(os.path.join(base, object))
    all_images = []
    all_depths = []
    count = 0
    count_folders = 0
    for folder in folders_contents:
        print(folder)
        for sub_folder in folders_contents[folder]:
            print(sub_folder)
            if sub_folder not in [".DS_Store"]:
                count_folders += 1
                files = os.listdir(os.path.join(base, folder, sub_folder))
                count += len(files) / 4
                depths, images = [], []
                for file in files:
                    if file[-10:] == "_depth.png":
                        depths.append(file)
                    elif file[-9:] != "_mask.png" and file[-4:] == ".png":
                        images.append(file)
                index = np.random.choice(np.arange(len(images)), sample, replace=False)
                try:
                    depths = [depths[i] for i in index]
                    images = [images[i] for i in index]

                    for image in images:
                        np_image = imread(os.path.join(base, folder, sub_folder, image))
                        all_images.append(np_image)
                    for depth in depths:
                        np_depth = imread(os.path.join(base, folder, sub_folder, depth))
                        np_depth = normalize_like_NYU(np_depth)
                        np_depth = np_depth.reshape((480, 640, 1))
                        all_depths.append(np_depth)
                except:
                    print(sub_folder + " ERROR")
    np.save("obj_depth_n_h_w_1", np.array(all_depths))
    np.save("obj_images_n_h_w_c", np.array(all_images))


if __name__ == "__main__":
    save_object_data_to_disk()
    # transformed_test_y = np.load('transformed_test_y.npy')
    # predicted_y = np.load('predicted_y.npy')
    # n_test = transformed_test_y.shape[0]
    # save_depth_images_to_disk(transformed_test_y.reshape((n_test, 55, 74)), "Test_Y")
    # save_depth_images_to_disk(predicted_y.reshape((n_test, 55, 74)), "Predicted_Y")

