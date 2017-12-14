from random import randrange

from scipy import signal
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
    np.save("obj_depth_n_h_w_1", np.array(all_depths, dtype="float16"))
    np.save("obj_images_n_h_w_c", np.array(all_images))

def convolve(single_channel, k, predicted_depth):
    n_rows = single_channel.shape[0]
    n_columns = single_channel.shape[1]
    new_image = single_channel.copy()
    for i in range(n_rows):
        for j in range(n_columns):
            if predicted_depth[i][j]>3 and i != 0 and i != n_rows - 1 and j != 0 and j != n_columns - 1:
                new_image[i][j] = single_channel[i - 1][j - 1] * k[0][0] + \
                                  single_channel[i - 1][j] * k[0][1] + \
                                  single_channel[i - 1][j + 1] * k[0][2] + \
                                  single_channel[i][j - 1] * k[1][0] + \
                                  single_channel[i][j] * k[1][0] + \
                                  single_channel[i][j + 1] * k[1][1] + \
                                  single_channel[i + 1][j - 1] * k[2][2] + \
                                  single_channel[i + 1][j] * k[2][1] + \
                                  single_channel[i + 1][j + 1] * k[2][2]
    return new_image

def portrait_mode(predicted_depth, actual_image, filename, root_dir):
    image = actual_image
    try:
        r,g,b,a = image[:, :, 0], image[:, :, 1], image[:, :, 2], image[:, :, 3]
    except:
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    no_blur_kernel = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

    blur_kernel = np.array([[0.0625, 0.125, 0.0625],
                       [0.125, 0.25, 0.125],
                       [0.0625, 0.125, 0.0625]])
    # blur_kernel = np.array([[0.0875, 0.125, 0.0875],
    #                         [0.125, 0.15, 0.125],
    #                         [0.0875, 0.125, 0.0875]])

    r = convolve(r, blur_kernel, predicted_depth)
    g = convolve(g, blur_kernel, predicted_depth)
    b = convolve(b, blur_kernel, predicted_depth)

    rgb = np.dstack((r, g, b))  # stacks 3 h x w arrays -> h x w x 3
    plt.imsave(os.path.join(root_dir, filename+"_portrait.png"), rgb/255)
    plt.imsave(os.path.join(root_dir, filename+".png"), image/255)


                    # print(type(image))
    # print(image.shape)
    # grad = signal.convolve2d(image, kernel, boundary='symm', mode='same')
    # H, W, C = image.shape
    # im = Image.new('RGB', (W, H))
    # im.putdata(format_pixel_data(grad, H, W))
    # im.save("portrait_" + actual_image)

def save_portraits(predicted_y, preprocessed_test_x, root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    n_test = predicted_y.shape[0]
    for i in range(n_test):
        print(os.path.join(root_dir, "%s.png" % i))
        portrait_mode(predicted_y[i], preprocessed_test_x[i], str(i), root_dir)

if __name__ == "__main__":
    # save_object_data_to_disk()

    transformed_test_y = np.load('transformed_test_y.npy')
    predicted_y = np.load('predicted_y.npy')
    preprocessed_test_x = np.load("preprocessed_test_x .npy")

    save_portraits(predicted_y, preprocessed_test_x, "Portrait")
    # n_test = transformed_test_y.shape[0]
    # save_depth_images_to_disk(transformed_test_y.reshape((n_test, 55, 74)), "Test_Y")
    # save_depth_images_to_disk(predicted_y.reshape((n_test, 55, 74)), "Predicted_Y")

