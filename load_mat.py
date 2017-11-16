import argparse
import os
import sys

import h5py
import numpy as np
from PIL import Image


def format_pixel_data(pixel_array, h, w):
    list = []
    for i in range(h):
        for j in range(w):
            list.append(tuple(pixel_array[:, j, i]))
    return list


def save_original_images_to_disk(data_file_dir):
    f = h5py.File(data_file_dir, 'r')
    images = np.array(f.get("images"))
    save_images_to_disk(images, "raw-images")


# saves all 1449 images in ./raw-images
def save_images_to_disk(images, root_dir):
    # print len(images)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for i in range(len(images)):
        print("saving image: %s" % i)
        image = images[i]
        C, W, H = image.shape
        im = Image.new('RGB', (W, H))
        im.putdata(format_pixel_data(image, H, W))
        im.save(os.path.join(root_dir, "%s.png" % i))

def save_original_depth_images_to_disk(data_file_dir):
    f = h5py.File(data_file_dir, 'r')
    images = np.array(f.get("depths"))
    save_depth_images_to_disk(images, "depth_images")

def format_pixel_data_depths(pixel_array, h, w):
    list = []
    for i in range(h):
        for j in range(w):
            list.append(int(pixel_array[j, i]))
    return list

# saves all 1449 images in ./raw-images
def save_depth_images_to_disk(images, root_dir):
    # print len(images)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for i in range(len(images)):
        print("saving image: %s" % i)
        print(images[i])
        image = images[i]
        W, H = image.shape
        im = Image.new('F', (W, H))
        im.putdata(format_pixel_data_depths(image, H, W))
        im.save(os.path.join(root_dir, "%s.png" % i))

def save_original_depth_arrays_to_disk(data_file_dir):
    f = h5py.File(data_file_dir, 'r')
    images = np.array(f.get("depths"))
    images = np.transpose(images, (0, 2, 1))
    n = images.shape[0]
    images = images.reshape((n, 480, 640, 1))
    np.save("depth_n_h_w", images)

def save_original_image_arrays_to_disk(data_file_dir):
    f = h5py.File(data_file_dir, 'r')
    images = np.array(f.get("images"))
    images = np.transpose(images, (0, 3, 2, 1))
    np.save("images_n_h_w_c", images)


# creates 894 sub-folders under ./single-object-images and put all images
# containing a specific object type in one of these 894 sub-folders
def save_single_object_images_to_disk(data_file_dir):
    f = h5py.File(data_file_dir, 'r')
    labels = ["".join([chr(j) for j in f[i].value]) for i in
              f.get("names").value[0]]
    pixel_labels = np.array(f.get("labels"))
    images = np.array(f.get("images"))
    num_images = images.shape[0]
    root_dir = "single-object-images"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for i in range(len(labels)):
        label = labels[i]
        label_sub_dir = os.path.join(os.path.expanduser(root_dir), label)
        if not os.path.exists(label_sub_dir):
            os.makedirs(label_sub_dir)
        images_to_save = []
        for j in range(num_images):
            image = images[j]
            pixel_label = pixel_labels[j]
            if np.sum(pixel_label == (i + 1)) > 0:
                images_to_save.append(image)
        save_images_to_disk(np.array(images_to_save), label_sub_dir)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--matlab_file_dir', type=str,
                        help='Directory of the matlab data file.',
                        default='./nyu_depth_v2_labeled.mat')
    return parser.parse_args(argv)


# crops all pixels of an image that are labeled as a specific object class
# using a rectangle and save this cropped image to 894 individual
# sub-folders under ./cropped-single-object-images. Since these images vary
# in sizes, we are not yet sure how this could be useful
def save_cropped_single_object_images_to_disk(data_file_dir):
    f = h5py.File(data_file_dir, 'r')
    labels = ["".join([chr(j) for j in f[i].value]) for i in
              f.get("names").value[0]]
    pixel_labels = np.array(f.get("labels"))
    images = np.array(f.get("images"))
    num_images = images.shape[0]
    root_dir = "cropped-single-object-images"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    for i in range(len(labels)):
        label = labels[i]
        label_sub_dir = os.path.join(os.path.expanduser(root_dir), label)
        if not os.path.exists(label_sub_dir):
            os.makedirs(label_sub_dir)
        images_to_save = []
        for j in range(num_images):
            image = images[j]
            pixel_label = pixel_labels[j]
            if np.sum(pixel_label == (i + 1)) > 0:
                print(j)
                images_to_save.append(cropped_image(image, pixel_label, i + 1))
        save_images_to_disk(images_to_save, label_sub_dir)


def cropped_image(image, pixel_label, index):
    c, w, h = image.shape
    x1, y1, x2, y2 = 10000, 10000, -1, -1

    for i in range(w):
        for j in range(h):
            if pixel_label[i, j] == index:
                x1 = min(x1, i)
                x2 = max(x2, i)
                y1 = min(y1, j)
                y2 = max(y2, j)

    return image[:, x1:x2 + 1, y1:y2 + 1]


if __name__ == '__main__':
    dir = parse_arguments(sys.argv[1:]).matlab_file_dir
    save_original_depth_arrays_to_disk(dir)
    #save_original_depth_images_to_disk(dir)
    # save_single_object_images_to_disk(dir)
    # save_cropped_single_object_images_to_disk(dir)