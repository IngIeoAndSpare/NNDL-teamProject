from PIL import Image
import numpy as np
from itertools import chain
import os


def get_original_images(root):
    images = []
    for path in os.listdir(root):
        for file in os.listdir(os.path.join(root, path)):
            filepath = '/'.join([root, path, file])
            images.append(filepath)
    return images


def error_vender_chooser(root, n_samples):
    choosen = []
    for path in os.listdir(root):
        temp = list(np.random.choice(os.listdir(os.path.join(root, path)), n_samples, replace=False))
        for i in range(len(temp)):
            filepath = '/'.join([root, path, temp[i]])
            temp[i] = filepath
        choosen.append(temp)

    return list(chain(*choosen))

def get_path_name(original, err_vendor):
    list_original_path = original.split('/')
    ori_subdir = list_original_path[-2]
    ori_filename = list_original_path[-1][:-4]
    del list_original_path

    list_err_vendor_path = err_vendor.split('/')
    err_type = list_err_vendor_path[-2]
    err_filename = list_err_vendor_path[-1][:-4]
    del list_err_vendor_path
    return ori_subdir, ori_filename, err_type, err_filename


def mask_error_original(original, err_vendor, root='err_images'):
    print(original, ' + ', err_vendor, ' processing ......')
    ori_subdir, ori_filename, err_type, err_filename = get_path_name(original, err_vendor)

    original = Image.open(original)
    err_vendor = Image.open(err_vendor)

    arr_original = np.array(original, dtype=np.int16)[:, :, :-1]  ## these have 8bit integer, to avoid overflow, revise the dtype to 16bit
    arr_err_vendor = np.array(err_vendor, dtype=np.int16)[:, :, :-1]  ##  RGBA channel to RGB channel

    arr_err_img = arr_original + arr_err_vendor
    arr_err_img = np.clip(arr_err_img, 0, 255)
    arr_err_img = arr_err_img.astype('uint8')

    filedir = os.path.join(root, ori_subdir, err_type)
    filename = ori_filename + '_' + err_filename + '.png'

    print(filedir, filename)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    err_img = Image.fromarray(arr_err_img)
    err_img.save(os.path.join(filedir, filename))


def main():
    ## set the directory and n_samples below here before run this script
    images = get_original_images('./data')

    for image in images:
        err_vendors = error_vender_chooser('./train_data', n_samples=10)
        for err_vendor in err_vendors:
            mask_error_original(image, err_vendor)
    print('done')


if __name__ == '__main__':
    main()
