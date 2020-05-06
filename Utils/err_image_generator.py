from PIL import Image
import numpy as np
from itertools import chain
import os

class ErrorImageGenerator:

    def __init__(self, image_root_path, train_image_path, error_image_context, sample_count = 10):
        self.image_root_path = image_root_path ## './data'
        self.train_image_path = train_image_path ## ./train_data
        self.error_image_context = error_image_context ## err_images
        self.samples_count = sample_count
        
        self.image_format = '.png'


    def get_error_image_from_mask(self):
        ## set the directory and n_samples below here before run this script
        images_path_arr = self._get_original_images_path()

        for image_path in images_path_arr:
            err_vendors = self._error_vender_chooser()
            for err_vendor in err_vendors:
                self._mask_error_original(image_path, err_vendor)
        print('done')


    def _get_original_images_path(self):
        images = []
        for path in os.listdir(self.image_root_path):
            for file in os.listdir(os.path.join(self.image_root_path, path)):
                filepath = '/'.join([self.image_root_path, path, file])
                images.append(filepath)
        return images


    def _error_vender_chooser(self):
        choosen = []
        for path in os.listdir(self.train_image_path):
            rand_err_image_list = list(
                np.random.choice(
                    os.listdir(os.path.join(self.train_image_path, path)),
                    self.samples_count, replace=False
                )
            )
            for i in range(len(rand_err_image_list)):
                file_path = '/'.join([self.train_image_path, path, rand_err_image_list[i]])
                rand_err_image_list[i] = file_path
            choosen.append(rand_err_image_list)

        return list(chain(*choosen))


    def _mask_error_original(self, original_image_path, err_vendor):
        print(original_image_path, ' + ', err_vendor, ' processing ......')
        ori_subdir, ori_filename, err_type, err_filename = self._get_path_name(
            original_image_path, err_vendor
        )

        original_image = Image.open(original_image_path)
        err_vendor = Image.open(err_vendor)

        ## These have 8bit integer, to avoid overflow, revise the dtype to 16bit
        arr_original = np.array(original_image, dtype=np.int16)[:, :, :-1]  
        
        ## RGBA channel to RGB channel
        arr_err_vendor = np.array(err_vendor, dtype=np.int16)[:, :, :-1]  

        arr_err_img = arr_original + arr_err_vendor
        arr_err_img = np.clip(arr_err_img, 0, 255)
        arr_err_img = arr_err_img.astype('uint8')

        filedir = os.path.join(self.error_image_context, ori_subdir, err_type)
        filename = ori_filename + '_' + err_filename + self.image_format

        print(filedir, filename)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        err_img = Image.fromarray(arr_err_img)
        err_img.save(os.path.join(filedir, filename))


    def _get_path_name(self, original_image_path, err_vendor):
        list_original_path = original_image_path.split('/')
        ori_subdir = list_original_path[-2]
        ori_filename = list_original_path[-1][:-4]
        del list_original_path

        list_err_vendor_path = err_vendor.split('/')
        err_type = list_err_vendor_path[-2]
        err_filename = list_err_vendor_path[-1][:-4]
        del list_err_vendor_path

        return ori_subdir, ori_filename, err_type, err_filename