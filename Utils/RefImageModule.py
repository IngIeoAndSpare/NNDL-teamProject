# image reader
import os
import requests

import cv2
from PIL import Image
import numpy as np


class RefImageModule:

    KEY = ""
    
    def __init__(self, file_path, image_encoding, get_method = True):
        
        ## Path params
        self.path_root_file_path = file_path
        
        ## Image params
        self.image_encoding = image_encoding # RGB, RGBA
        self.image_file_format = "png"

        self.url_context = "http://xdworld.vworld.kr:8080/XDServer/3DData?Version=2.0.0.0&Request=GetLayer&Layer=tile&"
        
        ## File getter 
        ## self.get_file_method_flag = get_method

    def get_ref_image(self, x_coordinate, y_coordinate, zoom_level):
        ref_image_array = []

        ## Get parent map tile params
        parent_x_coordinate = int(x_coordinate / 2)
        parent_y_coordinate = int(y_coordinate / 2)
        parent_zoom_level = zoom_level - 1

        ## Get child map tile params
        child_x_coordinate = x_coordinate * 2
        child_y_coordinate = y_coordinate * 2
        child_zoom_level = zoom_level + 1
        
        ref_image_array.append(
            self._get_image_from_local(
                parent_x_coordinate,
                parent_y_coordinate,
                parent_zoom_level
            )
        )

        for y in range(0, 1):
            for x in range(0, 1):
                ref_image_array.append(
                    self._get_image_from_local(
                        child_x_coordinate + x,
                        child_y_coordinate + y,
                        child_zoom_level
                    )
                )

        return ref_image_array

    def _get_image_from_local(self, x_coordinate, y_coordinate, zoom_level):
        ## TODO : get image use file path
        image_path = f"{self.path_root_file_path}\{zoom_level}\{y_coordinate}\{y_coordinate}_{x_coordinate}.{self.image_file_format}"
        return Image.open(image_path).convert(self.image_encoding)

    def _get_image_from_url(self, x_coordinate, y_coordinate, zoom_level):
        
        string_query_context = f"level={zoom_level}&IDX={x_coordinate}&IDY={y_coordinate}&Key={KEY}"
        url = ''.join([self.url_context, string_query_context])

        im_content = requests.get(url).content
        img_array = np.fromstring(im_content, np.uint8)

        ##https://stackoverflow.com/questions/17170752/python-opencv-load-image-from-byte-string
        img_np = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return img_np














