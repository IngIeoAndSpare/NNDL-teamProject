# image reader
from PIL import Image
import os

class RefImageModule:

    def __init__(self, file_path, image_encoding, get_method = True):
        
        ## Path params
        self.path_root_file_path = file_path
        
        ## Image params
        self.image_encoding = image_encoding # RGB, RGBA
        self.image_file_format = "png"
    
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

'''
    def _get_image_from_url(self, url):
        ## TODO : get image use vworld url
'''














