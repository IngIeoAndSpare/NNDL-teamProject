# image reader
from PIL import Image
import os

class RefImageModule:

    def __init__(self, file_path, image_encoding, get_method = True):

        ## Path params
        self.path_root_file_path = file_path
        
        ## Image params
        self.image_encoding = image_encoding
        self.image_fileFormat = "png"
    
        ## File getter 
        self.get_file_method_flag = get_method

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

        if self.get_file_method_flag :
            ref_image_array.append(
                self.get_file_method_flag(
                    parent_x_coordinate,
                    parent_y_coordinate,
                    parent_zoom_level
                )
            ) 

    def get_image_from_url(self, url):
        ## TODO : get image use vworld url


    def get_image_from_local(self, x_coordinate, y_coordinate, zoom_level):
        ## TODO : get image use file path

















