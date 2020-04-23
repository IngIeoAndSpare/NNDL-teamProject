## module
from Module.ImageClassificationModule import ClassificationModule
## from Module.ImageInpaintingModule import AutoEncoderModule

## utils
## from Utils.
## from Utils.

## util lib import
import os
import json



if __name__ == "__main__":

    APPLCATION_ROOT_PATH = r"..\\"

    ## file path ===================================
    
    ## Train image path 
    train_image_file_path = os.path.join(APPLCATION_ROOT_PATH, "train_data")

    ## Inpainting image path
    inpainting_image_file_path = os.path.join(APPLCATION_ROOT_PATH, "inpatinting")

    ## Reffer image set path
    ref_image_file_path = os.path.join(APPLCATION_ROOT_PATH, "ref_set")

    ## Classification network tr result file path
    result_cls_file_path = os.path.join(APPLCATION_ROOT_PATH, "network_result")

    ## Inpainting network tr result file path
    result_ip_file_path = os.path.join(APPLCATION_ROOT_PATH, "network_result")

    ## file path end ================================

