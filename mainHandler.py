## module
from Module.ImageClassificationModule import ClassificationModule
## from Module.ImageInpaintingModule import AutoEncoderModule

## util lib import
import os
import sys
import json

## util 
from Utils.err_image_generator import ErrorImageGenerator

if __name__ == "__main__":

    APPLCATION_ROOT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
    CLASSIFICATION_NETWORK_NAME = "ShuffleNet"
    
    TRAINING_FLAG = False

    ## File path ===================================
    
    ## Train image path 
    cl_train_image_file_path = os.path.join(APPLCATION_ROOT_PATH, "cl_train_data")

    ## Train image path 
    cl_test_image_file_path = os.path.join(APPLCATION_ROOT_PATH, "cl_test_data")
    
    ## Classification network tr result file path
    cl_result_networ_file_path = os.path.join(APPLCATION_ROOT_PATH, "network_result")


    ## Inpainting train image path
    ip_train_image_file_path = os.path.join(APPLCATION_ROOT_PATH, "ip_train_data")

    ip_test_image_file_path = os.path.join(APPLCATION_ROOT_PATH, "ip_test_data")

    ## Inpainting network tr result file path
    ip_result_network_file_path = os.path.join(APPLCATION_ROOT_PATH, "network_result")

    ## Reffer image set path
    ref_image_file_path = os.path.join(APPLCATION_ROOT_PATH, "ref_set")

    ## File path end ================================

    ## Error Image generator util example
    '''
    err_generator= ErrorImageGenerator(
        './data', './train_data', 'err_image'
    )
    err_generator.get_error_image_from_mask()
    '''
    ## example end ===========================

    if TRAINING_FLAG :
        ## Image classification training
        classificationModule = ClassificationModule (
                cl_train_image_file_path,
                cl_test_image_file_path,
                cl_result_networ_file_path,
                CLASSIFICATION_NETWORK_NAME
        )

    
    ## Image classification
    classificationModule.training_network()
    

    ## TODO : AutoEncoder training code