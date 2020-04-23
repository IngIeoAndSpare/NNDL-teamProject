## inpainting modules
## import ..

## Utils
## sys.path.append("..")
## from Utils.

## Network
from Network.AutoEncoderNetwork import AutoEncoderNetwork

## Util lib import
import os
import json

## image lib import
from PIL import Image

class AutoEncoderModule:
    '''
        CAUTION : this class is HANDLER. so, if you define NETWORK(nn) create network python file.
                  And write this file [from network.AutoEncoder import AutoEncoder] on head.

    '''

    def __init__(self):
    '''
     TODO : params init
     ex ->   self.{{PARAMS_NAME}} = {{PARAMS_INIT_VALUE}}
    '''
    
    def train_inpainting_module(self, params1):
        '''
            TODO : Inpainting module training
            you can ref Module.ImageClassificationModule training_network method(line 45~).
        '''


    def inpainting_image(self, input_image, ref_image):
        '''
            TODO : inpainting image use train network
        '''
    
    def _get_pt_file(self, file_path):
        '''
            TODO : load pt file
            you can ref ImageClassificationModel line 80~81

            examplp) network_model.load : pytroch.nn 

            code : 
            network_model.load_state_dict(torch.load({{PT_FILE_PATH}}))
            network_model.eval()
        '''

    def _get_network_model(self, params):
        '''
            TODO : return Network Object
            you can ref ImageClassificationModel line 157 (this case is lot of model)

            ex) return AutoEncoderNetwork(params...)
        '''

    '''
     def {{METHOD_NAME}}(self, [{{PARAMS}}]):
     XXX : Do something.
     if define method only use this module, just append '_' in method name head
     ex)   _get_params(self, [params...])
     
     if method call in this class, just use 'self'
     ex) self._get_params([params])  -> XXX : in this case you not write self in params area.
     call -> self._get_params(params1)   ||   define -> def _get_params(self, params1)
    '''