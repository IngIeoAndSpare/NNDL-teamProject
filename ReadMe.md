# NNDL 2020 team project #

**map tile error finder and recoved error image**

## run ##
run on the conda prompt

```
$ python mainHandler.py

```

## explain ##

### target data ###

* Map tile image
    * satellite photograph
    * aerial photograph

* Label
    * coordinate (tile x, y)
    * zoomlevel
    * missing error vender

* Data source
    * [naver](https://map.naver.com/v5/)
    * [kakao](https://map.kakao.com/)
    * [vworld](http://map.vworld.kr/map/maps.do)
    * [google](https://www.google.com/maps)

* Inpainting
    * autoencoder

### problem ###
* Map tile image's special feature
    * repeated image patterns
    * Shadow pattern about Building and Forest
    *  area
        * urban
        * marine
        * mountainous

* Missing error
    * line
    * dispersion
    * unspecified_shapes (NFP)
    * square


## require env ##
* Python : [3.8.1](https://www.python.org/downloads/release/python-381/)
* Cuda : [10.1_ver.2](https://developer.nvidia.com/cuda-10.1-download-archive-update2?arget_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork)
* Pytorch : [1.4.0](https://pytorch.org/get-started/locally/)
* Torchvision : [0.5.0](https://anaconda.org/pytorch/torchvision)

## error classification result ##


## ref paper ##

* Classification (network)
    * resNet [paper](https://arxiv.org/abs/1512.03385)
        * 18 
        * 34
        * 50
        * 101
    * googleNet [paper](https://arxiv.org/abs/1409.4842)
    * shuffleNet_V2 [paper](https://arxiv.org/abs/1807.11164)
    * AlexNet [paper](https://arxiv.org/abs/1404.5997)

* Satellite, Aerial photograph
    * GIS
        * [위성영상과 항공영상을 이용한 지형변화 탐지 사례 분석](http://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE07093505)
    * use network
        * [RNN to correct satellite Image classification Maps](https://ieeexplore.ieee.org/abstract/document/7938635)
        * [Multiscale Deep Features for High-Resolution Satellite Image](https://ieeexplore.ieee.org/abstract/document/8036413)
        * [Mask R-CNN을 이용한 항공영상에서 도로 균열 검출](http://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09308009)

