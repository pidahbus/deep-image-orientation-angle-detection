# Deep Image Orientation Angle Detection

## Introduction
Image Orientation Angle Detection Model (Deep-OAD) is a deep learning model to predict the orientation angle of the natural images. This repository allows to train, finetune and predict with a Deep-OAD model. The academic paper is available here:  https://arxiv.org/abs/2007.06709

## Data Preparation
Sample data has been provided in the data folder. The sample data is split in train, valid and test. For each set one CSV and one image directory must be created. Image directory contains the images. CSV should has two columns, image column contains the image name and angle column contains the true orientation angle of the image. 

## Model Training
Model training is very straightforward. Change the parameters in [config.py](https://github.com/pidahbus/deep-image-orientation-angle-detection/blob/main/config.py) and run in the terminal,
```
pip install -r requirements.txt
python train.py
```

## Model Finetuning
Set the path of a pretrained model in INIT_CHECKPOINT in [config.py](https://github.com/pidahbus/deep-image-orientation-angle-detection/blob/main/config.py) to finetune the model from the given checkpoint.


## Model Predictions
Sample prediction of orientation angles given images has been shown in [notebook.ipynb](https://github.com/pidahbus/deep-image-orientation-angle-detection/blob/main/notebook.ipynb). 


## Results
This method used the combination of CNN and a custom angluar loss function specially designed for angles that lead to a very promising result with respect to the recent works in this domain. Below is the result table picked from the above-mentioned academic [paper](https://arxiv.org/abs/2007.06709). It shows comparison between our OAD model and other image orientation angle estimation techniques. It is clearly seen that our OAD model outperforms other baseline methods and achieve very good results in terms of test mean absolute error (MAE).


| Task                                  | OAD-30 | Net-30 | OAD-45 | Net-45 | OAD-360 | Net-360 | Hough-var | Hough-pow | Fourier   |      
| ------------------------------------- |:------:|:------:|:------:|:------:|:-------:|:-------:|:---------:|:---------:|:---------:|
| Test images with ±30 degree rotation  |**1.52**|   3    |   -    |   -    |    -    |    -    |  11.41    |  10.62    |   10.66   |
| Test images with ±45 degree rotation  |   -    |   -    |**1.95**|  4.63  |    -    |    -    |  16.92    |  13.06    |   16.51   |
| Test images with ±180 degree rotation |   -    |   _    |   -    |   -    |**8.38** |  20.97  |     -     |     -     |     -     |



Below are some model predictions on natural images,


## Saved Model
You can download a trained Keras model [here](https://google.com). This model is trained on artifically created dataset using almost all of the images of Microsoft COCO. Additionally, this model is trained on task 3 that means this model is capable to predict orientation of images between 0° to 359° with test MAE of 8.38°. You can use this model for finetuning or model predictions using the above-mentioned methods.


## Citation
This paper is submitted for journal publication. If you are using this model then please use the below BibTeX to cite for now.

```
@ARTICLE{2020arXiv200706709M,
       author = {{Maji}, Subhadip and {Bose}, Smarajit},
        title = "{Deep Image Orientation Angle Detection}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning, Electrical Engineering and Systems Science - Image and Video Processing},
         year = 2020,
        month = jun,
          eid = {arXiv:2007.06709},
        pages = {arXiv:2007.06709},
archivePrefix = {arXiv},
       eprint = {2007.06709},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200706709M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

