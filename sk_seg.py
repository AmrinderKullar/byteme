#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:23:30 2019

@author: siddharthkrishnakumar
"""

import numpy as np
from PIL import Image
import cv2
from temp_sk_model import Deeplabv3

class semantSeg():
    def __init__(self):
        self.trained_image_width=512 
        self.mean_subtraction_value=127.5
        self.image = np.array([])
        self.fin = np.array([])
        self.path = 'imgs/image1.jpg'
        self.model = Deeplabv3()
        
    def orig_image(self, img):
        # self.image = cv2.imread(self.path)
        self.image = img
        return(self.image)
        
#    def clean_image(self):
##        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
#        kernel = np.ones((3,3),np.float32)/25
#        self.image = cv2.filter2D(self.image,-1,kernel)
#        return (self.image)
#        
    def get_image(self, img):  
        # self.image = cv2.imread(self.paimgth)
        self.image = img
        w, h, _ = self.image.shape
        ratio = float(self.trained_image_width) / np.max([w, h])
        resized_image = cv2.resize(self.image,(int(ratio * h), int(ratio * w)))
        resized_image = (resized_image / self.mean_subtraction_value) - 1.
        pad_x = int(self.trained_image_width - resized_image.shape[0])
        pad_y = int(self.trained_image_width - resized_image.shape[1])
        resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')
        res = self.model.predict(np.expand_dims(resized_image, 0))
        labels = np.argmax(res.squeeze(), -1)
        if pad_x > 0:
            labels = labels[:-pad_x]
        if pad_y > 0:
            labels = labels[:, :-pad_y]
        self.fin = np.array(Image.fromarray(labels.astype('uint8')).resize((h,w)))
        a = self.fin
        b = np.array([a,a,a]).transpose(1,2,0)
        aa = np.where(b== 15,self.image,0)
        return aa