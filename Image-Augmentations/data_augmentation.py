import random 
import numpy as np 
import cv2 
import os 
import sys
from utils import clip_bboxes

class RandomHorizontalFlip_(object):
    '''Radnomly horizontally flip the Image with the probability p.
    Args:
        p: (float) The probability with which the image is flipped.

    Output: 
        img: (np.array) The flipped image with shape (HxWxC)
        bboxes: (np.array) The transformed bounding boxe with shape (N, 4)
                n is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
    '''
    
    def __init__(self, p):
        self.p = p

    def __call__(self, img, bboxes):
        '''get the img_center coordinate''' 
        # img with shape (512, 256, 3) (H = 512, W = 256)
        # img_center = [128., 256.] 
        # [::-1] cause the img_center[0] is x_center and img_center[1] is y_center.
        img_center = np.array(img.shape[:2])[::-1] / 2
        #horizontal stack --> return standard format in image center
        # (x_center, y_center, x_center, y_center) like (x_min, y_min, x_max, y_max)
        img_center = np.hstack((img_center, img_center))
        bboxes = bboxes.copy()
        img = img.copy()
        if random.random() < self.p:
            # filp the img
            img = img[:, ::-1, :] # horizontal flip --> flip the width
            # https://stackoverflow.com/questions/59861233/cv2-rectangle-calls-overloaded-method-although-i-give-other-parameter
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # if dont --> make error when draw by opencv :>
            
            # flip the bboxes (Explain in document)
            # x_min, x_max change
            bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])
            bbox_w = abs(bboxes[:, 0] - bboxes[:, 2])
            # convert the x_min in the top left and x_max in the bottom right.
            # x_min
            bboxes[:, 0] -= bbox_w
            # x_max
            bboxes[:, 2] += bbox_w
        
        return img, bboxes
              

class RandomVerticalFlip_(object):
    '''Radnomly vertical flip the Image with the probability p.
    Args:
        p: (float) The probability with which the image is flipped.

    Output: 
        img: (np.array) The flipped image with shape (HxWxC)
        bboxes: (np.array) The transformed bounding boxe with shape (N, 4)
                n is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
    '''
    def __init__(self, p):
        self.p = p
    
    def __call__(self,img, bboxes):
        img_center = np.array(img.shape[:2])[::-1] / 2
        img_center = np.hstack((img_center, img_center))
        
        bboxes = bboxes.copy()
        img = img.copy()
        
        if random.random() < self.p:
            # filp the img
            img = img[::-1, ::, :] # vertical flip --> flip the height
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # if dont --> make error when draw by opencv 
            
            # flip the bboxes 
            # flip y_min, y_max
            bboxes[:, [1, 3]] += 2 * (img_center[[1, 3]] - bboxes[:, [1, 3]])
            bbox_h = abs(bboxes[:, 1] - bboxes[:, 3])

            # convert the y_min in the top left and y_max in the bottom right
            # x_min
            bboxes[:, 1] -= bbox_h
            # x_max
            bboxes[:, 3] += bbox_h
        
        return img, bboxes

    
class RandomScale(object):
    ''' Randomly scale an image
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped (removed). The resolution is maintained, and the remaining
    area if any is filled by black color.

    Args:
        scale: (float or tuple(float))
               If **float**, the image is scale by a factor from a range (1 - scale, 1 + scale)
               If **typle**, the image is scale by a factor from value specified by the tuple.

        diff: (bool) If False, remain the aspect ratio of image. Otherwise, if True
              the scale of image's height and image's width is not the same.
        p: (float) The probability with which the image is scaled.

    Outputs: 
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxe with shape (N, 4)
                n is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
    '''

    def __init__(self, scale=0.2, diff=False, p=0.5):
        self.scale=scale
        self.diff = diff
        self.p = p

        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"     
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)
    
    def __call__(self, img, bboxes, labels, color):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()
        scale_color = color
        scale_color = np.array(scale_color)

        if random.random() < self.p:
            img_shape = img.shape

            if self.diff: # not remain aspect ratio
                # https://docs.python.org/3/library/random.html#random.uniform
                # Return a random floating point number
                scale_x = random.uniform(*self.scale)
                scale_y = random.uniform(*self.scale)
            else:
                scale_x = random.uniform(*self.scale)
                scale_y = scale_x
            
            # factor scale 
            factor_scale_x = 1 + scale_x
            factor_scale_y = 1 + scale_y

            # scale image
            img = cv2.resize(img, dsize = None, # dsize: desired size for the output image
                            fx = factor_scale_x, fy = factor_scale_y)
            
            # scale bboxes
            bboxes[:, :4] *= [factor_scale_x, factor_scale_y, factor_scale_x, factor_scale_y]

            '''But we will keep the size constant. If we are going to scale down (smaller), 
            there would be remaining area. We will color it black.'''
            # First we start by creating a black image of the size of our original image. 
            canvas = np.zeros(shape = img_shape, dtype = np.uint8)

            # Then we determine the the size of our scaled image
            '''If it exceeds the dimensions of the original image (we are scaling up) , 
            then it needs to be cut off at the original dimensions. 
            We then "paste" the resized image on the canvas.'''
            x_lim = int(min(factor_scale_x, 1) * img_shape[1])
            y_lim = int(min(factor_scale_y, 1) * img_shape[0])

            canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
            img = canvas

            bboxes, labels, color = clip_bboxes(bboxes, clip_box = [0, 0, 1 + img_shape[1], 1 + img_shape[0]], labels = labels,
                                alpha = 0.25, color = scale_color)
        
        return img, bboxes, labels, color
        
