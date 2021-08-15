import random 
import numpy as np 
import cv2 
import os 
import sys
from utils import clip_bboxes, rotate_img, convert_2_to_4_corners, rotate_bboxes, convert_4_to_2_corners, letterbox_img

class RandomHorizontalFlip_(object):
    '''Radnomly horizontally flip the Image with the probability p.
    Args:
        p: (float) The probability with which the image is flipped.

    Output: 
        img: (np.array) The flipped image with shape (HxWxC)
        bboxes: (np.array) The transformed bounding boxe with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    '''
    
    def __init__(self, p):
        self.p = p

    def __call__(self, img, bboxes, labels):
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
        
        return img, bboxes, labels
              

class RandomVerticalFlip_(object):
    '''Radnomly vertical flip the Image with the probability p.
    Args:
        p: (float) The probability with which the image is flipped.

    Output: 
        img: (np.array) The flipped image with shape (HxWxC)
        bboxes: (np.array) The transformed bounding boxe with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    '''
    def __init__(self, p):
        self.p = p
    
    def __call__(self,img, bboxes, labels):
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
        
        return img, bboxes, labels

    
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
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
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
    
    def __call__(self, img, bboxes, labels):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()

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

            bboxes, labels = clip_bboxes(bboxes, clip_box = [0, 0, 1 + img_shape[1], 1 + img_shape[0]], labels = labels,
                                         alpha = 0.25)
        
        return img, bboxes, labels


class Scale(object):
    ''' Scale an image
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped (removed). The resolution is maintained, and the remaining
    area if any is filled by black color.

    Args:
        scale_x: (float) If **float**, the image's width is scale by a factor 1 + scale
        scale_y: (float) If **float**, the image's height is scale by a factor 1 + scale
        p: (float) The probability with which the image is scaled.

    Outputs: 
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxe with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    '''

    def __init__(self, scale_x=0.2, scale_y=0.2, p=0.5):
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.p = p
        assert self.scale_x > -1, "Scale factor can't be less than -1"
        assert self.scale_y > -1, "Scale factor can't be less than -1"
    
    def __call__(self, img, bboxes, labels):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()
        
        if random.random() < self.p:
            img_shape = img.shape
            
            # get factor scale accroding to x/y 
            factor_scale_x = 1 + self.scale_x
            factor_scale_y = 1 + self.scale_y

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

            bboxes, labels = clip_bboxes(bboxes, clip_box = [0, 0, 1 + img_shape[1], 1 + img_shape[0]], labels = labels,
                                         alpha = 0.25)
        
        return img, bboxes, labels


class RandomTranslate(object):
    ''' Randomly Translates the image
    Bounding boxes which have an area of less tahn 25% in the 
    remaining in the transformed image is dropped (removed). 
    The resolution is maintained, and the remaining area if any 
    is filled by black color.

    Args:
        translate: (float or tuple(float))
                   if **float**, the image is translated by a factor drawn
                   randomly from a range (1 - translate, 1 + translate)
                   if **tuple**, translate is drawn randomly from specified by the tuple.
        diff: (bool) If False, remain the aspect ratio of image. Otherwise, if True
              the scale of image's height and image's width is not the same.
        p: (float) The probability with which the image is translated.
    
    Output: 
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxe with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    '''

    def __init__(self, translate=0.2, diff=False, p=0.5):
        self.translate = translate
        self.diff = diff
        self.p = p

        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
            assert self.translate[0] > 0.0 and self.translate[0] < 1.0
            assert self.translate[1] > 0.0 and self.translate[1] < 1.0
        else: # self.translate == float
            assert self.translate > 0.0 and self.translate < 1.0
            self.translate = (- self.translate, self.translate)
    
    def __call__(self, img, bboxes, labels):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()

        if random.random() < self.p:
            img_shape = img.shape

            #percentage of the dimension of the image to translate
            translate_factor_x = random.uniform(*self.translate)
            translate_factor_y = random.uniform(*self.translate)

            if not self.diff:
                translate_factor_y = translate_factor_x
            
            # first initialise a black image about the size of our original image. 
            canvas = np.zeros(img_shape, dtype = np.uint8)
            
            '''Translate image'''
            # get the top-left corner co-ordinates of the shifted image.
            corner_x = int(translate_factor_x * img.shape[1])
            corner_y = int(translate_factor_y * img.shape[0])
            
            # the content's image is maintained after translated
            # img[top : bottom, left : right]
            mask = img[max(0, -corner_y) : min(img_shape[0], img_shape[0] - corner_y), 
                       max(0, -corner_x) : min(img_shape[1], img_shape[1] - corner_x)]
            
            # coordinate of translate image (x_min, y_min, x_max, y_max) in canvas
            coord_translate_img = [max(0, corner_x), max(0, corner_y), 
                                   min(img_shape[1], img_shape[1] + corner_x), min(img_shape[0], img_shape[0] + corner_y)]

            # add content's image is maintained after translated in canvas
            # canvas[top : bottom, left : right]
            canvas[coord_translate_img[1] : coord_translate_img[3], 
                   coord_translate_img[0] : coord_translate_img[2]] = mask
            img = canvas

            '''Translate bouding boxes'''
            # each box contain (x_min, y_min, x_max, y_max)
            bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

            bboxes, labels = clip_bboxes(bboxes, clip_box = [0, 0, 1 + img_shape[1], 1 + img_shape[0]], labels = labels,
                                         alpha = 0.25)

            return img, bboxes, labels

class Translate(object):
    ''' Randomly Translates the image
    Bounding boxes which have an area of less tahn 25% in the 
    remaining in the transformed image is dropped (removed). 
    The resolution is maintained, and the remaining area if any 
    is filled by black color.

    Args:
        translate_x: (float) If **float**, the image's width is translated with this factor
        translate_y: (float) If **float**, the image's height is translated with this factor     
        p: (float) The probability with which the image is translated.
    
    Output: 
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxe with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    '''

    def __init__(self, translate_x=0.2, translate_y=0.2, p=0.5):
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.p = p
        
        assert self.translate_x < 1.0 and self.translate_x > -1.0, "translate_x must be greater than -1 and smaller than 1"
        assert self.translate_y < 1.0 and self.translate_y > -1.0, "translate_y must be greater than -1 and smaller than 1"
    
    def __call__(self, img, bboxes, labels):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()

        if random.random() < self.p:
            img_shape = img.shape

            #percentage of the dimension of the image to translate
            translate_factor_x = self.translate_x
            translate_factor_y = self.translate_y 
            
            # first initialise a black image about the size of our original image. 
            canvas = np.zeros(img_shape, dtype = np.uint8)
            
            '''Translate image'''
            # get the top-left corner co-ordinates of the shifted image.
            corner_x = int(translate_factor_x * img.shape[1])
            corner_y = int(translate_factor_y * img.shape[0])
            
            # the content's image is maintained after translated
            # img[top : bottom, left : right]
            mask = img[max(0, -corner_y) : min(img_shape[0], img_shape[0] - corner_y), 
                       max(0, -corner_x) : min(img_shape[1], img_shape[1] - corner_x)]
            
            # coordinate of translate image (x_min, y_min, x_max, y_max) in canvas
            coord_translate_img = [max(0, corner_x), max(0, corner_y), 
                                   min(img_shape[1], img_shape[1] + corner_x), min(img_shape[0], img_shape[0] + corner_y)]

            # add content's image is maintained after translated in canvas
            # canvas[top : bottom, left : right]
            canvas[coord_translate_img[1] : coord_translate_img[3], 
                   coord_translate_img[0] : coord_translate_img[2]] = mask
            img = canvas

            '''Translate bouding boxes'''
            # each box contain (x_min, y_min, x_max, y_max)
            bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

            bboxes, labels = clip_bboxes(bboxes, clip_box = [0, 0, 1 + img_shape[1], 1 + img_shape[0]], labels = labels,
                                         alpha = 0.25)

            return img, bboxes, labels
    

class RandomRotate(object):
    ''' Randomly rotates an images
    Bounding boxes which habe an area og less than 25% in the remaining in the 
    transformed image is dropped (removed). The resolution is maintained, and the remaining
    area if nay is filled by black color.

    Args:
        angle: (float or tuple(float))
               If **float**, the image is rotated by a factor drawn
               randomly from a range (- angle, angle)
               If **tuple**, the angle is drawn randomly from value specified by the tuple
        p: (float) The probability with which the image is rotated.

    Output: 
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxe with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    '''

    def __init__(self, angle=20, p=0.5):
        self.angle = angle
        self.p = p

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"
        else:
            self.angle = (-self.angle, self.angle)
    
    def __call__(self, img, bboxes, labels):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()

        if random.random() < self.p:
            # get random angle from range
            angle = random.uniform(*self.angle)
            
            # get the width, height and center of point of image
            h, w = img.shape[0], img.shape[1]
            center_x, center_y = w // 2, h // 2

            # rotate the image
            img = rotate_img(img, angle)

            # convert 2 corners of boundig boxes to 4 corners (easy to calculate)
            corners = convert_2_to_4_corners(bboxes)
            
            # rotate bounding boxes
            corners = rotate_bboxes(corners_bboxes = corners, angle = angle, 
                                    center_x = center_x, center_y = center_y, 
                                    w = w, h = h)
            new_bboxes = convert_4_to_2_corners(corners)

            # calculate the scale_factor after rotation image (cause its dimension is change after rotation)
            scale_factor_x = img.shape[1] / w
            scale_factor_y = img.shape[0] / h

            # but we don't want to change the dimension of final image --> resize it to original dimension
            img = cv2.resize(img, (w, h))

            # then we must convert the coordinate of new_bboxes after rotation
            new_bboxes[:, :4] /= [scale_factor_x, scale_factor_y,scale_factor_x, scale_factor_y]
            bboxes = new_bboxes

            bboxes, labels = clip_bboxes(bboxes, clip_box = [0, 0, 1 + w, 1 + h], labels = labels,
                                                alpha = 0.25)
            
        return img, bboxes, labels
    

class Rotate(object):
    ''' Rotates an images
    Bounding boxes which habe an area og less than 25% in the remaining in the 
    transformed image is dropped (removed). The resolution is maintained, and the remaining
    area if nay is filled by black color.

    Args:
        angle: (float) Angle by which the image is to be rotated
        p: (float) The probability with which the image is rotated.

    Output: 
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxe with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    '''

    def __init__(self, angle=20, p=0.5):
        self.angle = angle
        self.p = p
    
    def __call__(self, img, bboxes, labels):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()

        if random.random() < self.p:
            # get random angle from range
            angle = self.angle
            
            # get the width, height and center of point of image
            h, w = img.shape[0], img.shape[1]
            center_x, center_y = w // 2, h // 2

            # rotate the image
            img = rotate_img(img, angle)

            # convert 2 corners of boundig boxes to 4 corners (easy to calculate)
            corners = convert_2_to_4_corners(bboxes)
            
            # rotate bounding boxes
            corners = rotate_bboxes(corners_bboxes = corners, angle = angle, 
                                    center_x = center_x, center_y = center_y, 
                                    w = w, h = h)
            new_bboxes = convert_4_to_2_corners(corners)

            # calculate the scale_factor after rotation image (cause its dimension is change after rotation)
            scale_factor_x = img.shape[1] / w
            scale_factor_y = img.shape[0] / h

            # but we don't want to change the dimension of final image --> resize it to original dimension
            img = cv2.resize(img, (w, h))

            # then we must convert the coordinate of new_bboxes after rotation
            new_bboxes[:, :4] /= [scale_factor_x, scale_factor_y,scale_factor_x, scale_factor_y]
            bboxes = new_bboxes

            bboxes, labels = clip_bboxes(bboxes, clip_box = [0, 0, 1 + w, 1 + h], labels = labels,
                                                alpha = 0.25)
            
        return img, bboxes, labels


class RandomShear(object):
    """ Randomly shears an image in the horizontal direction
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Args:
        shear_factor: (float or tuple(float))
                      If **float**, the image is sheared horizontally by a factor drawn 
                      randomly from a range (-`shear_factor`, `shear_factor`). 
                      If **tuple**, the `shear_factor` is drawn randomly from values specified by the 
                      tuple.
        p: (float) The probability with which the image is rotated.
    
    Output:
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxe with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    """

    def __init__(self, shear_factor=0.2, p=0.5):
        self.shear_factor = shear_factor
        self.p = p
        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for shearing factor"
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)
    
    def __call__(self, img, bboxes, labels):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()

        if random.random() < self.p:
            shear_factor = random.uniform(*self.shear_factor)

            # get the width, height of image
            h, w = img.shape[0], img.shape[1]

            # in document explain why we horizontal flip image when shear is negative
            '''
            if shear_factor < 0:
            1. Horizontal flip image.
            2. Shearing image with positive shear_factor.
            3. Horizontal flip image again
            '''

            if shear_factor < 0:
                img, bboxes = RandomHorizontalFlip_(p = 1)(img, bboxes)
            
            # create the shearing matrix
            shear_matrix = np.array([[1, abs(shear_factor), 0], 
                                     [0, 1, 0]])
            
            # when shearing, for maintaining the information of image 
            # then the image's width will change
            new_w = w + abs(shear_factor * h)

            # the cooridnate of bounding boxes (x_min, x_max) change
            bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

            # shearing the image 
            img = cv2.warpAffine(img, shear_matrix, dsize = (int(new_w), h))

            # if shear_factor < 0, we flip it again
            if shear_factor < 0:
                img, bboxes = RandomHorizontalFlip_(p = 1)(img, bboxes)

            # resize the image (after shearing) into original size
            img = cv2.resize(img, (w, h))

            # cause we resize the image --> resize the coordinates of bboxes
            scale_factor_x = new_w / w
            bboxes[:, :4] /= [scale_factor_x, 1, scale_factor_x, 1]

            # clip the bboxes
            bboxes, labels = clip_bboxes(bboxes, clip_box = [0, 0, 1 + w, 1 + h], labels = labels,
                                                alpha = 0.25)
        return img, bboxes, labels


class Shear(object):
    """ Randomly shears an image in the horizontal direction
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Args:
        shear_factor: (float or tuple(float))
                      If **float**, the image is sheared horizontally by a factor drawn 
                      randomly from a range (-`shear_factor`, `shear_factor`). 
                      If **tuple**, the `shear_factor` is drawn randomly from values specified by the 
                      tuple.
        p: (float) The probability with which the image is rotated.
    
    Output:
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxes with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    """

    def __init__(self, shear_factor=0.2, p=0.5):
        self.shear_factor = shear_factor
        self.p = p
    
    def __call__(self, img, bboxes, labels):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()

        if random.random() < self.p:
            shear_factor = self.shear_factor
            # get the width, height of image
            h, w = img.shape[0], img.shape[1]

            # in document explain why we horizontal flip image when shear is negative
            '''
            if shear_factor < 0:
            1. Horizontal flip image.
            2. Shearing image with positive shear_factor.
            3. Horizontal flip image again
            '''

            if shear_factor < 0:
                img, bboxes, labels = RandomHorizontalFlip_(p = 1)(img, bboxes, labels)
            
            # create the shearing matrix
            shear_matrix = np.array([[1, abs(shear_factor), 0], 
                                     [0, 1, 0]])
            
            # when shearing, for maintaining the information of image 
            # then the image's width will change
            new_w = w + abs(shear_factor * h)

            # the cooridnate of bounding boxes (x_min, x_max) change
            bboxes[:, [0, 2]] += ((bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

            # shearing the image 
            img = cv2.warpAffine(img, shear_matrix, dsize = (int(new_w), h))

            # if shear_factor < 0, we flip it again
            if shear_factor < 0:
                img, bboxes, labels = RandomHorizontalFlip_(p = 1)(img, bboxes, labels)

            # resize the image (after shearing) into original size
            img = cv2.resize(img, (w, h))

            # cause we resize the image --> resize the coordinates of bboxes
            scale_factor_x = new_w / w
            bboxes[:, :4] /= [scale_factor_x, 1, scale_factor_x, 1]

            # clip the bboxes
            bboxes, labels = clip_bboxes(bboxes, clip_box = [0, 0, 1 + w, 1 + h], labels = labels,
                                                alpha = 0.25)
        return img, bboxes, labels


class Resize(object):
    """Resize the image in accordance to `image_letter_box` function in darknet 
    The aspect ratio is maintained. The longer side is resized to the input 
    size of the network, while the remaining space on the shorter side is filled 
    with black color. **This should be the last transform**

    Args:
        inp_dim: (tuple(int)) with (width, height). Tuple containing the size to which the image will be resized.
    
    Outputs:
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxes with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    """

    def __init__(self, inp_dim):
        self.inp_dim = inp_dim
    
    def __call__(self, img, bboxes, labels):
        img = img.copy()
        bboxes = bboxes.copy()
        labels = labels.copy()

        desired_w, desired_h =  self.inp_dim
        img_w, img_h = img.shape[1], img.shape[0]

        # return the canvas with desired dimenion and contain information of image
        canvas = letterbox_img(img, self.inp_dim)
        #print(f'canvas = {canvas}')

        # calculate the scale_factor according to width or heigh --> get the minimum value
        # cause we want to maintain the information of orginal image
        scale_factor = min(desired_w / img_w, desired_h / img_h)

        # resize the coordinate of bboxes according to scale factor
        bboxes[:, :4] *= (scale_factor)

        # calculate the dimension of image (after resize) (not the dimension of canvas)
        new_w = scale_factor * img_w
        new_h = scale_factor * img_h

        # calculate the delta of height and width of image and canvas
        del_w = (desired_w - new_w) // 2
        del_h = (desired_h - new_h) // 2
        
        # create the translate_matrix for bbounding box to translate its coordinate 
        # correspoding to the location of image in canvas
        translate_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)

        #print(f'translate_matrix = {translate_matrix}')
        # translate the cooridnate of bounding boxes
        bboxes[:, :4] += translate_matrix

        img = canvas.astype(np.uint8)

        return img, bboxes, labels


class RandomHSV(object):
    """HSV transform to vary hue saturation and brightness
    Hue has a range of 0-179
    Saturation and Brightness have a range of 0-255. 
    Chose the amount you want to change thhe above quantities accordingly.

    Args:
        hue: (None or int or tuple (int))
             If None, the hue of the image is left unchanged. 
             If int, a random int is uniformly sampled from (-hue, hue) and added to the 
             hue of the image. 
             If tuple, the int is sampled from the range specified by the tuple.
        saturation: (None, or int or tuple(int))
                    If None, the saturation of the image is left unchanged. 
                    If int, a random int is uniformly sampled from (-saturation, saturation) 
                    and added to the hue of the image. 
                    If tuple, the int is sampled from the range  specified by the tuple. 
        brightness: (None or int or tuple(int))
                    If None, the brightness of the image is left unchanged. 
                    If int, a random int is uniformly sampled from (-brightness, brightness) 
                    and added to the hue of the image. 
                    If tuple, the int is sampled from the range  specified by the tuple.  

    Outputs:
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxes with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    """

    def __init__(self, hue=None, saturation=None, brightness=None):
        
        if hue:
            self.hue = hue
        else:
            self.hue = 0
        
        if saturation:
            self.saturation = saturation
        else:
            self.saturation = 0
        
        if brightness:
            self.brightness = brightness
        else:
            self.brightness = 0
        
        if type(self.hue) != tuple:
            self.hue = (-self.hue, self.hue)
            
        if type(self.saturation) != tuple:
            self.saturation = (-self.saturation, self.saturation)
        
        if type(brightness) != tuple:
            self.brightness = (-self.brightness, self.brightness)
    
    def __call__(self, img, bboxes, labels):
        # img which RGB
        # get random value 
        hue = random.randint(*self.hue)
        saturation = random.randint(*self.saturation)
        brightness = random.randint(*self.brightness)

        img = img.astype(int)

        #HSV
        hsv = np.array([hue, saturation, brightness]).astype(int)
        hsv = np.reshape(hsv, newshape = (1, 1, 3))
        
        # add hsv into image
        img += hsv

        # clip the image min = 0 and max = 255
        img = np.clip(img, 0, 255)

        # change the first channel (Red) with min = 0 and max = 179
        img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)

        img = img.astype(np.uint8)

        return img, bboxes, labels


class HSV(object):
    """HSV transform to vary hue saturation and brightness
    Hue has a range of 0-179
    Saturation and Brightness have a range of 0-255. 
    Chose the amount you want to change thhe above quantities accordingly.

    Args:
        hue: (None or int)
             If int, a random int is uniformly sampled and added to the image.
        saturation: (None, or int)
                    If None, the saturation of the image is left unchanged. 
                    If int, a random int is uniformly sampled and added to the hue of the image. 
        brightness: (None or int)
                    If int, a random int is uniformly sampled and added to the hue of the image. 
                    
    Outputs:
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxes with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    """

    def __init__(self, hue=None, saturation=None, brightness=None):
        
        if hue:
            self.hue = hue
        else:
            self.hue = 0
        
        if saturation:
            self.saturation = saturation
        else:
            self.saturation = 0
        
        if brightness:
            self.brightness = brightness
        else:
            self.brightness = 0
    
    def __call__(self, img, bboxes, labels):
        # img which RGB
        hue = self.hue
        saturation = self.saturation
        brightness = self.brightness
        
        img = img.astype(int)

        #HSV
        hsv = np.array([hue, saturation, brightness]).astype(int)
        hsv = np.reshape(hsv, newshape = (1, 1, 3))
        
        # add hsv into image
        img += hsv

        # clip the image min = 0 and max = 255
        img = np.clip(img, 0, 255)

        # change the first channel (Red) with min = 0 and max = 179
        img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)

        img = img.astype(np.uint8)

        return img, bboxes, labels


'''Combining multiple transformations'''
class Sequence_(object):
    '''Initialize squence object of transofrmation to the image/bboxes
    
    Args: 
        augmentations: (list) List containing the Transformation Object in Sequence
                       they are to be applied.
        probs: (float or list(float))
               If **float**, the probability with which each of the transformation will 
               be applied. 
               If **list**, the length must be equal to *augmentations*. 
               Each element of this list is the probability with which each 
               corresponding transformation is applied.
    
    Output:
        img: (ndarray) The scaled image with shape (H, W, C)
        bboxes: (ndarray) The transformed bounding boxes with shape (N, 4)
                N is number of bounding boxes and 4 represents x_min, y_min, x_max, y_max.
        labels: (ndarray) the transformed labels
    '''

    def __init__(self, augmentations, probs=1):
        self.augmentations = augmentations
        self.probs = probs

        if type(self.probs) == list:
            assert len(self.probs) == len(self.augmentations), "The length of probs and augmentations must be equal"
    
    def __call__(self, img, bboxes, labels):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs
            
            if random.random() < prob:
                img, bboxes, labels = augmentation(img, bboxes, labels)
        
        return img, bboxes, labels        