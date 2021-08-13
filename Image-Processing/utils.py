import numpy as np 
import cv2
# https://www.pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/
# https://learnopencv.com/otsu-thresholding-with-opencv/

def remove_black_bg(img, square=False):
    ''' Remove black background by image thresholding with thresh Otsu 
    Args:
        img: (np.array) with channels RGB
    Outputs:
        T
    '''
    img_tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    T, thresholded = cv2.threshold(img_tmp, 0, 255, cv2.THRESH_OTSU)
    bbox = cv2.boundingRect(thresholded)
    x, y, w, h = bbox
    
    if square:
        if w < h:
            reduant = h - w
            foreground = img[y : y + h, x : x + w + reduant]
        elif h < w:
            reduant = w - h
            foreground = img[y : y + h + reduant, x : x + w]
        else:
            foreground = img[y : y + h, x : x + w]
    else:
        foreground = img[y : y + h, x : x + w]
    
    return T, foreground