import numpy as np 
import cv2

### SOBEL ###
def sobel_function(img, blur_ksize=7, sobel_ksize=1, skipping_threshold=10, 
                   x_direction=True, y_direction=True):
    ''' sobel edge detection 
    Args:
        img: (np.array) img_array with RGB
        blur_ksize: (int) Kernel size parameter for Gaussian Blurry
        sobel_ksize: (int) size of extended Sobel Kernel. It should be 1, 3, 5, 7
        skipping_threshold: (int) ignore weakly edge
        x_direction: (bool) True with with edge enhanced in X-direction and otherwise
        y_direction: (bool) True with edge enhanced in Y-direction and otherwise
    Output:
        sobel_img: (np.array)
    '''

    # convert BGR to Gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    '''Sobel often uses a Gaussian filter to remove noise, smooth the image first 
    to make the edge detection algorithm work better.'''
    # https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    img_gaussian = cv2.GaussianBlur(gray_img, 
                                    ksize = (blur_ksize, blur_ksize), 
                                    sigmaX = 0)
    
    # sobel_img
    if x_direction:
        dx = 1
    else:
        dx = 0
    
    if y_direction:
        dy = 1
    else:
        dy = 0
    
    sobel_img = cv2.Sobel(src = img_gaussian, ddepth = cv2.CV_64F, 
                          dx = dx, dy = dy, ksize = sobel_ksize)
    #sobel_img = np.absolute(sobel_img)
    #sobel_img = np.uint8(sobel_img)
    
    # ignore weakly pixels
    for i in range(sobel_img.shape[0]):
        for j in range(sobel_img.shape[1]):
            if sobel_img[i][j] < skipping_threshold:
                sobel_img[i][j] = 0
            else:
                sobel_img[i][j] = 255
    
    return sobel_img

### CANNY EDGE ###
def canny_function(img, blur_ksize=7, threshold1=100, threshold2=200):
    ''' Canny Edge Detection
    Args:
        img: (np.array) img_array with RGB
        blur_ksize: (int) Kernel size parameter for Gaussian Blurry
        threshold1: (int) min threshold
        threshold2: (int) max threshold
    Output:
        canny_img: (np.array)
    '''

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, ksize = (blur_ksize, blur_ksize), sigmaX = 0)

    # canny edge
    canny_img = cv2.Canny(blurred_img, threshold1 = threshold1, threshold2 = threshold2)

    return canny_img

### Laplacian of Gaussian ### 
def laplacian_function(img, blur_ksize=7, lap_ksize=3, ddepth=cv2.CV_16S):
    ''' Laplacian of Gaussian
    Args: 
        img: (np.array) img_array with RGB
        blur_ksize: (int) Kernel size parameter for Gaussian Blurry
        lap_ksize: (int) Kernel size parameter for Laplacian
    Output:
        lap_img: (np.array)
    '''

    blurred_img = cv2.GaussianBlur(img, ksize = (blur_ksize, blur_ksize), sigmaX = 0)
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2GRAY)
   
    # laplacian
    lap_img = cv2.Laplacian(gray_img, ddepth = ddepth, ksize = lap_ksize)
    # converting back to uint8
    lap_img = cv2.convertScaleAbs(lap_img)
    
    return lap_img

### Difference of Gaussian ###
def dog_function(img, median_ksize=5, gaussian_ksize=0):
    ''' Difference of Gaussian
    Args:
        img: (np.array) img_array with RGB
        median_ksize: (int) Kernel size parameter for Median Bluury
        gaussian_ksize: (int) Kernel size parameter for Gaussian Bluury
    '''
    # median blur
    #https://theailearner.com/tag/cv2-medianblur/
    median_img = cv2.medianBlur(img, median_ksize)
    #median_img = cv2.cvtColor(median_img, cv2.COLOR_RGB2GRAY)

    # Calculate Gaussian blur with different sigma
    gaussian_3 = cv2.GaussianBlur(median_img, (gaussian_ksize, gaussian_ksize), 3)
    gaussian_5 = cv2.GaussianBlur(median_img, (gaussian_ksize, gaussian_ksize), 5)

    # calculate DoG
    dog_img = gaussian_3 - gaussian_5

    # remove negative value
    dog_img[dog_img < 0] = 0

    return dog_img
