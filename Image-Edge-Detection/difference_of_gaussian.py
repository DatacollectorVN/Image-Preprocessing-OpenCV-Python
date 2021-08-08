import cv2
import numpy as np 
import os
import argparse
from edge_detector import dog_function

def main(img_name, output_name, img_dir='./imgs', median_ksize=5, gaussian_ksize=0):
    IMAGE_PATH = os.path.join(img_dir, img_name)
    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    dog_img = dog_function(img, median_ksize, gaussian_ksize)
    cv2.imshow("window", dog_img)
    cv2.waitKey(0) 
    if output_name is not None:
        os.makedirs('./outputs', exist_ok = True)
        cv2.imwrite(os.path.join('./outputs', output_name), dog_img)
        print('Done saved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_name', dest = 'img_name', 
                        default = 'Input-image.jpg', type = str)
    parser.add_argument('--img_dir', dest = 'img_dir',
                        default = './imgs', type = str)
    parser.add_argument('--output_name', dest = 'output_name', 
                         default = None, type = str)
    parser.add_argument('--median_ksize', dest = 'median_ksize', 
                        default = 5, type = int)
    parser.add_argument('--gaussian_ksize', dest = 'gaussian_ksize', 
                        default = 0, type = int)
    args = parser.parse_args()
    main(args.img_name, args.output_name, args.img_dir, args.median_ksize, args.gaussian_ksize)
    

                    