import cv2
import numpy as np 
import os
import argparse
from src.edge_detector import canny_function

def main(img_name, output_name, img_dir='./imgs', threshold1=100, threshold2=200):
    IMAGE_PATH = os.path.join(img_dir, img_name) 
    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    canny_img = canny_function(img, threshold1 = threshold1, threshold2 = threshold2)
    cv2.imshow("window", canny_img)
    cv2.waitKey(0) 
    if output_name is not None:
        os.makedirs('./outputs', exist_ok = True)
        cv2.imwrite(os.path.join('./outputs', output_name), canny_img)
        print('Done saved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_name', dest = 'img_name', 
                        default = 'Input-image.jpg', type = str)
    parser.add_argument('--img_dir', dest = 'img_dir',
                        default = './imgs', type = str)
    parser.add_argument('--output_name', dest = 'output_name', 
                         default=None, type = str)
    parser.add_argument('--threshold1', dest = 'threshold1', 
                        default = 100, type = int)
    parser.add_argument('--threshold2', dest = 'threshold2', 
                        default = 200, type = int)
    args = parser.parse_args()
    main(args.img_name, args.output_name, args.img_dir, args.threshold1, args.threshold2)
    

                    