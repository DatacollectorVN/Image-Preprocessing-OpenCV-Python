import cv2
import numpy as np 
import os
import argparse
from edge_detector import laplacian_function

def main(img_name, output_name, img_dir='./imgs', lap_ksize=5):
    IMAGE_PATH = os.path.join(img_dir, img_name) 
    lap_img = laplacian_function(IMAGE_PATH, lap_ksize = lap_ksize)
    cv2.imshow("window", lap_img)
    cv2.waitKey(0) 
    if output_name is not None:
        os.makedirs('./outputs', exist_ok = True)
        cv2.imwrite(os.path.join('./outputs', output_name), lap_img)
        print('Done saved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_name', dest = 'img_name', 
                        default = 'Input-image.jpg', type = str)
    parser.add_argument('--img_dir', dest = 'img_dir',
                        default = './imgs', type = str)
    parser.add_argument('--output_name', dest = 'output_name', 
                         default = None, type = str)
    parser.add_argument('--lap_ksize', dest = 'lap_ksize', 
                        default = 3, type = int)
    args = parser.parse_args()
    main(args.img_name, args.output_name, args.img_dir, args.lap_ksize)
    

                    