import cv2
import numpy as np 
import os
import argparse
from edge_detector import sobel_function

def main(img_name, output_name, img_dir='./imgs', x_direction=True, y_direction=True):
    IMAGE_PATH = os.path.join(img_dir, img_name)
    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    sobel_img = sobel_function(img, x_direction = x_direction, y_direction = y_direction)
    cv2.imshow("window", sobel_img)
    cv2.waitKey(0) 
    if output_name is not None:
        os.makedirs('./outputs', exist_ok = True)
        cv2.imwrite(os.path.join('./outputs', output_name), sobel_img)
        print('Done saved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_name', dest = 'img_name', 
                        default = 'Input-image.jpg', type = str)
    parser.add_argument('--img_dir', dest = 'img_dir',
                        default = './imgs', type = str)
    parser.add_argument('--output_name', dest = 'output_name', 
                         default=None, type = str)
    parser.add_argument('--x_direction', dest = 'x_direction', 
                        action = 'store_true')
    parser.add_argument('-no-x_direction', dest = 'x_direction', 
                        action = 'store_false')
    parser.set_defaults(x_direction = True)
    parser.add_argument('--y_direction', dest = 'y_direction', 
                        action = 'store_true')
    parser.add_argument('-no-y_direction', dest = 'y_direction', 
                        action = 'store_false')
    parser.set_defaults(y_direction = True)
    args = parser.parse_args()
    main(args.img_name, args.output_name, args.img_dir, args.x_direction, args.y_direction)
    

                    