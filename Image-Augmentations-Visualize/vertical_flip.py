import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import os 
import sys
from src.utils import draw_bboxes
from src.data_augmentation import RandomVerticalFlip_
import argparse
img_file = '051132a778e61a86eb147c7c6f564dfe.jpg'
annotations_file = '051132a778e61a86eb147c7c6f564dfe.txt'
color = {'Cardiomegaly' : [255, 0, 0], 
         'Aortic enlargement' : [0, 255, 0], 
         'Pleural thickening' : [0, 0, 255]}

def main(img_file=img_file, annotations_file=annotations_file, color=color, output_name=None):
        img_dir = os.path.join('dataset', img_file)
        annotations_dir = os.path.join('dataset', annotations_file)

        annotations = np.loadtxt(annotations_dir, delimiter = ',', dtype=object)
        labels = annotations[:, 4]
        bboxes = np.array(annotations[:, :4], dtype = np.float32)
        
        print(f'annotations = \n{annotations}')
        print(f'labels = \n{labels}')
        print(f'bboxes = \n{bboxes}')

        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f'img.shape = {img.shape}')

        horizontal_flip = RandomVerticalFlip_(p = 1)
        flip_img, flip_bboxes, labels = horizontal_flip(img, bboxes, labels)
        print(f'flipp_bboxes = \n{flip_bboxes}')
        
        # draw original img
        for i in range(len(labels)):
            draw_bboxes(img, bbox = bboxes[i], label = labels[i], color = color[labels[i]], thickness = 3)
        
        # draw flipped img
        for i in range(len(labels)):
            draw_bboxes(flip_img, bbox = flip_bboxes[i], label = labels[i], color = color[labels[i]], thickness = 3)

        cv2.imshow('original-img', img)
        cv2.imshow('flip-img', flip_img)
        cv2.waitKey(0)

        if output_name is not None:
            os.makedirs('./outputs', exist_ok = True)
            cv2.imwrite(os.path.join('./outputs', 'original-img.jpg'), img)
            cv2.imwrite(os.path.join('./outputs', output_name), flip_img)
            print('Done saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', dest = 'img_file', default = img_file, type = str)
    parser.add_argument('--annot_file', dest = 'annot_file', default = annotations_file, type = str)
    parser.add_argument('--output_name', dest = 'output_name', default = None, type = str)
    args = parser.parse_args()
    
    main(args.img_file, args.annot_file, output_name = args.output_name)