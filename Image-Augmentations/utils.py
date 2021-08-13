# src: https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
# document: 
import cv2 
import numpy as np

def draw_bboxes(img, bbox, label, color, thickness):
    ''' draw the rectangle bboxes and its label
    Args:
        img: (np.array) with RGB and shape (H, W, C)
        bbox:(np.array) individual bbox with shape (4, ) with 4 represent to x_min, y_min, x_max. y_max
        label: (str): name of bbox
        color: (iterable) contain the color of its bbox 
        thickness: (int)
    '''
    cv2.rectangle(img, pt1 = (int(bbox[0]), int(bbox[1])), 
                  pt2 = (int(bbox[2]), int(bbox[3])), color = color, thickness = thickness)
    cv2.putText(img, text = label, org = (int(bbox[0]), int(bbox[1] - 5)), 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = color, 
                        thickness = 1, lineType = cv2.LINE_AA)

def bboxes_area(bboxes):
    '''Calculate the bounding boxes area
    Args:
        bboxes: (ndarray): contain bounding boxes with shape (N, 4) with
                N is number of boxes and 4 is represent to x_min, y_min, x_max, y_max.
    outputs: 
        area: (ndarray): contain the area of bounding boxes with shape (N, ) 
    '''
    area_bboxes = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    return area_bboxes
    
def clip_bboxes(bboxes, clip_box, labels, color, alpha):
    """ Clip the bounding boxes to the border of an image 
    Args:
        bboxes: (ndarray) Array contain bounding boxes with shape (N, 4)
                with N is number of bounding boxes and 4 represent to x_min, y_min, x_max. y_max.
        clip_box: (iterable) if array, it have shape (4, ) specifying the diagonal co-ordinates of the image.
                  The coordinates are represented in the formate x_min, y_min, x_max, y_max.
                  Almose case, clip_box = [0, 0, img.shape[1] (img_width), img.shape[0] (img_height)]
    Output:
        bboxes: (ndarray) Array containing **clipped** bounding boxes of shape `N X 4` where N is the 
                number of bounding boxes left are being clipped and the bounding boxes are represented in the
                format x_min, y_min, x_max, y_max.
    """

    bboxes = bboxes.copy()
    area_bboxes = bboxes_area(bboxes)
    
    # convert new x_min, y_min, x_max, y_max inside the border of an image after scale
    x_min = np.maximum(bboxes[:,0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bboxes[:,1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bboxes[:,2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bboxes[:,3], clip_box[3]).reshape(-1, 1)

    bboxes = np.hstack((x_min, y_min, x_max, y_max))

    # compute the percentage of bboxes area after
    print(f'bboxes_area(bboxes) = {bboxes_area(bboxes)}')
    print(f'area_bboxes = {area_bboxes}')
    percen_area_bboxes = bboxes_area(bboxes) / area_bboxes

    # the percentage of loss area
    percen_loss_area_bboxes = 1 - percen_area_bboxes
    print(f'loss_area_bboxes = {percen_loss_area_bboxes}')
    mask = (percen_loss_area_bboxes < alpha).astype(int)
    print(f'mask = {mask}')
    
    bboxes = bboxes[mask == 1, :]
    labels = labels[mask == 1]
    color = color[mask == 1]

    return bboxes, labels, color
