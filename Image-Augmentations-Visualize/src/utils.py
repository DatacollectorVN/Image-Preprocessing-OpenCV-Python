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
    
def clip_bboxes(bboxes, clip_box, labels, alpha):
    """ Clip the bounding boxes to the border of an image 
    Args:
        bboxes: (ndarray) Array contain bounding boxes with shape (N, 4)
                with N is number of bounding boxes and 4 represent to x_min, y_min, x_max. y_max.
        clip_box: (iterable) if array, it have shape (4, ) specifying the diagonal co-ordinates of the image.
                  The coordinates are represented in the formate x_min, y_min, x_max, y_max.
                  Almose case, clip_box = [0, 0, img.shape[1] (img_width), img.shape[0] (img_height)]
        labels: (ndarray) Array contatin the name of bboxes corresponding to bboxes.
        alpha: (float) The configuration. If the percentage loss area of bbox after transform is smaller 
               than alpha then drop this bbox. Otherwise, remain bbox.
    
    Output:
        bboxes: (ndarray) Array containing **clipped** bounding boxes of shape `N X 4` where N is the 
                number of bounding boxes left are being clipped and the bounding boxes are represented in the
                format x_min, y_min, x_max, y_max.
        labels (ndarray)
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
    
    # remain the boxes with satisfied condition with corespoding to index
    bboxes = bboxes[mask == 1, :]
    labels = labels[mask == 1]
    

    return bboxes, labels

def rotate_img(img, angle):
    '''Rotate the image
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Args:
        img: (nd.array) image with shape (H, W, C)
        angle: (float) angle by which the image is to be rotated. 
    
    Output:
        img: (nd.array) Rotated image.
    '''

    # get the coordinates of center point of image
    h, w = img.shape[0], img.shape[1]
    center_x, center_y = w // 2, h // 2

    # transform matrix (applying the negative of the
    # angle to rotate clockwise)
    rotation_matrix = cv2.getRotationMatrix2D(center = (center_x, center_y), 
                                              angle = angle, scale = 1.0)
    
    # get the sine and cosine (cosine = alpha and sine = beta cause scale = 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # calculate the new bounding boxes dimentions of the image. 
    # (explain in document)
    new_w = int((w * cos) + (h * sin))
    new_h = int((w * sin) + (h + cos))

    # calcuate the new center point with new width and height
    new_center_x, new_center_y = new_w // 2, new_h // 2

    '''For ensuring the the center of the new image does not move since it is the axis of rotation itself'''
    delta_center_x, delta_center_y = (new_center_x - center_x, new_center_y - center_y)

    # adjust the rotation matrix tot ake into account translation
    rotation_matrix[0, 2] += delta_center_x
    rotation_matrix[1, 2] += delta_center_y

    # perform the actual rotation and return the image
    img = cv2.warpAffine(img, rotation_matrix, (new_w, new_h))

    return img 

def convert_2_to_4_corners(bboxes):
    """Get 4 corrners of bounding boxes
    
    Args:
        bboxes: (ndarray) Array contain bounding boxes with shape (N, 4)
                with N is number of bounding boxes and 4 represent to x_min, y_min, x_max. y_max.
        
    Ouput: 
        corners: (ndarray) Array contatin bounding boxes with shape (N, 8)
                with N is number of bounding boxes and 8 represent to x1, y1, x2, y2, x3, y3, x4, y4
    """

    # get width and height of bboxes 
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    # top-left corner
    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    # top-right corner
    x2 = x1 + width
    y2 = y1

    # bottom-left corner
    x3 = x1
    y3 = y1 + height

    # bottom-right corner
    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    # stack all of these
    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners

def convert_4_to_2_corners(corners_bboxes):
    """ Get 2 corrners of bounding boxes
    
    Args:
        corners_bboxes: (ndarray) Numpy array of shape `N x 8` containing N bounding boxes each described by their 
                                          corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Output:
        bboxes: (ndarray)  Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
                                    number of bounding boxes and the bounding boxes are represented in the
                                    format x_min, y_min, x_max, y_max
    """
    x_axis = corners_bboxes[:, [0, 2, 4, 6]]
    y_axis = corners_bboxes[:, [1, 3, 5, 7]]

    # get x_min, y_min, x_max, y_max
    x_min = np.min(x_axis, axis = 1).reshape(-1, 1)
    y_min = np.min(y_axis, axis = 1).reshape(-1, 1)
    x_max = np.max(x_axis, axis = 1).reshape(-1, 1)
    y_max = np.max(y_axis, axis = 1).reshape(-1, 1)

    # convert it to 2 corners
    bboxes = np.hstack(tup = (x_min, y_min, x_max, y_max))

    return bboxes
    
def rotate_bboxes(corners_bboxes, angle, center_x, center_y, w, h):
    """Rotate the bounding boxes.
    
    Args: 
        corners_bboxes: (ndarray) Numpy array of shape (N , 8) containing N bounding boxes each described by their 
                        corner coordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        angle: (float) Angle by which the image is to be rotated
        center_x: (int) x coordinate of the center of image (about which the box will be rotated)
        center_y: (int) y coordinate of the center of image (about which the box will be rotated)
        w: (int) Width of the image
        h: (int) Height of the image
    
    Output:
        rotated_corners_bboxes: (ndarray) Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
                                corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    '''Exaplain detail this function in document'''
    # reshape the corners_bboxes 
    corners_bboxes = corners_bboxes.reshape(-1, 2)
    corners_bboxes = np.hstack((corners_bboxes, np.ones(shape = (corners_bboxes.shape[0], 1), dtype = np.float32)))

    # transform matrix (applying the negative of the
    # angle to rotate clockwise)
    rotation_matrix = cv2.getRotationMatrix2D(center = (center_x, center_y), 
                                              angle = angle, scale = 1.0)

    # get the sine and cosine (cosine = alpha and sine = beta cause scale = 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # calculate the new bounding boxes dimentions of the image. 
    # (explain in document)
    new_w = int((w * cos) + (h * sin))
    new_h = int((w * sin) + (h + cos))

    # calcuate the new center point with new width and height
    new_center_x, new_center_y = new_w // 2, new_h // 2

    '''For ensuring the the center of the new image does not move since it is the axis of rotation itself'''
    delta_center_x, delta_center_y = (new_center_x - center_x, new_center_y - center_y)

    # adjust the rotation matrix tot ake into account translation
    rotation_matrix[0, 2] += delta_center_x
    rotation_matrix[1, 2] += delta_center_y

    # corners_bboxes after transform (dot product)
    rotated_corners_bboxes = np.dot(rotation_matrix, corners_bboxes.T).T

    # reshape it to the original shape (N, 8)
    rotated_corners_bboxes = rotated_corners_bboxes.reshape(-1, 8)

    return rotated_corners_bboxes

def letterbox_img(img, inp_dim):
    """Resize image with unchanged aspect ratio using padding
    
    Args:
        img: (ndarray) Original image RBG with shape (H, W, C)
        inp_dim: (ndarray) with (width, height) 
    
    Output:
        canvas: (ndarray) Array with desired size and contain the image at the center
    """
    
    # get width and height of original image
    img_w, img_h = img.shape[1], img.shape[0]
    
    # get width and height of desierd dimension
    w, h = inp_dim 

    # calculate the new_w and new_h of content of image inside the desired dimension
    # calculate the scale_factor according to width or heigh --> get the minimum value
    # cause we want to maintain the information of orginal image.
    scale_factor = min(w / img_w, h / img_h)
    new_w = int(img_w * scale_factor)
    new_h = int(img_h * scale_factor)

    # resized orginal image
    # why cv2.INTER_CUBIC ?
    # https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
    # https://chadrick-kwag.net/cv2-resize-interpolation-methods/
    resized_img = cv2.resize(src = img, dsize = (new_w, new_h), 
                             interpolation = cv2.INTER_CUBIC)
    
    # create the canvas
    canvas = np.full(shape = (inp_dim[1], inp_dim[0], 3), fill_value = 128)
    
    # paste the image on the canvas (at center)
    # canvas[top : bottom, left : right, :]
    top = (h - new_h) // 2
    bottom = top + new_h
    left = (w - new_w) // 2
    right = left + new_w
    canvas[top : bottom, left : right, :] = resized_img

    return canvas

if __name__ == "__main__":
    corners_bboxes = np.array([[1, 1, 3, 1, 1, 3, 3, 3], 
                               [2, 2, 5, 2, 2, 5, 5, 5]])
    corners_bboxes = corners_bboxes.reshape(-1, 2)
    
    #print(f'corners_bboxes.reshape(-1, 2) = \n{corners_bboxes.reshape(-1, 2)}')
    corners_bboxes = np.hstack((corners_bboxes, np.ones((corners_bboxes.shape[0],1), dtype = np.float32)))

    print(f'corners_bboxes = \n{corners_bboxes}')
