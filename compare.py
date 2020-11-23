from skimage import metrics
import numpy as np
import cv2

def mse(img1, img2):
    '''
	:type: img1: narray
    :type: img2: narray
	:rtype: float
    Input image array
    Output mean square error
	'''
    er = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    er = er/float(img1.shape[0]*img1.shape[1])
    return er 

def resize_images(img1_path, img2_path):
    '''
	:type: img1_path: str
    :type: img2_path: str
	:rtype: narray ,narray, float
    Input image paths
    Output resized image arrays and new size
	'''
    image1 = cv2.imread(img1_path,cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(img2_path,cv2.IMREAD_UNCHANGED)
    shape1 = image1.shape
    shape2 = image2.shape
    # Define new shape
    shape0 = min(shape1[0], shape2[0])
    shape1 = min(shape1[1], shape2[1])
    new_size = (shape0, shape1)

    image1_rsize = cv2.resize(image1, new_size)
    image2_rsize = cv2.resize(image2, new_size) 
    # get the gray scaled images
    img1_n = image1_rsize[:,:,0]
    img2_n = image2_rsize[:,:,0]
    return img1_n, img2_n, new_size

def compare_images(image1, image2):
    
    img1, img2, new_size = resize_images(image1, image2)
    er = mse(img1, img2)
    s_index = metrics.structural_similarity(img1, img2)
    
    # threshold to detect similarity
    if er < 3000 and s_index > 0.3:
        return 1
    else:
        return 0
