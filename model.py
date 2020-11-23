from imageai.Detection import ObjectDetection
import os
import cv2
from PIL import Image
import numpy as np
import pytesseract
from pytesseract import Output
import shutil

def load_model(model = 'yolo'):
    '''
    :type: model: str
    :rtype: detector
    Input model name and output the corresponding model
    '''
    # Load object detection model
    execution_path = os.getcwd()
    detector = ObjectDetection()
    if model == 'yolo':
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(execution_path , "pretrained_model/yolo.h5"))
    elif model == 'resnet':
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(execution_path , "pretrained_model/resnet50_coco_best_v2.0.1.h5"))
    elif model == 'tinyyolo':
        detector.setModelTypeAsTinyYOLOv3()
        detector.setModelPath( os.path.join(execution_path , "pretrained_model/yolo-tiny.h5"))
    detector.loadModel()
    return detector

def object_detect(input_path, output_path):
    '''
	:type: input_path: str
    :type: output_path: str
	:rtype: list, image file
    Input input and output image path
    Output list of detected objects and image with bounding boxes
	'''
    # Detect object: furniture
    # Detect street signs
    detector = load_model()
    detections, extracted_images = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path, extract_detected_objects=True)
    return detections, extracted_images 

def extract_text(file, preprocess = 'thresh', lang = 'vie'):
    '''
	:type: file: image file
    :type: preprocess: str
    :type: lang: str
	:rtype: list
    Input image file, preprocess way and language
    Output list of textes being recognized
	'''
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # your path may be differen
    # Step 1: Preprocessing the image
    # convert the image to gray scale
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if preprocess == 'thresh':
    # Otsu's thresholding
        ret,img_pro = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif preprocess == 'blur':
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret, img_pro = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Step 2: Extract text from the images
    # extract text from the original image
    extracted_text_0 = pytesseract.image_to_string(img, lang = lang)
    # extract text from the first pre-processed image
    extracted_text_pro = pytesseract.image_to_string(img_pro, lang = lang)
    return extracted_text_pro

