from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
from model import object_detect, extract_text
from compare import compare_images
import shutil

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks

app = FastAPI()

@app.post("/detectObject")
def detectObject(file: UploadFile = File(...)):
	'''
	:type: file: image file
	:rtype: dict
    Input an image and output detected objects and image with bouding boxes
	'''
	# save the load image
	with open("images/image.jpg", "wb") as buffer:
		shutil.copyfileobj(file.file, buffer)
	
	# detect the objects
	input_path = "images/image.jpg"
	output_path = "images/imageObject.jpg"

	detections, extracted_images = object_detect(input_path, output_path) 
	file_path = "images/imageObject.jpg"
	response = {}
	for eachItem in detections:
		response[eachItem["name"]] = eachItem["percentage_probability"]

	return {'detections': response, 'extracted_images': FileResponse(file_path)}

@app.post("/extractText")
def extractText(file: UploadFile = File(...)):
	'''
	:type: file: image file
	:rtype: []
    Input an image and output text recognition
	'''
	with open("images/imageText.jpg", "wb") as buffer:
		shutil.copyfileobj(file.file, buffer)
	extracted_text_pro = extract_text("images/imageText.jpg", preprocess = 'thresh', lang = 'vie')
	splits_pro = extracted_text_pro.splitlines()
	return splits_pro


@app.post("/identityVerification")
def compareFace(identityPaper: UploadFile = File(...), photo: UploadFile = File(...)):
	'''
	:type: file: image file
	:type: file: image file
	:rtype: int
    Input images and output 0 if not match and 1 if match
	'''
	# save two images
	with open("images/identityPaper.jpg", "wb") as buffer:
		shutil.copyfileobj(identityPaper.file, buffer)
	with open("images/photo.jpg", "wb") as buffer:
		shutil.copyfileobj(photo.file, buffer)

	input_path1 = "images/identityPaper.jpg"
	output_path1 = "images/identityPaperExtracted.jpg"
	input_path2 = "images/photo.jpg"
	output_path2 = "images/photoExtracted.jpg"

	# Extract face out of identity paper
	detections1, extracted_images1 = object_detect(input_path1, output_path1)
	# Extract face out of slef taken photo
	detections2, extracted_images2 = object_detect(input_path2, output_path2)
	# Compare the two photo and output the results
	res = compare_images(extracted_images1[0], extracted_images2[0])
	return res


if __name__=="__main__":
	uvicorn.run("main:app", reload = True)


