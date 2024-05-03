import glob
import re
from collections import defaultdict

import cv2 as cv
import numpy as np
import pytesseract
import streamlit as st
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


def get_objects(processor, model, img: Image.Image) -> list[str]:
	# lower image resolution for faster processing
	img.thumbnail((400, 400), Image.Resampling.LANCZOS)
	inputs = processor(images=img, return_tensors="pt")
	outputs = model(**inputs)

	# convert outputs (bounding boxes and class logits) to COCO API
	# let's only keep detections with score > 0.9
	target_sizes = torch.tensor([img.size[::-1]])
	results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

	objs = []
	for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
		if score.item() > 0.5:
			objs.extend(model.config.id2label[label.item()].split())

	return objs

def is_img_dark(img: np.ndarray) -> bool:
	'''
	Given an image in numpy array format, return if the image is dark or not
	'''
	return cv.mean(img)[0] < 127

def get_text(img: Image.Image) -> list[str]:
	cv_img = np.array(img)
	cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)
	if is_img_dark(cv_img):
		cv_img = cv.bitwise_not(cv_img)
	text = pytesseract.image_to_string(cv_img)
	text = re.sub(r'\n+', ' ', text.strip())
	return text.split()

@st.cache_resource
def get_processor_and_model():
	print('loading model')
	return (
		DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm"),
		DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
	)

@st.cache_data
def build_img_index(img_directory):
	print('start building index')

	processor, model = get_processor_and_model()
	keyword2path = defaultdict(list)
	path2keyword = defaultdict(list)

	paths = glob.glob(img_directory)
	for path in paths:
		with Image.open(path).convert('RGB') as im:
			print(f'getting objects for {path}')
			objects = get_objects(processor, model, im)
			print(f'getting text in {path}')
			ocr_text = get_text(im)

			for obj in objects:
				keyword2path[obj.lower()].append(path)

			for text in ocr_text:
				keyword2path[text.lower()].append(path)

			path2keyword[path].extend(objects)
			path2keyword[path].extend(ocr_text)

	print('finish building index')

	return keyword2path, path2keyword


if __name__ == '__main__':
	img_directory = 'img/*'
	txt_directory = 'txt/*'
	keyword2path, path2keyword = build_img_index(img_directory)

	st.title('Content-Based Image Retrieval')
	wrapper = st.container(border=True)
	search_box = wrapper.text_input('Query')
	show_captions = wrapper.checkbox('Show captions')
	wrapper.write(f'Results for {search_box}')

	num_cols = 3
	cols = wrapper.columns(num_cols)

	if search_box:
		search_box = search_box.lower()
		if search_box in keyword2path:
			paths = keyword2path[search_box]
			for i, path in enumerate(paths):
				cols[i % num_cols].image(path, width=200, caption=', '.join(path2keyword[path]) if show_captions else None)
		else:
			wrapper.write('No results found')
	else:
		paths = glob.glob(img_directory)
		for i, path in enumerate(paths):
			cols[i % num_cols].image(path, width=200, caption=', '.join(path2keyword[path]) if show_captions else None)