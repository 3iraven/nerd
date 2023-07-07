import os
import urllib.request
import numpy as np
from app import app
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from docx import Document
from PIL import Image
import cv2

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docx'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def split_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (e.g., using Canny)
    edges = cv2.Canny(gray, 50, 150)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Find the first vertical line
    split_column = None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 5:  # Adjust the threshold as per your requirement
                split_column = x1
                break

    if split_column is not None:
        # Split the image based on the column index
        left_image = image[:, :split_column, :]
        right_image = image[:, split_column:, :]
        return left_image, right_image
    else:
        return None

class ParsingClass:
    def __init__(self):
        pass

    def parse_image(self, image):
        # Perform OCR using pytesseract
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        config_tesseract = '--psm 6 --oem 1'
        text = pytesseract.image_to_string(rgb, config=config_tesseract)
        return text

    def parse_word_document(self, docx_path):
        document = Document(docx_path)
        paragraphs = [p.text for p in document.paragraphs]
        parsed_text = ' '.join(paragraphs)
        return parsed_text

parsing_instance = ParsingClass()

@app.route('/file-upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp

    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if file_path.endswith('.docx'):
            parsed_text = parsing_instance.parse_word_document(file_path)
        else:
            # Split the image
            split_images = split_image(file_path)

            if split_images is not None:
                left_image, right_image = split_images

                # Process the left image
                left_text = parsing_instance.parse_image(left_image)

                # Process the right image
                right_text = parsing_instance.parse_image(right_image)

                parsed_text = "Left Image:\n" + left_text + "\n\nRight Image:\n" + right_text
            else:
                parsed_text = "Failed to split the image."

        response = {
            'message': 'File successfully uploaded',
            'file_name': filename,
            'right': right_text,
            'left': left_text
        }

        resp = jsonify(response)
        resp.status_code = 201
        return resp

    else:
        resp = jsonify({'message': 'Allowed file types are txt, pdf, png, jpg, jpeg, gif, docx'})
        resp.status_code = 400
        return resp

if __name__ == "__main__":
    app.run()

