from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import keras_ocr
import cv2

app = Flask(__name__)

def read_text_line_by_line(image_path):
    # Load the OCR model
    pipeline = keras_ocr.pipeline.Pipeline()

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform OCR
    predictions = pipeline.recognize([image])

    # Extract line-level text
    lines = []
    current_line = ""
    current_line_y = None
    for word in predictions[0]:
        text = word[0]
        _, _, _, y2 = word[1]
        if current_line_y is not None and (y2 - current_line_y)[0] < 10:
            current_line += " " + text
        else:
            lines.append(current_line.strip())
            current_line = text
        current_line_y = y2

    # Add the last line
    lines.append(current_line.strip())

    return lines

@app.route('/extract_text', methods=['POST'])
def extract_text():
    # Check if the 'image' file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'})

    image_file = request.files['image']

    # Save the uploaded image file
    if image_file:
        filename = secure_filename(image_file.filename)
        image_path = os.path.join('uploads', filename)
        image_file.save(image_path)

        # Extract text line by line
        lines = read_text_line_by_line(image_path)

        # Delete the temporary image file
        os.remove(image_path)

        return jsonify({'lines': lines})

    return jsonify({'error': 'Invalid request'})

if __name__ == '__main__':
    app.run()
