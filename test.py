import keras_ocr
import cv2

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

# Example usage
#image_path = 'path/to/your/image.jpg'
image_path = './business.png'
lines = read_text_line_by_line(image_path)

# Print the lines
for line in lines:
    print(line)

