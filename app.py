import cv2
import easyocr
import os
import csv
import uuid
import re
from flask import Flask, render_template, Response, send_from_directory

app = Flask(__name__)

# Ensure directories exist
PLATES_IMG_DIR = 'data/plates_img'
DATA_DIR = 'data'

for directory in [PLATES_IMG_DIR, DATA_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# CSV file for storing detected plates
CSV_FILE = os.path.join(DATA_DIR, 'plates.csv')
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['License Plate'])

# Load Haar Cascade for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Store detected plates to avoid repetition
detected_plates = set()

# Indian License Plate Regex
INDIAN_PLATE_REGEX = r'^[A-Z]{2}[\s]?[0-9]{2}[\s]?[A-Z]{2}[\s]?[0-9]{4}$'

def is_valid_plate(plate):
    """Check if the detected plate matches the Indian license plate format."""
    return re.match(INDIAN_PLATE_REGEX, plate)

def detect_license_plates():
    cap = cv2.VideoCapture(0)  # Open webcam
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect number plates using Haar Cascade
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 50))

        for (x, y, w, h) in plates:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the region containing the license plate
            plate_img = gray[y:y+h, x:x+w]

            # Try OCR on the cropped plate image
            text_results = reader.readtext(plate_img)

            # If OCR fails, apply preprocessing and try again
            if not text_results:
                print("OCR failed on raw image, applying preprocessing...")

                # Adaptive Thresholding
                processed_plate = cv2.adaptiveThreshold(plate_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                        cv2.THRESH_BINARY, 11, 2)

                text_results = reader.readtext(processed_plate)

            # Process OCR results
            for (_, text, _) in text_results:
                text = text.replace(" ", "").upper()

                if is_valid_plate(text) and text not in detected_plates:
                    detected_plates.add(text)

                    # **Save the full image (real-life evidence)**
                    filename = f"{text}_{uuid.uuid4().hex[:6]}.jpg"
                    filepath = os.path.join('data/plates_img', filename)
                    cv2.imwrite(filepath, frame)  # Save the whole frame
                    print(f"Saved Full Frame: {filepath}")

                    # Save to CSV
                    with open(CSV_FILE, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([text])
                        print(f"Saved to CSV: {text}")

        # Convert frame to JPEG for video streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videofeed')
def videofeed():
    return Response(detect_license_plates(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def results():
    plates = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            plates = [row[0] for row in reader]

    images = os.listdir(PLATES_IMG_DIR)  # Get list of detected plate images

    return render_template('results.html', plates=plates, images=images)

@app.route('/data/plates_img/<filename>')
def plates_files(filename):
    return send_from_directory(PLATES_IMG_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
