# License Plate Recognition 🚗🔍  

## About  
This is a basic license plate recognition system using **Flask, OpenCV, and EasyOCR**. It detects license plates from a live camera feed, extracts the numbers, and saves them in a CSV file along with captured images.  

## Requirements  
- Python  
- OpenCV (`opencv-python`)  
- EasyOCR (`easyocr`)  
- Flask (`flask`)   

## How to Run  
### 1. Clone the Repository
```sh
    git clone https://github.com/shwetabagade26/number_plate_recognition.git
    cd number_plate_recognition
```

### 2. Create a Virtual Environment 
```sh
    \myenv\Scripts\activate 
```

### 3. Install Dependencies
```sh
    pip install -r requirements.txt
```

### 4. Run the Application
```sh
    python app.py
```

### 6. Access the Application
Open your browser and go to:
(Remember to add app.run(debug=True,port=5000) in app.py)
```
    http://localhost:5000
```
## Output Screenshots
![Index Page Screenshot](images/Index.jpg)
![Results](images/Result.jpg)

### Why CUDA Wasn't Used
Currently, my laptop storage is limited, and downloading Visual Studio components required for CUDA Toolkit is not possible. After upgrading my storage space, I'll implement CUDA, which will improve OCR speed and accuracy.

### Future Plans
- Upgrade storage space to support CUDA installation.
- Implement GPU acceleration for faster OCR.
- Enhance OCR accuracy using deep learning models.
