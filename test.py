# Install dependencies (skip if already installed)
#!pip install easyocr opencv-python-headless matplotlib

import cv2
import easyocr
import numpy as np
import re
import matplotlib.pyplot as plt
from google.colab import files

# STEP 1: Upload your image
print("Please upload your water meter image:")
uploaded = files.upload()

# Get the filename
image_path = list(uploaded.keys())[0]
print(f"Image uploaded: {image_path}")

# STEP 2: Load and inspect the image
img = cv2.imread(image_path)
if img is None:
    print("ERROR: Could not load image. Please check the file.")
    exit()

print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")

# STEP 3: Display original image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

# STEP 4: Try different preprocessing methods
def preprocess_image(img, method='combined'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == 'simple':
        # Simple threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
        
    elif method == 'adaptive':
        # Adaptive threshold (better for varying lighting)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        return thresh
        
    elif method == 'combined':
        # CLAHE + Gaussian blur + Adaptive threshold
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        return thresh
        
    elif method == 'edge':
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    elif method == 'resize':
        # Resize image to make text larger
        h, w = gray.shape
        scale = 2.0  # Double the size
        resized = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

# STEP 5: Try all preprocessing methods
methods = ['simple', 'adaptive', 'combined', 'edge', 'resize']
reader = easyocr.Reader(['en'], gpu=False)  # Force CPU for compatibility

all_readings = []

for method in methods:
    print(f"\n{'='*50}")
    print(f"Trying preprocessing method: {method}")
    print('='*50)
    
    processed = preprocess_image(img, method)
    
    # Display processed image
    plt.figure(figsize=(8, 8))
    plt.imshow(processed, cmap='gray')
    plt.title(f"Preprocessed - {method}")
    plt.axis('off')
    plt.show()
    
    # Run OCR
    try:
        results = reader.readtext(processed, detail=1, paragraph=False)
        
        print(f"\nRaw OCR results for {method}:")
        for bbox, text, confidence in results:
            print(f"  Text: '{text}', Confidence: {confidence:.3f}")
        
        # Extract numbers
        for bbox, text, confidence in results:
            # Look for decimal numbers
            numbers = re.findall(r'\d+\.\d+', text)
            for num in numbers:
                all_readings.append(float(num))
            # Also look for integers
            int_numbers = re.findall(r'\b\d+\b', text)
            for num in int_numbers:
                if len(num) > 1:  # Avoid single digits
                    all_readings.append(float(num))
                    
    except Exception as e:
        print(f"Error with method {method}: {e}")

# STEP 6: Try without preprocessing (direct OCR on original image)
print(f"\n{'='*50}")
print("Trying direct OCR on original image")
print('='*50)

try:
    results = reader.readtext(img, detail=1, paragraph=False)
    print("\nRaw OCR results on original image:")
    for bbox, text, confidence in results:
        print(f"  Text: '{text}', Confidence: {confidence:.3f}")
        
        numbers = re.findall(r'\d+\.\d+', text)
        for num in numbers:
            all_readings.append(float(num))
        int_numbers = re.findall(r'\b\d+\b', text)
        for num in int_numbers:
            if len(num) > 1:
                all_readings.append(float(num))
except Exception as e:
    print(f"Error: {e}")

# STEP 7: Process the results
print(f"\n{'='*50}")
print("FINAL RESULTS")
print('='*50)

if all_readings:
    print(f"\nAll detected readings: {all_readings}")
    
    # Filter readings (remove outliers if needed)
    # Assuming readings should be between 0 and 10 for a typical meter
    valid_readings = [r for r in all_readings if 0 < r < 10]
    
    if valid_readings:
        # Take the most common value or median
        final_reading = np.median(valid_readings)
        print(f"\n✅ FINAL METER READING: {final_reading:.3f}")
        print(f"Based on {len(valid_readings)} valid readings: {valid_readings}")
    else:
        print("\n⚠️ No valid readings found in expected range (0-10)")
        print(f"Raw readings: {all_readings}")
else:
    print("\n❌ No readings detected at all!")
    print("\nTroubleshooting suggestions:")
    print("1. Make sure the image clearly shows the meter numbers")
    print("2. Try a different image with better lighting")
    print("3. Check if the image contains any text at all")
    print("4. Try cropping the image to just the meter dial")

# STEP 8: Save preprocessed images for inspection (optional)
cv2.imwrite('preprocessed_simple.jpg', preprocess_image(img, 'simple'))
cv2.imwrite('preprocessed_adaptive.jpg', preprocess_image(img, 'adaptive'))
cv2.imwrite('preprocessed_combined.jpg', preprocess_image(img, 'combined'))
print("\nSaved preprocessed images for inspection")