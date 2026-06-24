import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import os

class UniversalGaugeReader:
    def __init__(self):
        """Initialize the gauge reader with adaptive parameters"""
        # Tesseract configuration for digit recognition
        self.ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
        
    def preprocess_image(self, image):
        """Adaptive preprocessing for any gauge image"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance contrast using CLAHE (Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise while preserving edges
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        return denoised
    
    def detect_gauge_region(self, image):
        """Automatically detect the gauge region in the image"""
        gray = self.preprocess_image(image)
        
        # Use edge detection to find the gauge
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to close gaps in edges
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest circular/oval contour (likely the gauge)
        gauge_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter by area (remove too small)
            if area < 1000:
                continue
            
            # Check circularity (gauge is typically circular/oval)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Circularity between 0.3 and 1.0 for gauge-like shapes
                if 0.3 <= circularity <= 1.0:
                    if area > max_area:
                        max_area = area
                        gauge_contour = contour
        
        # If no circular contour found, use largest contour
        if gauge_contour is None and contours:
            gauge_contour = max(contours, key=cv2.contourArea)
        
        if gauge_contour is not None:
            return cv2.boundingRect(gauge_contour)
        
        # If still no contour, use whole image
        h, w = gray.shape
        return (0, 0, w, h)
    
    def detect_water_level(self, image):
        """Detect water level using multiple methods"""
        gray = self.preprocess_image(image)
        h, w = gray.shape
        
        # Method 1: Otsu thresholding to find dark regions (water)
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up noise
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        water_level_y = None
        water_contour = None
        
        if contours:
            # Find the largest dark region (likely water)
            # Water typically appears at the bottom of the gauge
            bottom_regions = []
            
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Filter by size and position
                if area > 500:  # Minimum area
                    # Check if region is at bottom half
                    if y + ch > h * 0.4:
                        bottom_regions.append((contour, area, y + ch))
            
            if bottom_regions:
                # Select the region with largest area or lowest position
                bottom_regions.sort(key=lambda x: x[2], reverse=True)
                water_contour = bottom_regions[0][0]
                x, y, cw, ch = cv2.boundingRect(water_contour)
                water_level_y = y  # Top of water
        
        # Method 2: If Otsu fails, try adaptive threshold
        if water_level_y is None:
            thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour in bottom half
                bottom_contours = [c for c in contours if cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] > h * 0.4]
                if bottom_contours:
                    water_contour = max(bottom_contours, key=cv2.contourArea)
                    x, y, cw, ch = cv2.boundingRect(water_contour)
                    water_level_y = y
        
        # Method 3: Edge detection for water line
        if water_level_y is None:
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                horizontal_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 10 or angle > 170:
                        horizontal_lines.append((y1 + y2) // 2)
                
                if horizontal_lines:
                    water_level_y = max(horizontal_lines)
        
        return water_level_y, water_contour
    
    def extract_numbers(self, image, gauge_bbox=None):
        """Extract all numbers from the gauge with their positions"""
        gray = self.preprocess_image(image)
        
        # If gauge bounding box is provided, crop to gauge region
        if gauge_bbox:
            x, y, w, h = gauge_bbox
            roi = gray[y:y+h, x:x+w]
        else:
            roi = gray
            x, y = 0, 0
        
        # Enhance for OCR
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use Tesseract to get data with positions
        data = pytesseract.image_to_data(thresh, config=self.ocr_config, 
                                        output_type=pytesseract.Output.DICT)
        
        numbers = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text and text.replace('.', '').isdigit():
                try:
                    num = float(text) if '.' in text else int(text)
                    # Get center Y position (relative to original image)
                    num_y = y + data['top'][i] + data['height'][i] // 2
                    num_x = x + data['left'][i] + data['width'][i] // 2
                    numbers.append({
                        'value': num,
                        'x': num_x,
                        'y': num_y,
                        'bbox': (x + data['left'][i], y + data['top'][i],
                                data['width'][i], data['height'][i])
                    })
                except ValueError:
                    continue
        
        return numbers
    
    def find_number_at_level(self, numbers, water_level_y):
        """Find the number closest to the water level"""
        if not numbers or water_level_y is None:
            return None
        
        # Find number closest to water level
        closest = None
        min_distance = float('inf')
        
        for num in numbers:
            distance = abs(num['y'] - water_level_y)
            if distance < min_distance:
                min_distance = distance
                closest = num
        
        return closest
    
    def read_gauge(self, image_path):
        """
        Main method to read gauge from any image
        Returns: (water_level_number, water_level_y, annotated_image)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        original = image.copy()
        result = image.copy()
        
        # Step 1: Detect gauge region
        gauge_bbox = self.detect_gauge_region(image)
        if gauge_bbox:
            x, y, w, h = gauge_bbox
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result, "Gauge Region", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Step 2: Detect water level
        water_level_y, water_contour = self.detect_water_level(image)
        
        if water_level_y is not None:
            # Draw water level line
            cv2.line(result, (0, water_level_y), (result.shape[1], water_level_y), 
                    (0, 0, 255), 3)
            cv2.putText(result, f"Water Level: Y={water_level_y}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if water_contour is not None:
                x, y, w, h = cv2.boundingRect(water_contour)
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Step 3: Extract numbers
        numbers = self.extract_numbers(image, gauge_bbox)
        
        # Draw all detected numbers
        for num in numbers:
            x, y, w, h = num['bbox']
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 0), 1)
            cv2.putText(result, str(num['value']), (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Step 4: Find number at water level
        if water_level_y is not None and numbers:
            closest_num = self.find_number_at_level(numbers, water_level_y)
            
            if closest_num:
                # Highlight the number at water level
                x, y, w, h = closest_num['bbox']
                cv2.rectangle(result, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 3)
                cv2.putText(result, f"Reading: {closest_num['value']}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                return closest_num['value'], water_level_y, result
        
        return None, water_level_y, result
    
    def batch_process(self, image_folder):
        """Process all images in a folder"""
        results = {}
        
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(image_folder, filename)
                try:
                    value, y, annotated = self.read_gauge(image_path)
                    results[filename] = {
                        'value': value,
                        'water_level_y': y,
                        'image': annotated
                    }
                    print(f"{filename}: {value if value else 'Not detected'}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    results[filename] = {'error': str(e)}
        
        return results

# Usage examples
def main():
    # Initialize the universal gauge reader
    reader = UniversalGaugeReader()
    
    # Option 1: Process a single image
    image_path = "test_2.webp"  # Make sure this file exists
    
    try:
        value, water_y, annotated = reader.read_gauge(image_path)
        
        if value is not None:
            print(f"✅ Water level reading: {value}")
            print(f"📏 Water level Y position: {water_y}")
            
            # Display result
            plt.figure(figsize=(12, 6))
            
            original = cv2.imread(image_path)
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            plt.subplot(1, 2, 1)
            plt.imshow(original_rgb)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(annotated_rgb)
            plt.title(f'Detected: {value}')
            plt.axis('off')
            
            plt.show()
        else:
            print("❌ Could not detect water level")
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Option 2: Process all images in a folder
    # results = reader.batch_process("path/to/image/folder")
    # for filename, data in results.items():
    #     print(f"{filename}: {data.get('value', 'Not detected')}")

if __name__ == "__main__":
    main()