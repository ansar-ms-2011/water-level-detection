import cv2
import numpy as np
import pytesseract
from scipy import ndimage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter

class AdaptiveGaugeReader:
    def __init__(self):
        """Initialize with minimal assumptions"""
        self.ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
        
    def preprocess_adaptive(self, image):
        """Adaptive preprocessing with multiple strategies"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Try multiple preprocessing methods
        processed = []
        
        # Method 1: Simple threshold
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed.append(thresh1)
        
        # Method 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        processed.append(thresh2)
        
        # Method 3: Edge detection
        edges = cv2.Canny(gray, 50, 150)
        processed.append(edges)
        
        # Method 4: Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        processed.append(magnitude)
        
        return processed, gray
    
    def detect_regions_by_intensity(self, image):
        """Detect regions based on intensity clustering (no color assumption)"""
        if len(image.shape) == 3:
            # Use multiple color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Get intensity from L channel (lightness)
            intensity = lab[:,:,0].flatten().reshape(-1, 1)
        else:
            intensity = image.flatten().reshape(-1, 1)
        
        # Cluster intensities into 2-3 groups
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(intensity)
        
        # Reshape to image shape
        if len(image.shape) == 3:
            labels = labels.reshape(image.shape[:2])
        else:
            labels = labels.reshape(image.shape)
        
        # Find the darkest and lightest regions
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)
        
        # Darkest region (likely water or background)
        dark_region = (labels == sorted_indices[0])
        # Lightest region (likely numbers or markings)
        light_region = (labels == sorted_indices[-1])
        
        return dark_region, light_region, labels
    
    def detect_water_without_assumptions(self, image):
        """
        Detect water level using multiple methods without assumptions
        about color, shape, or position
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        water_candidates = []
        
        # Method 1: Intensity-based region detection
        dark_region, _, _ = self.detect_regions_by_intensity(image)
        dark_region = dark_region.astype(np.uint8) * 255
        
        # Find contours in dark regions
        contours1, _ = cv2.findContours(dark_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours1:
            area = cv2.contourArea(cnt)
            if area > 100:  # Minimum area, can be adjusted
                x, y, cw, ch = cv2.boundingRect(cnt)
                water_candidates.append({
                    'contour': cnt,
                    'bbox': (x, y, cw, ch),
                    'area': area,
                    'y': y,
                    'method': 'intensity'
                })
        
        # Method 2: Edge-based detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find horizontal lines (potential water surface)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=20, maxLineGap=20)
        
        if lines is not None:
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Consider lines within 15 degrees of horizontal
                if angle < 15 or angle > 165:
                    # Line length
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length > 20:  # Minimum line length
                        avg_y = (y1 + y2) // 2
                        horizontal_lines.append({
                            'y': avg_y,
                            'length': length,
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                        })
            
            # Group similar lines
            if horizontal_lines:
                # Cluster by y position
                ys = np.array([line['y'] for line in horizontal_lines]).reshape(-1, 1)
                if len(ys) > 1:
                    kmeans_lines = KMeans(n_clusters=min(3, len(ys)), random_state=42, n_init=10)
                    line_labels = kmeans_lines.fit_predict(ys)
                    
                    # For each cluster, find the most prominent line
                    for cluster_id in range(min(3, len(ys))):
                        cluster_indices = np.where(line_labels == cluster_id)[0]
                        if len(cluster_indices) > 0:
                            cluster_lines = [horizontal_lines[i] for i in cluster_indices]
                            # Choose the longest line in cluster
                            best_line = max(cluster_lines, key=lambda x: x['length'])
                            water_candidates.append({
                                'y': best_line['y'],
                                'line': best_line,
                                'method': 'edge',
                                'confidence': len(cluster_indices) / len(horizontal_lines)
                            })
        
        # Method 3: Texture-based detection (water often has different texture)
        # Use local variance as texture measure
        kernel_size = 15
        local_var = cv2.Laplacian(gray, cv2.CV_64F)
        local_var = np.abs(local_var)
        local_var = cv2.GaussianBlur(local_var, (kernel_size, kernel_size), 0)
        
        # Find regions with low variance (uniform areas like water)
        low_var_threshold = np.percentile(local_var, 30)
        low_var_regions = local_var < low_var_threshold
        
        # Find contours of low variance regions
        low_var_uint8 = low_var_regions.astype(np.uint8) * 255
        contours2, _ = cv2.findContours(low_var_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours2:
            area = cv2.contourArea(cnt)
            if area > 500:  # Minimum area
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Check if region is large and uniform
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                mean_intensity = np.mean(gray[mask == 255])
                std_intensity = np.std(gray[mask == 255])
                
                if std_intensity < 30:  # Uniform region
                    water_candidates.append({
                        'contour': cnt,
                        'bbox': (x, y, cw, ch),
                        'area': area,
                        'y': y + ch,  # Bottom of region
                        'method': 'texture',
                        'mean_intensity': mean_intensity,
                        'std_intensity': std_intensity
                    })
        
        # Method 4: Gradient-based detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find horizontal gradient boundaries (water surface typically creates strong horizontal gradient)
        horizontal_gradient = np.abs(grad_y)
        horizontal_gradient = cv2.GaussianBlur(horizontal_gradient, (5, 5), 0)
        
        # Find peaks in horizontal gradient (potential water boundaries)
        for y in range(10, h-10):
            row_gradient = np.mean(horizontal_gradient[y-5:y+5, :])
            if row_gradient > np.percentile(horizontal_gradient, 90):  # High gradient
                water_candidates.append({
                    'y': y,
                    'method': 'gradient',
                    'strength': row_gradient
                })
        
        return water_candidates
    
    def extract_all_text_regions(self, image):
        """Extract all text/number regions without assumptions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use morphological operations to find text regions
        kernel = np.ones((3,3), np.uint8)
        
        # Different thresholding methods to capture text
        text_regions = []
        
        # Method 1: Otsu threshold
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_regions.append(thresh1)
        
        # Method 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        text_regions.append(thresh2)
        
        # Method 3: Morphological gradient
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, thresh3 = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_regions.append(thresh3)
        
        # Combine all methods
        combined = np.zeros_like(gray)
        for region in text_regions:
            combined = cv2.bitwise_or(combined, region)
        
        # Clean up
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        return combined
    
    def find_numbers_with_positions(self, image):
        """Find all numbers and their positions using multiple OCR passes"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        all_numbers = []
        
        # Multiple preprocessing strategies for better OCR
        preprocessed_images = []
        
        # Strategy 1: Original
        preprocessed_images.append(gray)
        
        # Strategy 2: Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        preprocessed_images.append(enhanced)
        
        # Strategy 3: Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(binary)
        
        # Strategy 4: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(adaptive)
        
        # Try OCR on all strategies
        for idx, proc_img in enumerate(preprocessed_images):
            try:
                # Use Tesseract to get data with positions
                data = pytesseract.image_to_data(proc_img, config=self.ocr_config,
                                               output_type=pytesseract.Output.DICT)
                
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    if text and text.replace('.', '').replace('-', '').isdigit():
                        try:
                            # Parse number
                            num = float(text) if '.' in text else int(text)
                            
                            # Get position
                            x = data['left'][i]
                            y = data['top'][i]
                            w = data['width'][i]
                            h = data['height'][i]
                            conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0
                            
                            # Only include high confidence detections
                            if conf > 30:
                                all_numbers.append({
                                    'value': num,
                                    'x': x,
                                    'y': y + h//2,  # Center Y
                                    'bbox': (x, y, w, h),
                                    'confidence': conf,
                                    'strategy': idx
                                })
                        except (ValueError, TypeError):
                            continue
            except Exception as e:
                continue
        
        # Group and deduplicate numbers
        if all_numbers:
            # Group by value and position proximity
            unique_numbers = []
            used_indices = set()
            
            for i, num1 in enumerate(all_numbers):
                if i in used_indices:
                    continue
                    
                # Find similar numbers
                group = [num1]
                for j, num2 in enumerate(all_numbers[i+1:], i+1):
                    if j in used_indices:
                        continue
                    
                    # Check if same value and close position
                    if num1['value'] == num2['value']:
                        x_diff = abs(num1['x'] - num2['x'])
                        y_diff = abs(num1['y'] - num2['y'])
                        if x_diff < 50 and y_diff < 30:  # Close together
                            group.append(num2)
                            used_indices.add(j)
                
                # Use the one with highest confidence
                best = max(group, key=lambda x: x['confidence'])
                unique_numbers.append(best)
                used_indices.add(i)
            
            return unique_numbers
        
        return []
    
    def find_number_at_level(self, numbers, candidates):
        """Find the number that best corresponds to water level"""
        if not numbers or not candidates:
            return None, None
        
        # Get all candidate y-positions
        candidate_ys = []
        for cand in candidates:
            if 'y' in cand:
                candidate_ys.append(cand['y'])
        
        if not candidate_ys:
            return None, None
        
        # Find the most likely water level (median or most common)
        candidate_ys = sorted(candidate_ys)
        
        # If we have multiple candidates, cluster them
        if len(candidate_ys) > 3:
            # Use histogram to find peaks
            hist, bins = np.histogram(candidate_ys, bins=min(len(candidate_ys)//2, 10))
            peak_bin = np.argmax(hist)
            water_level = (bins[peak_bin] + bins[peak_bin+1]) // 2
        else:
            # Use median
            water_level = int(np.median(candidate_ys))
        
        # Find number closest to water level
        closest_num = None
        min_distance = float('inf')
        
        for num in numbers:
            distance = abs(num['y'] - water_level)
            if distance < min_distance:
                min_distance = distance
                closest_num = num
        
        return closest_num, water_level
    
    def read_gauge(self, image_path):
        """Main method - completely adaptive with minimal assumptions"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        result = image.copy()
        h, w = image.shape[:2]
        
        print("🔍 Detecting water level candidates...")
        candidates = self.detect_water_without_assumptions(image)
        print(f"   Found {len(candidates)} water level candidates")
        
        # Extract numbers
        print("🔢 Extracting numbers from image...")
        numbers = self.find_numbers_with_positions(image)
        print(f"   Found {len(numbers)} numbers")
        
        # Find number at water level
        closest_num, water_level = self.find_number_at_level(numbers, candidates)
        
        # Visualize results
        # Draw all candidates
        for cand in candidates:
            if 'y' in cand:
                cv2.line(result, (0, cand['y']), (w, cand['y']), (255, 255, 0), 1)
        
        # Draw water level (best estimate)
        if water_level is not None:
            cv2.line(result, (0, water_level), (w, water_level), (0, 0, 255), 3)
            cv2.putText(result, f"Water Level: Y={water_level}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw all numbers
        for num in numbers:
            x, y, w_num, h_num = num['bbox']
            cv2.rectangle(result, (x, y), (x+w_num, y+h_num), (255, 255, 0), 1)
            cv2.putText(result, str(num['value']), (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Highlight the chosen number
        if closest_num:
            x, y, w_num, h_num = closest_num['bbox']
            cv2.rectangle(result, (x-5, y-5), (x+w_num+5, y+h_num+5), (0, 255, 0), 3)
            cv2.putText(result, f"✅ Reading: {closest_num['value']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            return closest_num['value'], water_level, result, candidates
        
        return None, water_level, result, candidates
    
    def analyze_confidence(self, candidates, numbers, water_level):
        """Analyze confidence of the detection"""
        confidence_score = 0
        factors = []
        
        # Factor 1: Number of candidates
        if len(candidates) > 2:
            confidence_score += 20
            factors.append("Multiple candidates found")
        
        # Factor 2: Numbers found
        if len(numbers) > 2:
            confidence_score += 20
            factors.append("Multiple numbers found")
        
        # Factor 3: Closest number proximity
        if water_level and numbers:
            closest = None
            min_dist = float('inf')
            for num in numbers:
                dist = abs(num['y'] - water_level)
                if dist < min_dist:
                    min_dist = dist
                    closest = num
            
            if closest and min_dist < 50:  # Close to water level
                confidence_score += 30
                factors.append(f"Number {closest['value']} close to water level (dist={min_dist})")
        
        # Factor 4: Distribution of candidates
        if candidates:
            ys = [c['y'] for c in candidates if 'y' in c]
            if len(ys) > 1:
                std_dev = np.std(ys)
                if std_dev < 20:  # Candidates are consistent
                    confidence_score += 30
                    factors.append("Candidates are consistent")
        
        return min(confidence_score, 100), factors

# Simple wrapper for easy use
def quick_gauge_read(image_path):
    """Quick and simple function to read any gauge"""
    reader = AdaptiveGaugeReader()
    value, water_level, result, candidates = reader.read_gauge(image_path)
    return value, result

# Example usage with detailed output
if __name__ == "__main__":
    reader = AdaptiveGaugeReader()
    
    # Process image
    image_path = "test_1.jpg"  # Replace with your test image path
    
    try:
        value, water_level, result, candidates = reader.read_gauge(image_path)
        
        print("\n" + "="*50)
        print("📊 RESULTS")
        print("="*50)
        print(f"Water Level Reading: {value if value else '❌ Not detected'}")
        print(f"Water Level Y-Position: {water_level}")
        print(f"Number of candidates: {len(candidates)}")
        
        # Confidence analysis
        numbers = reader.find_numbers_with_positions(cv2.imread(image_path))
        confidence, factors = reader.analyze_confidence(candidates, numbers, water_level)
        print(f"\nConfidence Score: {confidence}%")
        print("Factors:")
        for factor in factors:
            print(f"  ✓ {factor}")
        print("="*50)
        
        # Display result
        plt.figure(figsize=(15, 8))
        
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_rgb)
        plt.title('Original Image', fontsize=14)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_rgb)
        plt.title(f'Detection Result\nReading: {value if value else "Not detected"}', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")