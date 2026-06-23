import cv2
import numpy as np
import os
from ultralytics import YOLO

# 1. Load your custom trained model
MODEL_PATH = r"D:\PythonProjects\Demo1\runs\segment\train-3\weights\best.pt"
model = YOLO(MODEL_PATH)

def calculate_water_depth(image_path, config):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    img = cv2.imread(image_path)
    
    # FIX 1: Explicitly pass imgsz=640 to match training resolution
    # FIX 2: Lower the conf threshold to 0.15 so a model trained on 8 images can still trigger hits
    results = model(img, imgsz=640, conf=0.15, verbose=True)
    
    # Check if anything was detected at all
    if len(results) == 0 or results[0].masks is None:
        print(f"Warning: No water gauge polygons detected in {image_path}")
        return None

    # Extract detection info from the first result batch
    result = results[0]
    classes = result.boxes.cls.cpu().numpy()
    masks = result.masks.xy  # Pixel coordinates

    highest_wet_y = float('inf')
    lowest_dry_y = 0
    all_points = []

    # Map directly by ID numbers (0 and 1) to ignore label text spelling issues
    for idx, class_id in enumerate(classes):
        polygon = masks[idx]
        if len(polygon) == 0: 
            continue
            
        all_points.append(polygon)

        if int(class_id) == 1:  # wet class ID
            min_y = np.min(polygon[:, 1])
            if min_y < highest_wet_y:
                highest_wet_y = min_y
        elif int(class_id) == 0:  # dry class ID
            max_y = np.max(polygon[:, 1])
            if max_y > lowest_dry_y:
                lowest_dry_y = max_y

    # Fallback mechanisms for partial detections
    if highest_wet_y == float('inf'): highest_wet_y = lowest_dry_y
    if lowest_dry_y == 0: lowest_dry_y = highest_wet_y

    # Process overall geometry
    water_line_y = (highest_wet_y + lowest_dry_y) / 2.0
    flat_points = np.vstack(all_points)
    gauge_top_y = np.min(flat_points[:, 1])
    gauge_bottom_y = np.max(flat_points[:, 1])
    total_pixel_height = gauge_bottom_y - gauge_top_y

    if total_pixel_height == 0:
        return None

    submerged_ratio = (gauge_bottom_y - water_line_y) / total_pixel_height
    submerged_ratio = max(0.0, min(1.0, submerged_ratio))

    physical_range = config["max_meters"] - config["min_meters"]
    calculated_depth = config["min_meters"] + (submerged_ratio * physical_range)

    # Save validation image
    output_img = img.copy()
    cv2.line(output_img, (0, int(water_line_y)), (img.shape[1], int(water_line_y)), (0, 0, 255), 3)
    cv2.putText(output_img, f"Reading: {calculated_depth:.2f}m", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imwrite("latest_reading_output.jpg", output_img)
    print("Debug image saved as 'latest_reading_output.jpg'")

    return calculated_depth

# Base Configuration Profile
site_profiles = {
    "default_site": {
        "min_meters": 0.0,
        "max_meters": 2.5
    }
}

# Run execution test
test_image = "test_2.webp" # Make sure this file exists in D:\PythonProjects\Demo1\
water_level = calculate_water_depth(test_image, site_profiles["default_site"])

if water_level is not None:
    print(f"\n[SUCCESS] Level: {water_level:.2f} meters")
