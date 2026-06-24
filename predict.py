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
    
    # Run inference with lower confidence threshold for better detection
    results = model(img, imgsz=640, conf=0.15, verbose=True)
    
    # Check if anything was detected at all
    if len(results) == 0 or results[0].masks is None:
        print(f"Warning: No dry gauge polygons detected in {image_path}")
        return None

    # Extract detection info from the first result batch
    result = results[0]
    masks = result.masks.xy  # Pixel coordinates of all detected polygons
    
    # Since we only have one class (dry_gauge), we'll use all detected masks
    if len(masks) == 0:
        print(f"Warning: No mask polygons found in {image_path}")
        return None
    
    all_points = []
    
    # Collect all polygon points from all detected instances
    for polygon in masks:
        if len(polygon) > 0:
            all_points.append(polygon)
    
    # If we have multiple dry gauge detections, we'll use the largest one
    # or combine them to find the gauge boundaries
    if len(all_points) == 0:
        return None
    
    # If multiple detections, find the gauge that spans the most vertical space
    # (likely the full gauge)
    if len(all_points) > 1:
        # Find the polygon with the largest vertical extent
        max_extent = 0
        best_polygon = None
        for poly in all_points:
            extent = np.max(poly[:, 1]) - np.min(poly[:, 1])
            if extent > max_extent:
                max_extent = extent
                best_polygon = poly
        
        # Use the best polygon, or fallback to the first one
        primary_polygon = best_polygon if best_polygon is not None else all_points[0]
    else:
        primary_polygon = all_points[0]
    
    # Calculate gauge boundaries from the detected polygon
    gauge_top_y = np.min(primary_polygon[:, 1])
    gauge_bottom_y = np.max(primary_polygon[:, 1])
    total_pixel_height = gauge_bottom_y - gauge_top_y
    
    if total_pixel_height == 0:
        print("Warning: Zero height detected")
        return None
    
    # Since we only have dry gauge, we need to estimate the water level
    # Option 1: If the gauge has a visible water line, the dry part above water
    # will be detected. The water level is at the bottom of the dry detection.
    
    # Estimate water level at the bottom of the dry gauge detection
    # This assumes the dry part extends from water level upward
    water_line_y = gauge_bottom_y  # Water level is at the bottom of dry detection
    
    # Calculate submerged ratio (how much of the gauge is underwater)
    # Since we only detect dry part, the submerged ratio is inversely related
    # to how much of the gauge is dry
    submerged_ratio = 1.0 - (gauge_bottom_y - gauge_top_y) / total_pixel_height
    # This would be 0 if the entire gauge is dry, 1 if completely submerged
    
    # Alternative approach: Use the position of the dry detection relative to
    # the expected full gauge height. This requires knowing the full gauge height.
    # You can set a fixed full height or estimate it from the detection.
    
    # For now, we'll use a simple approach: assume the detection covers the
    # visible dry portion, and the water level is at its bottom
    physical_range = config["max_meters"] - config["min_meters"]
    
    # Calculate depth based on the assumption that the bottom of the dry gauge
    # detection represents the water level
    calculated_depth = config["min_meters"] + ((1.0 - submerged_ratio) * physical_range)
    
    # Clamp the value to valid range
    calculated_depth = max(config["min_meters"], min(config["max_meters"], calculated_depth))
    
    # Save validation image
    output_img = img.copy()
    
    # Draw the detected polygon
    cv2.polylines(output_img, [primary_polygon.astype(np.int32)], True, (0, 255, 0), 2)
    
    # Draw water line
    cv2.line(output_img, (0, int(water_line_y)), (img.shape[1], int(water_line_y)), (0, 0, 255), 3)
    
    # Add text information
    cv2.putText(output_img, f"Reading: {calculated_depth:.2f}m", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output_img, f"Gauge Top: {int(gauge_top_y)}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(output_img, f"Gauge Bottom: {int(gauge_bottom_y)}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
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
test_image = "test_1.jpg"  # Make sure this file exists
water_level = calculate_water_depth(test_image, site_profiles["default_site"])

if water_level is not None:
    print(f"\n[SUCCESS] Level: {water_level:.2f} meters")