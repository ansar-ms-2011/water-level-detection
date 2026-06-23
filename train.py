from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load the ultra-lightweight pretrained YOLOv8-Nano segmentation model
    # This model is fast, highly accurate for structural detection, and fits perfectly in 4GB VRAM
    model = YOLO('yolov8n-seg.pt')  

    # 2. Kick off training using our 4GB GPU optimized parameters
    model.train(
        data='dataset.yaml',      # Path to your dataset configuration file
        epochs=100,               # Run through the dataset 100 times for solid learning
        imgsz=640,                # Resize images to 640x640 internally
        batch=8,                  # Small batch size to safely prevent 4GB VRAM overloads
        workers=2,                # Keeps CPU/RAM usage stable on a laptop
        device=0                  # Explicitly force training on your NVIDIA GPU
    )
