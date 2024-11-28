from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.yaml')

    # Define the path to your dataset
    path = 'C:/Users/User/Desktop/PCB Detection/dataset/data.yaml'

    # Train the model
    results = model.train(
        data=path,         # Dataset location
        epochs=50,         # Number of epochs
        batch=16,          # Batch size
        imgsz=640,         # Image size
        lr0=0.001,         # Initial learning rate
        optimizer='SGD',   # Optimizer
        patience=10,       # Early stopping patience
        weight_decay=0.0005, # Weight decay
        device='cuda',     # Use GPU
    )

    # Validate the model
    results = model.val()

    # Export the trained model in ONNX format
    success = model.export(format='onnx')

if __name__ == '__main__':
    main()
