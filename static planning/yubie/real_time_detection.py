from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(
    '/Users/pranomvignesh/Workfolder/yubie/yubie/train/runs/detect/train/weights/best.pt')

# Define path to video file
source = '/Users/pranomvignesh/Workfolder/yubie/assets/video/trail_run_2_right.mp4'

# Run inference on the source
results = model(source, save=True)  # generator of Results objects

# for result in results:
#     print(result)
#     break

