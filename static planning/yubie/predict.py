import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(
    '/Users/pranomvignesh/Workfolder/yubie/yubie/train/runs/detect/train/weights/best.pt')

# Define path to video file
source = '/Users/pranomvignesh/Workfolder/yubie/assets/image/trail_run_1/frontleft_fisheye/frontleft_fisheye_image_0602.jpg'

# Run inference on the source
# results = model('/Users/pranomvignesh/Workfolder/yubie/assets/image/trail_run_1/frontleft_fisheye/frontleft_fisheye_image_0602.jpg')  # generator of Results objects


# Read the image
image = cv2.imread(source)


def detect_objects(image):
    results = model(image, classes=0, stream=True)
    for result in results:
        if len(result.boxes) > 0:
            for boxes in result.boxes:
                for box in boxes.xyxy:
                    bbox = [int(tensor.item()) for tensor in box]
                    x1, y1, x2, y2 = bbox
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    box_color = (100, 0, 255)
                    centroid_color = (0, 255, 100)
                    thickness = 2
                    radius = 5
                    cv2.circle(image, (cx, cy), radius, centroid_color, -1)
                    cv2.rectangle(image, (x1, y1), (x2, y2),
                                  box_color, thickness)
    return image


# Check if the image was successfully loaded
if image is not None:
    image = detect_objects(image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Could not load the image.")
