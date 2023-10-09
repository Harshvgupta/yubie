import time
import glob
import json
from pathlib import Path
from yubie import pubsub, Topics, VisionModule
from yubie.src.types import ImageObject
from PIL import Image

TIME_INTERVAL_IN_MS = 100


def publish_images(image_path):
    path_object = Path(image_path)
    file_name = path_object.name
    image = Image.open(image_path)
    image_object = ImageObject(
        label=file_name,
        data=image
    )
    pubsub.publish(Topics.ImageFromBot, image_object)


def main():
    cwd = Path(__file__).parent
    data_source = cwd.joinpath(
        '../../assets/image/trail_run_1/frontleft_fisheye/*.jpg')
    image_paths = glob.glob(str(data_source))
    detector = VisionModule.get_on_demand_detector()

    for image_path in image_paths:
        publish_images(image_path)
        time.sleep(TIME_INTERVAL_IN_MS/1000)
        output = detector.get_detections()
        if len(output[0]) == 0:
            print("No Detections")
        else:
            print(f"Detections: \n {json.dumps(output, indent = 2)}")


if __name__ == '__main__':
    main()
