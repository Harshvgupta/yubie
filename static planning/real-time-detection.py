# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple image display example."""

import argparse
import logging
import sys
import time
import os

import cv2
import numpy as np
from scipy import ndimage
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.time_sync import TimedOutError
from datetime import datetime
from ultralytics import YOLO

model = YOLO(
    '/Users/pranomvignesh/Workfolder/yubie/yubie/train/runs/detect/train/weights/best.pt')

CWD = Path(__file__).parent

_LOGGER = logging.getLogger(__name__)

VALUE_FOR_Q_KEYSTROKE = 113
VALUE_FOR_ESC_KEYSTROKE = 27

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}

current_date = datetime.now()
formatted_date = current_date.strftime("%b_%d_%y-%I_%M_%p")
IMAGE_SRC = f'./image_source/{formatted_date}'
IMAGE_DEST = f'./detections/{formatted_date}'


def create_dirs(rel_path):
    abs_path = CWD.joinpath(rel_path).absolute()
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)


def image_to_opencv(image, auto_rotate=True):
    """Convert an image proto message to an openCV image."""
    num_channels = 1  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
        extension = '.png'
    else:
        dtype = np.uint8
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_channels = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_channels = 4
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_channels = 1
            dtype = np.uint16
        extension = '.jpg'

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape(
                (image.shot.image.rows, image.shot.image.cols, num_channels))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    if auto_rotate:
        img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])

    return img, extension


def reset_image_client(robot):
    """Recreate the ImageClient from the robot object."""
    del robot.service_clients_by_name['image']
    del robot.channels_by_authority['api.spot.robot']
    return robot.ensure_client('image')


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
    return image, results


def main(argv):
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-sources',
                        help='Get image from source(s)', action='append')
    parser.add_argument('--image-service', help='Name of the image service to query.',
                        default=ImageClient.default_service_name)
    parser.add_argument('-j', '--jpeg-quality-percent', help='JPEG quality percentage (0-100)',
                        type=int, default=50)
    parser.add_argument('-c', '--capture-delay', help='Time [ms] to wait before the next capture',
                        type=int, default=100)
    parser.add_argument('-r', '--resize-ratio', help='Fraction to resize the image', type=float,
                        default=1)
    parser.add_argument(
        '--disable-full-screen',
        help='A single image source gets displayed full screen by default. This flag disables that.',
        action='store_true')
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')
    parser.add_argument('--save-source', help='saves the source image',
                        action='store_false')
    parser.add_argument('--save-detections', help='saves the detections',
                        action='store_false')
    options = parser.parse_args(argv)

    # Overwrite arguments

    options.disable_full_screen = True
    options.auto_rotate = True
    options.save_source = os.environ.get('SAVE_SOURCE', False)
    options.save_detections = os.environ.get('SAVE_DETECTIONS', False)

    if options.save_source:
        for image_src in options.image_sources:
            create_dirs(f'{IMAGE_SRC}/{image_src}')

    if options.save_detections:
        for image_src in options.image_sources:
            create_dirs(f'{IMAGE_DEST}/{image_src}')

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(options.image_service)
    requests = [
        build_image_request(source,
                            pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
                            quality_percent=options.jpeg_quality_percent,
                            resize_ratio=options.resize_ratio) for source in options.image_sources
    ]

    for image_source in options.image_sources:
        cv2.namedWindow(image_source, cv2.WINDOW_NORMAL)
        if len(options.image_sources) > 1 or options.disable_full_screen:
            cv2.setWindowProperty(
                image_source, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        else:
            cv2.setWindowProperty(
                image_source, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    keystroke = None
    timeout_count_before_reset = 0
    t1 = time.time()
    image_count = 0
    while keystroke != VALUE_FOR_Q_KEYSTROKE and keystroke != VALUE_FOR_ESC_KEYSTROKE:
        try:
            images_future = image_client.get_image_async(requests, timeout=0.5)
            while not images_future.done():
                keystroke = cv2.waitKey(25)
                if keystroke == VALUE_FOR_ESC_KEYSTROKE or keystroke == VALUE_FOR_Q_KEYSTROKE:
                    sys.exit(1)
            images = images_future.result()
        except TimedOutError as time_err:
            if timeout_count_before_reset == 5:
                # To attempt to handle bad comms and continue the live image stream, try recreating the
                # image client after having an RPC timeout 5 times.
                _LOGGER.info('Resetting image client after 5+ timeout errors.')
                image_client = reset_image_client(robot)
                timeout_count_before_reset = 0
            else:
                timeout_count_before_reset += 1
        except Exception as err:
            _LOGGER.warning(err)
            continue
        for i in range(len(images)):
            image, ext = image_to_opencv(images[i], options.auto_rotate)
            if options.save_source:
                cv2.imwrite(
                    f'{IMAGE_SRC}/{images[i].source.name}/image_{str(image_count).zfill(5)}{ext}', image)
            image, results = detect_objects(image)
            if options.save_detections:
                cv2.imwrite(
                    f'{IMAGE_DEST}/{images[i].source.name}/image_{str(image_count).zfill(5)}{ext}', image)
                for result in results:
                    result.save_txt(
                        f'{IMAGE_DEST}/{images[i].source.name}/image_{str(image_count).zfill(5)}.txt')
            cv2.imshow(images[i].source.name, image)
        keystroke = cv2.waitKey(options.capture_delay)
        image_count += 1
        # print(f'Mean image retrieval rate: {image_count/(time.time() - t1)}Hz')


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    if not main(sys.argv[1:]):
        sys.exit(1)

# Command to run
# python3 image_viewer.py 192.168.x.x --image-sources frontleft_fisheye_image
