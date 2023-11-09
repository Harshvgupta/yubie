import logging
import argparse
import sys
import os
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry

from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from yubie import VisionModule


ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


class ImageHandler():
    LOGGER = logging.getLogger()

    def __init__(self, options):
        cv2.namedWindow("Real time detections", cv2.WINDOW_NORMAL)
        self.detector = VisionModule.get_on_demand_detector()
        bosdyn.client.util.setup_logging(options.verbose)
        self.sdk = bosdyn.client.create_standard_sdk('image_handler')

        try:
            self.robot = self.sdk.create_robot(options.hostname)
            bosdyn.client.util.authenticate(self.robot)
            self.robot.time_sync.wait_for_sync()
        except bosdyn.client.RpcError as e:
            self.LOGGER.error(f'Failed to connect to robot: {e}')

        self.image_client = self.robot.ensure_client(
            ImageClient.default_service_name
        )

        # Spot requires a software estop to be activated.
        estop_client = self.robot.ensure_client(
            bosdyn.client.estop.EstopClient.default_service_name)
        self.estop_endpoint = bosdyn.client.estop.EstopEndpoint(
            client=estop_client, name='image_handler', estop_timeout=9.0)
        self.estop_endpoint.force_simple_setup()

        self.lease_client = self.robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name)
        try:
            self.lease = self.lease_client.acquire()
        except bosdyn.client.lease.ResourceAlreadyClaimedError as err:
            print(
                "ERROR: Lease cannot be acquired. Ensure no other client has the lease. Shutting down.")
            print(err)
            sys.exit()

    def listen_for_images(self):
        print('Listening for images...')
        try:
            with bosdyn.client.lease.LeaseKeepAlive(self.lease_client), bosdyn.client.estop.EstopKeepAlive(
                    self.estop_endpoint):
                try:
                    while True:
                        image_requests = [
                            image_pb2.ImageRequest(
                                image_source_name="frontleft_fisheye_image", image_format=image_pb2.Image.FORMAT_RAW)
                        ]
                        images = self.image_client.get_image(image_requests)
                        for image in images:
                            if image.status == image_pb2.ImageResponse.STATUS_OK:
                                if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                                    num_bytes = 3
                                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                                    num_bytes = 4
                                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                                    num_bytes = 1
                                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                                    num_bytes = 2
                                dtype = np.uint8
                                extension = '.jpg'
                                img = np.frombuffer(
                                    image.shot.image.data, dtype=dtype)
                                if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                                    try:
                                        # Attempt to reshape array into a RGB rows X cols shape.
                                        img = img.reshape(
                                            (image.shot.image.rows, image.shot.image.cols, num_bytes))
                                    except ValueError:
                                        # Unable to reshape the image data, trying a regular decode.
                                        img = cv2.imdecode(img, -1)
                                else:
                                    img = cv2.imdecode(img, -1)

                                # img = ndimage.rotate(
                                #     img, ROTATION_ANGLE[image.source.name])

                                # self.detector.render(Image.fromarray(img))
                                
                                # if image.shot.image.format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                                #     np_image = np.fromstring(
                                #         image.shot.image.data, dtype=dtype)
                                #     np_image = np_image.reshape(
                                #         image.shot.image.rows, image.shot.image.cols, 3)
                                #     result = self.detector.render(
                                #         Image.fromarray(np_image))
                                #     # cv2.imshow('Real time detections', result)
                                #     # key = cv2.waitKey(100)  # 100 ms
                                #     # # Break the loop if the 'q' key is pressed
                                #     # if key == ord("q"):
                                #     #     break
                                # pass
                except KeyboardInterrupt:
                    pass
        finally:
            self.lease_client.return_lease(self.lease)
        pass


def main(options):
    print(os.environ.get('BOSDYN_CLIENT_USERNAME'))
    print("Main Function")
    image_handler = ImageHandler(options)
    image_handler.listen_for_images()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_common_arguments(parser)
    options = parser.parse_args(sys.argv[1:])
    main(options)
