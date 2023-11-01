import logging
import argparse
import sys

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry

from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2

class ImageHandler():
    LOGGER = logging.getLogger()
    def __init__(self, options):
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
            print("ERROR: Lease cannot be acquired. Ensure no other client has the lease. Shutting down.")
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
                            image_pb2.ImageRequest(image_source_name="frontleft_fisheye_image", image_format=image_pb2.Image.FORMAT_RAW)
                        ]
                        images = self.image_client.get_image(image_requests)
                        for image in images:
                            if image.status == image_pb2.ImageResponse.STATUS_OK:
                                # Save Images from here
                                pass
                except KeyboardInterrupt:
                    pass
        finally:
            self.lease_client.return_lease(self.lease)
        pass


def main(options):
    print("Main Function")
    image_handler = ImageHandler(options)
    image_handler.listen_for_images()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_common_arguments(parser)
    options = parser.parse_args(sys.argv[1:])
    main()