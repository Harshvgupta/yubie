#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import tf2_ros 
import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.util
import bosdyn.geometry

import rospy
import geometry_msgs.msg
import std_msgs.msg
import sensor_msgs.msg
import visualization_msgs.msg
import spot_ros_msgs.msg
import spot_ros_srvs.srv

from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME


from dotenv import load_dotenv, find_dotenv
from grid_utils import get_terrain_markers
from enum import Enum

from cv_bridge import CvBridge

class Config(Enum):
    ROS_ONLY_MANIPULATES = 'ros_only_manipulates'
    ROS_COMPLETE = 'ros_complete'
    ROS_IMAGE_THROTTLED = 'ros_image_throttled'

INTERFACE_NAME = 'spot_middleware'
class EnvVar(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs):
        if not default and envvar:
            if envvar in os.environ:
                default = os.environ[envvar]
        if required and default:
            required = False
        super(EnvVar, self).__init__(
            default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_strings=None):
        setattr(namespace, self.dest, values)

def default_from_env(envvar):
    def wrapper(**kwargs):
        return EnvVar(envvar, **kwargs)
    return wrapper

class SpotMiddleware:
    
    def __init__(self, options):
        self.config = options.config
        self.check_spot_connection(options)
        bosdyn.client.util.setup_logging(options.verbose)
        self.create_robot_instance(options)
        pass
    
    def create_robot_instance(self, options):
        self.sdk = bosdyn.client.create_standard_sdk(INTERFACE_NAME)
        try:
            self.robot = self.sdk.create_robot(options.hostname)
            bosdyn.client.util.authenticate(self.robot)
            self.robot.time_sync.wait_for_sync()
        except bosdyn.client.RpcError as err:
            self.LOGGER.error("Failed to communicate with robot: %s", err)

        # Client to send cmds to Spot
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

        # Client to request images from Spot
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
    
        self.image_source_names = [
            src.name for src in self.image_client.list_image_sources() if "image" in src.name
        ]

        self.depth_image_sources = [
            src.name for src in self.image_client.list_image_sources() if "depth" in src.name
        ]
        # CV Bridge for converting cv image to ros 
        self.bridge = CvBridge()
        
        # Client to request robot state
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)

        # Client to request local occupancy grid
        self.grid_client = self.robot.ensure_client(LocalGridClient.default_service_name)
        self.local_grid_types = self.grid_client.get_local_grid_types()

        # Spot requires a software estop to be activated.
        estop_client = self.robot.ensure_client(
            bosdyn.client.estop.EstopClient.default_service_name)
        self.estop_endpoint = bosdyn.client.estop.EstopEndpoint(
            client=estop_client, name=INTERFACE_NAME, estop_timeout=9.0)
        self.estop_endpoint.force_simple_setup()

        # Only one client at a time can operate a robot.
        self.lease_client = self.robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name)
        try:
            self.lease = self.lease_client.acquire()
        except bosdyn.client.lease.ResourceAlreadyClaimedError as err:
            print("ERROR: Lease cannot be acquired. Ensure no other client has the lease. Shutting down.")
            print(err)
            sys.exit()
            
        # True for RViz visualization of Spot in 3rd person with occupancy grid
        self.third_person_view = True

        # Power on motors
        self.motors_on = options.motors_on.lower()!="n"

    
    def start_module(self, *args, **kwargs):
        self[self.config](*args, **kwargs)
        pass

    def initialize_services(self):
        rospy.Service("self_right_cmd", spot_ros_srvs.srv.Stand, self.self_right_cmd_srv)
        rospy.Service("stand_cmd", spot_ros_srvs.srv.Stand, self.stand_cmd_srv)
        rospy.Service("trajectory_cmd", spot_ros_srvs.srv.Trajectory, self.trajectory_cmd_srv)
        rospy.Service("velocity_cmd", spot_ros_srvs.srv.Velocity, self.velocity_cmd_srv)

    
    def initialize_publishers(self):
        self.kinematic_state_pub = rospy.Publisher(
            "kinematic_state", spot_ros_msgs.msg.KinematicState, queue_size=20)
        self.robot_state_pub = rospy.Publisher(
            "robot_state", spot_ros_msgs.msg.RobotState, queue_size=20)
        self.occupancy_grid_pub = rospy.Publisher(
            "occupancy_grid", visualization_msgs.msg.Marker, queue_size=20)

    
    def ros_complete(self):
        rospy.init_node('spot_middleware_ros_complete')
        rate = rospy.Rate(200)  # Update at 200 Hz

        self.initialize_services()
        self.initialize_publishers()
        
        # Single image publisher will publish all images from all Spot cameras
        
        # Publish tf2 from visual odometry frame to Spot's base link
        spot_tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Publish static tf2 from Spot's base link to front-left camera
        spot_tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()

        image_only_pub = rospy.Publisher(
            "spot_image", sensor_msgs.msg.Image, queue_size=20)

        camera_info_pub = rospy.Publisher(
            "spot_cam_info", sensor_msgs.msg.CameraInfo, queue_size=20)

        depth_image_pub = rospy.Publisher(
            "depth_image", sensor_msgs.msg.Image, queue_size=20)
        
        depth_array_pub = rospy.Publisher("depth_array", std_msgs.msg.Float64MultiArray, queue_size=20)

        # For RViz 3rd person POV visualization
        if self.third_person_view:
            joint_state_pub = rospy.Publisher(
                "joint_state_from_spot", sensor_msgs.msg.JointState, queue_size=20)

        try:
            with bosdyn.client.lease.LeaseKeepAlive(self.lease_client), bosdyn.client.estop.EstopKeepAlive(
                    self.estop_endpoint):
                rospy.loginfo("Acquired lease")
                if self.motors_on:
                    rospy.loginfo("Powering on robot... This may take a several seconds.")
                    self.robot.power_on(timeout_sec=20)
                    assert self.robot.is_powered_on(), "Robot power on failed."
                    rospy.loginfo("Robot powered on.")
                else:
                    rospy.loginfo("Not powering on robot, continuing")

                while not rospy.is_shutdown():
                    ''' Publish Robot State'''
                    kinematic_state, robot_state = self.get_robot_state()

                    self.kinematic_state_pub.publish(kinematic_state)
                    self.robot_state_pub.publish(robot_state)
                    
                    # Publish tf2 from the fixed vision_odometry_frame to the Spot's base_link
                    t = geometry_msgs.msg.TransformStamped()
                    t.header.stamp = rospy.Time.now()
                    t.header.frame_id = "vision_odometry_frame"
                    t.child_frame_id = "base_link"
                    t.transform.translation.x = kinematic_state.vision_tform_body.translation.x
                    t.transform.translation.y = kinematic_state.vision_tform_body.translation.y
                    t.transform.translation.z = kinematic_state.vision_tform_body.translation.z
                    t.transform.rotation.x = kinematic_state.vision_tform_body.rotation.x
                    t.transform.rotation.y = kinematic_state.vision_tform_body.rotation.y
                    t.transform.rotation.z = kinematic_state.vision_tform_body.rotation.z
                    t.transform.rotation.w = kinematic_state.vision_tform_body.rotation.w
                    spot_tf_broadcaster.sendTransform(t)
                    # print("printing",kinematic_state.joint_states)
                    if self.third_person_view:
                        joint_state_pub.publish(kinematic_state.joint_states)

                    ''' Publish Images'''
                    #img_reqs = [image_pb2.ImageRequest(image_source_name=source, image_format=image_pb2.Image.FORMAT_RAW) for source in self.image_source_names[2:3]]
                    def pixel_format_type_strings():
                        names = image_pb2.Image.PixelFormat.keys()
                        return names[1:]


                    def pixel_format_string_to_enum(enum_string):
                        return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)
                    
                    img_requests = [image_pb2.ImageRequest(image_source_name=source, image_format=image_pb2.Image.FORMAT_RAW, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8) for source in self.image_source_names]
                    image_list = self.image_client.get_image(img_requests)
                    depth_responses = self.image_client.get_image_from_sources(self.depth_image_sources)

                    for img in image_list:
                        if img.status == image_pb2.ImageResponse.STATUS_OK:

                            header = std_msgs.msg.Header()
                            header.stamp = t.header.stamp
                            header.frame_id = img.source.name

                            if img.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                                dtype = np.uint16
                            else:
                                dtype = np.uint8

                            if img.shot.image.format == image_pb2.Image.FORMAT_RAW:
                                # image = np.fromstring(img.shot.image.data, dtype=dtype)
                                image = np.frombuffer(img.shot.image.data, dtype=dtype)
                                image = image.reshape((img.shot.image.rows, img.shot.image.cols,3))

                            # Make Image component of ImageCapture
                            i = sensor_msgs.msg.Image()
                            i.header = header
                            i.width = img.shot.image.cols
                            i.height = img.shot.image.rows
                            i.data = img.shot.image.data if img.shot.image.format != image_pb2.Image.FORMAT_RAW else image.tobytes()
                            i.step = img.shot.image.cols
                            i.encoding = 'rgb8'

                            # CameraInfo
                            cam_info = sensor_msgs.msg.CameraInfo()
                            cam_info.header = i.header
                            cam_info.width = i.width
                            cam_info.height = i.height
                            cam_info.distortion_model = "plumb_bob"
                            cam_info.D = [0.0,0.0,0.0,0.0]
                            f = img.source.pinhole.intrinsics.focal_length
                            c = img.source.pinhole.intrinsics.principal_point
                            cam_info.K = \
                                [f.x, 0, c.x,  \
                                0, f.y, c.y,   \
                                0,   0,  1]
                            
                            # Depth
                            depth_array = std_msgs.msg.Float64MultiArray()
                            cv_depth = np.frombuffer(depth_responses[0].shot.image.data, dtype=np.uint16)
                            depth_array.data = cv_depth
                            cv_depth = cv_depth.reshape(depth_responses[0].shot.image.rows,
                                                        depth_responses[0].shot.image.cols)
                            cv_visual = cv2.imdecode(np.frombuffer(depth_responses[1].shot.image.data, dtype=np.uint8), -1)
                            visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(cv_visual, cv2.COLOR_GRAY2RGB)
                            min_val = np.min(cv_depth)
                            max_val = np.max(cv_depth)
                            depth_range = max_val - min_val
                            depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
                            depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
                            depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)  
                            out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)
                            if True:
                                if depth_responses[0].source.name[0:5] == 'front':
                                    out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)

                                elif depth_responses[0].source.name[0:5] == 'right':
                                    out = cv2.rotate(out, cv2.ROTATE_180)
                        
                            depth_image = self.bridge.cv2_to_imgmsg(out, "passthrough")


                            # Transform from base_link to camera for current img
                            body_tform_cam = get_a_tform_b(img.shot.transforms_snapshot,
                                BODY_FRAME_NAME,
                                img.shot.frame_name_image_sensor)
                            
                            # Generate camera to body Transform
                            body_tform_cam_tf = geometry_msgs.msg.Transform()
                            body_tform_cam_tf.translation.x = body_tform_cam.position.x
                            body_tform_cam_tf.translation.y = body_tform_cam.position.y
                            body_tform_cam_tf.translation.z = body_tform_cam.position.z
                            body_tform_cam_tf.rotation.x = body_tform_cam.rotation.x
                            body_tform_cam_tf.rotation.y = body_tform_cam.rotation.y
                            body_tform_cam_tf.rotation.z = body_tform_cam.rotation.z
                            body_tform_cam_tf.rotation.w = body_tform_cam.rotation.w

                            camera_transform_stamped = geometry_msgs.msg.TransformStamped()
                            camera_transform_stamped.header.stamp = header.stamp
                            camera_transform_stamped.header.frame_id = "base_link"
                            camera_transform_stamped.transform = body_tform_cam_tf
                            camera_transform_stamped.child_frame_id = img.source.name

                            # Publish body to camera static tf
                            spot_tf_static_broadcaster.sendTransform(camera_transform_stamped)

                            # Publish current image and camera info
                            image_only_pub.publish(i)
                            depth_image_pub.publish(depth_image)
                            camera_info_pub.publish(cam_info)
                            depth_array_pub.publish(depth_array)


                    ''' Publish occupancy grid'''
                    if occupancy_grid_pub.get_num_connections() > 0:
                        local_grid_proto = self.grid_client.get_local_grids(['terrain'])
                        markers = get_terrain_markers(local_grid_proto)
                        occupancy_grid_pub.publish(markers)

                    rospy.logdebug("Looping...")
                    rate.sleep()

        finally:
            # If we successfully acquired a lease, return it.
            self.lease_client.return_lease(self.lease)
        
    def check_spot_connection(self, options):
        try:
            with open(os.devnull, 'wb') as devnull:
                resp = subprocess.check_call(['ping', '-c', '1', options.hostname], stdout=devnull, stderr=subprocess.STDOUT)
                if resp != 0:
                    print ("ERROR: Cannot detect a Spot with IP: {}.\nMake sure Spot is powered on and on the same network".format(config.hostname))
                    sys.exit()
        except:
            print("ERROR: Cannot detect a Spot with IP: {}.\nMake sure Spot is powered on and on the same network".format(config.hostname))
            sys.exit()
    
        


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_common_arguments(parser)
    parser.add_argument('--motor_on', help='To enable the motors [Y/n]', default='Y')
    parser.add_argument(
        '--config', '-C',
        required= True,
        action= default_from_env('MIDDLEWARE_CONFIG'),
        default=Config.ROS_ONLY_MANIPULATES,
        help="""Provides the configuration in which the middleware is set to run."""
    )
    options = parser.parse_args(sys.argv[1:])
    try:
        robot = SpotMiddleware(options)
        robot.start_module()
    except rospy.ROSInterruptException:
        pass
