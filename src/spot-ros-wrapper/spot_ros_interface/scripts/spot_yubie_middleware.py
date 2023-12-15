#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import logging
import tf2_ros
import cv2
import numpy as np
import time

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.util
import bosdyn.geometry

import rospy
import geometry_msgs.msg
import std_msgs.msg
import sensor_msgs.msg
import visualization_msgs.msg
import diagnostic_msgs.msg
import spot_ros_msgs.msg
import spot_ros_srvs.srv


from scipy import ndimage
from bosdyn.client.time_sync import TimedOutError
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import get_a_tform_b, get_vision_tform_body, get_odom_tform_body, \
    BODY_FRAME_NAME, VISION_FRAME_NAME


from dotenv import load_dotenv, find_dotenv
from grid_utils import get_terrain_markers
from enum import Enum

from cv_bridge import CvBridge


class Config(Enum):
    ROS_ONLY_MANIPULATES = 'ros_only_manipulates'
    ROS_COMPLETE = 'ros_complete'
    ROS_IMAGE_THROTTLED = 'ros_image_throttled'


INTERFACE_NAME = 'spot_middleware'
PUBLISHING_FREQUENCY = 20


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
        self.default_publisher_mappings = {
            "kinematic_state": spot_ros_msgs.msg.KinematicState,
            "robot_state": spot_ros_msgs.msg.RobotState,
            "occupancy_grid": visualization_msgs.msg.Marker,
            "spot_image": sensor_msgs.msg.Image,
            "spot_cam_info": sensor_msgs.msg.CameraInfo,
            "depth_image": sensor_msgs.msg.Image,
            "depth_array": std_msgs.msg.Float64MultiArray,
            "centroid_positions": spot_ros_msgs.msg.CentroidPositions
        }
        self.default_service_mappings = {
            "self_right_cmd": spot_ros_srvs.srv.Stand,
            "stand_cmd": spot_ros_srvs.srv.Stand,
            "trajectory_cmd": spot_ros_srvs.srv.Trajectory,
            "velocity_cmd": spot_ros_srvs.srv.Velocity
        }
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
        self.command_client = self.robot.ensure_client(
            RobotCommandClient.default_service_name)

        # Client to request images from Spot
        self.image_client = self.robot.ensure_client(
            ImageClient.default_service_name)

        self.image_sources = [
            src.name for src in self.image_client.list_image_sources() if "image" in src.name
        ]

        self.depth_sources = [
            src.name for src in self.image_client.list_image_sources() if "depth" in src.name
        ]
        # CV Bridge for converting cv image to ros
        self.bridge = CvBridge()

        # Client to request robot state
        self.robot_state_client = self.robot.ensure_client(
            RobotStateClient.default_service_name)

        # Client to request local occupancy grid
        self.grid_client = self.robot.ensure_client(
            LocalGridClient.default_service_name)
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
            print(
                "ERROR: Lease cannot be acquired. Ensure no other client has the lease. Shutting down.")
            print(err)
            sys.exit()

        # True for RViz visualization of Spot in 3rd person with occupancy grid
        self.third_person_view = True

        # Power on motors
        self.motors_on = options.motors_on.lower() != "n"

    def self_right_cmd_service(self, stand):
        """ Callback that sends self-right cmd"""
        cmd = RobotCommandBuilder.selfright_command()
        ret = self.command_client.robot_command(cmd)
        rospy.loginfo("Robot self right cmd sent. {}".format(ret))

        return []

    def stand_cmd_service(self, stand):
        """Callback that sends stand cmd at a given height delta [m] from standard configuration"""

        cmd = RobotCommandBuilder.stand_command(
            body_height=stand.body_pose.translation.z,
            footprint_R_body=self.quat_to_euler(stand.body_pose.rotation)
        )
        ret = self.command_client.robot_command(cmd)
        rospy.loginfo("Robot stand cmd sent. {}".format(ret))

        return []

    def trajectory_cmd_service(self, trajectory):
        '''
        Callback that specifies waypoint(s) (Point) [m] with a final orientation [rad]

        The name of the frame that trajectory is relative to.
        The trajectory must be expressed in a gravity aligned frame, so either "vision", "odom", or "flat_body".
        Any other provided se2_frame_name will be rejected and the trajectory command will not be exectuted.
        '''
        # TODO: Support other reference frames (currently only VISION ref. frame)

        for pose in trajectory.waypoints.poses:
            x = pose.position.x
            y = pose.position.y
            heading = math.atan2(y, x)
            frame = VISION_FRAME_NAME

            cmd = RobotCommandBuilder.trajectory_command(
                goal_x=x,
                goal_y=y,
                goal_heading=heading,
                frame_name=frame,
            )
            self.command_client.robot_command(
                lease=None, command=cmd, end_time_secs=time.time() + self.TRAJECTORY_CMD_TIMEOUT)

        robot_state = self.get_robot_state()[0].vision_tform_body
        final_pose = geometry_msgs.msg.Pose()
        final_pose.position = robot_state.translation
        final_pose.orientation = robot_state.rotation

        spot_ros_srvs.srv.TrajectoryResponse(final_pose)

    def velocity_cmd_service(self, twist):
        """Callback that sends instantaneous velocity [m/s] commands to Spot"""

        v_x = twist.velocity.linear.x
        v_y = twist.velocity.linear.y
        v_rot = twist.velocity.angular.z

        cmd = RobotCommandBuilder.velocity_command(
            v_x=v_x,
            v_y=v_y,
            v_rot=v_rot
        )

        self.command_client.robot_command(
            cmd,
            end_time_secs=time.time() + self.VELOCITY_CMD_DURATION
        )
        rospy.loginfo(
            "Robot velocity cmd sent: v_x=${},v_y=${},v_rot${}".format(v_x, v_y, v_rot))
        return []

    ### Helper functions ###

    def block_until_pose_reached(self, cmd, goal):
        """Do not return until goal waypoint is reached, or TRAJECTORY_CMD_TIMEOUT is reached."""
        # TODO: Make trajectory_cmd_timeout part of the service request

        self.command_client.robot_command(
            cmd,
            end_time_secs=time.time()+self.TRAJECTORY_CMD_TIMEOUT if self.TRAJECTORY_CMD_TIMEOUT else None
        )

        start_time = time.time()
        current_time = time.time()
        while (not self.is_final_state(goal) and (current_time - start_time < self.TRAJECTORY_CMD_TIMEOUT if self.TRAJECTORY_CMD_TIMEOUT else True)):
            time.sleep(.25)
            current_time = time.time()
        return self.is_final_state(goal)

    def is_final_state(self, goal):
        """Check if the current robot state is within range of the specified position."""
        goal_x = goal[0]
        goal_y = goal[1]
        goal_heading = goal[2]
        robot_state = self.get_robot_state()[0].vision_tform_body
        robot_pose = robot_state.translation
        robot_angle = self.quat_to_euler((robot_state.rotation.x, robot_state.rotation.y,
                                          robot_state.rotation.z, robot_state.rotation.w)).yaw

        x_dist = abs(goal_x - robot_pose.x)
        y_dist = abs(goal_y - robot_pose.y)
        angle = abs(goal_heading - robot_angle)
        if ((x_dist < self.x_goal_tolerance) and (y_dist < self.y_goal_tolerance) and (angle < self.angle_goal_tolerance)):
            return True
        return False

    def quat_to_euler(self, quat):
        """Convert a quaternion to xyz Euler angles."""
        q = [quat.x, quat.y, quat.z, quat.w]
        roll = math.atan2(2 * q[3] * q[0] + q[1] * q[2],
                          1 - 2 * q[0]**2 + 2 * q[1]**2)
        pitch = math.atan2(2 * q[1] * q[3] - 2 * q[0]
                           * q[2], 1 - 2 * q[1]**2 - 2 * q[2]**2)
        yaw = math.atan2(2 * q[2] * q[3] + 2 * q[0] *
                         q[1], 1 - 2 * q[1]**2 - 2 * q[2]**2)
        return bosdyn.geometry.EulerZXY(yaw=yaw, roll=roll, pitch=pitch)

    # TODO: Unit test the get_state method conversion from pbuf to ROS msg (test repeated fields, etc)
    def get_robot_state(self):
        ''' Returns tuple of kinematic_state, robot_state
            kinematic_state:
                timestamp
                joint_states []
                ko_tform_body
                body_twist_rt_ko
                ground_plane_rt_ko
                vo_tform_body
            robot_state:
                power_state
                battery_states[]
                comms_states[]
                system_fault_state
                estop_states[]
                behavior_fault_state
        '''
        robot_state = self.robot_state_client.get_robot_state()
        rs_msg = spot_ros_msgs.msg.RobotState()

        ''' PowerState conversion '''
        rs_msg.power_state.header.stamp.secs = robot_state.power_state.timestamp.seconds
        rs_msg.power_state.header.stamp.nsecs = robot_state.power_state.timestamp.nanos
        # [enum]
        rs_msg.power_state.motor_power_state = robot_state.power_state.motor_power_state
        # [enum]
        rs_msg.power_state.shore_power_state = robot_state.power_state.shore_power_state
        # [google.protobuf.DoubleValue]
        rs_msg.power_state.locomotion_charge_percentage = robot_state.power_state.locomotion_charge_percentage.value
        # [google.protobuf.Duration]
        rs_msg.power_state.locomotion_estimated_runtime.secs = robot_state.power_state.locomotion_estimated_runtime.seconds

        ''' BatteryState conversion [repeated field] '''
        for battery_state in robot_state.battery_states:
            battery_state_msg = sensor_msgs.msg.BatteryState()

            header = std_msgs.msg.Header()
            header.stamp.secs = battery_state.timestamp.seconds
            header.stamp.nsecs = battery_state.timestamp.nanos
            header.frame_id = battery_state.identifier  # [string]

            battery_state_msg.header = header

            battery_state_msg.percentage = battery_state.charge_percentage.value / \
                100  # [double]
            # NOTE: Using battery_state_msg.charge as the estimated runtime in sec
            # [google.protobuf.Duration]
            battery_state_msg.charge = battery_state.estimated_runtime.seconds
            # [DoubleValue]
            battery_state_msg.current = battery_state.current.value
            # [DoubleValue]
            battery_state_msg.voltage = battery_state.voltage.value
            # NOTE: Ignoring battery_state.temperatures for now; no field in BatteryState maps directly to it
            # [enum]
            battery_state_msg.power_supply_status = battery_state.status

            rs_msg.battery_states.append(battery_state_msg)

        ''' CommsState conversion [repeated field] '''
        for comms_state in robot_state.comms_states:
            comms_state_msg = spot_ros_msgs.msg.CommsState()

            # [google.protobuf.Timestamp]
            comms_state_msg.header.stamp.secs = comms_state.timestamp.seconds
            # [google.protobuf.Timestamp]
            comms_state_msg.header.stamp.nsecs = comms_state.timestamp.nanos
            # [enum] Note: wifi_state is oneof
            comms_state_msg.wifi_mode = comms_state.wifi_state.current_mode
            comms_state_msg.essid = comms_state.wifi_state.essid  # [string]

            rs_msg.comms_states.append(comms_state_msg)

        ''' SystemFaultState conversion '''
        ### faults is Repeated ###
        for fault in robot_state.system_fault_state.faults:
            system_fault_msg = spot_ros_msgs.msg.SystemFault()

            system_fault_msg.header.frame_id = fault.name  # [string]
            # [google.protobuf.Timestamp]
            system_fault_msg.header.stamp.secs = fault.onset_timestamp.seconds
            # [google.protobuf.Timestamp]
            system_fault_msg.header.stamp.nsecs = fault.onset_timestamp.nanos
            # [google.protobuf.Duration]
            system_fault_msg.duration.secs = fault.duration.seconds
            # [google.protobuf.Duration]
            system_fault_msg.duration.nsecs = fault.duration.nanos
            system_fault_msg.code = fault.code  # [int32]
            system_fault_msg.uid = fault.uid  # [uint64]
            system_fault_msg.error_message = fault.error_message  # [string]
            system_fault_msg.attributes = fault.attributes  # [repeated-string]
            system_fault_msg.severity = fault.severity  # [enum]

            rs_msg.system_fault_state.faults.append(system_fault_msg)

        ### historical_faults is Repeated ###
        for historical_fault in robot_state.system_fault_state.faults:
            system_fault_msg = spot_ros_msgs.msg.SystemFault()

            # [string]
            system_fault_msg.header.frame_id = historical_fault.name
            # [google.protobuf.Timestamp]
            system_fault_msg.header.stamp.secs = historical_fault.onset_timestamp.seconds
            # [google.protobuf.Timestamp]
            system_fault_msg.header.stamp.nsecs = historical_fault.onset_timestamp.nanos
            # [google.protobuf.Duration]
            system_fault_msg.duration.secs = historical_fault.duration.seconds
            # [google.protobuf.Duration]
            system_fault_msg.duration.nsecs = historical_fault.duration.nanos
            system_fault_msg.code = historical_fault.code  # [int32]
            system_fault_msg.uid = historical_fault.uid  # [uint64]
            # [string]
            system_fault_msg.error_message = historical_fault.error_message
            # [repeated-string]
            system_fault_msg.attributes = historical_fault.attributes
            system_fault_msg.severity = historical_fault.severity  # [enum]

            rs_msg.system_fault_state.historical_faults.append(
                system_fault_msg)

        # [map<string,enum>]
        if robot_state.system_fault_state.aggregated:
            for key, value in robot_state.system_fault_state.aggregated.items():
                kv = diagnostic_msgs.msg.KeyValue()
                kv.key = key
                kv.value = value
                rs_msg.system_fault_state.aggregated.append(kv)

        ''' EStopState conversion [repeated field] '''
        for estop_state in robot_state.estop_states:
            estop_msg = spot_ros_msgs.msg.EStopState()

            # [google.protobuf.Timestamp]
            estop_msg.header.stamp.secs = estop_state.timestamp.seconds
            # [google.protobuf.Timestamp]
            estop_msg.header.stamp.nsecs = estop_state.timestamp.nanos
            estop_msg.header.frame_id = estop_state.name  # [string]
            estop_msg.type = estop_state.type  # [enum]
            estop_msg.state = estop_state.state  # [enum]
            # [string]
            estop_msg.state_description = estop_state.state_description

            rs_msg.estop_states.append(estop_msg)

        ''' KinematicState conversion '''
        ks_msg = spot_ros_msgs.msg.KinematicState()

        ks_msg.header.stamp.secs = robot_state.kinematic_state.acquisition_timestamp.seconds
        ks_msg.header.stamp.nsecs = robot_state.kinematic_state.acquisition_timestamp.nanos

        ### joint_states is repeated ###
        js = sensor_msgs.msg.JointState()
        js.header.stamp = ks_msg.header.stamp
        for joint_state in robot_state.kinematic_state.joint_states:
            js.name.append(joint_state.name)  # [string]
            # Note: angle in rad
            js.position.append(joint_state.position.value)
            js.velocity.append(joint_state.velocity.value)  # Note: ang vel
            # NOTE: ang accel. JointState doesn't have accel. Ignoring joint_state.acceleration for now.
            js.effort.append(joint_state.load.value)  # Note: Torque in N-m

        ks_msg.joint_states = js

        # SE3Pose representing transform of Spot's Body frame relative to the inertial Vision frame
        vision_tform_body = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot)

        ks_msg.vision_tform_body.translation.x = vision_tform_body.x
        ks_msg.vision_tform_body.translation.y = vision_tform_body.y
        ks_msg.vision_tform_body.translation.z = vision_tform_body.z

        ks_msg.vision_tform_body.rotation.x = vision_tform_body.rot.x
        ks_msg.vision_tform_body.rotation.y = vision_tform_body.rot.y
        ks_msg.vision_tform_body.rotation.z = vision_tform_body.rot.z
        ks_msg.vision_tform_body.rotation.w = vision_tform_body.rot.w

        # odom_tform_body: SE3Pose representing transform of Spot's Body frame relative to the odometry frame
        odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot)

        ks_msg.odom_tform_body.translation.x = odom_tform_body.x
        ks_msg.odom_tform_body.translation.y = odom_tform_body.y
        ks_msg.odom_tform_body.translation.z = odom_tform_body.z

        ks_msg.odom_tform_body.rotation.x = odom_tform_body.rot.x
        ks_msg.odom_tform_body.rotation.y = odom_tform_body.rot.y
        ks_msg.odom_tform_body.rotation.z = odom_tform_body.rot.z
        ks_msg.odom_tform_body.rotation.w = odom_tform_body.rot.w

        ''' velocity_of_body_in_vision '''
        ks_msg.velocity_of_body_in_vision.linear.x = robot_state.kinematic_state.velocity_of_body_in_vision.linear.x
        ks_msg.velocity_of_body_in_vision.linear.y = robot_state.kinematic_state.velocity_of_body_in_vision.linear.y
        ks_msg.velocity_of_body_in_vision.linear.z = robot_state.kinematic_state.velocity_of_body_in_vision.linear.z

        ks_msg.velocity_of_body_in_vision.angular.x = robot_state.kinematic_state.velocity_of_body_in_vision.angular.x
        ks_msg.velocity_of_body_in_vision.angular.y = robot_state.kinematic_state.velocity_of_body_in_vision.angular.y
        ks_msg.velocity_of_body_in_vision.angular.z = robot_state.kinematic_state.velocity_of_body_in_vision.angular.z

        ''' velocity_of_body_in_odom '''

        ks_msg.velocity_of_body_in_odom.linear.x = robot_state.kinematic_state.velocity_of_body_in_odom.linear.x
        ks_msg.velocity_of_body_in_odom.linear.y = robot_state.kinematic_state.velocity_of_body_in_odom.linear.y
        ks_msg.velocity_of_body_in_odom.linear.z = robot_state.kinematic_state.velocity_of_body_in_odom.linear.z

        ks_msg.velocity_of_body_in_odom.angular.x = robot_state.kinematic_state.velocity_of_body_in_odom.angular.x
        ks_msg.velocity_of_body_in_odom.angular.y = robot_state.kinematic_state.velocity_of_body_in_odom.angular.y
        ks_msg.velocity_of_body_in_odom.angular.z = robot_state.kinematic_state.velocity_of_body_in_odom.angular.z

        # BehaviorFaultState conversion
        '''faults is repeated'''
        for fault in robot_state.behavior_fault_state.faults:
            behaviour_fault_state_msg = spot_ros_msgs.msg.BehaviorFaultState()

            # [uint32]
            behaviour_fault_state_msg.header.frame_id = fault.behavior_fault_id
            # [google.protobuf.Timestamp]
            behaviour_fault_state_msg.header.stamp.secs = fault.onset_timestamp.seconds
            # [google.protobuf.Timestamp]
            behaviour_fault_state_msg.header.stamp.nsecs = fault.onset_timestamp.nanos
            behaviour_fault_state_msg.cause = fault.cause  # [enum]
            behaviour_fault_state_msg.status = fault.status  # [enum]

            rs_msg.behavior_fault_states.append(behaviour_fault_state_msg)

        # FootState conversion [repeated]
        for foot_state in robot_state.foot_state:
            foot_state_msg = spot_ros_msgs.msg.FootState()

            # [double]
            foot_state_msg.foot_position_rt_body.x = foot_state.foot_position_rt_body.x
            # [double]
            foot_state_msg.foot_position_rt_body.y = foot_state.foot_position_rt_body.y
            # [double]
            foot_state_msg.foot_position_rt_body.z = foot_state.foot_position_rt_body.z
            foot_state_msg.contact = foot_state.contact  # [enum]

            rs_msg.foot_states.append(foot_state_msg)

        return ks_msg, rs_msg  # kinematic state message, robot state message

    def start_module(self, *args, **kwargs):
        rospy.init_node(INTERFACE_NAME)
        rate = rospy.Rate(200)  # Update at 200 Hz

        services = self.default_service_mappings
        publishers = self.default_publisher_mappings
        if self.third_person_view:
            publishers['joint_state_from_spot'] = sensor_msgs.msg.JointState
        self.initialize_services(services)
        self.initialize_publishers(publishers)
        self.initialize_transformation_broadcaster()

        try:
            with bosdyn.client.lease.LeaseKeepAlive(self.lease_client), \
                    bosdyn.client.estop.EstopKeepAlive(self.estop_endpoint):
                rospy.loginfo("Acquired lease")
                if self.motors_on:
                    rospy.loginfo(
                        "Powering on robot... This may take a several seconds.")
                    self.robot.power_on(timeout_sec=20)
                    assert self.robot.is_powered_on(), "Robot power on failed."
                    rospy.loginfo("Robot powered on.")
                else:
                    rospy.loginfo("Not powering on robot, continuing")

                while not rospy.is_shutdown():
                    self[self.config](*args, **kwargs)
                    rospy.logdebug("Looping...")
                    rate.sleep()

        finally:
            # If we successfully acquired a lease, return it.
            self.lease_client.return_lease(self.lease)

    def initialize_services(self):
        for service_name, msg_type in self.default_publisher_mappings.items():
            rospy.Service(service_name, msg_type,
                          self[f'{service_name}_service'])

    def initialize_publishers(self):
        for topic, msg_type in self.default_publisher_mappings.items():
            self[f'{topic}_pub'] = rospy.Publisher(
                topic, msg_type, queue_size=PUBLISHING_FREQUENCY
            )

    def initialize_transformation_broadcaster(self):
        self.spot_tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.spot_tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()

    def publish_transformation(self, kinematic_state):
        # Publish tf2 from the fixed vision_odometry_frame to the Spot's base_link
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "vision_odometry_frame"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = kinematic_state.vision_tform_body.translation.x
        transform.transform.translation.y = kinematic_state.vision_tform_body.translation.y
        transform.transform.translation.z = kinematic_state.vision_tform_body.translation.z
        transform.transform.rotation.x = kinematic_state.vision_tform_body.rotation.x
        transform.transform.rotation.y = kinematic_state.vision_tform_body.rotation.y
        transform.transform.rotation.z = kinematic_state.vision_tform_body.rotation.z
        transform.transform.rotation.w = kinematic_state.vision_tform_body.rotation.w
        self.spot_tf_broadcaster.sendTransform(transform)
        if self.third_person_view:
            self.joint_state_pub.publish(kinematic_state.joint_states)

        return transform

    def publish_occupancy_grid(self):
        if self.occupancy_grid_pub.get_num_connections() > 0:
            local_grid_proto = self.grid_client.get_local_grids(['terrain'])
            markers = get_terrain_markers(local_grid_proto)
            self.occupancy_grid_pub.publish(markers)

    def publish_image_and_camera_info(self, transform):
        img_requests = [
            image_pb2.ImageRequest(image_source_name=source,
                                   image_format=image_pb2.Image.FORMAT_RAW,
                                   pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8)
            for source in self.image_sources[2:3]
        ]
        image_list = self.image_client.get_image(img_requests)
        depth_list = self.image_client.get_image_from_sources(
            ["frontleft_depth_in_visual_frame", "frontleft_fisheye_image"])

        def get_dtype(image):
            dtype = np.uint8
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
            return dtype

        def preprocess_image(image):
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                image = np.frombuffer(
                    image.shot.image.data, dtype=dtype)
                image = image.reshape(
                    (image.shot.image.rows, image.shot.image.cols, 3))
            return image

        def create_image_msg(header, image):
            image_msg = sensor_msgs.msg.Image()
            image_msg.header = header
            image_msg.width = image.shot.image.cols
            image_msg.height = image.shot.image.rows
            image_msg.data = image.shot.image.data if image.shot.image.format != image_pb2.Image.FORMAT_RAW else image.tobytes()
            image_msg.step = image.shot.image.cols
            image_msg.encoding = 'rgb8'
            return image_msg

        def create_camera_info(image, image_msg):
            camera_info = sensor_msgs.msg.CameraInfo()
            camera_info.header = image_msg.header
            camera_info.width = image_msg.width
            camera_info.height = image_msg.height
            camera_info.distortion_model = "plumb_bob"
            camera_info.D = [0.0, 0.0, 0.0, 0.0]
            f = image.source.pinhole.intrinsics.focal_length
            c = image.source.pinhole.intrinsics.principal_point
            camera_info.K = \
                [f.x, 0, c.x,
                    0, f.y, c.y,
                    0,   0,  1]

            return camera_info

        def create_depth_msg():
            depth_array = std_msgs.msg.Float64MultiArray()
            cv_depth = np.frombuffer(
                depth_list[0].shot.image.data, dtype=np.uint16)
            depth_array.data = cv_depth
            cv_depth = cv_depth.reshape(depth_list[0].shot.image.rows,
                                        depth_list[0].shot.image.cols)
            cv_visual = cv2.imdecode(np.frombuffer(
                depth_list[1].shot.image.data, dtype=np.uint8), -1)
            visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(
                cv_visual, cv2.COLOR_GRAY2RGB)
            min_val = np.min(cv_depth)
            max_val = np.max(cv_depth)
            depth_range = max_val - min_val
            depth8 = (255.0 / depth_range *
                      (cv_depth - min_val)).astype('uint8')
            depth8_rgb = cv2.cvtColor(
                depth8, cv2.COLOR_GRAY2RGB)
            depth_color = cv2.applyColorMap(
                depth8_rgb, cv2.COLORMAP_JET)
            out = cv2.addWeighted(
                visual_rgb, 0.5, depth_color, 0.5, 0)
            if True:
                if depth_list[0].source.name[0:5] == 'front':
                    out = cv2.rotate(
                        out, cv2.ROTATE_90_CLOCKWISE)

                elif depth_list[0].source.name[0:5] == 'right':
                    out = cv2.rotate(out, cv2.ROTATE_180)

            depth_msg = self.bridge.cv2_to_imgmsg(
                out, "passthrough")

            return depth_array, depth_msg

        for image in image_list:
            if image.status == image_pb2.ImageResponse.STATUS_OK:

                header = self.create_header_msg(image, transform)
                dtype = get_dtype(image)
                image = preprocess_image(image)

                image_msg = create_image_msg(header, image)
                camera_info = create_camera_info(image, image_msg)
                depth_array, depth_msg = create_depth_msg(image)

                publish_body_to_cam_tf(header, image)

                self.spot_image_pub.publish(image_msg)
                self.depth_image_pub.publish(depth_msg)
                self.depth_array_pub.publish(depth_array)
                self.spot_cam_info_pub.publish(camera_info)

    def publish_body_to_cam_tf(self, header, image):

        # Transform from base_link to camera for current img
        body_transform_cam = get_a_tform_b(image.shot.transforms_snapshot,
                                           BODY_FRAME_NAME,
                                           image.shot.frame_name_image_sensor)

        # Generate camera to body Transform
        body_to_cam_tf = geometry_msgs.msg.Transform()
        body_to_cam_tf.translation.x = body_transform_cam.position.x
        body_to_cam_tf.translation.y = body_transform_cam.position.y
        body_to_cam_tf.translation.z = body_transform_cam.position.z
        body_to_cam_tf.rotation.x = body_transform_cam.rotation.x
        body_to_cam_tf.rotation.y = body_transform_cam.rotation.y
        body_to_cam_tf.rotation.z = body_transform_cam.rotation.z
        body_to_cam_tf.rotation.w = body_transform_cam.rotation.w

        camera_transform_stamped = geometry_msgs.msg.TransformStamped()
        camera_transform_stamped.header.stamp = header.stamp
        camera_transform_stamped.header.frame_id = "base_link"
        camera_transform_stamped.transform = body_to_cam_tf
        camera_transform_stamped.child_frame_id = image.source.name

        # Publish body to camera static tf
        self.spot_tf_static_broadcaster.sendTransform(
            camera_transform_stamped)

    def create_header_msg(self, image, transform):
        header = std_msgs.msg.Header()
        header.stamp = transform.header.stamp
        header.frame_id = image.source.name
        return header

    def detect_and_publish(self, transform):
        # Have to work on this
        # Here we have to get the images from the spot and publish the detections
        # Not publishing the images
        """Simple image display example."""
        CWD = Path(__file__).parent
        model_path = CWD.joinpath('./model_weights/best.pt').absolute()
        model = YOLO(model_path)

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
                            cv2.circle(image, (cx, cy), radius,
                                       centroid_color, -1)
                            cv2.rectangle(image, (x1, y1), (x2, y2),
                                          box_color, thickness)
            return image, results

        self.robot.sync_with_directory()
        self.robot.time_sync.wait_for_sync()

        image_service = ImageClient.default_service_name
        jpeg_quality_percent = 50
        resize_ratio = 1
        image_sources = [
            'frontleft_fisheye_image'
        ]
        auto_rotate = True
        capture_delay = 100

        image_client = self.robot.ensure_client(image_service)
        requests = [
            build_image_request(source,
                                pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
                                quality_percent=jpeg_quality_percent,
                                resize_ratio=resize_ratio) for source in image_sources
        ]

        for image_source in image_sources:
            cv2.namedWindow(image_source, cv2.WINDOW_NORMAL)
            if len(image_sources) > 1:
                cv2.setWindowProperty(
                    image_source, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        keystroke = None
        timeout_count_before_reset = 0
        t1 = time.time()
        image_count = 0
        while keystroke != VALUE_FOR_Q_KEYSTROKE and keystroke != VALUE_FOR_ESC_KEYSTROKE:
            try:
                images_future = image_client.get_image_async(
                    requests, timeout=0.5)
                while not images_future.done():
                    keystroke = cv2.waitKey(25)
                    if keystroke == VALUE_FOR_ESC_KEYSTROKE or keystroke == VALUE_FOR_Q_KEYSTROKE:
                        sys.exit(1)
                images = images_future.result()
            except TimedOutError as time_err:
                if timeout_count_before_reset == 5:
                    # To attempt to handle bad comms and continue the live image stream, try recreating the
                    # image client after having an RPC timeout 5 times.
                    _LOGGER.info(
                        'Resetting image client after 5+ timeout errors.')
                    image_client = reset_image_client(self.robot)
                    timeout_count_before_reset = 0
                else:
                    timeout_count_before_reset += 1
            except Exception as err:
                _LOGGER.warning(err)
                continue
            for i in range(len(images)):
                image, ext = image_to_opencv(images[i], auto_rotate)
                image, results = detect_objects(image)
                # Have to publish the results from here
                cv2.imshow(images[i].source.name, image)
            keystroke = cv2.waitKey(capture_delay)
            image_count += 1
            print(
                f'Mean image retrieval rate: {image_count/(time.time() - t1)}Hz')

    def ros_complete(self):
        ''' Publish Robot State'''
        kinematic_state, robot_state = self.get_robot_state()
        self.kinematic_state_pub.publish(kinematic_state)
        self.robot_state_pub.publish(robot_state)

        '''Publish Transformation'''
        transform = self.publish_transformation(kinematic_state)

        ''' Publish Images'''
        self.publish_image_and_camera_info(transform)

        ''' Publish occupancy grid'''
        self.publish_occupancy_grid()

    def ros_only_manipulates(self):
        ''' Publish Robot State'''
        kinematic_state, robot_state = self.get_robot_state()
        self.kinematic_state_pub.publish(kinematic_state)
        self.robot_state_pub.publish(robot_state)

        '''Publish Transformation'''
        transform = self.publish_transformation(kinematic_state)

        ''' Publish Detections in Image'''
        self.detect_and_publish(transform)

        ''' Publish occupancy grid'''
        self.publish_occupancy_grid()

    def check_spot_connection(self, options):
        try:
            with open(os.devnull, 'wb') as devnull:
                resp = subprocess.check_call(
                    ['ping', '-c', '1', options.hostname], stdout=devnull, stderr=subprocess.STDOUT)
                if resp != 0:
                    print("ERROR: Cannot detect a Spot with IP: {}.\nMake sure Spot is powered on and on the same network".format(
                        options.hostname))
                    sys.exit()
        except:
            print("ERROR: Cannot detect a Spot with IP: {}.\nMake sure Spot is powered on and on the same network".format(
                options.hostname))
            sys.exit()


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_common_arguments(parser)
    parser.add_argument(
        '--motor_on', help='To enable the motors [Y/n]', default='Y')
    parser.add_argument(
        '--config', '-C',
        required=True,
        action=default_from_env('MIDDLEWARE_CONFIG'),
        default=Config.ROS_ONLY_MANIPULATES,
        help="""Provides the configuration in which the middleware is set to run."""
    )
    options = parser.parse_args(sys.argv[1:])
    try:
        robot = SpotMiddleware(options)
        robot.start_module()
    except rospy.ROSInterruptException:
        pass
