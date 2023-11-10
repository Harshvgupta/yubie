#!/usr/bin/env python3

#ROS imports
import rospy
import diagnostic_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import sensor_msgs.msg
import visualization_msgs.msg
import spot_ros_msgs.msg
import spot_ros_srvs.srv
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import PoseStamped
import tf
import spot_ros_srvs.srv
import math
import time
import numpy as np

class Follow():

    def __init__(self):

        self.ballpos_sub = rospy.Subscriber("/ball_pose",geometry_msgs.msg.Pose,self.trajectory_callback)
        self.trajectory_srv_req = spot_ros_srvs.srv.TrajectoryRequest()
        self.trajectory_srv_pub = rospy.ServiceProxy("trajectory_cmd", spot_ros_srvs.srv.Trajectory)
    def trajectory_callback(self,pose):
        print(pose)
        pose_arr = geometry_msgs.msg.PoseArray()
        pose_arr.poses.append(pose)
        
        self.trajectory_srv_req.waypoints.poses = pose_arr.poses

        try:
            rospy.wait_for_service("trajectory_cmd", timeout=2.0)
            self.trajectory_srv_pub(self.trajectory_srv_req)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

if __name__ == '__main__':
    try:
        rospy.init_node("follow")
        while not rospy.is_shutdown():
            follow = Follow()
    except rospy.ROSInterruptException:
        pass