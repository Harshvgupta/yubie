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
import tensorflow as tf
import numpy as np
import cv2

import warnings
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import pathlib
import tf2_ros

class Fetch():
    
    def __init__(self):
        
        # Subscribe images 
        self.img_sub = rospy.Subscriber("spot_image",sensor_msgs.msg.Image)
        # Subscribe depth
        self.depth_sub = rospy.Subscriber("depth_image",sensor_msgs.msg.Image)
        # Subscribe CameraInfo
        self.cam_info_sub = rospy.Subscriber("spot_cam_info", sensor_msgs.msg.CameraInfo)

    def get_obj_bounding_box(self):

        # detect_fn = tf.saved_model.load("/home/harsh/Documents/spot-sdk/python/examples/fetch/ball/exported-models5/ball_model/saved_model/")
        loaded_model = tf.saved_model.load("/home/harsh/Documents/spot-sdk/python/examples/fetch/ball/exported-models5/ball_model/saved_model/")
        detect_fn = loaded_model.signatures["serving_default"]
        print("hhhh")

        category_index = label_map_util.create_category_index_from_labelmap("/home/harsh/Documents/spot-sdk/python/examples/fetch/ball/labelball/label_map.pbtxt", use_display_name=True)
        warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings

        if len(self.img_sub.shape) < 3:
            # Single channel image, convert to RGB.
            img_sub = cv2.cvtColor(self.img_sub, cv2.COLOR_GRAY2RGB)

        input_tensor = tf.convert_to_tensor(img_sub)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)
        print(detections)
        # All outputs are batches of tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.

        boxes = detections['detection_boxes']

        box = (boxes[0].tolist())
        height, width = img_sub.shape[:2]
        abs_boxes = boxes * np.array([height, width, height, width])
        abs_boxes_xy = abs_boxes[[1,0, 3, 2]]
        box_center = (abs_boxes_xy[:2] + abs_boxes_xy[2:]) / 2
        print(box_center)

        obj_img_coord = box_center


        return box_center


    def get_world_pos(self,box_center, depth, focal_length):

        x = focal_length * (box_center[0] / depth )
        y = focal_length * (box_center[1] / depth )
        print(x,y)

        return x, y



    def get_obj_tfrom_cam(self,world_params):
        # Publish ball_tfrom_cam as tf
        pass

if __name__ == '__main__':
    try:
        robot = Fetch()
        robot.get_obj_bounding_box()
    except rospy.ROSInterruptException:
        pass