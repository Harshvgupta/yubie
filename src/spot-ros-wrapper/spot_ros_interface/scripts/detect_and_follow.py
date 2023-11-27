#!/usr/bin/env python3
import cv2
import warnings
# ROS imports
import rospy
import geometry_msgs.msg
import std_msgs.msg
import sensor_msgs.msg
import visualization_msgs.msg
import spot_ros_srvs.srv
import tf
import os
import time
import numpy as np
import threading
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from ultralytics import YOLO
import torch
from cv_bridge import CvBridge, CvBridgeError
from dotenv import load_dotenv, find_dotenv


REL_MODEL_PATH = os.path.expanduser('~/spot_ros_wrapper/src/spot-ros-wrapper/spot_ros_interface/scripts/models/best.pt')


class Fetch():
    def __init__(self):
        # print(os.getcwd())
        self.model = YOLO('/home/spottop/harsh_env/spot_ros_wrapper/src/spot-ros-wrapper/spot_ros_interface/scripts/model_weights/best.pt')
        # self.model = YOLO('./model_weights/best.pt')
        # print(self.model)  # Load model with custom weights

        self.img_sub = rospy.Subscriber(
            "/spot_image", sensor_msgs.msg.Image, self.image_callback, queue_size=20)
        # Subscribe depth
        self.depth_sub = rospy.Subscriber(
            "/depth_array", std_msgs.msg.Float64MultiArray, self.depth_callback, queue_size=20)
        # Subscribe CameraInfo
        self.cam_info_sub = rospy.Subscriber(
            "/spot_cam_info", sensor_msgs.msg.CameraInfo, self.camera_callback, queue_size=20)

        self.ball_pos = rospy.Publisher(
            "ball_pose", geometry_msgs.msg.Pose, queue_size=20)
        self.ball_pos_viz = rospy.Publisher(
            "ball_pose_viz", visualization_msgs.msg.Marker, queue_size=20)
        self.final_pose_pub = rospy.Publisher(
            "final_ball", PoseStamped, queue_size=20)

        self.stand_srv_req = spot_ros_srvs.srv.StandRequest()
        self.stand_srv_pub = rospy.ServiceProxy(
            "stand_cmd", spot_ros_srvs.srv.Stand)

        self.cam_info = None
        self.depth_array = None
        self.image = None
        self.focal_length = None
        self.bridge = CvBridge()
        self.pose = geometry_msgs.msg.Pose()
        self.final_pose = geometry_msgs.msg.Pose()
        self.rate = rospy.Rate(2)
        self.marker = visualization_msgs.msg.Marker()
        self.tf = tf.TransformListener()
        self.pose_stamp = geometry_msgs.msg.PoseStamped()
        self.x = 0
        self.y = 0
        self.depthc = 0
        self.box_center = [0, 0]
        self.pose_ball = geometry_msgs.msg.PoseStamped()
        self.condition = True

    def get_detections(self, image):
        # Preprocessing the image
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return self.model(image)

    def extract_info(self, detections):
        boundingBoxes = []
        centroids = []
        for output in detections:
            if len(output.boxes) > 0:
                for boxes in output.boxes:
                    for box in boxes.xyxy:
                        bbox = [int(tensor.item()) for tensor in box]
                        x1, y1, x2, y2 = bbox
                        boundingBoxes.append(bbox)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        centroids.append((cx, cy))
        print("bounding",boundingBoxes)
        return boundingBoxes, centroids

    # def preprocess_image(self, image):
    #     # Resize the image to the input size expected by the model (e.g., 640x640 for YOLOv5)
    #     resized_image = cv2.resize(image, (640, 480))
    #     image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
    #     # Convert BGR to RGB
    #     # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     cv2.imshow("Image Window", image)
    #     # cv2.waitKey(1)
    #     # Normalize the image
    #     normalized_image = image / 255.0

    #     # Add batch dimension and convert to tensor
    #     input_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1).unsqueeze(0).float()
    #     # print(input_tensor)
    #     return input_tensor

    # def get_detections(self, image):
    #     # Preprocessing the image
    #     preprocessed_image = self.preprocess_image(image)

    #     # Running the model with a timeout
    #     with torch.no_grad():
    #         results = self.model_with_timeout(func=self.model, args=(preprocessed_image,), timeout_duration=60)
    #         if results is None:
    #             print("Model timed out or encountered an error.")
    #             return None

    #     detections = results.xyxy[0]  # Adjust this according to your model's output format
    #     return detections
        
    # def extract_info(self, detections):
    #     boundingBoxes = []
    #     centroids = []
    #     # Assuming detections is a tensor of shape [1, number_of_detections, 6]
    #     # where each detection is [x1, y1, x2, y2, score, class]
    #     for det in detections[0]:
    #         x1, y1, x2, y2, score, class_id = det
    #         if score > 0.3:  # Threshold to filter weak detections
    #             bbox = [int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())]
    #             boundingBoxes.append(bbox)
    #             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    #             centroids.append((cx.item(), cy.item()))

    #     print(boundingBoxes,centroids)
    #     # print("hell")
    #     return boundingBoxes, centroids

    # def model_with_timeout(self, func, args=(), kwargs={}, timeout_duration=60, default=None):
    #     class InterruptableThread(threading.Thread):
    #         def __init__(self):
    #             threading.Thread.__init__(self)
    #             self.result = default

    #         def run(self):
    #             try:
    #                 self.result = func(*args, **kwargs)
    #             except Exception as e:
    #                 self.result = default
    #                 print("Error in thread:", e)

    #     it = InterruptableThread()
    #     it.start()
    #     it.join(timeout_duration)
    #     if it.is_alive():
    #         print("Model processing timed out")
    #         return default
    #     else:
    #         return it.result

    # def model_with_timeout(self, detections, timeout_duration=30):
    #     result_container = {}
    #     print("1")
    #     def run_model():
    #         try:
    #             result_container['result'] = self.model(detections)
    #         except Exception as e:
    #             result_container['error'] = e
    #     print("2")
    #     model_thread = threading.Thread(target=run_model)
    #     model_thread.start()
    #     model_thread.join(timeout_duration)

    #     if model_thread.is_alive():
    #         print("Model processing timed out")
    #         # Handle the timeout scenario
    #         print("3")
    #     else:
    #         if 'result' in result_container:
    #             print("4")
    #             return result_container['result']
    #         if 'error' in result_container:
    #             print("5")
    #             raise result_container['error']



    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            detections = self.get_detections(image)
            if detections is not None:
                boundingBoxes, centroids = self.extract_info(detections)

                print("BOUNDING BOXES",boundingBoxes)

        
            # @Harsh let me know how convert bounding boxes and centroid to pose and publish
            # cond, num = self.get_obj_bounding_box(self.image)
            if len(centroids) > 0:
                print("ball identified")
                # @Viki add centroid with high accuracy
                self.get_world_pos(centroids[0], self.depthc)
                self.x, self.y = centroids[0]
                self.get_obj_pose(self.x, self.y, self.depthc)
                print(self.depthc)
                t = self.tf.getLatestCommonTime(
                    "/vision_odometry_frame", "/frontleft_fisheye_image")
                self.pose_stamp.header.frame_id = "/frontleft_fisheye_image"
                # print(t)
                self.pose_stamp.pose = self.pose
                self.pose_stamp.header.stamp = t
                final_pose = self.tf.transformPose(
                    "/vision_odometry_frame", self.pose_stamp)
                self.pose_ball = final_pose
                # ball_in_vodom = self.transform_pose(self.pose,"/frontleft_fisheye_image","/vision_odometry_frame")
                self.visualize_ball(final_pose, "vision_odometry_frame")

                self.ball_pos.publish(final_pose.pose)
                self.ball_pos_viz.publish(self.marker)
                self.final_pose_pub.publish(final_pose)
                self.ball_pos.publish(self.pose_ball.pose)
                self.condition = False
                self.rate.sleep()
        except CvBridgeError as e:
            print(e)
        except Exception as e:
            print("An error occurred:", e)
        finally:
            # Ensure that the rate.sleep() is called even if an exception occurs
            self.rate.sleep()

    def transform_pose(self, input_pose, from_frame, to_frame):
        t = self.tf.getLatestCommonTime(from_frame, to_frame)

        self.pose_stamp.header.frame_id = from_frame
        self.pose_stamp.pose = input_pose
        self.pose_stamp.header.stamp = t
        final_pose = self.tf.transformPose(to_frame, self.pose_stamp)

        return final_pose

    def publish(self):
        self.ball_pos.publish(self.pose_ball.pose)

    def trajectory(self, pose):
        # self.trajectory_srv_req.waypoints.poses=[]
        pose_arr = geometry_msgs.msg.PoseArray()
        # print(pose_arr)
        pose_arr.poses.append(pose)
        # print(f"pose array -> {pose_arr}")

        self.trajectory_srv_req.waypoints.poses = pose_arr.poses

        try:
            rospy.wait_for_service("trajectory_cmd", timeout=2.0)

            # print(f"trajectory -> {self.trajectory_srv_req}")

            self.trajectory_srv_pub(self.trajectory_srv_req)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def depth_callback(self, msg):
        # self.image = self.bridge.imgmsg_to_cv2(msg)
        self.depth_array = np.array(msg.data)
        # print(self.depth_array)
        self.depth_array = self.depth_array.reshape(640, 480)
        # print(self.depth_array)
        self.depthc = self.depth_array[int(
            self.box_center[0]), int(self.box_center[1])]
        # print("depth", self.depthc)

        # print(self.depth_array)

    def camera_callback(self, msg):
        self.cam_info = msg.K
        self.focal_length = self.cam_info[0]
        # print(self.focal_length)

    def get_key(self, cond):
        if cond == True:
            key = input("Enter key: ")
        else:
            key = "u"
        return key

    def get_obj_bounding_box(self, image):

        category_index = label_map_util.create_category_index_from_labelmap(
            "/home/harsh/Documents/spot-sdk/python/examples/fetch/ball/labelball/label_map.pbtxt", use_display_name=True)
        warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings

        # if len(image.shape) < 3:
        #     # Single channel image, convert to RGB.
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # input_tensor = tf1.convert_to_tensor(image)
        # input_tensor = input_tensor[tf1.newaxis, ...]
        # detections = self.detect_fn(input_tensor)
        # # print(detections)
        # # All outputs are batches of tensors.
        # # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # # We're only interested in the first num_detections.
        # num_detections = int(detections.pop('num_detections'))
        # detections = {key: value[0, :num_detections].numpy()
        #               for key, value in detections.items()}
        # detections['num_detections'] = num_detections

        # # detection_classes should be ints.
        # boxes = detections['detection_boxes']

        # box = (boxes[0].tolist())
        # print(boxes,box)
        # image_np_with_detections = image.copy()
        # viz_utils.visualize_boxes_and_labels_on_image_array(
        #       image_np_with_detections,
        #       detections['detection_boxes'],
        #       detections['detection_classes'],
        #       detections['detection_scores'],
        #       category_index,
        #       use_normalized_coordinates=True,
        #       max_boxes_to_draw=200,
        #       min_score_thresh=.30,
        #       agnostic_mode=True)
        # height, width = image.shape[:2]

        # cv2.imshow("figure",image_np_with_detections)
        # cv2.waitKey(1)
        # abs_boxes = boxes * np.array([height, width, height, width])
        # abs_boxes_xy = abs_boxes[[1,0, 3, 2]]
        # box_center = (abs_boxes_xy[:2] + abs_boxes_xy[2:]) / 2

        # obj_img_coord = box_center
        # time.sleep(10)

        # return box_center
        # image=np.array(image)
        if len(image.shape) < 3:
            # Single channel image, convert to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        input_tensor = tf1.convert_to_tensor(image)
        input_tensor = input_tensor[tf1.newaxis, ...]
        detections = self.detect_fn(input_tensor)

        # All outputs are batches of tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        # print(len(detections))
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)

        image_np_with_detections = image.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=0.1,
            agnostic_mode=False
        )
        cv2.imshow("figure", image_np_with_detections)
        cv2.waitKey(1)

        # Compute the center based on the provided bounding boxes (from detections)
        boxes = detections['detection_boxes']
        scores = detections['detection_scores']

        # Check if there are any detections
        if len(boxes) > 0:
            height, width = image_np_with_detections.shape[:2]
            if self.depthc > 0:
                num = 1
            else:
                num = 0
            for box, score in zip(boxes, scores):
                # Check if score is greater than 0.50
                if score > 0.50:
                    abs_box = box * np.array([height, width, height, width])
                    # Reorder the coordinates
                    abs_box_xy = abs_box[[1, 0, 3, 2]]
                    self.box_center = (abs_box_xy[:2] + abs_box_xy[2:]) / 2
                    print(self.box_center)

                    return True, num
                else:
                    return False, num

    def get_world_pos(self, box_center, depthc):
        # cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
        # cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows,
        #                         image_responses[0].shot.image.cols)
        # print(cv_depth[0,:])
        # print(cv_depth[242][213])
        self.x = 331 * (box_center[0] / depthc)
        self.y = 331 * (box_center[1] / depthc)
        print("center", box_center[0], box_center[1])

        return self.x, self.y

    def get_obj_pose(self, x, y, depthc):
        self.pose.position.x = x/1000
        self.pose.position.y = y/1000
        self.pose.position.z = depthc/1000
        self.pose.orientation.x = 0
        self.pose.orientation.y = 0
        self.pose.orientation.z = 0
        self.pose.orientation.w = 1

    def visualize_ball(self, ball_pose, frame_id):
        self.marker.header.frame_id = frame_id
        self.marker.header.stamp = rospy.get_rostime()
        self.marker.id = 0
        self.marker.type = 2
        self.marker.pose.position = ball_pose.pose.position
        self.marker.pose.orientation = ball_pose.pose.orientation
        self.marker.scale.x = 0.1
        self.marker.scale.y = 0.1
        self.marker.scale.z = 0.1
        self.marker.color.r = 0.0
        self.marker.color.g = 0.0
        self.marker.color.b = 1.0
        self.marker.color.a = 1.0


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    try:
        rospy.init_node("fetch")
        robot = Fetch()
        rospy.spin()
    except rospy.ROSInterruptException:
        robot.publish()
