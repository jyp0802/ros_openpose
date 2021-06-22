#!/usr/bin/env python

# import modules
import sys
import cv2
import math
import rospy
import argparse
import message_filters
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import ColorRGBA
from ros_openpose.msg import Frame, Person, BodyPart, Pixel
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3, Point


# Import Openpose (Ubuntu)
rospy.init_node('ros_openpose')
py_openpose_path = rospy.get_param("~py_openpose_path")
try:
    # If you run `make install` (default path is `/usr/local/python` for Ubuntu)
    sys.path.append(py_openpose_path)
    from openpose import pyopenpose as op
except ImportError as e:
    rospy.logerr('OpenPose library could not be found. '
                 'Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


OPENPOSE1POINT7_OR_HIGHER = 'VectorDatum' in op.__dict__


class rosOpenPose:
    def __init__(self, frame_id, no_depth, pub_topic, color_topic, depth_topic, cam_info_topic, op_wrapper, display, skeleton_line_width, params):

        self.pub = rospy.Publisher(pub_topic, MarkerArray, queue_size=1)

        self.frame_id = frame_id
        self.no_depth = no_depth

        self.bridge = CvBridge()

        self.op_wrapper = op_wrapper

        self.display = display
        self.frame = None

        # Populate necessary K matrix values for 3D pose computation.
        cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo)
        self.fx = cam_info.K[0]
        self.fy = cam_info.K[4]
        self.cx = cam_info.K[2]
        self.cy = cam_info.K[5]

        # Obtain depth topic encoding
        encoding = rospy.wait_for_message(depth_topic, Image).encoding
        self.mm_to_m = 0.001 if encoding == "16UC1" else 1.

        # Function wrappers for OpenPose version discrepancies
        if OPENPOSE1POINT7_OR_HIGHER:
            self.emplaceAndPop = lambda datum: self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
            self.detect = lambda kp: kp is not None
        else:
            self.emplaceAndPop = lambda datum: self.op_wrapper.emplaceAndPop([datum])
            self.detect = lambda kp: kp.shape != ()

        image_sub = message_filters.Subscriber(color_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 0.01)
        self.ts.registerCallback(self.callback)

        self.colors = [ColorRGBA(0.12, 0.63, 0.42, 1.00),
                       ColorRGBA(0.98, 0.30, 0.30, 1.00),
                       ColorRGBA(0.26, 0.09, 0.91, 1.00),
                       ColorRGBA(0.77, 0.44, 0.14, 1.00),
                       ColorRGBA(0.92, 0.73, 0.14, 1.00),
                       ColorRGBA(0.00, 0.61, 0.88, 1.00),
                       ColorRGBA(1.00, 0.65, 0.60, 1.00),
                       ColorRGBA(0.59, 0.00, 0.56, 1.00)]

        self.fingers = 5
        self.count_keypoints_one_finger = 5
        self.total_finger_kepoints = self.fingers * self.count_keypoints_one_finger
        self.skeleton_line_width = skeleton_line_width

        """ OpenPose skeleton dictionary
        {0, "Nose"}, {13, "LKnee"}
        {1, "Neck"}, {14, "LAnkle"}
        {2, "RShoulder"}, {15, "REye"}
        {3, "RElbow"}, {16, "LEye"}
        {4, "RWrist"}, {17, "REar"}
        {5, "LShoulder"}, {18, "LEar"}
        {6, "LElbow"}, {19, "LBigToe"}
        {7, "LWrist"}, {20, "LSmallToe"}
        {8, "MidHip"}, {21, "LHeel"}
        {9, "RHip"}, {22, "RBigToe"}
        {10, "RKnee"}, {23, "RSmallToe"}
        {11, "RAnkle"}, {24, "RHeel"}
        {12, "LHip"}, {25, "Background"}
        """

        self.params = params

    def compute_3D_vectorized(self, kp, depth):
        # Create views (no copies made, so this remains efficient)
        U = kp[:, :, 0]
        V = kp[:, :, 1]

        # Extract the appropriate depth readings
        num_persons, body_part_count = U.shape
        XYZ = np.zeros((num_persons, body_part_count, 3), dtype=np.float32)
        for i in range(num_persons):
            for j in range(body_part_count):
                u, v = int(U[i, j]), int(V[i, j])
                if v < depth.shape[0] and u < depth.shape[1]:
                    XYZ[i, j, 2] = depth[v, u]

        XYZ[:, :, 2] *= self.mm_to_m  # convert to meters

        # Compute 3D coordinates in vectorized way
        Z = XYZ[:, :, 2]
        XYZ[:, :, 0] = (Z / self.fx) * (U - self.cx)
        XYZ[:, :, 1] = (Z / self.fy) * (V - self.cy)
        return XYZ
    
    def create_marker(self, index, color, marker_type, size, time):
        '''
        Function to create a visualization marker which is used inside RViz
        '''
        marker = Marker()
        marker.id = index
        marker.ns = "hand_marker"
        marker.color = color
        marker.action = Marker.ADD
        marker.type = marker_type
        marker.scale = Vector3(size, size, size)
        marker.header.stamp = time
        marker.header.frame_id = self.frame_id
        marker.lifetime = rospy.Duration(1)  # 1 second
        return marker

    def isValid(self, bodyPart):
        '''
        When should we consider a body part as a valid entity?
        We make sure that the score and z coordinate is a positive number.
        Notice that the z coordinate denotes the distance of the object located
        in front of the camera. Therefore it must be a positive number always.
        '''
        return bodyPart.score > 0 and not math.isnan(bodyPart.point.x) and not math.isnan(bodyPart.point.y) and not math.isnan(bodyPart.point.z) and bodyPart.point.z > 0

    def callback(self, ros_image, ros_depth):
        marker_array = MarkerArray()
        marker_counter = 0

        # Convert images to cv2 matrices
        image = depth = None
        try:
            image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(ros_depth, "passthrough")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        # Push data to OpenPose and block while waiting for results
        datum = op.Datum()
        datum.cvInputData = image
        self.emplaceAndPop(datum)

        pose_kp = datum.poseKeypoints
        lhand_kp = datum.handKeypoints[0]
        rhand_kp = datum.handKeypoints[1]

        # Set number of people detected
        if self.detect(pose_kp):
            num_persons = pose_kp.shape[0]
            body_part_count = pose_kp.shape[1]
        else:
            num_persons = 0
            body_part_count = 0

        # Check to see if hands were detected
        lhand_detected = False
        rhand_detected = False
        hand_part_count = 0

        if self.detect(lhand_kp):
            lhand_detected = True
            hand_part_count = lhand_kp.shape[1]

        if self.detect(rhand_kp):
            rhand_detected = True
            hand_part_count = rhand_kp.shape[1]

        # Handle body points
        if num_persons != 0:
            if lhand_detected:
                lh_XYZ = self.compute_3D_vectorized(lhand_kp, depth)
            if rhand_detected:
                rh_XYZ = self.compute_3D_vectorized(rhand_kp, depth)
        
        for person in range(num_persons):
            now = rospy.Time.now()
            marker_color = self.colors[person % len(self.colors)]

            # All joints on hands
            if self.params["hand_mode"] == "all":
                if self.params["hand_left"]:
                    left_hand = [self.create_marker(marker_counter + idx, marker_color, Marker.LINE_STRIP, self.skeleton_line_width, now) for idx in range(self.fingers)]
                    marker_counter += self.fingers
                if self.params["hand_right"]:
                    right_hand = [self.create_marker(marker_counter + idx, marker_color, Marker.LINE_STRIP, self.skeleton_line_width, now) for idx in range(self.fingers)]
                    marker_counter += self.fingers

                keypoint_counter = 0
                for idx in range(self.total_finger_kepoints):
                    strip_id = idx / self.count_keypoints_one_finger
                    temp_id = idx % self.count_keypoints_one_finger
                    if temp_id == 0:
                        point_id = temp_id
                    else:
                        keypoint_counter += 1
                        point_id = keypoint_counter
                    
                    if self.params["hand_left"]:
                        lh_part = BodyPart()
                        lh_part.pixel.x, lh_part.pixel.y, lh_part.score = lhand_kp[person, point_id]
                        lh_part.point.x, lh_part.point.y, lh_part.point.z = lh_XYZ[person, point_id]
                    if self.params["hand_right"]:
                        rh_part = BodyPart()
                        rh_part.pixel.x, rh_part.pixel.y, rh_part.score = rhand_kp[person, point_id]
                        rh_part.point.x, rh_part.point.y, rh_part.point.z = rh_XYZ[person, point_id]

                    if self.params["hand_left"] and self.isValid(lh_part):
                        left_hand[strip_id].points.append(lh_part.point)
                    if self.params["hand_right"] and self.isValid(rh_part):
                        right_hand[strip_id].points.append(rh_part.point)

                marker_array.markers.extend(left_hand)
                marker_array.markers.extend(right_hand)

            # Only hand wrist
            elif self.params["hand_mode"] == "wrist":
                if self.params["hand_left"]:
                    lh_part = BodyPart()
                    lh_part.pixel.x, lh_part.pixel.y, lh_part.score = lhand_kp[person, 0]
                    lh_part.point.x, lh_part.point.y, lh_part.point.z = lh_XYZ[person, 0]

                    if self.isValid(lh_part):
                        lh_marker = self.create_marker(marker_counter, marker_color, Marker.SPHERE, 0.1, now)
                        lh_marker.pose.position = lh_part.point
                        marker_array.markers.append(lh_marker)
                        marker_counter += 1

                if self.params["hand_right"]:
                    rh_part = BodyPart()
                    rh_part.pixel.x, rh_part.pixel.y, rh_part.score = rhand_kp[person, 0]
                    rh_part.point.x, rh_part.point.y, rh_part.point.z = rh_XYZ[person, 0]

                    if self.isValid(rh_part):
                        rh_marker = self.create_marker(marker_counter, marker_color, Marker.SPHERE, 0.1, now)
                        rh_marker.pose.position = rh_part.point
                        marker_array.markers.append(rh_marker)
                        marker_counter += 1

            # Only hand center
            elif self.params["hand_mode"] == "center":
                if self.params["hand_left"]:
                    left_exists = []
                    left_center = np.zeros(3)
                if self.params["hand_right"]:
                    right_exists = []
                    right_center = np.zeros(3)

                for i in [0,2,5,9,13,17]:
                    if self.params["hand_left"]:
                        lh_part = BodyPart()
                        lh_part.pixel.x, lh_part.pixel.y, lh_part.score = lhand_kp[person, i]
                        lh_part.point.x, lh_part.point.y, lh_part.point.z = lh_XYZ[person, i]

                        if self.isValid(lh_part):
                            if self.params["hand_joint"]:
                                lh_marker = self.create_marker(marker_counter, marker_color, Marker.SPHERE, 0.02, now)
                                lh_marker.pose.position = lh_part.point
                                marker_array.markers.append(lh_marker)
                                marker_counter += 1
                            left_exists.append(i)
                            left_center += lh_XYZ[person, i]

                    if self.params["hand_right"]:
                        rh_part = BodyPart()
                        rh_part.pixel.x, rh_part.pixel.y, rh_part.score = rhand_kp[person, i]
                        rh_part.point.x, rh_part.point.y, rh_part.point.z = rh_XYZ[person, i]

                        if self.isValid(rh_part):
                            if self.params["hand_joint"]:
                                rh_marker = self.create_marker(marker_counter, marker_color, Marker.SPHERE, 0.02, now)
                                rh_marker.pose.position = rh_part.point
                                marker_array.markers.append(rh_marker)
                                marker_counter += 1
                            right_exists.append(i)
                            right_center += rh_XYZ[person, i]
                
                if self.params["hand_left"] and 0 in left_exists:
                    lh_marker = self.create_marker(marker_counter, marker_color, Marker.SPHERE, 0.05, now)
                    left_center /= len(left_exists)
                    lh_marker.pose.position.x, lh_marker.pose.position.y, lh_marker.pose.position.z = left_center
                    marker_array.markers.append(lh_marker)
                    marker_counter += 1
                    if self.params["hand_vector"]:
                        p1 = p2 = 0
                        for i in [13,17]:
                            if i in left_exists:
                                p1 = i
                                break
                        for i in [5,2]:
                            if i in left_exists:
                                p2 = i
                                break
                        if p1 and p2:
                            p1 = lh_XYZ[person, p1]
                            p2 = lh_XYZ[person, p2]

                            v1 = p1 - left_center
                            v2 = p2 - left_center
                            nv = np.cross(v1, v2)
                            nv = nv / np.linalg.norm(nv)

                            a, b, c = nv
                            d = np.dot(nv, p2)

                            # print('plane equation:\n{:1.4f}x + {:1.4f}y + {:1.4f}z + {:1.4f} = 0'.format(a, b, c, d))

                            vec = self.create_marker(marker_counter, ColorRGBA(0.5, 0.5, 0.5, 1.00), Marker.ARROW, 0, now)
                            vec.scale.x = 0.01
                            vec.scale.y = 0.02
                            
                            vec.points.append(Point(*left_center))
                            vec.points.append(Point(*(left_center+nv*0.1)))

                            marker_array.markers.append(vec)
                            marker_counter += 1

                if self.params["hand_right"] and 0 in right_exists:
                    rh_marker = self.create_marker(marker_counter, marker_color, Marker.SPHERE, 0.05, now)
                    right_center /= len(right_exists)
                    rh_marker.pose.position.x, rh_marker.pose.position.y, rh_marker.pose.position.z = right_center
                    marker_array.markers.append(rh_marker)
                    marker_counter += 1
                        

            # Do only one person
            break

        if self.display: self.frame = datum.cvOutputData.copy()
        self.pub.publish(marker_array)


def main():
    frame_id = rospy.get_param("~frame_id")
    no_depth = rospy.get_param("~no_depth")
    pub_topic = rospy.get_param("~pub_topic")
    color_topic = rospy.get_param("~color_topic")
    depth_topic = rospy.get_param("~depth_topic")
    cam_info_topic = rospy.get_param("~cam_info_topic")
    skeleton_line_width = rospy.get_param('~skeleton_line_width')
    try:
        # Flags, refer to include/openpose/flags.hpp for more parameters
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()

        # Custom Params for OpenPose Wrapper
        params = dict()
        # Can manually set params like this as well
        # params["model_folder"] = "/home/asjchoi/Programs/openpose-1.6.0/models"
        params["display"] = 0

        # Custom Params for OpenPose Wrapper
        py_params = dict()
        py_params["hand_mode"] = "center"
        py_params["hand_left"] = True
        py_params["hand_right"] = False
        py_params["hand_joint"] = False
        py_params["hand_vector"] = True

        # Any more obscure flags can be found through this for loop
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params: params[key] = next_item

        # Starting OpenPose
        op_wrapper = op.WrapperPython()
        op_wrapper.configure(params)
        op_wrapper.start()

        display = True if 'display' not in params or int(params['display']) > 0 else False

        # Start ros wrapper
        rop = rosOpenPose(frame_id, no_depth, pub_topic, color_topic, depth_topic, cam_info_topic, op_wrapper, display, skeleton_line_width, py_params)

        if display:
            while not rospy.is_shutdown():
                if rop.frame is not None:
                    cv2.imshow("Ros OpenPose", rop.frame)
                    cv2.waitKey(1)
        else:
            rospy.spin()

    except Exception as e:
        rospy.logerr(e)
        sys.exit(-1)


if __name__ == "__main__":
    main()
