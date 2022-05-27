# -*- coding: utf-8 -*-
import math
import sys
import time

import cv2
import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
import numpy as np
import roslib
import rospy
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from scipy import ndimage

''' moveit commander settings '''
moveit_commander.roscpp_initialize(sys.argv)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
gripper = moveit_commander.MoveGroupCommander('gripper')
arm = moveit_commander.MoveGroupCommander('arm')
#gripper = moveit_commander.MoveGroupCommander('gripper')
arm.allow_replanning(True)
arm.set_planning_time(5)


class RobotOperator():
    def __init__ (self):
        ''' Motor settings '''
        twist = Twist()
        twist.linear.x = twist.linear.y = twist.linear.z = 0.0
        twist.angular.x = twist.angular.y = 0.0
        self.twist = twist

        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=20)

        self.before_direction = 1
        self.yolo_threshold = 0.70
        self.color_threshold = 0.33
        self.yolo_data = None
        self.lidar_data = None
        self.color_data = None
        self.current_state = None
        self.robot_state = ["sense_yolo", "sense_lidar", "sense_color",
                            "decide", "act_find", "act_approach", "act_grip", "halt"]

        self.bridge = CvBridge()
        self.image_fetch = np.zeros((1280, 720, 3))


    def joint(joint_diff=[0,0,0,0]):
        global arm, sleep_time

        joint_values = arm.get_current_joint_values()
        rospy.sleep(sleep_time)
        print (joint_values)
        for i in range(4):
            joint_values[i] = joint_values[i] + joint_diff[i]
        print (type(joint_values))
        print (joint_values)
        arm.go(joints=joint_values, wait=True)
        rospy.sleep(sleep_time)
        arm.stop()
        rospy.sleep(sleep_time)
        arm.clear_pose_targets()
        rospy.sleep(sleep_time)


    def gripper_move(self,coeff):
        global gripper

        #coeff : 1 or -1
        joint_values = gripper.get_current_joint_values()
        rospy.sleep(sleep_time)
        print ("gripper joint values : " + str(joint_values) )
        #gripper.set_joint_value_target([0.01])
        gripper.go([0.01*coeff,0.0], wait=True)
        rospy.sleep(sleep_time)

    def reset_grip(self):
        """로봇팔 초기 상태로 리셋
        """
        global sleep_time

        joint_values = arm.get_current_joint_values()
        chk = False
        for i in range(4):
            if(abs(joint_values[i]) > 0.5):
                chk = True
                joint_values[i] = 0.0
        if(chk):
            arm.go(joint_values,wait=True)
            rospy.sleep(sleep_time)
        self.gripper_move(1.5)

    def subscribe(self):
        rospy.init_node('RobotOperator', anonymous=True)
        rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.yolo_callback)
        rospy.Subscriber('/scan_heading', Float32, self.lidar_callback)
        rospy.Subscriber('/video_source/raw_2', Image, self.color_callback)

        while(self.current_state != "halt"):
            self.run_proc()
        #TODO : Add lidar callback, color filter callback
        rospy.spin()

    def yolo_callback(self, data):
        """Return x coordinates of center of box which is bottle on the **far right**
        """
        # assert (self.current_state == "sense")
        bottle_boxes = [
            each for each in data.bounding_boxes if each.Class == 'bottle']
        bottle_center_xs = [(each.xmin + each.xmax) //
                            2 for each in bottle_boxes]
        # -1: far right / 0: far left
        self.yolo_data = sorted(bottle_center_xs)[-1]

    def lidar_callback(self, data):
        # assert(self.current_state == "senser")
        self.lidar_data = data.data

    def color_callback(self,data):
        im = np.frombuffer(data.data, dtype=np.uint8).reshape(
            data.height, data.width, -1)
        im = cv2.blur(im, (25, 25))
        hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
        #print(im.shape)

        lower_color1 = np.array([160, 60, 60])
        upper_color1 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hls, lower_color1, upper_color1)
        result = cv2.bitwise_and(im, im, mask=mask1)

        com = ndimage.center_of_mass(result)
        std = ndimage.standard_deviation(result)
        if not math.isnan(com[0]):
            cv2.circle(result, (int(com[1]), int(com[0])), 5, (255, 255, 255), -1)
            self.color_data = data

        if np.any(result):
            nzeros = np.nonzero(result)
            x_max = np.max(nzeros[1])
            x_min = np.min(nzeros[1])
            y_max = np.max(nzeros[0])
            y_min = np.min(nzeros[0])
            print("nzeros",x_max,x_min,y_max,y_min)
            cv2.rectangle(result, (x_min,y_min),(x_max,y_max),(255,0,0),3)
            print("box size: ",x_max - x_min, y_max - y_min)

        #print(com)
        #print(std)
        #cv2.imshow('frame',result)
        #if cv2.waitKey(5) & 0xFF == 27:
        #    sys.exit()
        return (int(com[1]), int(com[0]))

    def set_next_state(self, next_act = None):
        # assert current state before change
        if (self.current_state == "decide"):
            assert next_act == "act_find" or next_act == "act_approach" or next_act == "act_grip"
            self.current_state = next_act
        elif (self.current_state == "act_find" or self.current_state == "act_approach"):
            self.current_state = "sense_yolo"
        elif (self.current_state == "act_grip"):
            self.current_state = "halt"

    def rotate_right(self):  # FIXME: before_direction을 곱하면 오른쪽으로 돌지 않고 이전에 돌던 방향으로 돕니다
        self.twist.angular.z = 0.1 * self.before_direction
        self.pub.publish(self.twist)
        return

    def find_target(self):
        # current state : act_find
        while(self.yolo_data is None):
            self.rotate_right()
        self.set_next_state("decide")
        return

    def match_direction(self):
        coordinates_criterion = None

        if self.lidar_data >= self.yolo_threshold:
            coordinates_criterion = self.yolo_data
        elif self.lidar_data >= self.color_threshold:
            coordinates_criterion = self.color_data and self.color_data[0]

        if coordinates_criterion is None:
            return

        move = float(640 - coordinates_criterion) / 640

        if abs(move) < 0.15:
            self.twist.angular.z = 0
        else:
            self.twist.angular.z = move * 0.5

        self.before_direction = -1 if move < 0 else 1
        self.pub.publish(self.twist)

    def go_front(self):
        pass

    def approach(self):
        while(self.lidar_data >= self.color_threshold):
            self.match_direction()
            self.go_front()
        self.set_next_state("decide")
        pass

    def gripper(self):
        pass
        self.set_next_state("halt")
        return

    def robot_halt(self):
        pass
        return

    def grip_condition_check(self, yolo_data, lidar_data, color_data):
        pass
        return True or False

    def sensor_init(self):
        self.yolo_data = None
        self.lidar_data = None
        self.color_data = None

    def run_proc(self):
        # current state : sense_color
        assert (self.current_state == "decide")

        if(self.yolo_data is None):
            # go to next state
            self.set_next_state("act_find")
            self.find_target()
            return

        assert (self.current_state == "decide")

        if(self.grip_condition_check(self.yolo_data, self.lidar_data, self.color_data)):
            self.set_next_state("act_gripper")
            self.gripper()
        else:
            self.set_next_state("act_approach")
            self.approach()


def main():
    operator = RobotOperator()
    operator.subscribe()


if __name__ == '__main__':
    main()
