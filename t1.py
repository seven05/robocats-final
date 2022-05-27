# -*- coding: utf-8 -*-
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
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String

from . import custom_msg

""" robot state variables """
robot_state = ["sense_yolo", "sense_lidar", "sense_color",
            "decide", "act_find", "act_approach", "act_grip","halt"]
success_state = [False,False,False,False,False,False,False]
current_state = robot_state[6] #state halt

""" sense data variables"""
bounding_box = None
lidar_distance = None

def next_state(next_act = None):
    if(current_state == "sense_yolo"):
        return "sense_lidar"
    elif (current_state == "sense_lidar"):
        return "sense_color"
    elif (current_state == "sense_color"):
        return "decide"
    elif (current_state == "decide"):
        assert next_act == "act_find" or next_act == "act_approach" or next_act == "act_grip"
        return next_act
    elif (current_state == "act_find" or current_state == "act_approach"):
        return "sense_yolo"
    elif (current_state == "act_grip"):
        return "halt"


"""
return None when no lidar connection or data
return distance when lidar data collected
"""
def get_lidar_distance(lidar_data):
    pass

def callback(yolo_data):
    global bounding_box, robot_state, success_state,current_state

    assert current_state == "sense_yolo"
    rospy.loginfo('current_step: ' + str(current_state))
    # yolo data에서 bottle 이름을 가진 label만 추출
    bottle_boxes = [each for each in yolo_data.bounding_boxes if each.Class == 'bottle']

    if len(bottle_boxes) == 0:  # bottle을 못찾으면 이전 진행 방향으로 계속 회전시킴
        bounding_box = None
        success_state[0] = False
        current_state = next_state()

    # box = bottle_boxes[0]  # legacy code; 항상 첫번째 bottle 추적 -> 경우에 따라 두 병의 순서가 계속 바뀌면 어떡할까?
    # bottle label이 여러개 발견 될 경우를 대비해서 xmin 좌표가 가장 작은 box 하나 추출
    box = sorted(bottle_boxes, key=lambda box: box.xmin)[0]

    bounding_box = box
    success_state[0] = True
    current_state = next_state()

def callback_lidar(lidar_data):
    global lidar_distance, success_state, current_state

    assert current_state == "sense_lidar"
    rospy.loginfo('current step: ' + str(current_state))

    # TODO : write get_lidar_data
    l_distance = get_lidar_distance(lidar_data)

    if(l_distance == None):
        lidar_distance = None
        success_state[1] = False
        current_state = next_state()
    else:
        lidar_distance = l_distance
        success_state[1] = True
        current_state = next_state()


class RobotOperator():
    def __init__ (self):
        self.lidar_threshold = None
        self.yolo_data = None
        self.lidar_data = None
        self.color_data = None
        self.current_state = None
        self.robot_state = ["sense_yolo", "sense_lidar", "sense_color",
                            "decide", "act_find", "act_approach", "act_grip", "halt"]
    def subscribe(self):
        rospy.init_node('RobotOperator', anonymous=True)
        rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.yolo_callback)
        rospy.Subscriber('/scan_heading', Float32, self.lidar_callback)
        rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.color_callback)  # FIXME: change queue name and class of queue

        while(self.current_state != "halt"):
            self.run_proc()
        #TODO : Add lidar callback, color filter callback
        rospy.spin()

    def yolo_callback(self, data):
        """Return x coordinates of center of box which is bottle on the **far right**
        """
        # assert (self.current_state == "sense")
        bottle_boxes = [each for each in data.bounding_boxes if each.Class == 'bottle']
        bottle_center_xs = [(each.xmin + each.xmax) // 2 for each in bottle_boxes]
        self.yolo_data = sorted(bottle_center_xs)[-1]  # -1: far right / 0: far left

    def lidar_callback(self, data):
        # assert(self.current_state == "senser")
        self.lidar_data = data.data

    def color_callback(self,data):  # TODO
        assert(self.current_state == "sense")
        self.set_next_state()
        self.color_data = data
        self.run_process()

    def set_next_state(self, next_act = None):
        # assert current state before change
        if(self.current_state == "sense"):
            self.current_state =  "decide"
        elif (self.current_state == "decide"):
            assert next_act == "act_find" or next_act == "act_approach" or next_act == "act_grip"
            self.current_state = next_act
        elif (self.current_state == "act_find" or current_state == "act_approach"):
            self.current_state = "sense_yolo"
        elif (self.current_state == "act_grip"):
            self.current_state = "halt"

    def rotate_right(self):
        pass
        return

    def find_target(self):
        # current state : act_find
        while(self.yolo_data is None):
            self.rotate_right()
        self.set_next_state("sense_yolo")
        return

    def match_direction(self):
        pass

    def get_front(self):
        pass

    def approach(self):
        while(self.lidar_data >= self.lidar_threshold):
            self.match_direction()
            self.get_front()
        self.set_next_state("sense_yolo")
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

        if(self.yolo_data is None and self.color_data is None):
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
