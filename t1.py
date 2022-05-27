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
sleep_time = 3


class RobotOperator():
    def __init__ (self):
        ''' Motor settings '''
        twist = Twist()
        twist.linear.x = twist.linear.y = twist.linear.z = 0.0
        twist.angular.x = twist.angular.y = 0.0
        self.twist = twist

        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=20)

        self.chk = False
        self.before_direction = 1
        self.yolo_threshold = 0.70
        self.color_threshold = 0.33
        self.yolo_data = None
        self.lidar_data = None
        self.color_data = None
        self.current_state = "decide"
        self.robot_state = ["decide", "act_find", "act_approach", "act_grip", "halt"]
        
        self.find_criterion = 'yolo'

        self.bridge = CvBridge()
        self.image_fetch = np.zeros((1280, 720, 3))


    def joint(self,joint1,joint2,joint3,joint4):
        global arm, sleep_time
        joint_diff = [joint1,joint2,joint3,joint4]
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
        global gripper, sleep_time

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
        
        self.current_state = "decide"
        
        while(self.current_state != "halt"):
            self.run_proc()
            
        self.robot_halt()
        #TODO : Add lidar callback, color filter callback
        rospy.spin()

    def yolo_callback(self, data):
        """Return x coordinates of center of box which is bottle on the **far right**
        """
        bottle_boxes = [
            each for each in data.bounding_boxes if each.Class == 'bottle']

        if len(bottle_boxes) == 0:
            self.yolo_data = None
            return

        bottle_center_xs = [(each.xmin + each.xmax) //
                            2 for each in bottle_boxes]
        # -1: far right / 0: far left
        self.yolo_data = sorted(bottle_center_xs)[-1]

    def lidar_callback(self, data):
        self.lidar_data = data.data

    def color_callback(self,data):
        """color filter image, return center of mass of cluster (com_x,com_y)
        """
        im = np.frombuffer(data.data, dtype=np.uint8).reshape(
            data.height, data.width, -1)
        im = cv2.blur(im, (25, 25))
        hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)

        lower_color1 = np.array([160, 60, 60])
        upper_color1 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hls, lower_color1, upper_color1)
        result = cv2.bitwise_and(im, im, mask=mask1)

        com = ndimage.center_of_mass(result)
        # std = ndimage.standard_deviation(result)
        if not math.isnan(com[0]):
            cv2.circle(result, (int(com[1]), int(com[0])), 5, (255, 255, 255), -1)
            self.color_data = int(com[1]), int(com[0])
        else:
            self.color_data = None

    def set_next_state(self, next_act = None):
        # assert current state before change
        if (self.current_state == "decide"):
            if(next_act == "act_find" or next_act == "act_approach" or next_act == "act_grip"):
                self.current_state = next_act
            else:
                print("Error in set_next_state")
            
        elif (self.current_state == "act_find" or self.current_state == "act_approach"):
            self.current_state = "decide"
        elif (self.current_state == "act_grip"):
            self.current_state = "halt"

    def rotate_previous_direction(self):
        self.twist.angular.z = 0.1 * self.before_direction
        self.pub.publish(self.twist)
        return

    def find_target(self):
        # current state : act_find
        if(self.current_state != "act_find"):
            print("ERROR : state is wrong, not act find, ",self.current_state)
        self.rotate_previous_direction()
        while(self.yolo_data is None):
            pass
        self.robot_halt()
        self.set_next_state("decide")
        return

    def match_direction(self):
        coordinates_criterion = None

        if self.lidar_data >= self.yolo_threshold:
            coordinates_criterion = self.yolo_data
        elif self.lidar_data >= self.color_threshold:
            if self.find_criterion == 'yolo':
                print('Change find criterion: yolo -> color')
                self.find_criterion = 'color'
            coordinates_criterion = self.color_data and self.color_data[0]

        if coordinates_criterion is None:
            return

        move = float(640 - coordinates_criterion) / 640

        if abs(move) < 0.1:
            self.twist.angular.z = 0
        else:
            self.twist.angular.z = move / abs(move) * 0.1

        self.before_direction = -1 if move < 0 else 1
        self.pub.publish(self.twist)
        #print("match_direction published")

    def go_front(self):
        self.twist.linear.x = 0.02
        self.pub.publish(self.twist)

    def approach(self):
        print("approach")
        while(self.lidar_data >= self.color_threshold):
            self.match_direction()
            self.go_front()
            time.sleep(0.01)
        self.robot_halt()
        self.set_next_state("decide")
        pass

    def gripper_execute(self):
        print("gripper_execute called")
        self.joint(0, 0.0, -0.8, 0.0)
        self.joint(0, 1.1, -0.0, 0.0)
        self.gripper_move(-1.0)
        self.joint(0, -0.8, 0.0, 0.0)
        self.set_next_state("halt")
        return

    def robot_halt(self):
        print("robot_halt")
        self.twist.linear.x = 0.0
        self.twist.linear.y = 0.0
        self.twist.linear.z = 0.0
        self.twist.angular.x = 0.0
        self.twist.angular.y = 0.0
        self.twist.angular.z = 0.0
        self.pub.publish(self.twist)
        return

    def grip_condition_check(self):
        return self.lidar_data <= self.color_threshold

    def sensor_init(self):
        self.yolo_data = None
        self.lidar_data = None
        self.color_data = None

    def run_proc(self):
        
        if (self.current_state != "decide"):
            return

        if(self.yolo_data is None and self.chk == False):
            # go to next state
            self.set_next_state("act_find")
            self.find_target()
            return

        if (self.current_state != "decide"):
            return
        
        self.chk=True

        if(self.grip_condition_check()):
            self.set_next_state("act_grip")
            self.gripper_execute()
            return
        else:
            self.set_next_state("act_approach")
            self.approach()
            return


def main():
    operator = RobotOperator()
    operator.subscribe()


if __name__ == '__main__':
    main()

