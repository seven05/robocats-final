# -*- coding: utf-8 -*-
import sys
import time

import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
import numpy as np
import rospy
from darknet_ros_msgs.msg import BoundingBoxes
from geometry_msgs.msg import Twist
# from std_msgs.msg import String

"""
1. 카메라로 bottle을 찾은 후 bottle을 인식하고 방향을 수정하면서
2. 카메라에 인식되는 bottle의 높이가 점점 커짐
3. bottle_height가 approach_threshold를 초과하면 bottle 인식 중지
4. 정해진 시간만큼 직진
5. 목표 시간만큼 직진 후 arm을 조작해서 bottle을 집기
6. bottle을 20cm 이상 들어올리기
"""

# parameters
approach_threshold = 500  # approach까지 근접을 확인할 병의 높이 -> direction matching 중지
linear_moving_speed = 0.02  # approach threshold 이후 직진 속도
moving_time = 17  # approach threshold 이후 직진 시간

current_step = 'detect'  # detect|approach|grip

# global vars
box_height = 0
pub = rospy.Publisher('cmd_vel', Twist, queue_size=20)
before_direction = -1

bottle_height_history = []
bottle_height_window_size = 5

sleep_time = 5
twist = None
approach_start_time = 0

robot = None
scene = None
gripper = None
arm = None


moveit_commander.roscpp_initialize(sys.argv)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
gripper = moveit_commander.MoveGroupCommander('gripper')
arm = moveit_commander.MoveGroupCommander('arm')
#gripper = moveit_commander.MoveGroupCommander('gripper')
arm.allow_replanning(True)
arm.set_planning_time(5)


def joint(joint_diff=[0,0,0,0]):
    global arm, sleep_time

    joint_values = arm.get_current_joint_values()
    rospy.sleep(sleep_time)
    print (joint_values)
    joint_temp = [0,0,0,0]
    for i in range(4):
        joint_values[i] = joint_values[i] + joint_diff[i]
    #joint_values[0] = -0.0
    #joint_values[1] = -0.0
    #joint_values[2] = -0.0
    #joint_values[3] = -0.0
    print (type(joint_values))
    print (joint_values)
    arm.go(joints=joint_values, wait=True)
    rospy.sleep(sleep_time)
    arm.stop()
    rospy.sleep(sleep_time)
    arm.clear_pose_targets()
    rospy.sleep(sleep_time)


def gripper_move(coeff):
    global gripper

    #coeff : 1 or -1
    joint_values = gripper.get_current_joint_values()
    rospy.sleep(sleep_time)
    print ("gripper joint values : " + str(joint_values) )
    #gripper.set_joint_value_target([0.01])
    gripper.go([0.01*coeff,0.0], wait=True)
    rospy.sleep(sleep_time)


def reset_grip():
    """로봇팔 초기 상태로 리셋
    """
    global sleep_time

    joint_values = arm.get_current_joint_values()
    chk = False
    for i in range(4):
        if(abs(joint_values[i]) > 0.1):
            chk = True
            joint_values[i] = 0.0
    if(chk):
        arm.go(joint_values,wait=True)
        rospy.sleep(sleep_time)
    gripper_move(1.5)


def match_direction(box):
    """방향을 bottle 방향으로 일치시킴
    """
    global twist, before_direction, box_height, approach_threshold, approach_start_time, current_step, linear_moving_speed, bottle_height_window_size, bottle_height_history
    rospy.loginfo('Call match direction')

    smoothed_box_height = sum(bottle_height_history) / len(bottle_height_history)
    if smoothed_box_height > approach_threshold:
        approach_start_time = time.time()
        current_step = 'approach'
        return

    box_center = (box.xmin + box.xmax) // 2
    box_center = float(box_center)
    move = (640 - box_center) / 640
    rospy.loginfo('correction/height/smooth: ' + str(move) + '\t' + str(box_height) + '\t' + str(smoothed_box_height))

    if abs(move) < 0.1:
        twist.angular.z = 0
        twist.linear.x = linear_moving_speed
        # # DEBUG: 찾으면 그냥 탈출하도록 함
        # current_step = 'approach'
        return
    else:
        twist.linear.x = 0
        twist.angular.z = move * 0.5

    before_direction = -1 if move < 0 else 1

    # pub.publish(twist)


def approach(box):
    """box_height가 vicinity_threashold를 넘을 때 까지 접근
    """
    global twist, box_height, current_step

    rospy.loginfo('Call approach')
    current_time = time.time()
    rospy.loginfo('current_time=' + str(current_time) + '\tapproach_start_time=' + str(approach_start_time) + '\tmoving_time=' + str(moving_time))
    if current_time - approach_start_time >= moving_time:
        twist.linear.x = 0
        current_step = 'grip'
        return

    rospy.loginfo('linear x: ' + str(linear_moving_speed))
    twist.linear.x = linear_moving_speed

    # pub.publish(twist)


def grip_bottle():
    """box_height가 vicinity_threshold를 초과하면 동작함

    무지성으로 병 잡고 20cm 이상 들어야 함
    """
    global current_step

    rospy.loginfo('Call grip')
    rospy.loginfo('grip bottle')
    joint(joint_diff=[0, 0.0, -0.8, 0.0])
    joint(joint_diff=[0, 1.1, -0.0, 0.0])
    gripper_move(-1.0)
    joint(joint_diff=[0, -0.8, 0.0, 0.0])
    #joint(joint_diff=[0, 0.0, -0.8, 0.0])


def callback(yolo_data):
    global twist, before_direction, box_height, pub, bottle_height_history, bottle_height_window_size

    # yolo data에서 bottle 이름을 가진 label만 추출
    bottle_boxes = [each for each in yolo_data.bounding_boxes if each.Class == 'bottle']

    if len(bottle_boxes) == 0:  # bottle을 못찾으면 이전 진행 방향으로 계속 회전시킴
        twist.angular.z = 0.1 * before_direction
        pub.publish(twist)
        return

    # box = bottle_boxes[0]  # legacy code; 항상 첫번째 bottle 추적 -> 경우에 따라 두 병의 순서가 계속 바뀌면 어떡할까?
    # bottle label이 여러개 발견 될 경우를 대비해서 xmin 좌표가 가장 작은 box 하나 추출
    box = sorted(bottle_boxes, key=lambda box: box.xmin)[0]

    box_height = (box.ymax - box.ymin)  # box size 이용해서 근접 계산
    bottle_height_history = [box_height] + bottle_height_history
    bottle_height_history = bottle_height_history[:bottle_height_window_size]

    rospy.loginfo('current_step: ' + str(current_step))

    if current_step == 'detect':
        match_direction(box)
    elif current_step == 'approach':
        approach(box)
    elif current_step == 'grip':
        grip_bottle()

    pub.publish(twist)


def main():
    global twist, robot, scene, gripper, arm

    twist = Twist()
    twist.linear.x = twist.linear.y = twist.linear.z = 0.0
    twist.angular.x = twist.angular.y = 0.0

    rospy.init_node('listener', anonymous=True)

    # moveit_commander.roscpp_initialize(sys.argv)
    # robot = moveit_commander.RobotCommander()
    # scene = moveit_commander.PlanningSceneInterface()
    # gripper = moveit_commander.MoveGroupCommander('gripper')
    # arm = moveit_commander.MoveGroupCommander('arm')
    # #gripper = moveit_commander.MoveGroupCommander('gripper')
    # arm.allow_replanning(True)
    # arm.set_planning_time(5)

    reset_grip()

    rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, callback)

    rospy.spin()


if __name__ == '__main__':
    main()
