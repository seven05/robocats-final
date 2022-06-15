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
from sensor_msgs.msg import LaserScan

from scipy import ndimage

""" moveit commander settings """
moveit_commander.roscpp_initialize(sys.argv)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
gripper = moveit_commander.MoveGroupCommander('gripper')
arm = moveit_commander.MoveGroupCommander('arm')
# gripper = moveit_commander.MoveGroupCommander('gripper')
arm.allow_replanning(True)
arm.set_planning_time(5)
sleep_time = 3


class RobotOperator:
    def __init__(self):
        """ Motor settings """
        twist = Twist()
        twist.linear.x = twist.linear.y = twist.linear.z = 0.0
        twist.angular.x = twist.angular.y = 0.0
        self.twist = twist

        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=20)

        self.chk = False
        self.before_direction = 1
        self.yolo_threshold = 0.70
        self.color_threshold = 0.3
        self.yolo_data = None
        self.lidar_data = None
        self.color_data = None
        self.current_state = 'decide'
        self.robot_state = ['decide', 'act_find', 'act_approach', 'act_grip', 'halt']
        self.need_default_direction = False
        self.now_move_default_direction = False

        self.find_criterion = 'yolo'

        self.bridge = CvBridge()
        self.image_fetch = np.zeros((1280, 720, 3))

        gripper.go([0.015, 0.0], wait=True)
        rospy.sleep(sleep_time)

    def joint(self, joint1, joint2, joint3, joint4):
        global arm, sleep_time
        joint_diff = [joint1, joint2, joint3, joint4]
        joint_values = arm.get_current_joint_values()
        rospy.sleep(sleep_time)
        print(joint_values)
        for i in range(4):
            joint_values[i] = joint_values[i] + joint_diff[i]
        print(type(joint_values))
        print(joint_values)
        arm.go(joints=joint_values, wait=True)
        rospy.sleep(sleep_time)
        arm.stop()
        rospy.sleep(sleep_time)
        arm.clear_pose_targets()
        rospy.sleep(sleep_time)

    def gripper_move(self, coeff):
        global gripper, sleep_time

        # coeff : 1 or -1
        joint_values = gripper.get_current_joint_values()
        rospy.sleep(sleep_time)
        print('gripper joint values : ' + str(joint_values))
        # gripper.set_joint_value_target([0.01])
        gripper.go([0.01 * coeff, 0.0], wait=True)
        rospy.sleep(sleep_time)

    def reset_grip(self):
        """로봇팔 초기 상태로 리셋
        """
        global sleep_time

        joint_values = arm.get_current_joint_values()
        chk = False
        joint_sign_map = [1, -1, 1, -1]
        for joint_idx, joint_sign in enumerate(joint_sign_map):
            if abs(joint_values[joint_idx]) < 0.6:
                chk = True
                joint_values[joint_idx] = 1.2 * joint_sign
            else:
                joint_values[joint_idx] = 0
        if chk:
            arm.go(joint_values, wait=True)
            rospy.sleep(sleep_time)
            time.sleep(5)  # 팔 꺾기까지 딜레이 줘서 그 전에 값 읽는 것 방지
        arm.go([0, 0, 0, 0], wait=True)
        rospy.sleep(sleep_time)
        time.sleep(5)
        if gripper.get_current_joint_values() < 1.2:
            self.gripper_move(1.5)
            rospy.sleep(sleep_time)

    def move_default_direction_callback(self, data):
        """self.need_default_direction가 True일 경우 경기장 가장 긴 대각선 방향을 바라보도록 함
        """
        if not self.need_default_direction:  # 필요하지 않다면 실행하지 않음
            return
        ranges = data.ranges
        direction_distance_map = sorted(list(zip(list(range(len(ranges))), ranges)), key=lambda x: x[1], reverse=True)
        direction_distance_map = [each for each in direction_distance_map if each[1] < 3.6]
        long_direction, long_distance = direction_distance_map[0]
        if 5 <= long_direction < 180:
            # move left
            print('[move_default_direction_callback] move left %5d %f'%(long_direction, long_distance))
            self.twist.angular.z = 0.2
        elif 180 <= long_direction < 355:
            # move right
            print('[move_default_direction_callback] move right %5d %f'%(long_direction, long_distance))
            self.twist.angular.z = -0.2
        elif long_distance < 5 or long_direction >= 355:
            print('[move_default_direction_callback] stop %5d %f'%(long_direction, long_distance))
            self.twist.angular.z = 0
            print('[move_default_direction_callback] direction process done')
            self.robot_halt()
            self.need_default_direction = False  # 완료되었으므로 더이상 실행하지 않음
            self.now_move_default_direction = False
        self.pub.publish(self.twist)
        time.sleep(0.01)

    def subscribe(self):
        rospy.init_node('RobotOperator', anonymous=True)
        rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.yolo_callback)
        rospy.Subscriber('/scan_heading', Float32, self.lidar_callback)
        rospy.Subscriber('/video_source/raw_2', Image, self.color_callback)
        rospy.Subscriber('/scan', LaserScan, self.move_default_direction_callback)
        self.reset_grip()
        self.current_state = 'decide'

        while self.current_state != 'halt':
            self.run_proc()

        self.robot_halt()
        # TODO : Add lidar callback, color filter callback
        rospy.spin()

    def yolo_callback(self, data):
        """Return x coordinates of center of box which is bottle on the **far right**
        """
        bottle_boxes = [each for each in data.bounding_boxes if each.Class == 'bottle']

        if len(bottle_boxes) == 0:
            self.yolo_data = None
            return

        # 너무 작은 bottle은 인식하지 않도록 함
        bottle_center_xs = [(each.xmin + each.xmax) // 2 for each in bottle_boxes if (each.xmax - each.xmin) > 68 and (each.ymax - each.ymin) > 204]
        if len(bottle_center_xs) > 0:  # 필터링되어 list size = 0이 되면 못찾은걸로 인식함
            # -1: far right / 0: far left
            self.yolo_data = sorted(bottle_center_xs)[-1]
        else:
            self.yolo_data = None
            return

    def lidar_callback(self, data):
        self.lidar_data = data.data

    def color_callback(self, data):
        """color filter image, return center of mass of cluster (com_x,com_y)
        """
        im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
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

    def set_next_state(self, next_act=None):
        # assert current state before change
        if self.current_state == 'decide':
            if next_act == 'act_find' or next_act == 'act_approach' or next_act == 'act_grip':
                self.current_state = next_act
            else:
                print('Error in set_next_state')

        elif self.current_state == 'act_find' or self.current_state == 'act_approach':
            self.current_state = 'decide'
        elif self.current_state == 'act_grip':
            self.current_state = 'halt'

    def rotate_previous_direction(self):
        self.twist.angular.z = 0.1 * self.before_direction
        self.pub.publish(self.twist)
        return

    def move_default_direction(self):
        """경기장에서 가장 긴 (but, 3.6m 이내인) 방향을 바라보도록 함

        flag를 True로 변경해서 move_default_direction_callback()이 실행되도록 함
        """
        self.need_default_direction = True
        self.now_move_default_direction = True
        # move_default_direction_callback()이 종료되면 self.now_move_default_direction flag가 False로 바뀜
        # 코드가 동기적으로 돌아야하므로 종료될 때 까지 loop 돌림
        while self.now_move_default_direction:
            time.sleep(0.01)

    def deg2rad(self, deg):
        return deg * np.pi / 180

    def turn_deg(self, direction, degree):
        """direction 방향(left: +, right: -)으로 degree만큼 이동
        """
        print('Turn %s %f deg' % (direction, degree))
        direction_sign = 1 if direction == 'left' else -1
        turn_radian = self.deg2rad(degree)
        start_searching_time = time.time()

        self.twist.angular.z = 0.2 * direction_sign
        target_turn_time = turn_radian / abs(self.twist.angular.z)
        self.pub.publish(self.twist)

        find_yolo = False

        while time.time() - start_searching_time < target_turn_time:
            if self.yolo_data is not None:  # yolo는 callback으로 찾으므로 데이터 조회해보면 됨
                print('[Turn right %f deg] Find bottle using YOLO' % (degree,))
                find_yolo = True
                break
            time.sleep(0.005)
        else:
            print('[Turn right %f deg] Cannot found bottle using YOLO' % (degree,))
        self.robot_halt()
        return find_yolo

    def forward_meter(self, meter):
        """앞으로 meter만큼 이동
        """
        print('Forward %fm' % (meter,))
        start_searching_time = time.time()

        self.twist.linear.x = 0.05
        target_turn_time = meter / abs(self.twist.linear.x)
        self.pub.publish(self.twist)

        find_yolo = False

        while time.time() - start_searching_time < target_turn_time:
            if self.yolo_data is not None:  # yolo는 callback으로 찾으므로 데이터 조회해보면 됨
                print('[Forward %fm] Find bottle using YOLO' % (meter,))
                find_yolo = True
                break
            time.sleep(0.005)
        else:
            print('[Forward %fm] Cannot found bottle using YOLO' % (meter,))
        self.robot_halt()
        return find_yolo

    def forward_heading_lidar(self, meter):  # TODO: FIXME: heading이 아니라 lidar 최대값 (3.6m 이하) 기준으로 접근하는 방식으로 변경 필요
        """heading 방향 lidar가 meter가 될 때 까지 전진
        """
        print('Forward headling lidar %fm' % (meter,))

        self.twist.linear.x = 0.05
        self.pub.publish(self.twist)

        find_yolo = False

        while self.lidar_data > meter:
            if self.yolo_data is not None:  # yolo는 callback으로 찾으므로 데이터 조회해보면 됨
                print('[Forward %fm] Find bottle using YOLO' % (meter,))
                find_yolo = True
                break
            time.sleep(0.005)
        else:
            print('[Forward headling lidar %fm] Cannot found bottle using YOLO' % (meter,))
        self.robot_halt()
        return find_yolo

    def found_target_routine(self):
        """find_target()에서 bottle을 찾았을 때 실행할 루틴
        """
        self.robot_halt()
        self.set_next_state('decide')

    def find_target(self):
        # current state : act_find
        if self.current_state != 'act_find':
            print('ERROR : state is wrong, not act find, ', self.current_state)
        # Find step #1
        self.move_default_direction()
        # self.rotate_previous_direction()
        # while self.yolo_data is None:
        #     pass
        if self.turn_deg('left', 120) or self.turn_deg('right', 240):  # 앞 루틴 True이면 뒤 루틴 실행 안함
            self.found_target_routine()
            return

        # Find step #2
        print('[find_target] Reset position because not found')
        self.move_default_direction()
        # self.forward_meter(0.5)
        self.forward_heading_lidar(2.3)  # 시작 위치에 따라 0.5m 이동 결과가 달라지므로 최장 길이 기준으로 역산
        print('[find_target] Find bottle routine in step 2')
        if self.turn_deg('left', 120) or self.turn_deg('right', 240):  # 앞 루틴 True이면 뒤 루틴 실행 안함
            self.found_target_routine()
            return

        # Find step #3
        if self.turn_deg('left', 85):  # 75 deg, but bias calibrated degree
            self.found_target_routine()
            return
        self.forward_meter(0.9)
        print('[find_target] Find bottle routine in step 3')
        if self.turn_deg('right', 90) or self.turn_deg('left', 190):  # 앞 루틴 True이면 뒤 루틴 실행 안함, calibrated 180 deg
            self.found_target_routine()
            return

        # Find step #4
        self.forward_meter(0.9)
        print('[find_target] Find bottle routine in step 4')
        if self.turn_deg('right', 90) or self.turn_deg('left', 190):  # 앞 루틴 True이면 뒤 루틴 실행 안함, calibrated 180 deg
            self.found_target_routine()
            return

        # Find step #5
        self.forward_meter(0.9)
        print('[find_target] Find bottle routine in step 5')
        if self.turn_deg('right', 90) or self.turn_deg('left', 190):  # 앞 루틴 True이면 뒤 루틴 실행 안함, calibrated 180 deg
            self.found_target_routine()
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
            self.twist.angular.z = move / abs(move) * 0.05

        self.before_direction = -1 if move < 0 else 1
        self.pub.publish(self.twist)
        time.sleep(0.01)
        # print("match_direction published")

    def go_front(self):
        self.twist.linear.x = 0.02
        self.pub.publish(self.twist)
        time.sleep(0.01)

    def approach(self):
        while self.lidar_data >= self.color_threshold:
            self.match_direction()
            self.go_front()
        self.robot_halt()
        self.set_next_state('decide')
        pass

    def gripper_execute(self):
        print('gripper_execute called')
        self.joint(0, 0.0, -0.8, 0.0)
        self.joint(0, 1.1, -0.0, 0.0)
        self.gripper_move(-1.0)
        self.joint(0, -0.8, 0.0, 0.0)
        self.set_next_state('halt')
        return

    def robot_halt(self):
        print('robot_halt')
        self.twist.linear.x = 0.0
        self.twist.linear.y = 0.0
        self.twist.linear.z = 0.0
        self.twist.angular.x = 0.0
        self.twist.angular.y = 0.0
        self.twist.angular.z = 0.0
        self.pub.publish(self.twist)
        time.sleep(0.01)
        return

    def grip_condition_check(self):
        return self.lidar_data <= self.color_threshold

    def sensor_init(self):
        self.yolo_data = None
        self.lidar_data = None
        self.color_data = None

    def run_proc(self):

        if self.current_state != 'decide':
            return

        if self.yolo_data is None and self.chk == False:
            # go to next state
            self.set_next_state('act_find')
            self.find_target()
            return

        if self.current_state != 'decide':
            return

        self.chk = True

        if self.grip_condition_check():
            self.set_next_state('act_grip')
            self.gripper_execute()
            return
        else:
            self.set_next_state('act_approach')
            self.approach()
            return


def main():
    operator = RobotOperator()
    operator.subscribe()


if __name__ == '__main__':
    main()
