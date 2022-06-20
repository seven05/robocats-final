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


"""add ros odometry"""
#https://www.theconstructsim.com/ros-qa-know-pose-robot-python/
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

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

"""odometry global variables"""
ang = None
sub = None

odom_pose = None
init_odom = None
eps = 0.01

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
        self.yolo_width = None
        self.yolo_height = None
        self.lidar_data = None
        self.color_data = None
        self.current_state = 'decide'
        self.robot_state = ['decide', 'act_find', 'act_approach', 'act_grip', 'act_return', 'act_drop_bottle', 'halt']
        self.need_default_direction = False
        self.now_move_default_direction = False
        self.approach_speed = 0.1  # 접근하면서 변경되는 속도 -> yolo: 접근하면서 감소, color: 0.02 고정
        self.angular_calibration_value = 0.04  # 로봇이 왼쪽으로 틀어지는 현상 보정하기 위해 더해주는 값
        self.approach_fix_speed_threshold = 0.5
        self.yolo_height_approach_threshold = 430.0  # yolo 높이가 이 기준 이상이면 느리게 움직임
        
        """odometry varaibles"""
        self.init_pose = None
        self.deg45_pose = None
        self.drop_count = 0

        self.find_criterion = 'yolo'

        self.bridge = CvBridge()
        self.image_fetch = np.zeros((1280, 720, 3))

        gripper.go([0.015, 0.0], wait=True)
        rospy.sleep(sleep_time)
        
    """odometry methods start"""
    def get_odom_yaw(self, odom):
        orientation_q = odom.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
        if(yaw < 0):
            yaw += np.pi * 2
        
        return yaw
    
    def print_odom_pose(self):
        global odom_pose
        print("Odometry pose: ",odom_pose)
        print("odom x: ", odom_pose.position.x)
        print("odom y: ", odom_pose.position.y)
        print("odom yaw : ", self.get_odom_yaw(odom_pose))
        
    def odom_callback(self, odom_msg):
        global odom_pose
        odom_pose = odom_msg.pose.pose
        
    def translate(self, x, y):
        global sub,ang
        mat = np.matrix([
            [np.cos(ang), -np.sin(ang)],
            [np.sin(ang), np.cos(ang)]
        ])
        
        t = np.dot((x - sub[0], y - sub[1]), mat)[0]
        
        print("trasnlated coordinate", t)
        return t[0, 0], t[0, 1]
    
    def manual_move(self, linear=0, angular=0, stop=False):

        if(stop):
            self.twist.linear.x = 0
        else:
            self.twist.linear.x = linear
        self.twist.linear.y = 0.0
        self.twist.linear.z = 0.0
        self.twist.angular.x = 0.0
        self.twist.angular.y = 0.0
        if(stop):
            self.twist.angular.z = 0
        else:
            self.twist.angular.z = angular
        #if(linear is not None):
            #twist.angular.z += (linear*0.09)

        self.pub.publish(self.twist)
        time.sleep(0.01)
        
    def go_one_meter_front(self, speed=0.05):
        
        front_speed = speed
        self.manual_move(linear=front_speed)
        
        moving_time = 0.5 / abs(front_speed)
        start_time = time.time()
        rospy.sleep(sleep_time)
        
        while(True):
            cur_time = time.time() 
            if(cur_time - start_time > moving_time):
                break
            time.sleep(0.001)
        
        self.manual_move(stop=True)
        
    def return_origin2(self):
        print("return origin2 start")
        global odom_pose, eps
        
        start_odom = odom_pose
        x_1 = start_odom.position.x
        y_1 = start_odom.position.y
        t_x,t_y = self.translate(x_1, y_1)
        t_x += 0.1
        alpha = np.arctan2(t_y, t_x)
        
        """turn 180 degrees back"""
        self.manual_move(angular=0.1)
        rot_time = time.time()
        while(True):
            cur_time = time.time()
            if(cur_time - rot_time > (np.pi / 0.1) + eps):
                break
            
        cnt = 0
        while(True):
            cnt += 1
            now_coor = self.translate(odom_pose.position.x, odom_pose.position.y)
            if(abs(now_coor[0]) < (0.4) and abs(now_coor[1]) < 0.2 ):
                self.manual_move(stop=True)
                break
            if(cnt > 300):
                break
                
            self.manual_move(linear=0.03)
            start_time = time.time()
            while(True):
                now = time.time()
                if(now - start_time > 0.5):
                    break
                time.sleep(0.01)
            self.manual_move(stop=True)
            time.sleep(0.01)
            
            odom_now = odom_pose
            x_2 = odom_now.position.x
            y_2 = odom_now.position.y
            t_x2,t_y2 = self.translate(x_2, y_2)
            alpha2 = np.arctan2(t_y2, t_x2)
            
            print("alpha2 : ",alpha2, "alpha ", alpha)
            print(abs(alpha2 - alpha)*4, abs(alpha2 - alpha)*2)
            
            if(alpha2 > alpha + eps):
                self.manual_move(angular=0.1)
                rot_time = time.time()
                while(True):
                    now = time.time()
                    if(now - rot_time > abs(alpha2 - alpha)*4):
                        break
                    time.sleep(0.01)
                self.manual_move(stop=True)
                time.sleep(0.01)
            elif(alpha2 < alpha - eps):
                self.manual_move(angular= -0.1)
                rot_time = time.time()
                while(True):
                    now = time.time()
                    if(now - rot_time > abs(alpha2 - alpha)*2):
                        break
                    time.sleep(0.01)
                self.manual_move(stop=True)
                time.sleep(0.01)
            else:
                self.manual_move(linear=0.05)
                rot_time = time.time()
                while(True):
                    now = time.time()
                    if(now - rot_time > 0.5):
                        break
                    time.sleep(0.01)
                self.manual_move(stop=True)
                time.sleep(0.01)

    def drop_bottle(self):
        print("drop bottle started")
        global odom_pose
        init_odom_yaw = self.get_odom_yaw(self.deg45_pose) + np.pi
        if(init_odom_yaw > 2*np.pi + eps):
            init_odom_yaw -= 2*np.pi
        self.manual_move(angular=0.1)
        while(True):
            cur_odom_yaw = self.get_odom_yaw(odom_pose)
            if(abs(init_odom_yaw - cur_odom_yaw) < eps*10):
                break
            time.sleep(0.001)
        self.manual_move(stop=True)
        
        """release gripper"""
        #self.joint(0, 1.1, -0.0, 0.0)
        self.gripper_move(1.5)
        
        self.drop_count += 1
        
        """TODO : actual drop bottle to move arm"""

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
        joint_sign_map = [1, -1, 1, -1]
        need_far_threshold = [0.05, 0.1, 0.1, 0.1]  # 최소값 (절대값)

        for try_count in range(3):
            print('[reset_grip] reset grip try: %d' % (try_count + 1))
            joint_values = arm.get_current_joint_values()
            print('[reset_grip] Read current joint values: %s' % (', '.join([str(each) for each in joint_values])))

            all_smaller_than_threshold = True
            for joint_idx, joint_value in enumerate(joint_values):
                if abs(joint_value) >= need_far_threshold[joint_idx]:
                    all_smaller_than_threshold = False
            if all_smaller_than_threshold:
                print('[reset_grip] All joint angle smaller than threshold so, break reset grip loop')
                break

            need_0_joint = []  # 0으로 보내기 필요한 joint 확인
            need_far_joint = []  # 멀리 보내기 필요한 joint 확인
            for joint_idx, joint_value in enumerate(joint_values):
                if abs(joint_value) > 0.61:
                    need_0_joint.append(joint_idx)
                elif abs(joint_value) > need_far_threshold[joint_idx]:
                    need_far_joint.append(joint_idx)

            print('[reset_grip] Move joint angle to 0: %s'%(', '.join([str(each) for each in need_0_joint])))
            print('[reset_grip] Move joint angle to far: %s'%(', '.join([str(each) for each in need_far_joint])))
            target_joint_values = []
            for joint_idx, joint_value in enumerate(joint_values):
                if joint_idx in need_0_joint:
                    target_joint_values.append(-joint_value)
                elif joint_idx in need_far_joint:
                    if joint_value * joint_sign_map[joint_idx] > 0:  # 부호 같음
                        target_joint_values.append(joint_sign_map[joint_idx] * 0.7)
                    elif joint_value * joint_sign_map[joint_idx] < 0:  # 부호 다름
                        target_joint_values.append(joint_sign_map[joint_idx] * 0.7 - joint_value)
                    else:  # 그럴 일은 없겠지만 만약 0이라면
                        target_joint_values.append(0)
                else:
                    target_joint_values.append(0)

            print('[reset_grip] Execute joint move')
            self.joint(*target_joint_values)
            time.sleep(5)
            joint_values = arm.get_current_joint_values()
            print('[reset_grip] After move read current joint values: %s' % (', '.join([str(each) for each in joint_values])))

        gripper_value = gripper.get_current_joint_values()[0]
        print('[reset grip] Current gripper value: %f' % (gripper_value,))
        if gripper_value < 1.2:
            print('  >>> [reset grip] gripper move to: 1.5')
            self.gripper_move(1.5)
            rospy.sleep(sleep_time)

        joint_values = arm.get_current_joint_values()
        return joint_values[0] < 0.05

    def reset_direction_grip(self):
        """제일 첫번째 gripper motor만 리셋함

        reset_grip() 결과가 False일 떄 실행
        """
        joint_values = arm.get_current_joint_values()
        print('[reset_direction_grip] Before calibration joint value is ' + ', '.join([str(each) for each in joint_values]))
        self.joint(joint_values[0] + 0.7, 0, 0, 0)
        print('[reset_direction_grip] Add 0.7 rad to 1st motor')
        joint_values = arm.get_current_joint_values()
        self.joint(-joint_values[0], 0, 0, 0)
        joint_values = arm.get_current_joint_values()
        print('[reset_direction_grip] After calibration joint value is ' + ', '.join([str(each) for each in joint_values]))

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
        global sub, ang
        rospy.init_node('RobotOperator', anonymous=True)
        rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.yolo_callback)
        rospy.Subscriber('/scan_heading', Float32, self.lidar_callback)
        rospy.Subscriber('/video_source/raw_2', Image, self.color_callback)
        rospy.Subscriber('/scan', LaserScan, self.move_default_direction_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        """TODO : ADD 45 deg rotate"""
        
        if not self.reset_grip():
            print('[subscribe] 1st motor has very small error, reset again')
            self.reset_direction_grip()
        self.current_state = 'decide'

        self.deg45_pose = odom_pose
        """turn 45 degrees by time"""
        print("turn 45 degrees")
        self.manual_move(angular= -0.1)
        rot_time = time.time()
        while(True):
            cur_time = time.time()
            if(abs(cur_time - rot_time) > (0.785398 / 0.1) + eps):
                self.manual_move(stop=True)
                break
            time.sleep(0.001)
            
        """initialize coordinates"""
        print("start coordinate initialization")
        self.init_pose = odom_pose
        x_0 = self.init_pose.position.x
        y_0 = self.init_pose.position.y
        
        self.go_one_meter_front()
        rospy.sleep(sleep_time)
        
        cur_odom = odom_pose
        x_1 = cur_odom.position.x
        y_1 = cur_odom.position.y
        
        print("dist : ", math.sqrt(abs(x_0-x_1)**2 + abs(y_0-y_1)**2))
        
        sub = (x_0 , y_0)
        ang = np.arctan2(y_1 - y_0, x_1 - x_0)
        print("sub ",sub, "ang", ang)
        self.translate(x_1, y_1)
        
        self.manual_move(linear = -0.05)
        start_time = time.time()
        while(True):
            cur_time = time.time()
            if(abs(cur_time - start_time) > 10.0):
                self.manual_move(stop=True)
                break
            time.sleep(0.001)
        
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
        # bottle_center_xs = [(each.xmin + each.xmax) // 2 for each in bottle_boxes if (each.xmax - each.xmin) > 68 and (each.ymax - each.ymin) > 204]
        bottles = [each for each in bottle_boxes if (each.xmax - each.xmin) > 68 and (each.ymax - each.ymin) > 204]
        # bottle_center_xs = [(each.xmin + each.xmax) // 2 for each in bottles]
        if len(bottles) > 0:  # 필터링되어 list size = 0이 되면 못찾은걸로 인식함
            # -1: far right / 0: far left
            target_bottle = sorted(bottles, key=lambda bottle: (bottle.xmin + bottle.xmax) // 2)[-1]
            # self.yolo_data = sorted(bottle_center_xs)[-1]
            self.yolo_data = (target_bottle.xmin + target_bottle.xmax) // 2
            self.yolo_width = target_bottle.xmax - target_bottle.xmin
            self.yolo_height = target_bottle.ymax - target_bottle.ymin
        else:
            self.yolo_data = None
            self.yolo_width = None
            self.yolo_height = None
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

        elif self.current_state == 'act_find' or self.current_state == 'act_approach' or self.current_state=='act_drop_bottle':
            self.current_state = 'decide'
        elif self.current_state == 'act_grip':
            self.current_state = 'act_return'
        elif self.current_state == 'act_return':
            self.current_state = 'act_drop_bottle'

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

    def turn_deg(self, direction, degree, angular_speed=0.2):
        """direction 방향(left: +, right: -)으로 degree만큼 이동
        """
        print('Turn %s %f deg' % (direction, degree))
        direction_sign = 1 if direction == 'left' else -1
        turn_radian = self.deg2rad(degree)
        start_searching_time = time.time()

        self.twist.angular.z = angular_speed * direction_sign
        self.twist.angular.z += self.angular_calibration_value
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

    def forward_meter(self, meter, speed=0.05):
        """앞으로 meter만큼 이동
        """
        print('Forward %fm' % (meter,))
        start_searching_time = time.time()

        self.twist.linear.x = speed
        target_move_time = meter / abs(speed)
        self.twist.angular.z += self.angular_calibration_value
        self.pub.publish(self.twist)

        find_yolo = False

        while time.time() - start_searching_time < target_move_time:
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
        # TODO: FIXME: 빠르게 이동하면서 찾기
        # Plan A: 처음에 좌우로 10도씩 둘러봐서 못찾으면 80cm 이동해서 둘러보고 못찾으면 또 80cm 이동해서 둘러보고
        # Plan B: 처음부터 그냥 직진하면서 찾기 -> 속도는 빠른데 시야에서 살짝 벗어나면?

        # current state : act_find
        if self.current_state != 'act_find':
            print('ERROR : state is wrong, not act find, ', self.current_state)

        # 주위를 둘러볼 각도 (한쪽 방향으로)
        STEP_1_DEG = 60
        STEP_3_DEG = 60
        STEP_5_DEG = 60
        STEP_7_DEG = 60
        STEP_9_DEG = 60
        TURN_ANGULAR_SPEED = 0.2

        # Find step #1
        # 대각 방향으로 정렬
        self.move_default_direction()

        command_set = (
            # Step 1
            self.turn_deg('left', STEP_1_DEG, TURN_ANGULAR_SPEED) or \
            self.turn_deg('right', STEP_1_DEG * 2, TURN_ANGULAR_SPEED) or \
            self.turn_deg('left', STEP_1_DEG, TURN_ANGULAR_SPEED) or \
            # Step 2
            self.forward_meter(1.0, 0.1) or \
            # Step 3
            self.turn_deg('left', STEP_3_DEG, TURN_ANGULAR_SPEED) or \
            self.turn_deg('right', STEP_3_DEG * 2, TURN_ANGULAR_SPEED) or \
            self.turn_deg('left', STEP_3_DEG - 45, TURN_ANGULAR_SPEED) or \
            # Step 4
            self.forward_meter(0.8, 0.1) or \
            # Step 5
            self.turn_deg('right', STEP_5_DEG, TURN_ANGULAR_SPEED) or \
            self.turn_deg('left', STEP_5_DEG + 90, TURN_ANGULAR_SPEED) or \
            # Step 6
            self.forward_meter(0.8, 0.1) or \
            # Step 7
            self.turn_deg('right', STEP_7_DEG, TURN_ANGULAR_SPEED) or \
            self.turn_deg('left', STEP_7_DEG + 90, TURN_ANGULAR_SPEED) or \
            # Step 8
            self.forward_meter(0.8, 0.1) or \
            # Step 9
            self.turn_deg('right', STEP_9_DEG, TURN_ANGULAR_SPEED) or \
            self.turn_deg('left', STEP_9_DEG + 90, TURN_ANGULAR_SPEED)
        )

        if command_set:
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
            angular_speed_factor = (self.approach_speed * 50) ** 0.5  # sqrt(현재 직선 속도(0.02 ~ 0.1))에 비례해서 각속도 변화
            self.twist.angular.z = move / abs(move) * 0.05 * angular_speed_factor

        self.before_direction = -1 if move < 0 else 1
        self.pub.publish(self.twist)
        time.sleep(0.01)
        # print("match_direction published")

    def go_front(self):
        # approach_fix_speed_threshold 거리보다 길 때 남은 거리에 따라 속도 변화
        if self.yolo_height > self.yolo_height_approach_threshold and self.lidar_data >= self.approach_fix_speed_threshold:
            # approach_fix_speed_threshold보다 가까운 거리에서 lidar 값이 튈 경우 빨라질 수 있음
            # 따라서 0.05보다 낮은 속도로 접근중이었다면 업데이트 하지 않음
            if self.approach_speed > 0.05:
                self.approach_speed = 0.05
        elif self.lidar_data >= self.approach_fix_speed_threshold:
            MAX_SPEED = 0.1 if self.lidar_data < self.yolo_threshold else 0.05  # color filter 부터는 0.05로 최대 속도 제한
            self.is_yolo_height_approach_print = False
            new_approach_speed = max(min(0.08 * self.lidar_data - 0.02, MAX_SPEED), 0.02)  # speed range: 0.02 ~ 0.1
            if new_approach_speed < self.approach_speed:
                print('[go_front] Update approach speed: distance=%f\tbefore_spee=%f\tspeed=%f' % (self.lidar_data, self.approach_speed, new_approach_speed))
                self.approach_speed = new_approach_speed  # 작아지는 방향으로만 update
        else:
            self.approach_speed = 0.02

        # self.twist.linear.x = 0.02
        self.twist.linear.x = self.approach_speed  #
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
        self.manual_move(linear = -0.05)
        start_time = time.time()
        while(True):
            cur_time = time.time()
            if(abs(cur_time - start_time) > 4.0):
                self.manual_move(stop=True)
                break
            time.sleep(0.001)
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
        
        if self.current_state == 'act_return':
            self.return_origin2()
            self.set_next_state('act_drop_bottle')
            return
        
        if self.current_state == 'act_drop_bottle':
            self.drop_bottle()
            if(self.drop_count < 2):
                self.reset_grip()
                init_odom_yaw = self.get_odom_yaw(self.deg45_pose)
                self.manual_move(angular=0.1)
                while(True):
                    cur_odom_yaw = self.get_odom_yaw(odom_pose)
                    if(abs(init_odom_yaw - cur_odom_yaw) < eps*10):
                        break
                    time.sleep(0.001)
                self.manual_move(stop=True)
                self.find_criterion = 'yolo'
                self.need_default_direction = False
                self.now_move_default_direction = False
                self.approach_speed = 0.1  # 접근하면서 변경되는 속도 -> yolo: 접근하면서 감소, color: 0.02 고정
                
                self.sensor_init()
                self.chk = False
                
                self.set_next_state('decide')
                return
            else:
                self.set_next_state('halt')
                return

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
    try:
        operator = RobotOperator()
        operator.subscribe()
    except KeyboardInterrupt:
        operator.robot_halt()
        sys.exit(0)


if __name__ == '__main__':
    main()
