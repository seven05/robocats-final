# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String
from darknet_ros_msgs.msg import BoundingBoxes
from geometry_msgs.msg import Twist

pub = rospy.Publisher('cmd_vel', Twist, queue_size=20)
before_direction = -1

"""
1. 카메라로 bottle을 찾은 후 bottle을 인식하고 방향을 수정하면서
2. 카메라에 인식되는 bottle의 높이가 점점 커짐
3. bottle_height가 approach_threshold를 초과하면 bottle 인식 중지
4. bottle_height가 vicinity_threshold를 초과할 때 까지 직진
5. vicinity_threshold를 초과하면 arm을 조작해서 bottle을 집기
"""

approach_threshold = 150  # approach까지 근접을 확인할 병의 높이 -> direction matching 중지
vicinity_threshold = 200  # direction matching 중지 이후 로봇이 근접해서 카메라에서 인식되는 병의 높이 -> 팔로 잡으면 됨
box_height = 0
twist = None


def match_direction(box):
    """방향을 bottle 방향으로 일치시킴
    """
    global twist, before_direction, box_height, approach_threshold

    if box_height > approach_threshold:
        return

    box_center = (box.xmin + box.xmax) // 2
    box_center = float(box_center)
    move = (640 - box_center) / 640
    rospy.loginfo('correction: ' + str(move))

    if abs(move) < 0.15:
        twist.angular.z = 0
    else:
        twist.angular.z = move * 0.5

    before_direction = -1 if move < 0 else 1

    pub.publish(twist)


def approach(box):
    """box_height가 vicinity_threashold를 넘을 때 까지 접근
    """
    global twist, box_height, vicinity_threshold

    if box_height > vicinity_threshold:
        return

    linear_speed = (vicinity_threshold - box_height) / 100 * 0.05
    linear_speed = min(linear_speed, 0.1)  # 최대 속도: 0.1
    linear_speed = max(linear_speed, 0.01)  # 최대 속도: 0.01
    twist.linear = linear_speed

    pub.publish(twist)


def grip_bottle():
    """box_height가 vicinity_threshold를 초과하면 동작함

    무지성으로 병 잡고 20cm 이상 들어야 함
    """
    pass  # TODO: FIXME:



def callback(yolo_data):
    global twist, before_direction, box_height

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

    match_direction(box)
    approach(box)
    grip_bottle()


def main():
    twist = Twist()
    twist.linear.x = twist.linear.y = twist.linear.z = 0.0
    twist.angular.x = twist.angular.y = 0.0

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, callback)

    rospy.spin()


if __name__ == '__main__':
    main()
