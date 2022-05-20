# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String
from darknet_ros_msgs.msg import BoundingBoxes
from geometry_msgs.msg import Twist

pub = rospy.Publisher('cmd_vel', Twist, queue_size=20)
before_direction = -1

def callback(data):
    global before_direction
    twist = Twist()
    twist.linear.x = twist.linear.y = twist.linear.z = 0.0
    twist.angular.x = twist.angular.y = 0.0

    bottle_boxes = [each for each in data.bounding_boxes if each.Class == 'bottle']
    if len(bottle_boxes) == 0:
        twist.angular.z = 0.1 * before_direction
        pub.publish(twist)
        return
    box = bottle_boxes[0]
    box_height = (box.ymax - box.ymin)
    box_center = (box.xmin + box.xmax) // 2
    box_center = float(box_center)
    move = (640 - box_center) / 640
    rospy.loginfo('move: ' + str(move))
    rospy.loginfo('height: ' + str(box_height))

    if abs(move) < 0.15:
        twist.angular.z = 0
    else:
        twist.angular.z = move * 0.5

    before_direction = -1 if move < 0 else 1
    pub.publish(twist)

rospy.init_node('listener', anonymous=True)
rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, callback)

rospy.spin()

