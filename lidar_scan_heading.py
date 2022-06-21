#!/usr/bin/env python
from __future__ import print_function
import sys

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

before_distance = None
pub = rospy.Publisher('/scan_heading', Float32, queue_size=10)


def laser_callback(data):
    global pub, before_distance

    ranges = data.ranges
    intensities = data.intensities

    range_count = 2

    heading_ranges = ranges[-range_count:] + ranges[:range_count + 1]
    heading_intensities = intensities[-range_count:] + intensities[:range_count + 1]

    target_heading_ranges = []
    for i in range(len(heading_ranges)):
        if heading_intensities[i] == 0:
            continue
        target_heading_ranges.append(heading_ranges[i])

    heading_distance = sorted(target_heading_ranges)[0]
    if heading_distance < 0.12:
        return

    if before_distance is not None and abs(before_distance - heading_distance) > 0.5:
        before_distance = heading_distance
        return

    distance_object = Float32()
    distance_object.data = heading_distance
    pub.publish(distance_object)
    rospy.loginfo('heading distance: ' + str(heading_distance))
    before_distance = heading_distance


rospy.init_node('lidar_scan_heading', anonymous=True)
rospy.Subscriber('/scan', LaserScan, laser_callback)
rospy.spin()