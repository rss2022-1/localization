#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseWithCovarianceStamped
import math
def simple_publisher():
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rospy.init_node('simple_publisher', anonymous=True)
    initial = PoseWithCovarianceStamped()
    initial.header.stamp = rospy.Time.now()
    initial.header.frame_id = "/map"
    initial.pose.pose.position.x = 0
    initial.pose.pose.position.y = 0
    initial.pose.pose.position.z = 0
    initial.pose.pose.orientation.x = 0
    initial.pose.pose.orientation.y = 0
    initial.pose.pose.orientation.z = 1
    initial.pose.pose.orientation.w = math.pi/3.
    pub.publish(initial)
    rospy.loginfo("Initial Pose Sent!")

if __name__ == '__main__':
    try:
        simple_publisher()
    except rospy.ROSInterruptException:
        pass
