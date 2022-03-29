#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel
import numpy as np
import time
import threading
from sklearn.cluster import DBSCAN

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TwistWithCovarianceStamped


class ParticleFilter:

    def __init__(self):
        # Initialize class variables
        self.last_update = time.time()

        # Setup
        self.lock = threading.Lock()
        self.num_particles = 100 # TODO: Initialize particles
        self.particles = np.zeros((1, self.num_particles))
        # Get parameters
        self.particle_filter_frame = \
                rospy.get_param("~particle_filter_frame")

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan, self.lidar_callback, queue_size=1)
        self.odom_sub  = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initialpose_callback, queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def get_average_pose(self, particles):
        """ Compute the "average" pose of the particles. """

        """ NOTE: As for determining the "average pose" of all of your particles,
        be careful with taking the average of  𝜃
        values. See this page: mean of circular quantities.
        Also consider the case where your distribution is multi modal - an average could pick
        a very unlikely pose between two modes. What better notions of "average" could you use? """
        poses = np.array([[x, y, np.cos(theta), np.sin(theta)] for x, y, theta in particles])
        clusters = DBSCAN.fit_predict(poses, eps=0.5, min_samples=5) # TODO: Tweak the eps and min_samples values based off our data
        rospy.loginfo("Number of clusters: %d", len(set(clusters)))
        rospy.loginfo("Clusters %s", set(clusters))


    def lidar_callback(self, msg):
        """ Compute particle likelihoods and resample particles. """
        # Takes in lidar data and calls sensor_model to get particle likelihoods
        ranges = np.array(msg.ranges)
        particle_likelihoods = self.sensor_model.evaluate(self.particles, ranges)

        # Resample particles based on likelihoods
        num_particles = self.num_particles # TODO: Check to make sure this makes sense
        sampled_particles = np.random.choice(self.particles, num_particles, p=particle_likelihoods)
        with self.lock:
            self.particles = sampled_particles

            # Publish the "average" pose as a transform between the map and the car's expected base_link
            avg_pose = self.get_average_pose(sampled_particles)
            self.publish_pose(avg_pose)

    def odom_callback(self, msg):
        """ Update particle positions based on odometry."""
        # Takes in odometry data then calls motion_model to update the particles
        # Twist gets us the Linear/Angular velocities
        vx, vy = msg.twist.linear[:2]
        vtheta = msg.twist.angular[-1]

        #   Use a Set dt to find dx, dy, and dtheta
        curr_time = time.time()
        dt = curr_time - self.last_update
        self.last_update = curr_time
        odom = np.array([vx*dt, vy*dt, vtheta*dt])
        propogated_particles = self.motion_model.evaluate(self.particles, odom)
        with self.lock:
            self.particles = propogated_particles

            # Determine the "average" particle pose
            avg_pose = self.get_average_pose(self.particles)

            # Publish this "average" pose as a transform between the map and the car's expected base_link
            self.publish_pose(avg_pose)

    def publish_pose(self, pose):
        """ Publish a transform between the map and the base_link frome of the given pose. """
        # Publish this "average" pose as a transform between the map and the car's expected base_link
        new_pose = PoseWithCovarianceStamped()
        new_pose.pose.point = [pose[0], pose[1], 0]
        new_pose.pose.quaternion = [0,0,1,pose[2]]
        # create covariance matrix somehow
        self.odom_pub.publish(new_pose)

    def initialpose_callback(self, msg):
        """ Initialize particle positions based on an initial pose estimate. """
        pass

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
)