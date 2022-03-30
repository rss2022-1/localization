#!/usr/bin/env python2
import rospy
from sensor_model import SensorModel
from motion_model import MotionModel
import numpy as np
import time
import threading
# from sklearn.cluster import DBSCAN

from sensor_msgs.msg import LaserScan, PointCloud
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TwistWithCovarianceStamped, Point32, Point, Vector3
from ackermann_msgs.msg import AckermannDriveStamped
import tf2_ros
import tf.transformations as transformations
from tf import TransformListener


class ParticleFilter:

    def __init__(self):
        # Initialize class variables
        self.map_initialized = False
        self.last_update = time.time()
        self.sim = True
        self.previous_scan = None
        self.flag = False
        self.num_lidar_scans = 100 if self.sim else 897
        self.full_ranges = np.array([-1. for i in range(self.num_lidar_scans)])
        self.velocity = 0.5
        # Setup
        self.lock = threading.Lock()
        self.num_particles = rospy.get_param("~num_particles")
        self.particles = np.zeros((self.num_particles, 3))
        self.initialized = False
        # Get parameters
        self.particle_filter_frame = rospy.get_param("~particle_filter_frame")

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
        self.cloud_topic = rospy.get_param("~cloud_topic", "cloud_map")
        self.estimation_topic = rospy.get_param("~estimation_topic", "/estim_marker")
        self.odom_pub  = rospy.Publisher("/pf/pose/odom", Odometry, queue_size = 1)
        self.cloud_publisher = rospy.Publisher(self.cloud_topic, PointCloud, queue_size=10)
        self.estimation_publisher = rospy.Publisher(self.estimation_topic, Marker, queue_size=10)
        self.drive_publisher = rospy.Publisher("/vesc/high_level/ackermann_cmd_mux/input/nav_0", AckermannDriveStamped, queue_size=10)

        # Testing publishers
        self.actual_positions_pub = rospy.Publisher("actual_position", Vector3, queue_size=10)
        self.pred_positions_pub = rospy.Publisher("predicted_position", Vector3, queue_size=10)
        self.tf = TransformListener()

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

        self.map_subscriber = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)

    def map_callback(self, map_msg):
        """ Callback for the map. """
        self.map_initialized = True

    def combine_scans(self, ranges):
        for i in range(len(ranges)):
            self.full_ranges[i] = min(self.previous_scan[i], ranges[i])/2.0
        
    def get_ranges(self, data):
        if not self.flag:
            self.previous_scan = np.array(data.ranges)
            self.flag = True
            return [], []
        else:
            self.combine_scans(data.ranges)
            self.flag = False
            self.previous_scan = None

        ranges = self.full_ranges.roll(-self.num_lidar_scans/6) # roll the array to the left to handle lidar offset
        return np.array(ranges)

    def get_average_pose(self, particles):
        """ Compute the "average" pose of the particles. """

        """ NOTE: As for determining the "average pose" of all of your particles,
        be careful with taking the average of theta
        values. See this page: mean of circular quantities.
        Also consider the case where your distribution is multi modal - an average could pick
        a very unlikely pose between two modes. What better notions of "average" could you use? """
        # Cluster poses
        # Use sin and cos to avoid circular mean problems
        poses = np.array([[x, y, np.sin(theta), np.cos(theta)] for x, y, theta in particles])
        # clusters = DBSCAN(eps=0.05, min_samples=100).fit_predict(poses) # TODO: Tweak the eps and min_samples values based off our data
        # rospy.loginfo("Number of clusters: %d", len(set(clusters)))
        # rospy.loginfo("Clusters %s", set(clusters))

        # Get poses in largest cluster
        # labels, counts = np.unique(clusters, return_counts=True)
        # largest_cluster_label = np.argmax(counts)
        # indices = [i for i in range(len(labels)) if clusters[i] == largest_cluster_label]
        # largest_cluster = poses[indices]

        # Get average of poses in largest cluster
        # Uses arctan2 method for circular mean as seen on Wikipedia
        avg = np.mean(poses, axis=0)
        x_bar, y_bar = avg[0], avg[1]
        theta_bar = np.arctan2(avg[2], avg[3])

        try:
            t = self.tf.getLatestCommonTime("map", "base_link")
            position, quaternion = self.tf.lookupTransform("map", "base_link", t)
            actual = Vector3()
            actual.x = position[0]
            actual.y = position[1]
            actual.z = position[2]
            self.actual_positions_pub.publish(actual)
            predicted = Vector3()
            predicted.x = x_bar
            predicted.y = y_bar
            predicted.z = 0
            self.pred_positions_pub.publish(predicted)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, rospy.exceptions.ROSTimeMovedBackwardsException):
            rospy.loginfo("Could not get positions")

        return x_bar, y_bar, theta_bar

    def lidar_callback(self, msg):
        """ Compute particle likelihoods and resample particles. """
        # Takes in lidar data and calls sensor_model to get particle likelihoods
        if not self.initialized and not self.map_initialized:
            return
        
        ranges = self.get_ranges(msg)
        if len(ranges) > 1:
            rospy.loginfo("Full Lidar data received.")
            particle_likelihoods = self.sensor_model.evaluate(self.particles, ranges)
            particle_likelihoods = particle_likelihoods / np.sum(particle_likelihoods) # Normalize to sum to 1

            # Resample particles based on likelihoods
            sampled_particles_indices = np.random.choice(len(self.particles), self.num_particles, p=particle_likelihoods)
            sampled_particles = self.particles[sampled_particles_indices]
            with self.lock:
                self.particles = sampled_particles

                # Publish the "average" pose as a transform between the map and the car's expected base_link
                avg_pose = self.get_average_pose(sampled_particles)
                self.pub_point_cloud()
                self.estimated_pose = avg_pose
                self.pub_pose_estimation(avg_pose)
                self.publish_pose(avg_pose)
                self.full_ranges = np.array([-1. for i in range(self.NUM_LIDAR_SCANS)])
        else:
            rospy.loginfo("Waiting on full lidar data...")

    def odom_callback(self, msg):
        """ Update particle positions based on odometry."""
        # Takes in odometry data then calls motion_model to update the particles
        if not self.initialized and not self.map_initialized:
            return
        # Twist gets us the Linear/Angular velocities
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vtheta = msg.twist.twist.angular.z

        #   Use a Set dt to find dx, dy, and dtheta
        curr_time = time.time()
        # dt = curr_time - self.last_update
        dt = 1/20.
        self.last_update = curr_time
        odom = np.array([vx*dt, vy*dt, vtheta*dt])
        propogated_particles = self.motion_model.evaluate(self.particles, odom)
        with self.lock:
            self.particles = propogated_particles

            # Determine the "average" particle pose
            avg_pose = self.get_average_pose(self.particles)
            self.pub_point_cloud()
            self.estimated_pose = avg_pose
            self.pub_pose_estimation(avg_pose)

            # Publish this "average" pose as a transform between the map and the car's expected base_link
            self.publish_pose(avg_pose)

    def publish_pose(self, avg_pose):
        """ Publish a transform between the map and the base_link frome of the given pose. """
        # Publish this "average" pose as a transform between the map and the car's expected base_link
        odom = Odometry()
        odom.header.frame_id = self.particle_filter_frame
        odom.header.stamp = rospy.Time.now()
        odom.pose.pose.position.x = avg_pose[0]
        odom.pose.pose.position.y = avg_pose[1]
        odom.pose.pose.position.z = 0
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = 1
        odom.pose.pose.orientation.w = avg_pose[2]

        # create covariance matrix somehow
        self.odom_pub.publish(odom)

    def initialpose_callback(self, msg):
        """ Initialize particle positions based on an initial pose estimate. """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        base_noise = .5
        self.particles[:, 0] = x + np.random.normal(0, base_noise, self.num_particles)
        self.particles[:, 1] = y + np.random.normal(0, base_noise, self.num_particles)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
        self.last_update = time.time()
        self.initialized = True

    def pub_point_cloud(self):
        ''' Publishes the point cloud of the particles '''
        cloud = PointCloud()
        cloud.header.frame_id = "/map"
        cloud.points = [Point32() for i in range(self.num_particles)]
        for i in range(self.num_particles):
            cloud.points[i].x = self.particles[i, 0]
            cloud.points[i].y = self.particles[i, 1]
            cloud.points[i].z = 0

        self.cloud_publisher.publish(cloud)

    def pub_pose_estimation(self, avg_pose):
        ''' Publishes the current estimated pose of the car '''
        estimation = Marker()
        estimation.header.frame_id = "/map"
        estimation.header.stamp = rospy.Time.now()
        estimation.ns = "estimation_marker"
        estimation.id = 0
        estimation.type = estimation.ARROW
        estimation.action = estimation.ADD
        estimation.points = [Point(), Point()]
        # Start
        estimation.points[0].x = avg_pose[0]
        estimation.points[0].y = avg_pose[1]
        estimation.points[0].z = 0
        # End
        estimation.points[1].x = np.cos(avg_pose[2]) + avg_pose[0]
        estimation.points[1].y = np.sin(avg_pose[2]) + avg_pose[1]
        estimation.points[1].z = 0
        estimation.color.a = 1.0
        estimation.color.g = 1.0
        estimation.scale.x = .2
        estimation.scale.y = .2
        self.estimation_publisher.publish(estimation)

    def create_ackermann_msg(self, steering_angle):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = '/map'
        msg.drive.steering_angle = steering_angle
        msg.drive.steering_angle_velocity = 0.0
        msg.drive.speed = self.VELOCITY
        msg.drive.acceleration = 0.5
        msg.drive.jerk = 0.0
        return msg

if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()