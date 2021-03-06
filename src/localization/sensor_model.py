import numpy as np
from scan_simulator_2d import PyScanSimulator2D

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle", 100)
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization", 500)
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view", 4.71)
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale", 1.0)
        self.map_resolution = 1

        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        self.z_max = self.table_width - 1.0
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = np.zeros((self.table_width, self.table_width))
        self.precompute_sensor_model()
        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_resolution = 1.0
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)
        rospy.loginfo("Initialized sensor model")

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        p_hit_table = np.zeros((self.table_width, self.table_width))
        epsilon = .1

        # Loop through all z and d values and fill in lookup table
        for z in range(self.table_width):
            for d in range(self.table_width):
                p_hit = 1.0/np.sqrt(2*np.pi*self.sigma_hit**2) * np.exp(-((float(z)-float(d))**2)/(2*self.sigma_hit**2))
                p_short = 2.0/d * (1-float(z)/float(d)) if (z <= d and d != 0) else 0.0
                p_max = float(z == self.z_max)
                p_rand = 1.0/self.z_max if z <= self.z_max else 0.0

                result_without_hit = self.alpha_short * p_short + self.alpha_max * p_max + self.alpha_rand * p_rand
                self.sensor_model_table[z][d] = result_without_hit
                p_hit_table[z][d] = p_hit

        # Normalize p_hit values
        column_sums = p_hit_table.sum(axis=0, keepdims=True)
        normalized_p_hits = p_hit_table/column_sums

        # Add p_hit values to main table
        self.sensor_model_table = self.sensor_model_table + self.alpha_hit * normalized_p_hits

        # Normalize the whole table
        column_sums = self.sensor_model_table.sum(axis=0, keepdims=True)
        self.sensor_model_table = self.sensor_model_table/column_sums

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        # Convert scan values from meters to pixels and clip values
        indices = np.arange(0, len(observation), (len(observation)//self.num_beams_per_particle)).astype(np.uint16)
        observation = observation[indices]

        stacked_scans = self.scan_sim.scan(particles)
        stacked_scans /= float(self.map_resolution * self.lidar_scale_to_map_scale)
        stacked_scans = np.clip(stacked_scans, 0, self.z_max) # clip
        stacked_scans = np.rint(stacked_scans) # discretize
        stacked_scans = stacked_scans.astype(np.uint16)


        # Convert ground truth scan values from meters to pixels and clip values
        observation = np.divide(observation, float(self.map_resolution * self.lidar_scale_to_map_scale))
        observation = np.clip(observation, 0, self.z_max) # clip
        observation = np.rint(observation) # discretize
        observation = observation.astype(np.uint16)


        particle_likelihoods = np.prod(self.sensor_model_table[observation, stacked_scans], axis=1)

        # Scan likelihood given by product of all likelihoods
        # particle_likelihoods = np.ones(len(particles))
        # rospy.loginfo('length of particles: %f', len(particles))
        # rospy.loginfo(self.sensor_model_table.shape)
        # for i in range(len(stacked_scans)):
        #     scan = stacked_scans[i]
        #     for j in range(len(scan)):
        #         # d = int(observation[j]) # ground truth
        #         d = observation[j].astype(int) # ground truth
        #         z = int(scan[j])
        #         particle_likelihoods[i] *= self.sensor_model_table[d][z]

        return particle_likelihoods**(1.0/2.2)

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Add map resolution info
        self.map_resoultion = map_msg.info.resolution

        # Make the map set
        self.map_set = True
        self.map_resolution = map_msg.info.resolution

        print("Map initialized")
