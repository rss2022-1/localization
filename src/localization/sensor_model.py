import numpy as np
from scan_simulator_2d import PyScanSimulator2D

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale")

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
        z_max = self.table_width - 1.0

        # Loop through all z and d values and fill in lookup table
        for z in range(self.table_width):
            for d in range(self.table_width):
                p_hit = 1.0/np.sqrt(2*np.pi*self.sigma_hit**2) * np.exp(-((z-d)**2)/(2*self.sigma_hit**2))
                p_short = 2.0/d * (1-z/d) if (z <= d and d != 0) else 0.0
                p_max = 1.0/epsilon if (z >= z_max -.1 and z <= z_max) else 0.0
                p_rand = 1.0/z_max if z <= z_max else 0.0
                
                result_without_hit = self.alpha_short * p_short + self.alpha_max * p_max + self.alpha_random * p_rand
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

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scans = self.scan_sim.scan(particles)
        scaled_scans = scans / (self.map_resolution*self.lidar_scale_to_map_scale)

        z_max = self.table_width - 1
        np.clip(scaled_scans, 0, z_max)

        ####################################

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

        print("Map initialized")
