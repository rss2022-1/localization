import numpy as np
import rospy



class MotionModel:

    def __init__(self):

        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.scale = .05
        self.deterministic = rospy.get_param("~deterministic", "/cloud_map")

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        # TODO
        res = []
        x, y, theta = odometry
        odometry_mx = np.array([[np.cos(theta), -np.sin(theta), x],
                                [np.sin(theta), np.cos(theta), y],
                                [0, 0, 1]])
        for particle in particles:
            px, py, p_theta = particle
            particle_mx = np.array([[np.cos(p_theta), -np.sin(p_theta), px],
                                [np.sin(p_theta), np.cos(p_theta), py],
                                [0, 0, 1]])
            new_pos = np.dot(particle_mx, odometry_mx)
            if self.deterministic:
                res.append(np.array([new_pos[0][2], new_pos[1][2], theta + particle[-1]]))
            else:
                res.append(np.array([new_pos[0][2] + self.noise(self.scale), new_pos[1][2] + self.noise(self.scale), theta + particle[-1] + self.noise(self.scale)]))
        return np.array(res)

        ####################################

    def noise(self, scale):
        return np.random.normal(0, scale)
