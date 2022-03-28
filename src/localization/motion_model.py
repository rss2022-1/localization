import numpy as np



class MotionModel:

    def __init__(self):

        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.scale = .05

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
        odometry_mx = np.zeros([3,3])
        odometry_mx[0,0] = np.cos(odometry[2])
        odometry_mx[0,1] = -np.sin(odometry[2])
        odometry_mx[1,0] = np.sin(odometry[2])
        odometry_mx[1,1] = np.cos(odometry[2])
        odometry_mx[0,2] = odometry[0]
        odometry_mx[1,2] = odometry[1]
        odometry_mx[2,2] = 1
        for particle in particles:
            particle_mx = np.zeros([3,3])
            particle_mx[0,0] = np.cos(particle[2])
            particle_mx[0,1] = -np.sin(particle[2])
            particle_mx[1,0] = np.sin(particle[2])
            particle_mx[1,1] = np.cos(particle[2])
            particle_mx[0,2] = particle[0]
            particle_mx[1,2] = particle[1]
            particle_mx[2,2] = 1
            new_pos = np.dot(particle_mx, odometry_mx)
            res.append(np.array([new_pos[0,2] + self.noise(self.scale), new_pos[1,2] + self.noise(self.scale), np.arccos(new_pos[0,0]) + self.noise(self.scale)]))
            # res.append(np.array([new_pos[0,2], new_pos[1,2], np.arccos(new_pos[0,0])]))
        return np.array(res)

        ####################################

    def noise(self, scale):
        return np.random.normal(0, scale)
