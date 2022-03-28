
import numpy as np



class MotionModel:

    def __init__(self):

        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
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
        xy = particles[:, :2]
        xy_odom = np.array([odometry[0:2]]).T
        xy_odom = np.tile(xy_odom, (particles.shape[0],1,1))
        theta = np.array([particles[:, -1]]).T # Nx1
        theta_odom = np.array([odometry[2]]) # 1x1

        # add noise if deterministic is false
        noise_coeff = self.noise_coeff * int(not self.deterministic)
        noise = noise_coeff * np.random.rand(particles.shape[0], 2, 1)
        xy_odom += noise

        # create rotation matrices for each theta + dtheta
        theta_new = np.ones_like(theta)*theta_odom + theta # Nx1
        odom_rot_mats = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]).T
        odom_rot_mats = odom_rot_mats.transpose(0,1,3,2) # 1xNx2x2
        rot_mats = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]).T
        rot_mats = rot_mats.transpose(0,1,3,2) # 1xNx2x2

        # create 4x4 pose matrices with pos and ori
        xy_odom = np.expand_dims(xy_odom, axis=0)
        xy = np.expand_dims([xy], axis=3)
        odom_pose = np.concatenate((odom_rot_mats, xy_odom), axis=3)[0]
        init_pose = np.concatenate((rot_mats, xy), axis=3)[0]

        n = xy.shape[1]
        z = np.array([[0,0,1]])
        z = np.tile(z, (n,1))
        z = np.expand_dims(z, axis=1)
        init_pose = np.concatenate((init_pose, z), axis=1)
        odom_pose = np.concatenate((odom_pose, z), axis=1)

        # transform initial poses with odometry transformations
        final_pose = np.matmul(init_pose, odom_pose)
        final = final_pose[:, :2, -1]
        particles = np.concatenate((final, theta_new), axis=1)
        return particles

