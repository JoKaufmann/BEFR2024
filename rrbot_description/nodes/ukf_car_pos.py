import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped, PointStamped
import numpy as np
import tf

# helper functions
def qv_mult(q1, v1):
    ''' Rotate vector v1 by quaternion q1 '''
    q2 = list(v1)
    q2.append(0.0)
    return tf.transformations.quaternion_multiply(
        tf.transformations.quaternion_multiply(q1, q2), 
        tf.transformations.quaternion_conjugate(q1)
    )[:3]

def qv_inv(q1):
    ''' Inverse of quaternion q1 '''
    q2=q1*1.0
    q2[:3]*=-1
    return q2

class UKF:
    def __init__(self):
        # Initialization flags
        self.state_initialized = False

        # Initialize camera and measurement vector
        # /rrbot/camera/SigmaX: 4.0
        # /rrbot/camera/SigmaY: 4.0
        # /rrbot/camera/SigmaZ: 0.25
        self.sensor_size = np.array([640, 480])     # sensor/image size
        self.pp = self.sensor_size*0.5              # principal point
        self.f = 300                                # focal length
        self.depth_factor = 1                       # depth factor
        self.cam_offset = np.zeros(3)               # camera offset in drone frame
        # camera direction on robot as quaternion from [roll, pitch, yaw]
        self.cam_dir = tf.transformations.quaternion_from_euler(0, np.pi/4, 0)     # ToDo: change pitch back to pi/4, also in world file
        self.dcam_initialized = False
        self.dcam = np.zeros(6)                     # target position and velocity in camera frame

        # Sigma point parameters
        self.dim_x = 6
        self.alpha = 1e-3   # suggested by Wan and van der Merwe
        self.beta = 2.0     # 2.0 is optimal for Gaussian priors
        self.kappa = 0
        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        self.gamma = np.sqrt(self.dim_x + self.lambda_)
        # Sigma point weights
        self.Wm = np.zeros(2*self.dim_x + 1)
        self.Wc = np.zeros(2*self.dim_x + 1)
        self.Wm[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.dim_x + self.lambda_) + (1 - self.alpha**2 + self.beta)
        for i in range(1, 2*self.dim_x + 1):
            self.Wm[i] = self.Wc[i] = 1 / (2*(self.dim_x + self.lambda_))
        # Todo: check self.wm, for debugging, create pos. weight vector
        self.Wm = np.concatenate((np.array([self.dim_x*2]), np.array([1]*int(self.dim_x*2))))/(self.dim_x*4)
        self.Wc = self.Wm

        # Initialize state vector
        self.gt_car = np.zeros(10)
        self.gt_drone = np.zeros(10)
        self.x = np.zeros(self.dim_x)       # (target_pos, target_vel)

        # Initialize covariance matrix
        self.Sigma = np.eye(self.dim_x)                             #TODO: Define Covariance
        self.sigma_points = np.zeros((self.dim_x, 2*self.dim_x+1))  # dimension len(x) x 2*len(x) + 1
        
        # Initialize noise matrices
        self.R = np.diag(np.array([1, 1, 1, 1, 1, 1]))*10              #TODO: Define process noise
        self.Q = np.diag(np.array([4, 4., 0.25, 1, 1, 1]))*0.01          #TODO: Define measurement noise

        # Set initial timestamp
        #? currently done in groundtruth_car_callback once the first groundtruth data is received
        # self.prev_time = rospy.Time.now()

        # Initialize subscriber
        rospy.Subscriber("rrbot/fake_detection", PointStamped, self.depth_cam_callback)
        rospy.Subscriber("rrbot/position_groundtruth", Odometry, self.groundtruth_drone_callback)
        rospy.Subscriber("car/position_groundtruth", Odometry, self.groundtruth_car_callback)
        
        # Initialize publisher
        self.pub = rospy.Publisher('car/pos_estimate', Odometry, queue_size=10)
    
    def prediction_step(self, dt):
        if self.state_initialized is False:
            return
        
        # Compute sigma points
        self.get_sigma_points()

        # Calculate transition of sigma points through state transition model
        sigma_points_pred = np.zeros((self.dim_x, 2*self.dim_x + 1))
        for i in range(2*self.dim_x + 1):
            sigma_points_pred[:, i] = self.state_transition(self.sigma_points[:, i], dt)

        # Compute predicted mean
        self.x = np.sum(self.Wm*sigma_points_pred, axis=1)
        # self.x = np.average(sigma_points_pred, axis=1, weights=self.Wm)

        # Compute predicted covariance
        self.Sigma = np.zeros((self.dim_x, self.dim_x))
        for i in range(2*self.dim_x + 1):
            self.Sigma += self.Wc[i]*np.outer(sigma_points_pred[:, i]-self.x, sigma_points_pred[:, i]-self.x)
        self.Sigma += self.R
        
    def update_step(self, dt):
        if self.state_initialized is False:
            return
        
        # Initialize sigma points in measurement space
        Z_sigma_old = np.zeros((3, 2*self.dim_x + 1))

        # transform old sigma points to measurement space
        for i in range(2*self.dim_x + 1):
            Z_sigma_old[:, i] = self.measurement_model(self.sigma_points[:3, i])

        # get new sigma points from pred. state and state covariance
        self.get_sigma_points()

        # transform sigma points to measurement space
        Z_sigma = np.zeros((6, 2*self.dim_x + 1))
        for i in range(2*self.dim_x + 1):
            Z_sigma[:3, i] = self.measurement_model(self.sigma_points[:3, i])
        Z_sigma[3:, :] = (Z_sigma[:3, :] - Z_sigma_old[:3, :])/dt

        # mean of predicted measurement
        z_sigma_mean = np.sum(self.Wm*Z_sigma, axis=1)
        # z_sigma_mean = np.average(Z_sigma, axis=1, weights=self.Wm)

        # covariance of predicted measurement
        S = np.zeros((self.dim_x, self.dim_x))
        for i in range(2*self.dim_x + 1):
            S += self.Wc[i]*np.outer(Z_sigma[:, i]-z_sigma_mean, Z_sigma[:, i]-z_sigma_mean)
        S += self.Q

        # cross-covariance between state and measurement
        Sigma_hat = np.zeros((self.dim_x, 6))
        for i in range(2*self.dim_x + 1):
            Sigma_hat += self.Wc[i]*np.outer(self.sigma_points[:, i]-self.x, Z_sigma[:, i]-z_sigma_mean)
        # print(f'S: {S}\n')
        print(f'Sigma_hat: {Sigma_hat}\n')
        
        # Kalman gain
        K = Sigma_hat@np.linalg.inv(S)
        # print(f'K: {K}\n')
        # update state and covariance
        self.x += K@(self.dcam - z_sigma_mean)
        self.Sigma -= K@S@K.T
    
    def state_transition(self, x: np.array, dt: float) -> np.array:
        ''' State transition model for target
        Args:
            x (np.array): state vector
            dt (float): time step
        Returns:
            np.array: predicted state vector
        '''
        pos_pred = x[0:3]# + np.random.randn(3)*0.01
        vel_pred = x[3:6]# + np.random.randn(3)*0.001  # assumption of constant velocity
        # pos_pred = x[0:3] + dt*x[3:6]
        # vel_pred = x[3:6]               # assumption of constant velocity

        return np.concatenate((pos_pred, vel_pred))

    def measurement_model(self, x: np.array) -> np.array:
        ''' Measurement model for camera
        Args:
            x (np.array): position of target in world frame
        Returns:
            np.array: measurement vector (u, v, d) in image frame
        '''
        # calc. current cam position and direction
        cam_pos = self.gt_drone[0:3] + qv_mult(self.gt_drone[6:10], self.cam_offset)
        cam_dir = tf.transformations.quaternion_multiply(self.cam_dir, self.gt_drone[6:10])
        # calc. target position in camera frame
        tar_to_cam_world = x - cam_pos
        tar_pos_cam = qv_mult(qv_inv(cam_dir), tar_to_cam_world)
        # calc. target position in image frame
        u = (-self.f*tar_pos_cam[1]/tar_pos_cam[0])+self.pp[0]
        v = (-self.f*tar_pos_cam[2]/tar_pos_cam[0])+self.pp[1]
        d = self.depth_factor*np.linalg.norm(tar_pos_cam)
        return np.array([u, v, d])
        
    def get_sigma_points(self):
        self.sigma_points = np.tile(self.x, (self.dim_x*2+1, 1)).T
        L = np.diag(np.diag(np.linalg.cholesky(self.gamma**2*self.Sigma)))    # cholesky decomposition of Sigma (already returns the sqrt of Sigma, since A=LL*)
        self.sigma_points[:, 1:1+self.dim_x] += L
        self.sigma_points[:, 1+self.dim_x:] -= L

    def groundtruth_drone_callback(self, data):
        # position x,y,z
        self.gt_drone[0] = data.pose.pose.position.x
        self.gt_drone[1] = data.pose.pose.position.y
        self.gt_drone[2] = data.pose.pose.position.z

        # velocity x,y,z
        self.gt_drone[3] = data.twist.twist.linear.x
        self.gt_drone[4] = data.twist.twist.linear.y
        self.gt_drone[5] = data.twist.twist.linear.z

        # quaternion x,y,z,w
        self.gt_drone[6] = data.pose.pose.orientation.x
        self.gt_drone[7] = data.pose.pose.orientation.y
        self.gt_drone[8] = data.pose.pose.orientation.z
        self.gt_drone[9] = data.pose.pose.orientation.w

    def groundtruth_car_callback(self, data):
        # position x,y,z
        self.gt_car[0] = data.pose.pose.position.x
        self.gt_car[1] = data.pose.pose.position.y
        self.gt_car[2] = data.pose.pose.position.z

        # velocity x,y,z
        self.gt_car[3] = data.twist.twist.linear.x
        self.gt_car[4] = data.twist.twist.linear.y
        self.gt_car[5] = data.twist.twist.linear.z

        # quaternion x,y,z,w
        self.gt_car[6] = data.pose.pose.orientation.x
        self.gt_car[7] = data.pose.pose.orientation.y
        self.gt_car[8] = data.pose.pose.orientation.z
        self.gt_car[9] = data.pose.pose.orientation.w

        # init state vector
        if self.state_initialized is False:
            self.x = self.gt_car[0:6]
            self.state_initialized = True
            self.prev_time = data.header.stamp.secs

    def depth_cam_callback(self, data):
        if self.state_initialized is False:
            return
        # Calculate time difference
        timestamp = data.header.stamp.secs + data.header.stamp.nsecs*1e-9
        dt = (timestamp - self.prev_time)
        self.prev_time = timestamp
        # Get new depth camera data
        u = data.point.x
        v = data.point.y
        d = data.point.z
        # Calculate change in measurement
        if self.dcam_initialized is False:
            u_dot = 0
            v_dot = 0
            d_dot = 0
            self.dcam_initialized = True
        else:
            u_dot = (u - self.dcam[0])/dt
            v_dot = (v - self.dcam[1])/dt
            d_dot = (d - self.dcam[2])/dt

        # Update measurement data
        self.dcam = np.array([u, v, d, u_dot, v_dot, d_dot])
        
        # Call the UKF algorithm
        self.prediction_step(dt)
        self.update_step(dt)
        # print(f'x:    {self.x}\n')
        # print(f'x_gt: {self.gt_car}\n')
        self.publish_data()

    def publish_data(self):
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"

        # Set the position
        odom.pose.pose.position.x = self.x[0]
        odom.pose.pose.position.y = self.x[1]
        odom.pose.pose.position.z = self.x[2]

        # Set the velocity
        odom.twist.twist.linear.x = self.x[3]
        odom.twist.twist.linear.y = self.x[4]
        odom.twist.twist.linear.z = self.x[5]

        self.pub.publish(odom)

if __name__ == '__main__':
    np.set_printoptions(precision=4)
    rospy.init_node('ukf_node')
    ukf = UKF()
    rospy.spin()