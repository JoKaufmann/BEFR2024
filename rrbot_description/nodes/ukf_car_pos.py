#!/usr/bin/env python
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
        self.state_car_initialized = False
        self.state_drone_initialized = False

        # Initialize camera and measurement vector
        # ToDo: check where the camera is defined to get correct direction and focal length
        # /rrbot/camera/SigmaX: 4.0
        # /rrbot/camera/SigmaY: 4.0
        # /rrbot/camera/SigmaZ: 0.25
        # /rrbot/camera/f: 300.0
        # /rrbot/camera/pitch: 0.7853975
        self.size = np.array([640, 480])    # sensor/image size
        self.pp = self.size*0.5             # principal point
        self.f = 300                        # focal length
        self.dcam = np.zeros(3)             # target position in camera frame

        # Sigma point parameters
        self.dim_x = 13
        self.alpha = 1e-3 # suggested by Wan and van der Merwe
        self.beta = 2.0 # 2.0 is optimal for Gaussian priors
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.dim_x + self.kappa) - self.dim_x
        self.gamma = np.sqrt(self.dim_x + self.lambda_)
        # Sigma point weights
        self.Wm = np.zeros(2*self.dim_x + 1)
        self.Wc = np.zeros(2*self.dim_x + 1)
        self.Wm[0] = self.lambda_ / (self.dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.dim_x + self.lambda_) + (1 - self.alpha**2 + self.beta)
        for i in range(1, 2*self.dim_x + 1):
            self.Wm[i] = self.Wc[i] = 1 / (2*(self.dim_x + self.lambda_))
        
        # Initialize state vector
        self.gt_car = np.zeros(10)
        self.gt_drone = np.zeros(10)
        self.x = np.zeros(self.dim_x) # (target_pos, target_vel, cam_pos, cam_orientation) #TODO: Define State

        # Initialize covariance matrix
        self.Sigma = np.eye(self.dim_x) #TODO: Define Covariance

        # Initialize noise matrices
        self.R = np.eye(self.dim_x) #TODO: Define process noise
        self.Q = np.eye(self.dim_x) #TODO: Define measurement noise

        # Set initial timestamp
        self.prev_time = rospy.Time.now()

        # Initialize subscriber
        rospy.Subscriber("rrbot/fake_detection", PointStamped, self.depth_cam_callback)
        rospy.Subscriber("rrbot/position_groundtruth", Odometry, self.groundtruth_drone_callback)
        rospy.Subscriber("car/position_groundtruth", Odometry, self.groundtruth_car_callback)
        
        # Initialize publisher
        self.pub = rospy.Publisher('car/pos_estimate', PointStamped, queue_size=10)
    
    def prediction_step(self, dt):
        if self.state_initialized is False:
            return
        
        # Compute sigma points
        sigma_points = np.zeros((13, 27)) # dimension len(x) x 2*len(x) + 1
        sigma_points[:, 0] = self.x
        for i in range(1, 1+self.dim_x):
            sigma_points[:, i] = self.x + self.gamma*np.sqrt(self.Sigma[i-1])
            sigma_points[:, i+self.dim_x] = self.x - self.gamma*np.sqrt(self.Sigma[i-1])

        # Predict sigma points through process model #TODO Implement process model
        sigma_points_pred = np.zeros((13, 27))

        # Compute predicted mean
        x_pred = self.Wm[0]*sigma_points_pred[:, 0]
        for i in range(1, 2*self.dim_x + 1):
            x_pred += self.Wm[i]*sigma_points_pred[:, i]

        # Compute predicted covariance #TODO Define process noise
        P_pred = self.Wc[0]*(sigma_points_pred[:, 0]-x_pred)@(sigma_points_pred[:, 0]-x_pred).T + self.R
        for i in range(1, 2*self.dim_x + 1):
            P_pred += self.Wc[i]*(sigma_points_pred[:, i]-x_pred)@(sigma_points_pred[:, i]-x_pred).T + self.R #TODO Add R each time?


    def update_step(self):
        if self.state_initialized is False:
            return
        # ToDo: Implement update step

        self.publish_data()
    
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

        if self.state_initialized is False:
            self.x[6:9] = self.gt_drone[0:3]
            self.x[9:14] = self.gt_drone[6:10]
            self.state_drone_initialized = True
            if self.state_car_initialized is True:
                self.state_initialized = True

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

        # init state    # TODO: Init state
        if self.state_initialized is False:
            self.x[0:6] = self.gt_car[0:6]
            self.state_car_initialized = True
            if self.state_drone_initialized is True:
                self.state_initialized = True

    def depth_cam_callback(self, data):
        if self.state_initialized is False:
            return
        # get depth camera data
        u = data.point.x
        v = data.point.y
        d = data.point.z

        # calculate target position in camera frame # ToDo: Check sign
        x_cam = d*(-self.pp[0] + u)*np.sqrt(1/(self.f**2 + self.pp[0]**2 - 2*self.pp[0]*u + self.pp[1]**2 - 2*self.pp[1]*v + u**2 + v**2))
        y_cam = d*(-self.pp[1] + v)*np.sqrt(1/(self.f**2 + self.pp[0]**2 - 2*self.pp[0]*u + self.pp[1]**2 - 2*self.pp[1]*v + u**2 + v**2))
        z_cam = -d*self.f*np.sqrt(1/(self.f**2 + self.pp[0]**2 - 2*self.pp[0]*u + self.pp[1]**2 - 2*self.pp[1]*v + u**2 + v**2))
        
        # x_cam = d*(self.pp[0] - u)*np.sqrt(1/(self.f**2 + self.pp[0]**2 - 2*self.pp[0]*u + self.pp[1]**2 - 2*self.pp[1]*v + u**2 + v**2))
        # y_cam = d*(self.pp[1] - v)*np.sqrt(1/(self.f**2 + self.pp[0]**2 - 2*self.pp[0]*u + self.pp[1]**2 - 2*self.pp[1]*v + u**2 + v**2))
        # z_cam = d*self.f*np.sqrt(1/(self.f**2 + self.pp[0]**2 - 2*self.pp[0]*u + self.pp[1]**2 - 2*self.pp[1]*v + u**2 + v**2))
        target_cam = np.array([x_cam, y_cam, z_cam])

        # calculate target position in world frame # ToDo: add camera offset and orientation on drone (default all 0), see fake_visual_detector
        q = self.gt_drone[6:10]
        p = self.gt_drone[0:3]
        self.dcam = qv_mult(q, target_cam) + p

        print('x_calc: {} \t x_gt: {} \n y_calc: {} \t y_gt: {} \n z_calc: {} \t z_gt: {} \n'.format(self.dcam[0], self.gt_car[0], self.dcam[1], self.gt_car[1], self.dcam[2], self.gt_car[2]))
        
        self.update_step()

    def publish_data(self):
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.point.x = self.dcam[0]
        msg.point.y = self.dcam[1]
        msg.point.z = self.dcam[2]

        self.pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('ukf_node_2')
    ukf = UKF()
    rospy.spin()