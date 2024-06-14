#!/usr/bin/env python
import rospy
import std_msgs.msg
import sensor_msgs.msg
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped, PointStamped
import numpy as np
from pyproj import Transformer
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from enum import Enum
import sys

# Reference GPS coordinate for (0,0,0) in cartesian coordinates
REFERENCE_LAT = 48.75 
REFERENCE_LON = 9.105
REFERENCE_ALT = 0

# Magnetic field constants
# MAG = np.array([6e-06, 2.3e-05, -4.2e-05])
MAG = np.array([2.3e-05, 6e-06, -4.2e-05])      # y and x changed, due to reference heading 0 -> x = east, north = y
MAG = MAG/np.linalg.norm(MAG)                   # normalize to compare with the magnetometer measurement

def quaternion_product(q1, q2):
    qr = np.zeros(4)
    qr[0] = q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1]
    qr[1] = q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0]
    qr[2] = q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3]
    qr[3] = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2]

    return qr

def quaternion_from_omega(q, omega, bias, dt):
    q_dot = 0.5 * np.array([[ q[3], -q[2],  q[1]],
                            [ q[2],  q[3], -q[0]],
                            [-q[1],  q[0],  q[3]],
                            [-q[0], -q[1], -q[2]]]) @ (omega - bias)
    # print('dq: ', q_dot * dt)
    q = q + q_dot * dt
    return q

def rotate_vector(q, v):
    q_inv = q.copy()
    q_inv[:-1] = q_inv[:-1]*(-1)
    qv = np.concatenate((v, np.array([0])))
    return (quaternion_product(quaternion_product(q, qv), q_inv))[:-1]

def partial_q(q, v):
    # calculates the partial derivative of a rotated vector q*v*q^-1 with respect to the quaternion q
    return np.array([[ q[0]*v[0]+q[1]*v[1]+q[2]*v[2],  q[3]*v[2]+q[0]*v[1]-q[1]*v[0], -q[3]*v[1]+q[0]*v[2]-q[2]*v[0],  q[3]*v[0]+q[1]*v[2]-q[2]*v[1]],
                     [-q[3]*v[2]-q[0]*v[1]+q[1]*v[0],  q[0]*v[0]+q[1]*v[1]+q[2]*v[2],  q[3]*v[0]+q[1]*v[2]-q[2]*v[1],  q[3]*v[1]-q[0]*v[2]+q[2]*v[0]],
                     [ q[3]*v[1]-q[0]*v[2]+q[2]*v[0], -q[3]*v[0]-q[1]*v[2]+q[2]*v[1],  q[0]*v[0]+q[1]*v[1]+q[2]*v[2], -q[3]*v[2]-q[0]*v[1]+q[1]*v[0]]])*2

class Sensor(Enum):
    IMU = 1
    GPS_POS = 2
    GPS_VEL = 3
    MAGNETOMETER = 4
    BAROMETER = 5

class EKF:
    def __init__(self):
        # Initialize GPS to UTM transformer
        self.wsg84_to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)

        # Initialize measurements
        self.omega = np.zeros(3)
        self.p_GPS = np.zeros(3)
        self.v_GPS = np.zeros(3)
        self.p_z_B = 0
        self.m_M = np.zeros(3)

        # Initialize state vector and groundtruth
        self.gt = np.zeros(10)
        self.x = np.zeros(17)
        self.state_initialized = False

        # Initialize process noise covariance
        imu_dt = 0.01   # 100Hz
        R_pos = np.ones(3)*0.1
        R_vel = np.ones(3)
        R_q = np.ones(4)
        R_b_gyro = np.ones(3)*0.01
        R_b_p_gps = np.ones(3)*0.01
        R_b_baro = np.array([0.01])
        self.R = np.diag(np.concatenate((R_pos, R_vel, R_q, R_b_gyro, R_b_p_gps, R_b_baro)))*imu_dt**2
        
        # Initialize measurement noise covariance
        Q_GPS_p = np.array([0.01**2, 0.01**2, 0.01**2])
        Q_GPS_v = np.array([0.05**2, 0.05**2, 0.05**2])
        Q_baro = np.array([0.1**2])
        Q_mag = np.array([1.3e-2**2, 1.3e-2**2, 1.3e-2**2])
        self.Q = np.diag(np.concatenate((Q_GPS_p, Q_GPS_v, Q_baro, Q_mag)))
        
        # Initialize state covariance
        Sigma_p_init = np.ones(3)                   
        Sigma_v_init = np.ones(3)                   
        Sigma_q_init = np.ones(4)*0.1               
        Sigma_b_gyro_init = np.ones(3)
        Sigma_b_p_gps_init = np.ones(3)
        Sigma_b_baro_init = np.array([1])    
        self.Sigma = np.diag(np.concatenate((Sigma_p_init, Sigma_v_init, Sigma_q_init, Sigma_b_gyro_init, Sigma_b_p_gps_init, Sigma_b_baro_init)))

        # Set timestamp
        self.prev_time = rospy.Time.now()

        # Initialize subscriber
        rospy.Subscriber("rrbot/sim_imu", Imu, self.imu_callback)
        rospy.Subscriber("rrbot/sim_magnetometer", Vector3Stamped, self.magnetometer_callback)
        rospy.Subscriber("rrbot/sim_gps_fix", NavSatFix, self.gps_pos_callback)
        rospy.Subscriber("rrbot/sim_gps_vel", Vector3Stamped, self.gps_vel_callback)
        rospy.Subscriber("rrbot/sim_barometer", PointStamped, self.barometer_callback)
        rospy.Subscriber("rrbot/position_groundtruth", Odometry, self.groundtruth_callback)
        
        # Initialize publisher
        self.pub = rospy.Publisher('rrbot/state_estimate', Odometry, queue_size=10)

    def groundtruth_callback(self, data):
        # position x,y,z
        self.gt[0] = data.pose.pose.position.x
        self.gt[1] = data.pose.pose.position.y
        self.gt[2] = data.pose.pose.position.z

        # velocity x,y,z
        self.gt[3] = data.twist.twist.linear.x
        self.gt[4] = data.twist.twist.linear.y
        self.gt[5] = data.twist.twist.linear.z

        # quaternion x,y,z,w
        self.gt[6] = data.pose.pose.orientation.x
        self.gt[7] = data.pose.pose.orientation.y
        self.gt[8] = data.pose.pose.orientation.z
        self.gt[9] = data.pose.pose.orientation.w

        # init state
        if self.state_initialized is False:
            self.x[0:10] = self.gt[0:10]
            self.state_initialized = True

    def prediction_step(self, dt):
        if self.state_initialized is False:  # if state is not initialized
            return
        # predict the state vector
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]
        b_gyro = self.x[10:13]
        b_p_gps = self.x[13:16]
        b_baro = np.array([self.x[16]])

        q_new = quaternion_from_omega(q, self.omega, b_gyro, dt)
        q_new = q_new/np.linalg.norm(q_new) # normalize to unit quaternion
        v_new = v + (rotate_vector(q, self.a) + np.array([0, 0, -9.8066]))*dt
        p_new = p + v*dt + 0.5*(rotate_vector(q, self.a) + np.array([0, 0, -9.8066]))*dt**2
        self.x = np.concatenate((p_new, v_new, q_new, b_gyro, b_p_gps, b_baro))
        
        # and its jacobian
        ## pos. part
        dpdp = np.eye(3)
        dpdv = np.eye(3)*dt
        dpdq = partial_q(q, self.a)*dt**2
        ## vel. part
        dvdv = np.eye(3)
        dvdq = partial_q(q, self.a)*dt
        ## q part
        dqdq = np.eye(4) + np.array([   [ 0                         , self.omega[2]-b_gyro[2]   ,-self.omega[1]+b_gyro[1]   , self.omega[0]-b_gyro[0]   ],
                                        [-self.omega[2]+b_gyro[2]   , 0                         , self.omega[0]+b_gyro[0]   , self.omega[1]+b_gyro[1]   ],
                                        [ self.omega[1]-b_gyro[1]   ,-self.omega[0]-b_gyro[0]   , 0                         , self.omega[2]+b_gyro[2]   ],
                                        [-self.omega[0]+b_gyro[0]   ,-self.omega[1]-b_gyro[1]   ,-self.omega[2]-b_gyro[2]   , 0                         ]])*0.5*dt
        dqdb_gyro = np.array([  [-q[3], -q[2],  -q[1]],
                                [-q[2], -q[3],   q[0]],
                                [ q[1], -q[0],  -q[3]],
                                [ q[0],  q[1],   q[2]]])*dt*0.5
        ## bias part
        db_gyrodb_gyro = np.eye(3)
        db_p_gpsdb_p_gps = np.eye(3)
        db_barodb_baro = np.array([1])
        ## assemble the jacobian
        G = np.block([  [ dpdp               , dpdv  , dpdq  , np.zeros((3, 3)) , np.zeros((3, 3))  , np.zeros((3, 1))  ],
                        [ np.zeros((3, 3))   , dvdv  , dvdq  , np.zeros((3, 3)) , np.zeros((3, 3))  , np.zeros((3, 1))  ],
                        [ np.zeros((4, 6))           , dqdq  , dqdb_gyro        , np.zeros((4, 3))  , np.zeros((4, 1))  ],
                        [ np.zeros((3, 10))                  , db_gyrodb_gyro   , np.zeros((3, 3))  , np.zeros((3, 1))  ],
                        [ np.zeros((3, 13))                                     , db_p_gpsdb_p_gps  , np.zeros((3, 1))  ],
                        [ np.zeros((1, 16))                                                         , db_barodb_baro    ]])

        # predict the state covariance
        self.Sigma = G @ self.Sigma @ G.T + self.R

        # Publish the odometry
        self.publish_odometry()

    def update_step(self, sensor_type=None):
        if self.state_initialized is False:  # if state is not initialized
            return
        # calculate the predicted measurement
        q = self.x[6:10]
        b_p_gps = self.x[13:16]
        b_baro = np.array([self.x[16]])
        p_GPS_pred = self.x[0:3] + b_p_gps
        v_GPS_pred = self.x[3:6]
        p_z_B_pred = np.array([self.x[2]]) + b_baro
        m_M_pred = 2*np.array([ [ MAG[0]*(0.5 - q[1]**2 - q[2]**2) + MAG[1]*(q[3]*q[2] + q[0]*q[1])      + MAG[2]*(q[0]*q[2] - q[3]*q[1])   ],
                                [ MAG[0]*(q[0]*q[1] - q[3]*q[2])   + MAG[1]*(0.5 - q[0]**2 - q[2]**2)    + MAG[2]*(q[3]*q[0] + q[1]*q[2])   ],
                                [ MAG[0]*(q[3]*q[1] + q[0]*q[2])   + MAG[1]*(q[1]*q[2] - q[3]*q[0])      + MAG[2]*(0.5 - q[0]**2 - q[1]**2) ]]).flatten()
        
        # and its jacobian
        dp_GPSdp = np.eye(3)
        dp_GPSdb_gps = np.eye(3)
        dv_GPSdv = np.eye(3)
        
        dBdp_z = np.array([0, 0, 1])
        dBdb_B = np.array([1])
        
        dMdq = np.array([[2*MAG[1]*q[1] + 2*MAG[2]*q[2]                ,-4*MAG[0]*q[1] + 2*MAG[1]*q[0] - 2*MAG[2]*q[3] ,-4*MAG[0]*q[2] + 2*MAG[1]*q[3] + 2*MAG[2]*q[0] , 2*MAG[1]*q[2] - 2*MAG[2]*q[1]], 
                         [2*MAG[0]*q[1] - 4*MAG[1]*q[0] + 2*MAG[2]*q[3], 2*MAG[0]*q[0] + 2*MAG[2]*q[2]                 ,-2*MAG[0]*q[3] - 4*MAG[1]*q[2] + 2*MAG[2]*q[1] ,-2*MAG[0]*q[2] + 2*MAG[2]*q[0]], 
                         [2*MAG[0]*q[2] - 2*MAG[1]*q[3] - 4*MAG[2]*q[0], 2*MAG[0]*q[3] + 2*MAG[1]*q[2] - 4*MAG[2]*q[1] , 2*MAG[0]*q[0] + 2*MAG[1]*q[1]                 , 2*MAG[0]*q[1] - 2*MAG[1]*q[0]]])

        # assemble jacobian      
        H = np.block([  [ dp_GPSdp          , np.zeros((3, 3))  , np.zeros((3, 4))  , np.zeros((3, 3))  , dp_GPSdb_gps      , np.zeros((3, 1))  ],
                        [ np.zeros((3, 3))  , dv_GPSdv          , np.zeros((3, 4))  , np.zeros((3, 3))  , np.zeros((3, 3))  , np.zeros((3, 1))  ],
                        [ dBdp_z            , np.zeros((1, 3))  , np.zeros((1, 4))  , np.zeros((1, 3))  , np.zeros((1, 3))  , dBdb_B            ],
                        [ np.zeros((3, 3))  , np.zeros((3, 3))  , dMdq              , np.zeros((3, 3))  , np.zeros((3, 3))  , np.zeros((3, 1))  ]])

        innovation = np.concatenate((self.p_GPS - p_GPS_pred, self.v_GPS - v_GPS_pred, self.p_z_B - p_z_B_pred, self.m_M - m_M_pred))
        
        if sensor_type == Sensor.GPS_POS:
            # without height from GPS
            # s_i = [0, 1, 2, 13, 14, 15]     # state index
            # m_i = [0, 1, 2]                 # measurement index
            # without height from GPS, height from barometer
            s_i = [0, 1, 13, 14]            # state index
            m_i = [0, 1]                    # measurement index
        elif sensor_type == Sensor.GPS_VEL:
            s_i = [3, 4, 5]
            m_i = [3, 4, 5]
        elif sensor_type == Sensor.BAROMETER:
            s_i = [2, 16]
            m_i = [6]
        elif sensor_type == Sensor.MAGNETOMETER:
            s_i = [6, 7, 8, 9]
            m_i = [7, 8, 9]
        else:
            return

        # calculate the kalman gain .reshape((len(i),len(m_i)))
        K = self.Sigma[np.ix_(s_i, s_i)] @ H[np.ix_(m_i, s_i)].T @ np.linalg.inv(H[np.ix_(m_i, s_i)] @ self.Sigma[np.ix_(s_i, s_i)] @ H[np.ix_(m_i, s_i)].T + self.Q[np.ix_(m_i, m_i)])

        # update the state vector and state covariance
        self.x[s_i] = self.x[s_i] + K @ innovation[m_i]
        self.Sigma[np.ix_(s_i, s_i)] = (np.eye(len(s_i)) - (K @ H[np.ix_(m_i, s_i)])) @ self.Sigma[np.ix_(s_i, s_i)]

        # normalize q
        self.x[6:10] = self.x[6:10]/np.linalg.norm(self.x[6:10])

        # Publish the odometry
        self.publish_odometry()

    def imu_callback(self, data):
        dt = (rospy.Time.now() - self.prev_time).to_sec()
        self.prev_time = rospy.Time.now()
        
        # convert to x,y,z
        self.a = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])
        self.omega = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])
        
        # Perform the prediction step
        self.prediction_step(dt)
    
    def magnetometer_callback(self, data):
        # convert to x,y,z
        self.m_M = np.array([data.vector.x, data.vector.y, data.vector.z])
        self.m_M = self.m_M/np.linalg.norm(self.m_M)                        # normalize to compare with magnetic field vector

        # Perform the update step
        self.update_step(Sensor.MAGNETOMETER)

    def gps_pos_callback(self, data):
        x_ref, y_ref, z_ref = self.wsg84_to_utm.transform(REFERENCE_LAT, REFERENCE_LON, REFERENCE_ALT)
        x_gps, y_gps, z_gps = self.wsg84_to_utm.transform(data.latitude, data.longitude, data.altitude)

        # Update the latest GPS data
        self.p_GPS = np.array([x_gps - x_ref, -y_gps + y_ref, z_gps - z_ref])

        # Perform the update step
        self.update_step(Sensor.GPS_POS)
    
    def gps_vel_callback(self, data):
        self.v_GPS = np.array([data.vector.x, data.vector.y, data.vector.z])

        # Perform the update step
        self.update_step(Sensor.GPS_VEL)

    def barometer_callback(self, data):
        self.p_z_B = data.point.z

        # Perform the update step
        self.update_step(Sensor.BAROMETER)

    def publish_odometry(self):
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"

        # Set the position
        odom.pose.pose.position.x = self.x[0]
        odom.pose.pose.position.y = self.x[1]
        odom.pose.pose.position.z = self.x[2]

        # Set the orientation
        odom.pose.pose.orientation.x = self.x[6]
        odom.pose.pose.orientation.y = self.x[7]
        odom.pose.pose.orientation.z = self.x[8]
        odom.pose.pose.orientation.w = self.x[9]

        # Set the covariance for position and orientation (only im part for quaternion approx euler cov.)
        odom.pose.covariance = np.block([   [ self.Sigma[0:3, 0:3]  , self.Sigma[0:3, 6:9]  ],
                                            [ self.Sigma[6:9, 0:3]  , self.Sigma[6:9, 6:9]  ]]).flatten(order='C') # row-major

        # Set the velocity
        odom.twist.twist.linear.x = self.x[3]
        odom.twist.twist.linear.y = self.x[4]
        odom.twist.twist.linear.z = self.x[5]

        # Set the angular velocity
        odom.twist.twist.angular.x = self.omega[0]
        odom.twist.twist.angular.y = self.omega[1]
        odom.twist.twist.angular.z = self.omega[2]

        # Set the covariance for velocity and angular velocity
        ## uncertainty in angular velocity = uncertainty in bias
        odom.twist.covariance = np.block([  [self.Sigma[3:6, 3:6]   , self.Sigma[3:6, 10:13]   ],
                                            [self.Sigma[10:13, 3:6] , self.Sigma[10:13, 10:13] ]]).flatten(order='C')   # row-major
        
        # Publish the odometry
        self.pub.publish(odom)

if __name__ == '__main__':
    rospy.init_node('ekf_node')
    ekf = EKF()
    rospy.spin()