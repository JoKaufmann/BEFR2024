#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped, PointStamped
import numpy as np

class UKF:
    def __init__(self):
        # Initialize measurement vector
        self.dcam = np.zeros(3)
        self.gt_drone = np.zeros(10)
        
        # Initialize state vector
        self.gt_car = np.zeros(10)
        self.state_initialized = False
        self.x = np.zeros(10) # TODO: Define State

        # Set initial timestamp
        self.prev_time = rospy.Time.now()

        # Initialize subscriber
        rospy.Subscriber("rrbot/fake_detection", PointStamped, self.depth_cam_callback)
        rospy.Subscriber("rrbot/position_groundtruth", Odometry, self.groundtruth_callback)
        rospy.Subscriber("car/position_groundtruth", Odometry, self.groundtruth_car_callback)
        
        # Initialize publisher
        self.pub = rospy.Publisher('car/pos_estimate', PointStamped, queue_size=10)
    
    def prediction_step(self, dt):
        pass

    def update_step(self):
        pass
    
    def groundtruth_callback(self, data):
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

        # init state    # TODO: Init state
        # if self.state_initialized is False:
        #     self.x[0:10] = self.gt[0:10]
        #     self.state_initialized = True

    def depth_cam_callback(self, data):
        self.dcam[0] = data.x   # u
        self.dcam[1] = data.y   # v
        self.dcam[2] = data.z   # depth

        self.update_step()

    def publish_data(self):
        pass

if __name__ == '__main__':
    rospy.init_node('ukf_node')
    ukf = UKF()
    rospy.spin()