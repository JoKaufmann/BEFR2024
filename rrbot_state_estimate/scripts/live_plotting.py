import rospy
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

len_deque = 10000
# Initialize lists to store data
state_estimation_time = deque(maxlen=len_deque)
state_estimate_pos_x = deque(maxlen=len_deque)
state_estimate_pos_y = deque(maxlen=len_deque)
state_estimate_pos_z = deque(maxlen=len_deque)
state_estimate_orientation_x = deque(maxlen=len_deque)
state_estimate_orientation_y = deque(maxlen=len_deque)
state_estimate_orientation_z = deque(maxlen=len_deque)

groundtruth_time = deque(maxlen=len_deque)
groundtruth_pos_x = deque(maxlen=len_deque)
groundtruth_pos_y = deque(maxlen=len_deque)
groundtruth_pos_z = deque(maxlen=len_deque)
groundtruth_orientation_x = deque(maxlen=len_deque)
groundtruth_orientation_y = deque(maxlen=len_deque)
groundtruth_orientation_z = deque(maxlen=len_deque)

# Define callback functions for ROS subscribers
def state_estimate_callback(data):
    state_estimation_time.append(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)
    state_estimate_pos_x.append(data.pose.pose.position.x)
    state_estimate_pos_y.append(data.pose.pose.position.y)
    state_estimate_pos_z.append(data.pose.pose.position.z)
    
    # get orientation in euler angles
    euler_x, euler_y, euler_z = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
    state_estimate_orientation_x.append(euler_x)
    state_estimate_orientation_y.append(euler_y)
    state_estimate_orientation_z.append(euler_z)

def groundtruth_callback(data):
    groundtruth_time.append(data.header.stamp.secs + data.header.stamp.nsecs * 1e-9)
    groundtruth_pos_x.append(data.pose.pose.position.x)
    groundtruth_pos_y.append(data.pose.pose.position.y)
    groundtruth_pos_z.append(data.pose.pose.position.z)

    # get orientation in euler angles
    euler_x, euler_y, euler_z = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
    groundtruth_orientation_x.append(euler_x)
    groundtruth_orientation_y.append(euler_y)
    groundtruth_orientation_z.append(euler_z)

# Initialize ROS node
rospy.init_node('live_plot_node', anonymous=True)

# Subscribe to the topics
rospy.Subscriber('/rrbot/state_estimate', Odometry, state_estimate_callback)
rospy.Subscriber('/rrbot/position_groundtruth', Odometry, groundtruth_callback)

# Set up live plots with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Position subplot
line1, = ax1.plot([], [], 'r-', label='pos_x_est', linewidth=2)
line2, = ax1.plot([], [], 'r-', label='pos_x_groundtruth', linewidth=2, linestyle='--')
line3, = ax1.plot([], [], 'm-', label='pos_y_est', linewidth=2)
line4, = ax1.plot([], [], 'm-', label='pos_y_groundtruth', linewidth=2, linestyle='--')
line5, = ax1.plot([], [], 'b-', label='pos_z_est', linewidth=2)
line6, = ax1.plot([], [], 'b-', label='pos_z_groundtruth', linewidth=2, linestyle='--')
ax1.set_ylim(-100, 100)
ax1.set_title('Position')
ax1.legend()

# Orientation subplot
line7, = ax2.plot([], [], 'r-', label='orientation_x_est', linewidth=2)
line8, = ax2.plot([], [], 'r-', label='orientation_x_groundtruth', linewidth=2, linestyle='--')
line9, = ax2.plot([], [], 'm-', label='orientation_y_est', linewidth=2)
line10, = ax2.plot([], [], 'm-', label='orientation_y_groundtruth', linewidth=2, linestyle='--')
line11, = ax2.plot([], [], 'b-', label='orientation_z_est', linewidth=2)
line12, = ax2.plot([], [], 'b-', label='orientation_z_groundtruth', linewidth=2, linestyle='--')
ax2.set_ylim(-90, 90)
ax2.set_title('Orientation')
ax2.legend()

def update(frame):
    ax1.set_xlim(state_estimation_time[0], state_estimation_time[-1])
    ax2.set_xlim(state_estimation_time[0], state_estimation_time[-1])
    
    
    line1.set_data(state_estimation_time    , state_estimate_pos_x)
    line2.set_data(groundtruth_time         , groundtruth_pos_x)
    line3.set_data(state_estimation_time    , state_estimate_pos_y)
    line4.set_data(groundtruth_time         , groundtruth_pos_y)
    line5.set_data(state_estimation_time    , state_estimate_pos_z)
    line6.set_data(groundtruth_time         , groundtruth_pos_z)
    
    line7.set_data(state_estimation_time    , state_estimate_orientation_x)
    line8.set_data(groundtruth_time         , groundtruth_orientation_x)
    line9.set_data(state_estimation_time    , state_estimate_orientation_y)
    line10.set_data(groundtruth_time        , groundtruth_orientation_y)
    line11.set_data(state_estimation_time   , state_estimate_orientation_z)
    line12.set_data(groundtruth_time        , groundtruth_orientation_z)
    
    return line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12

ani = FuncAnimation(fig, update, interval=100)
plt.tight_layout()
plt.show()

# Keep the script running
rospy.spin()