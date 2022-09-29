#!/usr/bin/env python3
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from control import Turtlebot3Path
import math

class Turtlebot3PositionControl(Node):

    def __init__(self):
        super().__init__('position_control')

        self.odom = Odometry()
        self.curr_pos_x = 0.0
        self.curr_pos_y = 0.0
        self.goal_pos_x = 1.0
        self.goal_pos_y = 0.0
        self.step = 0

        self.init_odom_state = False  # To get the initial pose at the beginning

        qos = QoSProfile(depth=10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            qos)

        self.update_timer = self.create_timer(0.010, self.update_callback)  # unit: s

        self.get_logger().info("Turtlebot3 position control node has been initialised.")


    def odom_callback(self, msg):
        self.curr_pos_x = msg.pose.pose.position.x
        self.curr_pos_y = msg.pose.pose.position.y
        self.init_odom_state = True

    def update_callback(self):
        if self.init_odom_state is True:
            self.generate_path()

    def generate_path(self):
        twist = Twist()

        linear_velocity = 0.1
        
        if self.step == 0:
            distance = math.sqrt(
                        (self.goal_pos_x - self.curr_pos_x)**2 +
                        (self.goal_pos_y - self.curr_pos_y)**2)
            
            twist, self.step = Turtlebot3Path.go_straight(distance, linear_velocity, self.step)

        self.cmd_vel_pub.publish(twist)

