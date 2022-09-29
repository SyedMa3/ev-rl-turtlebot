#!/usr/bin/env python3
import math
from geometry_msgs.msg import Twist

class Turtlebot3Path():
    
    def go_straight(distance, linear_velocity, step):
        twist = Twist()

        if distance > 0.01:
            twist.linear.x = linear_velocity
        else:
            step += 1

        return twist, step