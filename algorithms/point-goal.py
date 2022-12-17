import numpy as np
from turtlebot3_msgs.srv import Dqn
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
import sys
import time


# n = 3, m = 2

class optAgent(Node):
    def __init__(self, stage):
        super().__init__('optAgent')

        self.last_pose_x = 2.0
        self.last_pose_y = 2.0
        self.last_pose_theta = 0.0

        qos = QoSProfile(depth=10)

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            qos)

        self.dqn_com_client = self.create_client(Dqn, 'dqn_com')

        self.process()

    def odom_callback(self, msg):
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)


    def getU():
        return 1

    def process(self):
        K = []
        u = np.array([-0.9, -0.3])
        u_k = np.array(u)

        phi = None
        init = True
        Q = np.identity(3)
        R = np.identity(2)
        H = np.random.rand(5,5)
        flag = 0

        Y = None
        tt = time.time()
        x_k = np.array([self.last_pose_x, self.last_pose_y, self.last_pose_theta])
        next_state = []
        # r_k = np.array([self.last_pose_x, 0.0, 0.0])
        # x_k = np.hstack((x_k, r_k))
        for episode in range(1, 10000):
            print(f'episode ---- {episode}')
            # r_k = np.array([self.last_pose_x, 0.0, 0.0])
            req = Dqn.Request()
            t = time.time() - tt
            # u = [1.0, 1.0]
            z_k = np.hstack((x_k, u_k))
            e = np.array([-1 * (6*np.sin(7*t)+np.cos(5*t)+6*np.cos(11*t))/30, -1 * (2*np.sin(7*t)+7*np.cos(5*t)-2*np.cos(11*t))/30])
            
            if(flag == 0):
                u_k = u_k + e
            # u = u+e
            # print(int(action))
            req.action = u_k.tolist()
            req.init = init
            while not self.dqn_com_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')

            future = self.dqn_com_client.call_async(req)
            while rclpy.ok():
                rclpy.spin_once(self)
                if future.done():
                    if future.result() is not None:
                        # Next state and reward
                        next_state = future.result().state
                        done = future.result().done
                        init = False
                    else:
                        self.get_logger().error(
                            'Exception while calling service: {0}'.format(future.exception()))
                    break
            
            next_state = np.array(next_state)
            # next_state[2] = np.arctan(np.sin(next_state[2])/ np.cos(next_state[2]))
            # next_state = np.hstack((next_state, r_k))
            # print(z_k - next_state.T)
            z = np.hstack((next_state, u_k))
            # print(z_k, '-----', z)
            # print(next_state)
            # print(u_k)
            # print((z_k - z))
            if episode == 1:
                phi = np.kron((z_k - z), (z_k - z))
                # print(u_k)
                Y = (x_k.T @ Q @ x_k) + (u_k.T @ R @ u_k)
            else:
                # print((x_k.T @ Q @ x_k) + (u_k.T @ R @ u_k))
                phi = np.c_[phi, np.kron((z_k - z), (z_k - z))]
                Y = np.c_[Y, (x_k.T @ Q @ x_k) + (u_k.T @ R @ u_k)]

            if episode > 15:

                # print(phi)
                H_new, _,rr, _ = np.linalg.lstsq(phi.T, Y.T, rcond=None)
                print(f'rank - {rr}')
                phi = np.delete(phi, 0, 1)
                Y = np.delete(Y, 0, 1)
                print(H_new.shape)
                print(np.linalg.norm(H_new-H.reshape(25,1)))
                if(np.linalg.norm(H_new-H.reshape(25,1)) < 500 and flag == 0):
                    flag = 1
                    print('---------------converged--------------')
                H_new = H_new.reshape(5,5)
                # print(H)
                H_uu = H_new[-2:, -2:]
                H_ux = H_new[-2:,:3]
                # print(H_uu)
                # print(H_ux)
                u = -1 * (np.linalg.inv(H_uu) @ H_ux @ x_k)
                H = H_new
                # print('theta---', np.degrees(self.last_pose_theta))
                # exit()
            x_k = next_state

            time.sleep(0.05)

    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quat = [x, y, z, w]
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=sys.argv[1]):
    rclpy.init(args=list(args))
    opt_agent = optAgent(args)
    rclpy.spin(opt_agent)

    opt_agent.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
