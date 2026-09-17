# Model Predictive Control
# Author: Yuansj
# Time : 2022/02/24
import numpy as np
import time
import math
import casadi as ca
from cvxopt import matrix, solvers

# class MPCSOLVE:
#     """
#     MPC solver used for merge scenario. Highway-env config "ContinuousAction" action
#     """

#     def __init__(self, rollout_space:int):
#         self.N = rollout_space  # rollout space
#         self.dt = 0.1  # time step
#         self.state_n = 4 # features
#         self.control_n = 2 # control num
#         self.Length = 5  # m
#         self.max_a = 5 # m/s^2
#         self.max_steering_angle = np.pi / 180 * 5 # rad

#         self.A_cell = None
#         self.B_cell = None
#         self.Esk = None

#     def compute_matrix(self, car_info, ref_info) -> None:
#         '''
#         this function used to update matirx at the beginning of each control period
#         input: k step velocity, heading, steering angle
#         return: matrix p, matrix q, matrix g, matrix h, matrix a, matrix b
#         '''
#         position_k = car_info['position']
#         velocity_k = car_info['velocity']
#         steering_angle = car_info['steering_angle']
#         heading_k = car_info['heading']
#         E_s_k = np.mat([[position_k[0] - ref_info['p_ref'][0]],
#                         [position_k[1] - ref_info['p_ref'][1]],
#                         [velocity_k - ref_info['v_ref']],
#                         [heading_k - ref_info['h_ref']]])
        
#         v_k = velocity_k  # k step velocity
#         beta = np.arctan(1 / 2 * np.tan(steering_angle))

#         A = np.mat([[0, 0, np.cos(heading_k + beta), -v_k * np.sin(heading_k + beta)],
#                     [0, 0, np.sin(heading_k + beta), v_k * np.cos(heading_k + beta)],
#                     [0, 0, 0, 0],
#                     [0, 0, np.sin(beta) / (self.Length / 2), 0]])

#         B = np.mat([[0, -v_k * np.sin(heading_k + beta) * (2 / (1 + 3 * math.pow(np.cos(steering_angle), 2)))],
#                     [0, v_k * np.cos(heading_k + beta) * (2 / (1 + 3 * math.pow(np.cos(steering_angle), 2)))],
#                     [1, 0],
#                     [0, v_k * np.cos(beta) * 2 / (self.Length / 2 * (1 + 3 * math.pow(np.cos(steering_angle), 2)))]])

#         I = np.mat(np.eye(4))

#         Akt = A * self.dt + I
#         Bkt = B * self.dt

#         # compute P and Q matrix
#         current_A = Akt
#         A_cell = current_A
#         for i in range(self.N - 1):
#             current_A = np.dot(current_A, Akt)
#             A_cell = np.vstack((A_cell, current_A))

#         B_cell_column = Bkt
#         for j in range(self.N):
#             current_b = Bkt
#             if j == 0:
#                 for k in range(self.N - 1):
#                     current_b = np.dot(Akt, current_b)
#                     B_cell_column = np.vstack((B_cell_column , current_b))
#                 B_cell = B_cell_column
#             else:
#                 B_cell_column = np.zeros([4*j,2])
#                 for k in range(self.N - j):
#                     B_cell_column = np.vstack((B_cell_column , current_b))
#                     current_b = np.dot(Akt, current_b)
#                 B_cell = np.hstack((B_cell, B_cell_column))
        
#         A_cell = np.mat(A_cell)
#         B_cell = np.mat(B_cell)
#         Q = np.mat(np.eye(self.N * self.state_n))
#         R = np.mat(np.eye(self.N * self.control_n))

#         P = 2*B_cell.transpose()*Q*B_cell + R
#         Q = 2*(E_s_k.transpose()*A_cell.transpose()*Q*B_cell).transpose()

#         # compute constraint matrix
#         g_matrix_1 = np.eye(self.N * self.control_n)
#         h_matrix_1 = np.zeros([self.N * self.control_n, 1])
#         for i in range(len(h_matrix_1)):
#             if i % 2 == 0:
#                 h_matrix_1[i][0] = self.max_a
#             elif i % 2 == 1:
#                 h_matrix_1[i][0] = self.max_steering_angle
        
#         h_matrix_2 = h_matrix_1
#         g_matrix_2 = - g_matrix_1

#         G = np.mat(np.vstack((g_matrix_1, g_matrix_2)))
#         H = np.mat(np.vstack((h_matrix_1, h_matrix_2)))

#         self.A_cell = A_cell
#         self.B_cell = B_cell
#         self.Esk = E_s_k

#         return P, Q, G, H

#     def solve(self, P, Q, G, H, A=[], B=[]):
#         '''
#         solve the QP
#         input: 
#         P, Q is the matrix of objective function
#         G, H is the inequality constraint equation coefficient
#         A, B is the equality constraint equation coefficient
#         return: optimal control u
#         '''
#         p = matrix(P)
#         q = matrix(Q)
#         g = matrix(G)
#         h = matrix(H)
#         a = matrix(A)
#         b = matrix(B)

#         settings = dict(show_progress=False)
#         sol = solvers.qp(p,q,g,h,options=settings)
#         result = sol['x']

#         return result



# class MPCSOLVE:
#     def __init__(self, rollout_space: int):
#         self.N = rollout_space  # prediction horizon
#         self.dt = 0.1  # time step
#         self.state_n = 4  # number of state variables
#         self.control_n = 2  # number of control variables
#         self.Length = 5  # vehicle length (m)
#         self.max_a = 5  # maximum acceleration (m/s^2)
#         self.max_steering_angle = np.pi / 180 * 5  # maximum steering angle (rad)
        
#     def compute_mpc(self, car_info, ref_info):
#         # Define CasADi symbols for states and controls
#         X = ca.SX.sym('X', self.state_n, self.N + 1)  # states (x, y, v, heading)
#         U = ca.SX.sym('U', self.control_n, self.N)  # controls (acceleration, steering)
        
#         # Cost function
#         cost = 0
#         for k in range(self.N):
#             # State error
#             e_x = X[0, k] - ref_info['p_ref'][0]
#             e_y = X[1, k] - ref_info['p_ref'][1]
#             e_v = X[2, k] - ref_info['v_ref']
#             e_heading = X[3, k] - ref_info['h_ref']
#             cost += e_x**2 + e_y**2 + e_v**2 + e_heading**2
#             # Control effort cost (penalizing large controls)
#             cost += U[0, k]**2 + U[1, k]**2

#         # System dynamics constraints
#         constraints = []
#         for k in range(self.N):
#             x_k = X[:, k]  # current state
#             u_k = U[:, k]  # current control

#             # Compute next state using discrete dynamics
#             v_k = x_k[2]
#             steering_angle = u_k[1]
#             beta = ca.atan(1 / 2 * ca.tan(steering_angle))  # slip angle

#             # Vehicle dynamics model (discretized using Euler's method)
#             A_k = ca.SX([
#                 [0, 0, ca.cos(x_k[3] + beta), -v_k * ca.sin(x_k[3] + beta)],
#                 [0, 0, ca.sin(x_k[3] + beta), v_k * ca.cos(x_k[3] + beta)],
#                 [0, 0, 0, 0],
#                 [0, 0, ca.sin(beta) / (self.Length / 2), 0]
#             ])
            
#             B_k = ca.SX([
#                 [0, -v_k * ca.sin(x_k[3] + beta) * (2 / (1 + 3 * ca.cos(steering_angle)**2))],
#                 [0, v_k * ca.cos(x_k[3] + beta) * (2 / (1 + 3 * ca.cos(steering_angle)**2))],
#                 [1, 0],
#                 [0, v_k * ca.cos(beta) * 2 / (self.Length / 2 * (1 + 3 * ca.cos(steering_angle)**2))]
#             ])
            
#             # Next state equation: x_{k+1} = x_k + (A_k * x_k + B_k * u_k) * dt
#             x_next = x_k + (ca.mtimes(A_k, x_k) + ca.mtimes(B_k, u_k)) * self.dt
#             constraints.append(X[:, k+1] - x_next)

#         # Control input constraints
#         for k in range(self.N):
#             constraints.append(U[0, k] - self.max_a)   # max acceleration constraint
#             constraints.append(U[1, k] - self.max_steering_angle)  # max steering angle constraint
#             constraints.append(-U[0, k])  # min acceleration
#             constraints.append(-U[1, k])  # min steering angle

#         # Setup NLP problem for IPOPT
#         nlp = {'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
#                'f': cost,
#                'g': ca.vertcat(*constraints)}

#         # Solver options
#         opts = {'ipopt.print_level': 0, 'print_time': 0}
#         solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

#         # Initial conditions and bounds
#         x0 = np.zeros((self.state_n * (self.N + 1) + self.control_n * self.N, 1))
#         lbx = -np.inf * np.ones_like(x0)
#         ubx = np.inf * np.ones_like(x0)
#         lbg = np.zeros((self.state_n * (self.N) + self.control_n * self.N, 1))
#         ubg = np.zeros((self.state_n * (self.N) + self.control_n * self.N, 1))

#         # Solve the optimization problem
#         solution = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
#         opt_solution = solution['x']

#         return opt_solution

class MPCSOLVE:
    """
    MPC solver used for merge scenario. Highway-env config "ContinuousAction" action
    """

    def __init__(self, rollout_space: int):
        self.N = rollout_space  # rollout space
        self.dt = 0.1  # time step
        self.state_n = 4  # features
        self.control_n = 2  # control num
        self.Length = 5  # m
        self.max_a = 5  # m/s^2
        self.max_ax = 5
        self.min_ax = -5
        self.max_ay = 2
        self.min_ay = -2
        self.max_steering_angle = np.pi / 180 * 5
        self.min_steering_angle = -np.pi / 180 * 5
        self.delta_max_ax = 0.25
        self.delta_min_ax = -0.25
        self.delta_max_ay = 0.1
        self.delta_min_ay = -0.1
        self.delta_max_steering_angle = np.pi / 180 * 1
        self.delta_min_steering_angle = -np.pi / 180 * 1
        self.beta = None
        self.A_cell = None
        self.B_cell = None
        self.Esk = None
        self.future_ref = None
        self.road_boundary = None

    def compute_matrix(self, car_info, ref_info, future_ref, road_boundary, sur_cars_info) -> None:
        '''
        this function used to update matrix at the beginning of each control period
        input: k step velocity, heading, steering angle
        return: matrix p, matrix q, matrix g, matrix h, matrix a, matrix b
        '''

        self.future_ref = future_ref
        self.sur_cars_info = sur_cars_info
        self.road_boundary = road_boundary

        position_k = car_info['position']
        velocity_k = car_info['velocity']
        steering_angle = car_info['steering_angle']
        heading_k = car_info['heading']
        E_s_k = np.mat([[position_k[0] - ref_info['p_ref'][0]],
                         [position_k[1] - ref_info['p_ref'][1]],
                         [velocity_k - ref_info['v_ref']],
                         [heading_k - ref_info['h_ref']]])

        v_k = velocity_k  # k step velocity
        self.beta = np.arctan(1 / 2 * np.tan(steering_angle))

        A = np.mat([[0, 0, np.cos(heading_k + self.beta), -v_k * np.sin(heading_k + self.beta)],
                     [0, 0, np.sin(heading_k + self.beta), v_k * np.cos(heading_k + self.beta)],
                     [0, 0, 0, 0],
                     [0, 0, np.sin(self.beta) / (self.Length / 2), 0]])

        B = np.mat([[0, -v_k * np.sin(heading_k + self.beta) * (2 / (1 + 3 * np.cos(steering_angle)**2))],
                     [0, v_k * np.cos(heading_k + self.beta) * (2 / (1 + 3 * np.cos(steering_angle)**2))],
                     [1, 0],
                     [0, v_k * np.cos(self.beta) * 2 / (self.Length / 2 * (1 + 3 * np.cos(steering_angle)**2))]])

        I = np.mat(np.eye(4))

        Akt = A * self.dt + I
        Bkt = B * self.dt

        # compute P and Q matrix
        current_A = Akt
        A_cell = current_A
        for i in range(self.N - 1):
            current_A = np.dot(current_A, Akt)
            A_cell = np.vstack((A_cell, current_A))

        B_cell_column = Bkt
        for j in range(self.N):
            current_b = Bkt
            if j == 0:
                for k in range(self.N - 1):
                    current_b = np.dot(Akt, current_b)
                    B_cell_column = np.vstack((B_cell_column, current_b))
                B_cell = B_cell_column
            else:
                B_cell_column = np.zeros([4*j, 2])
                for k in range(self.N - j):
                    B_cell_column = np.vstack((B_cell_column, current_b))
                    current_b = np.dot(Akt, current_b)
                B_cell = np.hstack((B_cell, B_cell_column))

        A_cell = np.mat(A_cell)
        B_cell = np.mat(B_cell)
        Q = np.mat(np.eye(self.N * self.state_n))
        R = np.mat(np.eye(self.N * self.control_n))

        P = 2 * B_cell.transpose() * Q * B_cell + R
        Q = 2 * (E_s_k.transpose() * A_cell.transpose() * Q * B_cell).transpose()

        # compute constraint matrix
        g_matrix_1 = np.eye(self.N * self.control_n)
        h_matrix_1 = np.zeros([self.N * self.control_n, 1])
        for i in range(len(h_matrix_1)):
            if i % 2 == 0:
                h_matrix_1[i][0] = self.max_a
            elif i % 2 == 1:
                h_matrix_1[i][0] = self.max_steering_angle

        h_matrix_2 = h_matrix_1
        g_matrix_2 = -g_matrix_1

        G = np.mat(np.vstack((g_matrix_1, g_matrix_2)))
        H = np.mat(np.vstack((h_matrix_1, h_matrix_2)))

        self.A_cell = A_cell
        self.B_cell = B_cell
        self.Esk = E_s_k

        return P, Q, G, H

    def solve(self, P, Q, G, H, A=[], B=[]):
        '''
        solve the QP using CasADi and IPOPT
        input: 
        P, Q is the matrix of objective function
        G, H is the inequality constraint equation coefficient
        A, B is the equality constraint equation coefficient
        return: optimal control u
        '''

        # Problem dimensions
        control_dim = self.N * self.control_n

        # Define CasADi variables (optimization variables: control actions)
        U = ca.SX.sym('U', control_dim)

        # Define the objective function (quadratic form)
        obj = 0.5 * ca.mtimes([U.T, P, U]) + ca.mtimes([Q.T, U])

        constraints = []
        lbx = []
        ubx = []
        lbg = []
        ubg = []

        # 控制量约束
        for k in range(self.N):
            constraints.append(U[2*k] * ca.cos(self.beta))  # 纵向加速度
            constraints.append(U[2*k] * ca.sin(self.beta))  # 横向加速度
            constraints.append(U[2*k+1])  # 方向盘转角
            lbg += [self.min_ax, self.min_ay, self.min_steering_angle]
            ubg += [self.max_ax, self.max_ay, self.max_steering_angle]
        # 控制量变化量约束
        for k in range(1, self.N):
            constraints.append((U[2*k] - U[(k-1)*2]) * ca.cos(self.beta))  # 纵向加速度变化量
            constraints.append((U[2*k] - U[(k-1)*2]) * ca.sin(self.beta))  # 横向加速度变化量
            constraints.append(U[2*k+1] - U[(k-1)*2+1])  # 方向盘转角变化量
            lbg += [self.delta_min_ax, self.delta_min_ay, self.delta_min_steering_angle]
            ubg += [self.delta_max_ax, self.delta_max_ay, self.delta_max_steering_angle]
        
        # 状态量获取
        position_x = []
        position_y = []
        speed = []
        heading = []
        A_cell_SX = ca.SX(self.A_cell)
        B_cell_SX = ca.SX(self.B_cell)
        Esk_SX = ca.SX(self.Esk)
        future_ref = ca.SX(self.future_ref)
        Y = A_cell_SX @ Esk_SX + B_cell_SX @ U
        future_state = future_ref + Y[0:self.N * self.state_n, 0]  # 形状(40,1)
        for i in range(future_state.shape[0] // 4):
            position_x.append(future_state[i*4, 0])
            position_y.append(future_state[i*4+1,0])
            speed.append(future_state[i*4+2, 0])
            heading.append(future_state[i*4+3, 0])

        # 速度约束
        for i in range(10):
            constraints.append(speed[i])
            lbg += [12]
            ubg += [30]

        # # 边界约束
        # for i in range(10):
        #     constraints.append(position_x[i])
        #     constraints.append(position_y[i])
        #     lbg += [self.road_boundary['min_x'], self.road_boundary['min_y']]
        #     ubg += [self.road_boundary['max_x'], self.road_boundary['max_y']]

        # # 安全约束
        # safe_distance = 2.5  # 安全距离
        # cbf_rate = 0.5  # CBF增益
        # obs_num = len(self.sur_cars_info)
        # obs_position_x = []
        # obs_position_y = []
        # obs_speed = []
        # obs_heading = []
        # for sur_car in self.sur_cars_info:
        #     obs_position_x.append(ca.SX(sur_car.position[0]))
        #     obs_position_y.append(ca.SX(sur_car.position[1]))
        #     obs_speed.append(ca.SX(sur_car.speed))
        #     obs_heading.append(ca.SX(sur_car.heading))
        # for i in range(self.N-1):
        #     for j in range(obs_num):
        #         car_pos = ca.vertcat(position_x[i], position_y[i])
        #         car_pos_next = ca.vertcat(position_x[i+1], position_y[i+1])
        #         obs_pos = ca.vertcat(obs_position_x[j], obs_position_y[j])
        #         distance = ca.norm_2(car_pos - obs_pos)
        #         distance_next = ca.norm_2(car_pos_next - obs_pos)
        #         B = distance - safe_distance
        #         B_next = distance_next - safe_distance
        #         # 添加CBF约束
        #         constraints.append(B_next - B + cbf_rate*B)
        #         lbg += [0]
        #         ubg += [ca.inf]


        # Setup the optimization problem
        constraints = ca.vertcat(*constraints)
        nlp = {'x': U, 'f': obj, 'g': constraints}

        # IPOPT options (optional: tune for performance)
        opts_setting = {
            'ipopt.max_iter': 300,
            'ipopt.print_level': 0,
            'print_time': False,
            'print_in': False,
            'print_out': False,
        }

        # Create IPOPT solver
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts_setting)

        # Solve the optimization problem
        sol = solver(lbg=lbg, ubg=ubg)

        # Extract the solution
        result = sol['x'].full()
        
        # print(result)

        return result
