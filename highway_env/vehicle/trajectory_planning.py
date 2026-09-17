import numpy as np
import time
import os
import math
import casadi as ca

######################### 五次多项式 ################################
def polynomial_5order(t1, t2, node1_state, node2_state):
    # 边界点条件
    t_init = t1
    t_final = t2

    init_state = node1_state
    final_state = node2_state

    x_init, y_init, vx_init, vy_init, ax_init, ay_init = init_state
    x_final, y_final, vx_final, vy_final, ax_final, ay_final = final_state

    # 参数向量
    X_para = np.array([x_init, vx_init, ax_init, x_final, vx_final, ax_final])
    Y_para = np.array([y_init, vy_init, ay_init, y_final, vy_final, ay_final])

    # 五次多项式求解的变量矩阵
    T_init = np.array([t_init**i for i in range(6)])  # 时间变量向量
    DT_init = np.array([0, 1, 2*t_init, 3*t_init**2, 4*t_init**3, 5*t_init**4])  # 时间一阶导变量向量
    DDT_init = np.array([0, 0, 2, 6*t_init, 12*t_init**2, 20*t_init**3])  # 时间二阶导变量向量

    T_final = np.array([t_final**i for i in range(6)])  # 时间变量向量
    DT_final = np.array([0, 1, 2*t_final, 3*t_final**2, 4*t_final**3, 5*t_final**4])  # 时间一阶导变量向量
    DDT_final = np.array([0, 0, 2, 6*t_final, 12*t_final**2, 20*t_final**3])  # 时间二阶导变量向量

    T_solving_matrix = np.vstack((T_init, DT_init, DDT_init, T_final, DT_final, DDT_final))

    # 横纵向基于五次多项式轨迹的系数求解
    P = np.linalg.solve(T_solving_matrix, X_para)
    Q = np.linalg.solve(T_solving_matrix, Y_para)

    # 降幂输出
    P = P[::-1]
    Q = Q[::-1]

    return P, Q

######################### 多边形膨胀 ################################
def inflate_polygon(vertices, R):
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    new_vertices = []
    num_vertices = len(vertices)
    # 遍历每个顶点
    for i in range(num_vertices):
        # 获取当前顶点及其前一个和下一个顶点
        prev_vertex = vertices[i - 1]
        curr_vertex = vertices[i]
        next_vertex = vertices[(i + 1) % num_vertices]
        # 计算与前一个点形成的边和与下一个点形成的边
        edge1 = np.array(curr_vertex) - np.array(prev_vertex)
        edge2 = np.array(next_vertex) - np.array(curr_vertex)
        # 计算边的法向量 (顺时针旋转90度)
        normal1 = np.array([-edge1[1], edge1[0]])
        normal2 = np.array([-edge2[1], edge2[0]])
        # 归一化法向量
        normal1 = normalize(normal1)
        normal2 = normalize(normal2)
        # 计算当前顶点的平均法向量
        avg_normal = normalize(normal1 + normal2)
        # 沿法向量方向移动当前顶点
        inflated_vertex = np.array(curr_vertex) + R * avg_normal
        new_vertices.append(inflated_vertex)
    return np.array(new_vertices)

##################### 碰撞检测 #################################
def collision_check(obb_a, obb_b):
    def get_axes(obb):
        axes = []
        for i in range(4):
            p1 = obb[i]
            p2 = obb[(i + 1) % 4]
            edge = np.array(p2) - np.array(p1)
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal / np.linalg.norm(normal))
        return axes
    def project_obb(obb, axis):
        projections = [np.dot(vertex, axis) for vertex in obb]
        return min(projections), max(projections)
    axes_a = get_axes(obb_a)
    axes_b = get_axes(obb_b)
    for axis in axes_a + axes_b:
        min_a, max_a = project_obb(obb_a, axis)
        min_b, max_b = project_obb(obb_b, axis)
        if max_a < min_b or max_b < min_a:
            return False
    return True
    
##################### 行车走廊 #################################
def generate_driving_corridor(center_point, ego_car_info, sur_cars_info):
    length_step1 = 1
    length_step2 = 0.1
    length_max = 5.0
    expand_length = [0, 0, 0, 0]  # 左、上、右、下方向的拓展总长度
    direction = [set([0, 1, 2, 3]), set([0, 1, 2, 3])]  # 两次扩张方向集合
    center_x = center_point[0]
    center_y = center_point[1]
    length_step = [length_step1, length_step2]
    box_x_min, box_x_max = center_x, center_x
    box_y_min, box_y_max = center_y, center_y
    box = []
    temp_box = []
    # 障碍物多边形生成
    obstacles = []
    inflated_radius = 0.5
    for sur_car in sur_cars_info:
        obs_polygon = sur_car.polygon()
        inflated_obs_polygon = inflate_polygon(obs_polygon, inflated_radius)
        obstacles.append(inflated_obs_polygon)
    # 两阶段行车走廊局部矩形生成
    collision_flag = 0
    for i in range(2):
        while direction[i]:
            for index in list(direction[i]):
                # 矩形拓展
                if  index == 0:   # 左
                    expand_length[0] += length_step[i]
                    box_x_min = center_x - expand_length[0]
                elif index == 1:  # 上
                    expand_length[1] += length_step[i]
                    box_y_max = center_y + expand_length[1]
                elif index == 2:  # 右
                    expand_length[2] += length_step[i]
                    box_x_max = center_x + expand_length[2]
                elif index == 3:  # 下
                    expand_length[3] += length_step[i]
                    box_y_min = center_y - expand_length[3]
                # 碰撞检测
                temp_box = [[box_x_min, box_y_min],[box_x_min, box_y_max],[box_x_max, box_y_max],[box_x_max, box_y_min]]
                if (box_x_min != box_x_max) and (box_y_min != box_y_max):
                    for obs in obstacles:
                        collision_flag =  collision_check(temp_box, obs)
                        if collision_flag: 
                            break
                else:
                    collision_flag = 0
                # 扩展回撤
                if collision_flag or expand_length[index] >= length_max:
                    direction[i].remove(index)
                    if  index == 0:
                        expand_length[0] -= length_step[i]
                        box_x_min = center_x - expand_length[0]
                    elif index == 1:
                        expand_length[1] -= length_step[i]
                        box_y_max = center_y + expand_length[1]
                    elif index == 2:
                        expand_length[2] -= length_step[i]
                        box_x_max = center_x + expand_length[2]
                    elif index == 3:
                        expand_length[3] -= length_step[i]
                        box_y_min = center_y - expand_length[3]
    # 最终的局部矩形输出
    box = {
        'min_x': box_x_min,
        'max_x': box_x_max,
        'min_y': box_y_min,
        'max_y': box_y_max,
    }
    return box

##################### IPOPT求解器 ################################
class MPCSOLVE_IPOPT:
    def __init__(self, rollout_space: int):
        self.N = rollout_space
        self.dt = 0.1
        self.state_n = 4  # 状态的维度 [x, y, v, heading]
        self.control_n = 2  # 控制输入的维度 [加速度, 转向角]
        self.Length = 5  # 车辆长度
        self.max_ax = 5
        self.min_ax = -5
        self.max_ay = 5
        self.min_ay = -5
        self.max_steering_angle = np.pi / 180 * 30
        self.min_steering_angle = -np.pi / 180 * 30
        self.delta_max_ax = 2.5
        self.delta_min_ax = -2.5
        self.delta_max_ay = 2.5
        self.delta_min_ay = -2.5
        self.delta_max_steering_angle = np.pi / 180 * 5
        self.delta_min_steering_angle = -np.pi / 180 * 5
        self.w1_diag = [1, 1, 0.1, 1]
        self.w2_diag = [1, 1]
        self.w3_diag = [1, 1]

    def solve_nmpc(self, traj_init, ego_car_info, sur_cars_info, road_boundary):
        # # 车辆初始状态
        # x_init = traj_init['x'][0]
        # y_init = traj_init['y'][0]
        # v_init = traj_init['v'][0]
        # h_init = traj_init['fai'][0]
        # state_init = [x_init, y_init, v_init, h_init]
        # state_init = ca.vertcat(*state_init)
        # state_init = ca.MX(state_init)

        # # 车辆终止状态
        # x_final = traj_init['x'][self.N]
        # y_final = traj_init['y'][self.N]
        # v_final = traj_init['v'][self.N]
        # h_final = traj_init['fai'][self.N]
        # state_final = [x_final, y_final, v_final, h_final]
        # state_final = ca.vertcat(*state_final)
        # state_final = ca.MX(state_final)

        # 车辆初始状态
        x_init = ego_car_info['position'][0]
        y_init = ego_car_info['position'][1]
        v_init = ego_car_info['speed']
        h_init = ego_car_info['heading']
        state_init = [x_init, y_init, v_init, h_init]
        state_init = ca.vertcat(*state_init)
        state_init = ca.MX(state_init)

        # 车辆参考状态
        x_ref = ca.horzcat(*traj_init['x'])
        y_ref = ca.horzcat(*traj_init['y'])
        v_ref = ca.horzcat(*traj_init['v'])
        h_ref = ca.horzcat(*traj_init['fai'])
        state_ref = ca.vertcat(x_ref, y_ref, v_ref, h_ref)
        state_ref = ca.MX(state_ref)

        # 动力学方程中的A和B矩阵
        steering_angle = ego_car_info['steering']
        beta = np.arctan(1 / 2 * np.tan(steering_angle))
        A = ca.vertcat(ca.horzcat(0, 0, ca.cos(h_init + beta), -v_init * ca.sin(h_init + beta)),
                       ca.horzcat(0, 0, ca.sin(h_init + beta), v_init * ca.cos(h_init + beta)),
                       ca.horzcat(0, 0, 0, 0),
                       ca.horzcat(0, 0, ca.sin(beta) / (self.Length / 2), 0))

        B = ca.vertcat(ca.horzcat(0, -v_init * ca.sin(h_init + beta) * (2 / (1 + 3 * ca.cos(steering_angle)**2))),
                       ca.horzcat(0, v_init * ca.cos(h_init + beta) * (2 / (1 + 3 * ca.cos(steering_angle)**2))),
                       ca.horzcat(1, 0),
                       ca.horzcat(0, v_init * ca.cos(beta) * 2 / (self.Length / 2 * (1 + 3 * ca.cos(steering_angle)**2))))

        # 定义优化变量
        X = ca.MX.sym('X', self.state_n, self.N+1)    # 状态量
        U = ca.MX.sym('U', self.control_n, self.N)    # 控制量
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # 定义目标函数和约束条件
        objective = 0
        constraints = []
        lbx = []
        ubx = []
        lbg = []
        ubg = []
        w1_diag = ca.diag(ca.MX(self.w1_diag))
        w2_diag = ca.diag(ca.MX(self.w2_diag))
        w3_diag = ca.diag(ca.MX(self.w3_diag))
        # 起始状态约束
        constraints.append(X[:, 0] - state_init)  
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]
        # # 终止状态约束
        # constraints.append(X[:, self.N] - state_final)
        # lbg += [0, 0, 0, 0]
        # ubg += [0, 0, 0, 0]
        # 运动学约束
        for k in range(self.N):
            X_k = X[:, k]
            U_k = U[:, k]
            X_next = X[:, k+1]
            X_next_pred = X_k + self.dt * (A @ X_k + B @ U_k)  # 欧拉法预测下一个状态
            constraints.append(X_next - X_next_pred)
            lbg += [0, 0, 0, 0]
            ubg += [0, 0, 0, 0]
        # 控制量约束
        for k in range(self.N):
            constraints.append(U[0, k] * ca.cos(beta))  # 纵向加速度
            constraints.append(U[0, k] * ca.sin(beta))  # 横向加速度
            constraints.append(U[1, k])  # 方向盘转角
            lbg += [self.min_ax, self.min_ay, self.min_steering_angle]
            ubg += [self.max_ax, self.max_ay, self.max_steering_angle]
        # # 控制量变化量约束
        # for k in range(1, self.N):
        #     constraints.append((U[0, k] - U[0, k]) * ca.cos(beta))  # 纵向加速度变化量
        #     constraints.append((U[0, k] - U[0, k]) * ca.sin(beta))  # 横向加速度变化量
        #     constraints.append(U[1, k] - U[1, k])  # 方向盘转角变化量
        #     lbg += [self.delta_min_ax, self.delta_min_ay, self.delta_min_steering_angle]
        #     ubg += [self.delta_max_ax, self.delta_max_ay, self.delta_max_steering_angle]
        # # 状态量约束
        # for k in range(0, self.N+1):
        #     constraints.append(X[2, k] * ca.cos(beta + X[3, k]))
        #     constraints.append(X[2, k] * ca.sin(beta + X[3, k]))
        #     lbg += [0, 0]
        #     ubg += [100, 100]
        # # 边界约束
        # for k in range(1, self.N):
        #     constraints.append(X[0, k])
        #     constraints.append(X[1, k])
        #     lbg += [road_boundary['min_x'], road_boundary['min_y']]
        #     ubg += [road_boundary['max_x'], road_boundary['max_y']]
        # # 安全约束
        # center_points = [[traj_init['x'][k], traj_init['y'][k]] for k in range(1, self.N)]  # 起始点与终止点不算
        # boxes = []
        # for center_point in center_points:
        #     box = generate_driving_corridor(center_point, ego_car_info, sur_cars_info)
        #     boxes.append(box)
        # for k in range(1, self.N):
        #     constraints.append(X[0, k])
        #     constraints.append(X[1, k])
        #     lbg += [boxes[k-1]['min_x'], boxes[k-1]['min_y']]
        #     ubg += [boxes[k-1]['max_x'], boxes[k-1]['max_y']]
        # 转换约束为向量形式
        constraints = ca.vertcat(*constraints)

        # 目标函数
        for k in range(self.N):
            X_k = X[:, k]
            U_k = U[:, k]
            state_ref_k = state_ref[:, k]
            # 代价函数：状态误差和控制代价
            if (k >= 0) and (k <= self.N):
                objective += ca.mtimes([(X_k - state_ref_k).T, w1_diag, (X_k - state_ref_k)])
            # if (k >= 0) and (k < self.N):
            #     objective += ca.mtimes([U_k.T, w2_diag, U_k])
            # if (k >= 0) and (k < self.N - 1):
            #     delta_U_k = U[:, k+1] - U[:, k]
            #     objective += ca.mtimes([delta_U_k.T, w3_diag, delta_U_k])

        # 优化变量上下界
        for _ in range(self.N + 1):
            lbx += [-ca.inf, -ca.inf, -ca.inf, -ca.inf]  # 状态变量下界
            ubx += [ca.inf, ca.inf, ca.inf, ca.inf]      # 状态变量上界
        for _ in range(self.N):
            lbx += [-ca.inf, -ca.inf]  # 控制输入下界
            ubx += [ca.inf, ca.inf]    # 控制输入上界

        # 定义非线性规划问题
        nlp = {'x': opt_vars, 'f': objective, 'g': constraints}

        # 使用IPOPT求解器求解MPC问题
        opts_setting = {
            'ipopt.max_iter': 300,
            'ipopt.print_level': 0,
            'print_time': False,
            'print_in': False,
            'print_out': False,
        }
        # solver = ca.nlpsol('solver', 'ipopt', nlp)
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts_setting)

        # 求解问题
        sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        # 获取最优状态量和控制量
        X_sol = ca.reshape(sol['x'][:self.state_n*(self.N+1)], self.state_n, self.N+1)
        U_sol = ca.reshape(sol['x'][self.state_n*(self.N+1):], self.control_n, self.N)

        # print("Optimal state trajectory:\n", X_sol)
        # print("Optimal control inputs:\n", U_sol)

        return X_sol, U_sol