import numpy as np
import matplotlib.pyplot as plt

# 定义势场函数
def potential_field(x, y, x_obs, y_obs, A, sigma_x, sigma_y, beta):
    term_x = ((x - x_obs)**2 / sigma_x**2)**beta
    term_y = ((y - y_obs)**2 / sigma_y**2)**beta
    return A * np.exp(-0.5 * (term_x + term_y))

# 设置参数
A = 5.0        # 势场强度
sigma_x = 1.0  # x方向的标准差
sigma_y = 1.5  # y方向的标准差
beta = 1.0     # 势场形状控制参数
x_obs = 1.0    # 障碍物x位置
y_obs = 0.0    # 障碍物y位置

# 创建网格
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

# 计算势场
Z = potential_field(X, Y, x_obs, y_obs, A, sigma_x, sigma_y, beta)

# 绘制势场图
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Potential Field')
plt.title('2D Potential Field')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # 定义道路边界势场函数
# def lane_boundary_potential(y, y_left, y_right, A, sigma):
#     left_term = np.exp(-((y - y_left)**2) / (2 * sigma**2))
#     right_term = np.exp(-((y - y_right)**2) / (2 * sigma**2))
#     return A * (left_term + right_term)

# # 设置参数
# A = 5.0        # 势场强度
# sigma = 0.5    # 控制势场的宽度
# y_left = -4.0  # 左车道边界
# y_right = 2.0  # 右车道边界

# # 创建y轴网格
# y = np.linspace(-5, 5, 400)

# # 计算道路边界势场
# Z_lane = lane_boundary_potential(y, y_left, y_right, A, sigma)

# # 绘制势场图
# plt.figure(figsize=(8, 4))
# plt.plot(y, Z_lane, label='Lane Boundary Potential')
# plt.axvline(x=y_left, color='r', linestyle='--', label='Left Boundary')
# plt.axvline(x=y_right, color='b', linestyle='--', label='Right Boundary')
# plt.title('Lane Boundary Potential Field')
# plt.xlabel('y (lateral position)')
# plt.ylabel('Potential Field')
# plt.legend()
# plt.grid(True)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Constants
# A = 100.0          # Amplitude
# kv = 0.5         # Constant for velocity effect
# alpha = 0.5      # Constant (0 ≤ α ≤ 1)
# xobs = 0.0       # Obstacle's x position
# yobs = 0.0       # Obstacle's y position
# Lobs = 1.0       # Obstacle's length
# vobs = 1.0       # Obstacle's longitudinal velocity
# v = 2.0          # Ego vehicle's longitudinal velocity
# sigma_v = kv * np.abs(vobs - v)  # Velocity related term

# # Grid setup
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# X, Y = np.meshgrid(x, y)

# # Compute Umov
# def Umov(X, Y, xobs, yobs, sigma_v, vobs, v, alpha):
#     relv = np.where(vobs > v, 1, -1)
#     numerator = A * np.exp(((X - xobs)**2 / sigma_v**2) + ((Y - yobs)**2 / (sigma_v**2)))
#     denominator = 1 + np.exp(relv * (X - xobs - alpha * Lobs * relv))
#     return numerator / denominator

# # Calculate Umov values
# U_values = Umov(X, Y, xobs, yobs, sigma_v, vobs, v, alpha)

# # Plotting
# plt.figure(figsize=(10, 8))
# contour = plt.contourf(X, Y, U_values, levels=50, cmap='viridis')
# plt.colorbar(contour)
# plt.title('Umov Contour Plot')
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.grid()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Constants
# A = 5.0          # Amplitude
# kv = 0.5         # Constant for velocity effect
# alpha = 0.2      # Constant (0 ≤ α ≤ 1)
# xobs = 0.0       # Obstacle's x position
# yobs = 0.0       # Obstacle's y position
# Lobs = 1.0       # Obstacle's length
# vobs = -10.0     # Obstacle's longitudinal velocity
# v = 1.0          # Ego vehicle's longitudinal velocity
# sigma_v = kv * np.abs(vobs - v)  # Velocity related term
# sigma_y = 1.0

# # Grid setup with increased resolution
# x = np.linspace(-10, 10, 500)  # Increase the number of points
# y = np.linspace(-10, 10, 500)  # Increase the number of points
# X, Y = np.meshgrid(x, y)

# # Compute Umov
# def Umov(X, Y, xobs, yobs, sigma_v, sigma_y, vobs, v, alpha):
#     relv = np.where(vobs > v, 1, -1)
#     numerator = A * np.exp(-((X - xobs)**2 / sigma_v**2) - ((Y - yobs)**2 / (sigma_y**2)))
#     denominator = 1 + np.exp(relv * (X - xobs - alpha * Lobs * relv))
#     return numerator / denominator

# # Calculate Umov values
# U_values = Umov(X, Y, xobs, yobs, sigma_v, sigma_y, vobs, v, alpha)

# # Check for NaN or Inf values
# if np.any(np.isnan(U_values)) or np.any(np.isinf(U_values)):
#     print("Warning: U_values contains NaN or Inf values.")

# # Plotting 3D surface
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, U_values, cmap='viridis', edgecolor='none', rstride=1, cstride=1, alpha=1.0)
# fig.colorbar(surf)
# ax.set_title('Umov 3D Surface Plot')
# ax.set_xlabel('X Position')
# ax.set_ylabel('Y Position')
# ax.set_zlabel('Umov')
# ax.set_xlim([-10, 10])  # Set x-axis limits
# ax.set_ylim([-10, 10])  # Set y-axis limits
# ax.set_zlim([0, 15])    # Set z-axis limits based on U_values

# # Disable grid lines
# ax.grid(False)
# plt.show()



# # Plotting
# plt.figure(figsize=(10, 8))
# contour = plt.contourf(X, Y, U_values, levels=50, cmap='viridis')
# plt.colorbar(contour)
# plt.title('Umov Contour Plot')
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.grid()
# plt.show()


# def evaluate(self, state, action_mask=None):
#     """
#     评估状态并返回动作
#     Args:
#         state: 状态输入
#         action_mask: 动作掩码，用于屏蔽非法动作
#     """
#     action_probs = self.forward(state)
    
#     # 如果提供了动作掩码，则应用掩码
#     if action_mask is not None:
#         if action_mask.shape != action_probs.shape:
#             print(f"Mask shape: {action_mask.shape}, Probs shape: {action_probs.shape}")
#             raise ValueError("Action mask shape does not match action probabilities shape.")
            
#         # 应用掩码并重新归一化
#         masked_probs = action_probs * action_mask
#         # 确保每个样本的和不为0
#         sum_probs = masked_probs.sum(dim=-1, keepdim=True)
#         valid_mask = (sum_probs > 0).float()
        
#         # 对有效的样本进行归一化，无效样本保持均匀分布
#         action_probs = torch.where(
#             valid_mask == 1,
#             masked_probs / (sum_probs + 1e-10),
#             torch.ones_like(masked_probs) / masked_probs.size(-1)
#         )

#     # 数值稳定性处理
#     action_probs = torch.clamp(action_probs, min=1e-10, max=1.0)
#     action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

#     # 检查 action_probs 的有效性
#     if (action_probs < 0).any():
#         print("Warning: Negative probabilities detected")
#         action_probs = F.relu(action_probs)
#         action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
#     if torch.isnan(action_probs).any():
#         print("Warning: NaN values detected in probabilities")
#         action_probs = torch.where(torch.isnan(action_probs), 
#                                  torch.ones_like(action_probs) / action_probs.size(-1),
#                                  action_probs)
        
#     # 使用更宽松的容差检查
#     if not torch.allclose(action_probs.sum(dim=-1), torch.ones_like(action_probs.sum(dim=-1)), rtol=1e-4, atol=1e-4):
#         print(f"Sum of probabilities: {action_probs.sum(dim=-1)}")
#         print(f"Action probabilities: {action_probs}")
#         raise ValueError("Action probabilities do not sum to 1.")

#     # 创建分布并采样
#     dist = Categorical(action_probs)
#     action = dist.sample()
    
#     # 计算对数概率
#     log_action_probabilities = torch.log(action_probs + 1e-10)
    
#     return action.detach().cpu(), action_probs, log_action_probabilities

