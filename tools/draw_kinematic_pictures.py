import gym
import sys
import os

# 获取项目根目录并添加到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'src/agent'))
sys.path.append(os.path.join(root_dir, 'highway_env'))

import core
import highway_env
import torch
from matplotlib import pyplot as plt
import numpy as np
import random
import time

# 首先添加这些全局变量到文件开头的配置部分
label_size = 50
ticks_size = 48
legend_size = 30
line_size = 4
title_size = 55
title_order = ['(a)','(b)','(c)', '(d)', '(e)', '(f)']

font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size': label_size,}

font_title = {'family' : 'Times New Roman',
'weight' : 'normal',
'size': title_size,}


def initial_env():
    merge_env = gym.make('merge_game_env-v0')
    # merge_env = merge_env.unwrapped
    merge_env.configure({
        "simulation_frequency": 10,
        "policy_frequency": 2,
        "screen_width": 720,
        "screen_height": 540,
        "scaling": 6,
        "manual_control":False,
        "mpc_control": True,
        "show_mpc_trajectory": True,
        "show_other_vehicles_predict": False,  # False  True
        'show_trajectories': True,
        'show_risk_field': False,  # False  True
        "only_show_ego_history": True,
        'show_history_frequency': 8,
        "show_history_duration": 0.8
    })
    merge_env.reset()
    return merge_env


def initial_agent(agent_name):
    agent_path = "trained_models/" + agent_name + ".pt"
    agent = torch.load(agent_path)
    return agent

def calculate_derivatives(data, time):
    """计算数据的导数"""
    # 使用numpy的diff计算差分
    data_diff = np.diff(data)
    time_diff = np.diff(time)
    # 计算导数
    derivative = data_diff / time_diff
    return derivative

def process_agent_data(episodes_data, episodes_time=None, calculate_rate=False):
    """处理多个episodes的数据，返平均值和标准差"""
    # 找到最短的episode长度
    min_length = min(len(episode) for episode in episodes_data)
    
    # 截断所有episodes到相同长度
    truncated_data = [episode[:min_length] for episode in episodes_data]
    
    if calculate_rate and episodes_time is not None:
        # 截断时间数据
        truncated_time = [episode[:min_length] for episode in episodes_time]
        # 计算每个episode的导数
        derivatives = [calculate_derivatives(data, time) 
                      for data, time in zip(truncated_data, truncated_time)]
        # 确保所有导数数据长度相同（因为差分会减少一个点）
        min_deriv_length = min(len(d) for d in derivatives)
        derivatives = [d[:min_deriv_length] for d in derivatives]
        # 计算导数的平均值和标准差
        deriv_array = np.array(derivatives)
        mean_deriv = np.mean(deriv_array, axis=0)
        std_deriv = np.std(deriv_array, axis=0)
        return mean_deriv, std_deriv
    
    # 转换为numpy数组便于计算
    data_array = np.array(truncated_data)
    
    # 计算平均值和标准差
    mean_data = np.mean(data_array, axis=0)
    std_data = np.std(data_array, axis=0)
    
    return mean_data, std_data

def plot_selected_agents(all_time, all_speed, all_acc, all_st_angle, all_jerk, all_angle_rate, all_heading, agent_name, selected_agents):
    """绘制选定agent的对比图"""
    # 检查 selected_agents 是否在有效范围内
    valid_agents = [i for i in selected_agents if i < len(all_time)]
    if not valid_agents:
        print("警告: 没有有效的代理可供绘制。")
        return

    # 筛选数据
    selected_times = [all_time[i] for i in valid_agents]
    selected_speeds = [all_speed[i] for i in valid_agents]
    selected_accs = [all_acc[i] for i in valid_agents]
    selected_st_angles = [all_st_angle[i] for i in valid_agents]
    selected_jerks = [all_jerk[i] for i in valid_agents]
    selected_angle_rates = [all_angle_rate[i] for i in valid_agents]
    selected_headings = [all_heading[i] for i in valid_agents]
    selected_names = [agent_name[i] for i in valid_agents]

    # 创建大图
    plt.figure(figsize=(16 * 2, 12 * 2))
    
    # 绘制六种状态的对比图
    data_y = [selected_speeds, selected_accs, selected_jerks, 
              selected_st_angles, selected_angle_rates, selected_headings]
    name = ['speed', 'acc', 'jerk', 'st_angle', 'angle_rate', 'heading']
    
    for k in range(len(name)):
        plot(k, selected_times, data_y[k], label_name=name[k], agent_names=selected_names)
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    
    # 如果需要保存图片
    save_dir = './pictures/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'kinematic_comparison'), dpi=600, format='pdf', bbox_inches='tight')
    plt.show()

def plot(i, data_x_list, data_y_list, label_name, agent_names):
    num = 231 + i  # 改为 2x3 的子图布局
    plt.subplot(num)
    
    for idx, (data_x, data_y, agent_name) in enumerate(zip(data_x_list, data_y_list, agent_names)):
        plt.plot(data_x[:-1] if label_name in ['jerk', 'angle_rate'] else data_x, 
                data_y, 
                linewidth=line_size,
                label=agent_name.strip('/'))
    
    # 设置x轴刻度值为0,2,4,6,8,10,12,14
    x_ticks = np.arange(0, 15, 2)  # 从0到14，步长为2
    plt.xticks(x_ticks, fontsize=ticks_size, fontproperties='Times New Roman')
    plt.yticks(fontsize=ticks_size, fontproperties='Times New Roman')
    
    plt.title(title_order[i], font_title, y=-0.3)
    plt.xlabel('Time (s)', font)
    
    # 设置横坐标范围，确保能显示所有刻度
    plt.xlim(0, 140)  # 修改为0到14，与刻度范围一致
    
    if i == 0:
        plt.legend(prop={'family': 'Times New Roman', 'size': legend_size})
        
    if label_name == 'speed':
        plt.ylabel('Velocity (m/s)', font, labelpad=20)
    elif label_name == 'acc':
        plt.ylabel('Acceleration (m/s$^2$)', font, labelpad=20)
    elif label_name == 'jerk':
        plt.ylabel('Jerk (m/s$^3$)', font, labelpad=20)
    elif label_name == 'st_angle':
        plt.ylabel('Steering Angle (rad)', font, labelpad=20)
    elif label_name == 'angle_rate':
        plt.ylabel('Steering Rate (rad/s)', font, labelpad=20)
    elif label_name == 'heading':
        plt.ylabel('Heading Angle (rad)', font, labelpad=20)

def collect_episode_data(env, agent, re_action):
    """收集单个episode的数据"""
    obs = env.reset()
    episode_speed = []
    episode_heading = []
    episode_acc = []
    episode_st_angle = []
    episode_time = []
    
    for k in range(1000):
        print(k)
        obs = obs[1:].flatten()
        a, _, _, _ = agent.step(torch.from_numpy(obs).float())
        
        if re_action:
            # a = env.check_action(int(a), 'simple')
            a = env.check_action(int(a), 'mpc')
            
        obs_, r, d, info = env.step(int(a))
        
        episode_speed.append(env.vehicle.speed)
        episode_heading.append(env.vehicle.heading)
        episode_st_angle.append(env.vehicle.action['steering'])
        episode_acc.append(env.vehicle.action['acceleration'])
        episode_time.append(env.time)
        
        obs = obs_
        if d:
            break
            
    return episode_speed, episode_heading, episode_acc, episode_st_angle, episode_time

if __name__ == "__main__":
    import argparse
    perser = argparse.ArgumentParser()
    perser.add_argument('--total_episodes', type=int, default=1)
    perser.add_argument('--seed', type=int, default=123)
    perser.add_argument('--render', type=bool, default=True)
    perser.add_argument('--re_action', type=bool, default=True)
    perser.add_argument('--plot_state', type=bool, default=True)
    perser.add_argument('--save_freq', type=int, default=1)
    args = perser.parse_args()
    # agent_name = ['/SAC_simple_check_sac_simple_nstep_0.01_5']
    agent_name = ['/Augmented_Lagrangian_MPC_SACD_1']

    all_speed = []
    all_acc = []
    all_heading = []
    all_st_angle = []
    all_time = []
    all_jerk = []
    all_angle_rate = []

    total_episodes = args.total_episodes
    seed = args.seed
    render = args.render
    re_action = args.re_action
    save_freq = args.save_freq
    plot_state = args.plot_state

    # 修改数据收集部分
    for count, i in enumerate(agent_name):
        np.random.seed(seed)
        env = initial_env()
        env = env.unwrapped
        agent = initial_agent(i)
        
        # 存储当前agent的所有episodes数据
        agent_speeds = []
        agent_headings = []
        agent_accs = []
        agent_st_angles = []
        agent_times = []
        
        # 收集多个episodes的数据
        for j in range(total_episodes):
            speed, heading, acc, st_angle, time_data = collect_episode_data(env, agent, re_action)
            agent_speeds.append(speed)
            agent_headings.append(heading)
            agent_accs.append(acc)
            agent_st_angles.append(st_angle)
            agent_times.append(time_data)
        
        # 计算平均值和标准差
        mean_speed, std_speed = process_agent_data(agent_speeds)
        mean_acc, std_acc = process_agent_data(agent_accs)
        mean_st_angle, std_st_angle = process_agent_data(agent_st_angles)
        mean_heading, std_heading = process_agent_data(agent_headings)
        mean_time, _ = process_agent_data(agent_times)
        
        # 计算jerk和angle rate
        mean_jerk, std_jerk = process_agent_data(agent_accs, agent_times, calculate_rate=True)
        mean_angle_rate, std_angle_rate = process_agent_data(agent_st_angles, agent_times, calculate_rate=True)
        
        # 存储处理后的数据
        all_speed.append(mean_speed)
        all_acc.append(mean_acc)
        all_st_angle.append(mean_st_angle)
        all_time.append(mean_time)
        all_jerk.append(mean_jerk)
        all_angle_rate.append(mean_angle_rate)
        all_heading.append(mean_heading)

    # 绘图部分
    if plot_state:
        selected_agents = [0, 1]
        plot_selected_agents(all_time, all_speed, all_acc, all_st_angle, 
                           all_jerk, all_angle_rate, all_heading, agent_name, selected_agents)
    
