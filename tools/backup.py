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

font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size': 16,}

def save_picture(picture, id, agent_name):
    save_dir = './draw_picture/' + agent_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, id)
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.imshow(picture)
    plt.axis('off')
    plt.savefig(save_path, dpi=800, format='svg')

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

def plot(i, data_x_list, data_y_list, label_name, agent_names):
    plt.figure(i)
    for idx, (data_x, data_y, agent_name) in enumerate(zip(data_x_list, data_y_list, agent_names)):
        plt.plot(data_x, data_y, label=agent_name.strip('/'))
    
    plt.xlabel('Time(s)', font)
    if label_name == 'speed':
        plt.ylabel('Velocity(m/s)', font)
    if label_name == 'acc':
        plt.ylabel('Acceleration(m/s$^2$)', font)
    if label_name == 'st_angle':
        plt.ylabel('Steering Angle(rad)', font)
    plt.legend(prop=font)

def plot_selected_agents(all_time, all_speed, all_acc, all_st_angle, agent_name, selected_agents):
    """
    绘制选定agent的对比图
    selected_agents: 要对比的agent的索引列表
    """
    # 筛选数据
    selected_times = [all_time[i] for i in selected_agents]
    selected_speeds = [all_speed[i] for i in selected_agents]
    selected_accs = [all_acc[i] for i in selected_agents]
    selected_st_angles = [all_st_angle[i] for i in selected_agents]
    selected_names = [agent_name[i] for i in selected_agents]

    # 绘制三种状态的对比图
    data_y = [selected_speeds, selected_accs, selected_st_angles]
    name = ['speed', 'acc', 'st_angle']
    for k in range(len(name)):
        plot(k, selected_times, data_y[k], label_name=name[k], agent_names=selected_names)
    plt.show()

if __name__ == "__main__":
    import argparse
    perser = argparse.ArgumentParser()
    perser.add_argument('--total_episodes', type=int, default=10)
    perser.add_argument('--seed', type=int, default=123)
    perser.add_argument('--render', type=bool, default=True)
    perser.add_argument('--safe_pic', type=bool, default=True)
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

    total_episodes = args.total_episodes
    seed = args.seed
    render = args.render
    save = args.safe_pic
    re_action = args.re_action
    save_freq = args.save_freq
    plot_state = args.plot_state

    # 先收集所有agent的数据
    for count, i in enumerate(agent_name):
        np.random.seed(seed)
        env = initial_env()
        env = env.unwrapped
        agent = initial_agent(i)

        for j in range(total_episodes):
            obs = env.reset()
            temp_speed = []
            temp_heading = []
            temp_acc = []
            temp_st_angle = []
            temp_time = []
            
            for k in range(1000):
                print(k)
                obs = obs[1:].flatten()
                time_1 = time.time()
                a, _, _, _ = agent.step(torch.from_numpy(obs).float())

                if re_action:
                    a = env.check_action(int(a), 'simple')
                    # a = env.check_action(int(a), 'mpc')
                time_2 = time.time()
                print('------')
                print('delta time', time_2 - time_1)
                obs_, r, d, info = env.step(int(a))

                temp_speed.append(env.vehicle.speed)
                temp_heading.append(env.vehicle.heading)
                temp_st_angle.append(env.vehicle.action['steering'])
                temp_acc.append(env.vehicle.action['acceleration'])
                temp_time.append(env.time)

                obs = obs_
                if render:
                    env.render()
                if save:
                    if k % save_freq == 0 and k > 40:
                        # episode num_steps_decision
                        id = str(j) + '_' + str(k) + '_' + str(a)
                        save_picture(env.render(mode="rgb_array"), id, i)
                if d:
                    break
                    
        all_speed.append(temp_speed)
        all_acc.append(temp_acc)
        all_st_angle.append(temp_st_angle)
        all_heading.append(temp_heading)
        all_time.append(temp_time)

    # 数据收集完成后，选择要对比的agent进行绘图
    if plot_state:
        # 例如：对比第0个和第1个agent
        selected_agents = [0, 1]  # 可以根据需要修改要对比的agent索引
        plot_selected_agents(all_time, all_speed, all_acc, all_st_angle, agent_name, selected_agents)