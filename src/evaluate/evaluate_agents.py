import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/agent')))

import highway_env
import core
import argparse
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from utils.mpi_tools import mpi_fork, mpi_sum, mpi_avg, num_procs, proc_id
import random

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha=0.0003, state_dim=50, action_dim=5, fc1_dim=256, fc2_dim=256):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        V = self.V(x)
        A = self.A(x)

        return V, A

def merge_eval(env_fn,
               render=False,
               Max_episode_len=1000,
               eval_episodes=100,
               logger:EpochLogger=None,
               print_freq=50,
               choose_agent="ppo_baseline",
               safe_protect=False,
               seed=0,
               data_file=None,
               exp_name=None):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # set env seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # create merge env
    merge_env = env_fn()
    merge_env = merge_env.unwrapped
    merge_env.action_space.seed(seed)
    merge_env.configure({
        "simulation_frequency": 10,
        "policy_frequency": 2,
        "screen_width": 2000,
        "screen_height": 600,
        "scaling": 10,
        "cooperative_prob": 0.8,
        "mpc_control": True
    })

    ##### choose an trained agent #######
    agent_path = "trained_models/" + choose_agent + ".pt"
    if 'DuelingDQN' in choose_agent:
        agent = DuelingDeepQNetwork()
        agent = torch.load(agent_path)
    else:
        agent = torch.load(agent_path)
    
    # save data to txt file
    save_dir = os.path.join('eval_result', data_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_dir = os.path.join(save_dir, exp_name + '_' + choose_agent)
    head_list = ['Agent', 'Episode', 'Crashed', 'Success', 'Costs', 'Rewards', 'Length',
                 'AvgSpeed', 'MaxSpeed', 'MinSpeed', 'SpeedStd',
                 'AvgAcc', 'MaxAcc', 'MinAcc', 'AccStd',
                 'AvgJerk', 'MaxJerk', 'JerkStd',
                 'AvgSteerAngle', 'MaxSteerAngle', 'SteerAngleStd',
                 'AvgSteerRate', 'MaxSteerRate', 'SteerRateStd\n']
    head_str = '\t'.join(head_list)
    with open(file_dir + '/process.txt', 'w+') as f:
        f.write(head_str)
    # Sync params across processes
    sync_params(agent)
    # logger.store(agent=choose_agent)

    # initialize counter
    episode_counter = 0
    crash_counter = 0
    success_counter = 0
    cost_counter = 0
    re_action = safe_protect

    # EVALUATE #
    while True:
        obs = merge_env.reset()
        ep_costs = 0
        ep_rewards = 0
        ep_len = 0
        crash = 0
        reach = 0
        
        # 初始化状态记录列表
        speeds = []
        accelerations = []
        jerks = []
        steer_angles = []
        steer_rates = []
        
        last_speed = None
        last_acc = None
        last_steer = None
        dt = 0.1  # 时间步长
        
        if episode_counter == int(eval_episodes / num_procs()):
            break
            
        for step in range(int(Max_episode_len / num_procs())):
            obs = obs[1:].flatten()
            if 'Original_DuelingDQN' in choose_agent :
                state = torch.tensor([obs], dtype=torch.float)
                _, A = agent.forward(state)
                a = torch.argmax(A).item()
            else:
                a, _, _, _ = agent.step(torch.from_numpy(obs).float())

            if re_action:
                a = merge_env.check_action(int(a), check_type="mpc")

            obs_, r, d, info = merge_env.step(int(a))
            ep_rewards += r
            
            # 记录车辆状态
            current_speed = merge_env.vehicle.speed
            current_steer = merge_env.vehicle.action['steering']
            current_acc = merge_env.vehicle.action['acceleration']
            
            speeds.append(current_speed)
            steer_angles.append(current_steer)
            accelerations.append(current_acc)
            
            # 计算方向盘角速度
            if last_steer is not None:
                steer_rate = (current_steer - last_steer) / dt
                steer_rates.append(steer_rate)
            
            # 计算加加速度
            if last_acc is not None:
                jerk = (current_acc - last_acc) / dt
                jerks.append(jerk)
            
            last_speed = current_speed
            last_acc = current_acc
            last_steer = current_steer
            
            cost = info['cost']
            ep_costs += cost
            ep_len += 1
            if render:
                merge_env.render()
            if info['crashed']:
                crash = 1
                crash_counter += 1
            elif info['success']:
                reach = 1
                if ep_costs >= 0.3:
                    cost_counter += 1
                else:
                    success_counter += 1
            obs = obs_
            if d:
                episode_counter += 1
                
                # 计算统计量
                speed_stats = {
                    'avg': np.mean(speeds) if speeds else 0,
                    'max': np.max(speeds) if speeds else 0,
                    'min': np.min(speeds) if speeds else 0,
                    'std': np.std(speeds) if speeds else 0
                }
                
                acc_stats = {
                    'avg': np.mean(accelerations) if accelerations else 0,
                    'max': np.max(accelerations) if accelerations else 0,
                    'min': np.min(accelerations) if accelerations else 0,
                    'std': np.std(accelerations) if accelerations else 0
                }
                
                jerk_stats = {
                    'avg': np.mean(jerks) if jerks else 0,
                    'max': np.max(np.abs(jerks)) if jerks else 0,
                    'std': np.std(jerks) if jerks else 0
                }
                
                steer_angle_stats = {
                    'avg': np.mean(steer_angles) if steer_angles else 0,
                    'max': np.max(np.abs(steer_angles)) if steer_angles else 0,
                    'std': np.std(steer_angles) if steer_angles else 0
                }
                
                steer_rate_stats = {
                    'avg': np.mean(steer_rates) if steer_rates else 0,
                    'max': np.max(np.abs(steer_rates)) if steer_rates else 0,
                    'std': np.std(steer_rates) if steer_rates else 0
                }
                
                # 写入数据
                data_list = [
                    agent_name, str(episode_counter), str(crash), str(reach), 
                    str(ep_costs), str(ep_rewards), str(ep_len),
                    str(speed_stats['avg']), str(speed_stats['max']), 
                    str(speed_stats['min']), str(speed_stats['std']),
                    str(acc_stats['avg']), str(acc_stats['max']), 
                    str(acc_stats['min']), str(acc_stats['std']),
                    str(jerk_stats['avg']), str(jerk_stats['max']), 
                    str(jerk_stats['std']),
                    str(steer_angle_stats['avg']), str(steer_angle_stats['max']), 
                    str(steer_angle_stats['std']),
                    str(steer_rate_stats['avg']), str(steer_rate_stats['max']), 
                    str(steer_rate_stats['std']) + '\n'
                ]
                
                head_str = '\t'.join(data_list)
                with open(file_dir + '/process.txt', 'a+') as f:
                    f.write(head_str)
                
                # 记录到logger
                logger.store(
                    AverageCost=ep_costs,
                    EpisodeReward=ep_rewards,
                    AverageLen=ep_len,
                    AverageSpeed=speed_stats['avg'],
                    MaxSpeed=speed_stats['max'],
                    MinSpeed=speed_stats['min'],
                    SpeedStd=speed_stats['std'],
                    AverageAcc=acc_stats['avg'],
                    MaxAcc=acc_stats['max'],
                    MinAcc=acc_stats['min'],
                    AccStd=acc_stats['std'],
                    AverageJerk=jerk_stats['avg'],
                    MaxJerk=jerk_stats['max'],
                    JerkStd=jerk_stats['std'],
                    AverageSteerAngle=steer_angle_stats['avg'],
                    MaxSteerAngle=steer_angle_stats['max'],
                    SteerAngleStd=steer_angle_stats['std'],
                    AverageSteerRate=steer_rate_stats['avg'],
                    MaxSteerRate=steer_rate_stats['max'],
                    SteerRateStd=steer_rate_stats['std']
                )
                
                # 添加到logger输出
                logger.log_tabular("Agent", choose_agent)
                logger.log_tabular("Episode", episode_counter)
                logger.log_tabular("Crashed", mpi_sum(crash))
                logger.log_tabular("Success", mpi_sum(reach))
                logger.log_tabular("Costs", mpi_avg(ep_costs))
                logger.log_tabular("Rewards", mpi_avg(ep_rewards))
                logger.log_tabular("Length", mpi_avg(ep_len))
                logger.log_tabular("AverageSpeed", mpi_avg(speed_stats['avg']))
                logger.log_tabular("MaxSpeed", mpi_avg(speed_stats['max']))
                logger.log_tabular("MinSpeed", mpi_avg(speed_stats['min']))
                logger.log_tabular("SpeedStd", mpi_avg(speed_stats['std']))
                logger.log_tabular("AverageAcc", mpi_avg(acc_stats['avg']))
                logger.log_tabular("MaxAcc", mpi_avg(acc_stats['max']))
                logger.log_tabular("MinAcc", mpi_avg(acc_stats['min']))
                logger.log_tabular("AccStd", mpi_avg(acc_stats['std']))
                logger.log_tabular("AverageJerk", mpi_avg(jerk_stats['avg']))
                logger.log_tabular("MaxJerk", mpi_avg(jerk_stats['max']))
                logger.log_tabular("JerkStd", mpi_avg(jerk_stats['std']))
                logger.log_tabular("AverageSteerAngle", mpi_avg(steer_angle_stats['avg']))
                logger.log_tabular("MaxSteerAngle", mpi_avg(steer_angle_stats['max']))
                logger.log_tabular("SteerAngleStd", mpi_avg(steer_angle_stats['std']))
                logger.log_tabular("AverageSteerRate", mpi_avg(steer_rate_stats['avg']))
                logger.log_tabular("MaxSteerRate", mpi_avg(steer_rate_stats['max']))
                logger.log_tabular("SteerRateStd", mpi_avg(steer_rate_stats['std']))
                logger.dump_tabular()
                
                break
    
    print('CrashRatio', mpi_sum(crash_counter) / eval_episodes,
          'CrashCounter', mpi_sum(crash_counter),
          'SuccessRatio', mpi_sum(success_counter) / eval_episodes,
          'SuccessCounter', mpi_sum(success_counter),
          'HighCostsRatio', mpi_sum(cost_counter) / eval_episodes,
          'HighCostsCounter', mpi_sum(cost_counter))

    # logger.log_tabular("Agent", choose_agent)
    # logger.log_tabular("CrashRatio", mpi_sum(crash_counter) / eval_episodes)
    # logger.log_tabular("AverageCost", average_only=True)
    # logger.log_tabular("AverageLen", average_only=True)
    # logger.log_tabular("CrashCounter", mpi_sum(crash_counter))
    # logger.log_tabular("SuccessCounter", mpi_sum(success_counter))
    # logger.log_tabular("SuccessRatio", mpi_sum(success_counter) / eval_episodes)
    # logger.log_tabular("CostCounter", mpi_sum(cost_counter))
    # logger.log_tabular("CostRatio", mpi_sum(cost_counter) / eval_episodes)
    # logger.log_tabular("FailCounter", mpi_sum(cost_counter) + mpi_sum(crash_counter))
    # logger.dump_tabular()

if __name__ == "__main__":
    # "merge_eval_low_density-v0" "merge_eval_high_density-v0" "merge_game_env-v0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="merge_game_env-v0") 
    parser.add_argument('--eval_episodes', type=int, default=500)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=10)  # 5
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--exp_name', type=str, default='eval_in_medium')
    parser.add_argument('--render', action='store_true', help='Render evaluation')
    parser.add_argument('--safe_protect', action='store_true', help='use the action sheild module')
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--agents', nargs='+', default=['Augmented_Lagrangian_MPC_SACD_1',
                                                        'Augmented_Lagrangian_MPC_SACD_2',
                                                        'Augmented_Lagrangian_MPC_SACD_3',
                                                        'Augmented_Lagrangian_MPC_SACD_4',
                                                        'Augmented_Lagrangian_MPC_SACD_5',
                                                        ])
    # parser.add_argument('--agents', nargs='+', default=['Baseline_SACD_MPC_1',
    #                                                     'Baseline_SACD_MPC_2',
    #                                                     'Baseline_SACD_MPC_3',
    #                                                     'Baseline_SACD_MPC_4',
    #                                                     'Baseline_SACD_MPC_5',
    #                                                     ])

    parser.add_argument('--data_file', type=str, default='SARMI')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from utils.run_utils import setup_logger_kwargs

    for i, agent_name in enumerate(args.agents):

        logger_kwargs = setup_logger_kwargs(args.exp_name + '_' + agent_name, 
                                            data_dir=os.path.join('eval_result',args.data_file))

        # Set up logger and save configuration
        logger = EpochLogger(**logger_kwargs)

        args.safe_protect = False
        print('agent_name:', agent_name)
        print('Safe protect', args.safe_protect)
        # if agent_name == 'sac_mpc' or agent_name == 'sac_mpc_nstep':
        # if agent_name == 'SAC_sac_mpc_nsteps_high_2' or agent_name == 'SAC_sac_mpc_nsteps_low_2':
        if agent_name == 'Augmented_Lagrangian_MPC_SACD_1':
            args.safe_protect = True
            print('Safe protect', args.safe_protect)

        merge_eval(lambda: gym.make(args.env),
                   Max_episode_len=args.steps,
                   eval_episodes=args.eval_episodes,
                   print_freq=args.print_freq,
                   logger=logger,
                   choose_agent=agent_name,
                   safe_protect=args.safe_protect,
                   render=args.render,
                   seed=args.seed,
                   data_file=args.data_file,
                   exp_name=args.exp_name)
