import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

Episode_num = 500

def get_evaluation_data(all_data):
    data = pd.concat(all_data, axis=0, join='inner', ignore_index=True)
    
    # Convert all numeric columns to float
    numeric_columns = ['Crashed', 'Costs', 'Success', 'Length', 'Reward',
                      'AvgSpeed', 'MaxSpeed', 'MinSpeed', 'SpeedStd',
                      'AvgAcc', 'MaxAcc', 'MinAcc', 'AccStd',
                      'AvgJerk', 'MaxJerk', 'JerkStd',
                      'AvgSteerAngle', 'MaxSteerAngle', 'SteerAngleStd',
                      'AvgSteerRate', 'MaxSteerRate', 'SteerRateStd']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].astype(float)
    
    agent_name = data['Agent'].unique()
    for index, name in enumerate(agent_name):
        agent_data = data.loc[data['Agent'] == name]
        success_data = agent_data.loc[agent_data['Success'] == 1]
        
        print('-------------')
        print('Agent:', name)
        
        # 1. Basic Metrics
        collision_num = agent_data.loc[agent_data['Crashed'] == 1, 'Crashed'].sum()
        success_num = success_data.shape[0]
        total_inter_num = success_data['Length'].sum()
        average_time = success_data['Length'].mean() * 0.1 * 5
        
        print('\n1. Basic Performance Metrics:')
        print('- Collision Count:', collision_num)
        print('- Collision Rate: {:.2f}%'.format(collision_num / Episode_num * 100))
        print('- Success Rate: {:.2f}%'.format(success_num / Episode_num * 100))
        print('- Average Completion Time: {:.2f}s'.format(average_time))
        print('- Total Interactions:', total_inter_num)
        print('- Average Cost: {:.4f}'.format(agent_data['Costs'].mean()))
        print('- Average Reward: {:.4f}'.format(agent_data['Rewards'].mean()))
        
        # 2. Speed Metrics
        print('\n2. Speed Metrics:')
        print('- Average Speed: {:.2f} m/s'.format(agent_data['AvgSpeed'].mean()))
        print('- Maximum Speed: {:.2f} m/s'.format(agent_data['MaxSpeed'].mean()))
        print('- Minimum Speed: {:.2f} m/s'.format(agent_data['MinSpeed'].mean()))
        print('- Speed Std: {:.4f}'.format(agent_data['SpeedStd'].mean()))
        
        # 3. Acceleration Metrics
        print('\n3. Acceleration Metrics:')
        print('- Average Acceleration: {:.2f} m/s²'.format(agent_data['AvgAcc'].mean()))
        print('- Maximum Acceleration: {:.2f} m/s²'.format(agent_data['MaxAcc'].mean()))
        print('- Minimum Acceleration: {:.2f} m/s²'.format(agent_data['MinAcc'].mean()))
        print('- Acceleration Std: {:.4f}'.format(agent_data['AccStd'].mean()))
        
        # 4. Jerk Metrics
        print('\n4. Jerk Metrics:')
        print('- Average Jerk: {:.2f} m/s³'.format(agent_data['AvgJerk'].mean()))
        print('- Maximum Jerk: {:.2f} m/s³'.format(agent_data['MaxJerk'].mean()))
        print('- Jerk Std: {:.4f}'.format(agent_data['JerkStd'].mean()))
        
        # 5. Steering Metrics
        print('\n5. Steering Metrics:')
        print('- Average Steering Angle: {:.4f} rad'.format(agent_data['AvgSteerAngle'].mean()))
        print('- Maximum Steering Angle: {:.4f} rad'.format(agent_data['MaxSteerAngle'].mean()))
        print('- Steering Angle Std: {:.4f}'.format(agent_data['SteerAngleStd'].mean()))
        print('- Average Steering Rate: {:.4f} rad/s'.format(agent_data['AvgSteerRate'].mean()))
        print('- Maximum Steering Rate: {:.4f} rad/s'.format(agent_data['MaxSteerRate'].mean()))
        print('- Steering Rate Std: {:.4f}'.format(agent_data['SteerRateStd'].mean()))
        
        # 6. Successful Episodes Only
        print('\n6. Metrics for Successful Episodes:')
        print('- Success Average Speed: {:.2f} m/s'.format(success_data['AvgSpeed'].mean()))
        print('- Success Average Acceleration: {:.2f} m/s²'.format(success_data['AvgAcc'].mean()))
        print('- Success Average Jerk: {:.2f} m/s³'.format(success_data['AvgJerk'].mean()))
        print('- Success Average Steering: {:.4f} rad'.format(success_data['AvgSteerAngle'].mean()))

def get_all_data(file_path):
    all_data = []
    for root, dirs, files in os.walk(file_path):
        if 'process.txt' in files:
            data_path = os.path.join(root, 'process.txt')
            # data_path = os.path.join(root, 'progress.txt')
            exp_data = pd.read_table(data_path)
            exp_data = exp_data[3:]
            all_data.append(exp_data)

    return all_data

def show_evaluate(data_set):
    for agent_data in data_set:
        if agent_data.empty:  # 检查 agent_data 是否为空
            print("警告: agent_data 为空，跳过该代理。")
            continue
        
        metrics_data = agent_data[['Success', 'Crashed', 'Costs', 
                                    'Rewards', 'AvgSpeed', 'Length']]
        plt.figure(figsize=(12, 6))
        plt.title(f'Boxplot for {agent_data["Agent"].unique()[0]}')
        sns.boxplot(data=metrics_data)
        plt.ylabel('Values')
        plt.xlabel('Metrics')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--file_path", type=str, default='eval_result/original/low')
    # parser.add_argument("--file_path", type=str, default='eval_result/ppo/mixed')
    parser.add_argument("--file_path", type=str, default='eval_result/baseline/eval_in_high_Baseline_SACD_MPC_1')
    
    args = parser.parse_args()

    all_data = get_all_data(args.file_path)
    get_evaluation_data(all_data)
    show_evaluate(all_data)  # 显示箱体图