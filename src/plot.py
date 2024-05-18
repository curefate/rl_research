import json
import matplotlib.pyplot as plt
import os

def read_train_data(file_path):
    steps = []
    rewards = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            steps.append(data['step'])
            rewards.append(data['episode_reward'])
    return steps, rewards

def read_eval_data(file_path):
    steps = []
    eval_rewards = []
    eval_test_rewards = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            steps.append(data['step'])
            eval_rewards.append(data['episode_reward'])
            eval_test_rewards.append(data['episode_reward_test_env'])
    return steps, eval_rewards, eval_test_rewards

task = 'walker_walk'
seed = 42
method = 'resnet18_SmoothGradCAMpp_mode_3'
train_steps, train_rewards = read_train_data(f'./../logs/{task}/soda/{method}/{seed}/train.log')
steps, eval_rewards, eval_test_rewards = read_eval_data(f'./../logs/{task}/soda/{method}/{seed}/eval.log')

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_rewards, color='blue')
plt.plot(steps, eval_rewards, color='red')
plt.plot(steps, eval_test_rewards, color='green')

plt.xlabel('Step')
plt.ylabel('Reward')
plt.savefig(f'reward_{method}_{seed}.png')
