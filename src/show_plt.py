import matplotlib.pyplot as plt


def show_train(file_paths, plt_names):
    # read data
    all_data = []
    for file_path in file_paths:
        steps = []
        episode_rewards = []
        critic_losses = []
        actor_losses = []
        with open(file_path, 'r') as file:
            for line in file:
                data = eval(line)  # 使用eval函数将字符串转换为字典
                steps.append(data.get("step", 0))
                episode_rewards.append(data.get("episode_reward", 0))
                critic_losses.append(data.get("critic_loss", 0))
                actor_losses.append(data.get("actor_loss", 0))
        all_data.append((steps, episode_rewards, critic_losses, actor_losses))

    # show plt
    plt.figure(figsize=(10, 10))
    for idx, (steps, episode_rewards, critic_losses, actor_losses) in enumerate(all_data):
        # 将步数转换为百万步
        steps = [step / 1e6 for step in steps]

        # 绘制奖励图表
        plt.subplot(3, 1, 1)
        plt.plot(steps, episode_rewards, label=plt_names[idx])
        plt.xlabel('Steps (M)')
        plt.ylabel('Episode Reward')
        plt.title('Training Process - Episode Reward')
        plt.legend()

        # 绘制评论者损失图表
        plt.subplot(3, 1, 2)
        plt.plot(steps, critic_losses, label=plt_names[idx])
        plt.xlabel('Steps (M)')
        plt.ylabel('Critic Loss')
        plt.title('Training Process - Critic Loss')
        plt.legend()

        # 绘制演员损失图表
        plt.subplot(3, 1, 3)
        plt.plot(steps, actor_losses, label=f'Training {idx + 1}')
        plt.xlabel('Steps (M)')
        plt.ylabel('Actor Loss')
        plt.title('Training Process - Actor Loss')
        plt.legend()

    plt.tight_layout()
    plt.show()


def show_eval(file_paths, plt_names):
    # read data
    all_data = []
    for file_path in file_paths:
        steps = []
        episode_rewards = []
        episode_reward_test_env = []
        with open(file_path, 'r') as file:
            for line in file:
                data = eval(line)  # 使用eval函数将字符串转换为字典
                steps.append(data.get("step", 0))
                episode_rewards.append(data.get("episode_reward", 0))
                episode_reward_test_env.append(data.get("episode_reward_test_env", 0))
        all_data.append((steps, episode_rewards, episode_reward_test_env))

    # show plt
    plt.figure(figsize=(10, 10))
    for idx, (steps, episode_rewards, episode_reward_test_env) in enumerate(all_data):
        # 将步数转换为百万步
        steps = [step / 1e6 for step in steps]

        plt.subplot(2, 1, 1)
        plt.plot(steps, episode_rewards, label=plt_names[idx])
        plt.xlabel('Steps (M)')
        plt.ylabel('Episode Reward')
        plt.title('Evaluating Process - Episode Reward')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(steps, episode_reward_test_env, label=plt_names[idx])
        plt.xlabel('Steps (M)')
        plt.ylabel('Episode Reward')
        plt.title('Evaluating Process - Episode Reward In Test Env')
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    file_paths = ['../logs/walker_walk/sac/114514/train.log', '../logs/walker_walk/sac/114514/train_cam.log']
    plt_names = ['Baseline', 'Ours']
    show_train(file_paths, plt_names)
    file_paths = ['../logs/walker_walk/sac/114514/eval.log', '../logs/walker_walk/sac/114514/eval_cam.log']
    show_eval(file_paths, plt_names)
