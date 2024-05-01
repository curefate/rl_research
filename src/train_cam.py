import torch
import os
import re
import numpy as np
import gym
import utils
import time
from arguments_cam import parse_args
from env.wrappers_cam import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
import gc
import json


def evaluate(env, agent, video, num_episodes, L, step, test_env=False):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        if L is not None:
            _test_env = '_test_env' if test_env else ''
            video.save(f'{step}{_test_env}.mp4')
            L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


def main(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode='train',
        intensity=0.,
        cam_model=args.cam_model,
        cam_extractor=args.cam_extractor,
        cam_layer=args.cam_layer,
        cam_mode=args.cam_mode
    )
    test_env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed + 42,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.eval_mode,
        intensity=args.distracting_cs_intensity,
        cam_model=args.cam_model,
        cam_extractor=args.cam_extractor,
        cam_layer=args.cam_layer,
        cam_mode=args.cam_mode
    ) if args.eval_mode is not None else None

    # Create working directory
    cam_dir = args.cam_model + '_' + args.cam_extractor + '_mode_' + str(args.cam_mode)
    if args.cam_layer is not None:
        cam_dir += '_' + args.cam_layer
    work_dir = os.path.join(args.log_dir, args.domain_name + '_' + args.task_name, args.algorithm, cam_dir,
                            str(args.seed))
    if args.work_dir is not None:
        work_dir = args.work_dir
    print('Working directory:', work_dir)
    # assert not os.path.exists(os.path.join(work_dir, 'train.log')) or args.ckpt_path is not None, \
    #     'specified working directory already exists'
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    utils.write_info(args, os.path.join(work_dir, 'info.log'))
    rerun_dir = utils.make_dir(os.path.join(work_dir, 'rerun'))

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    obs_shape = env.observation_space.shape
    if args.cam_mode == 1:
        obs_shape = list(obs_shape)
        obs_shape[0] += args.frame_stack
        obs_shape = tuple(obs_shape)
    replay_buffer = utils.ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size
    )
    cropped_obs_shape = (3 * args.frame_stack, args.image_crop_size, args.image_crop_size)
    if args.cam_mode == 1:
        cropped_obs_shape = (4 * args.frame_stack, args.image_crop_size, args.image_crop_size)
    print('Observations:', env.observation_space.shape)
    print('Cropped observations:', cropped_obs_shape)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=args
    )
    # if args.ckpt_path is not None:
    #     ckpt_path = args.ckpt_path
    #     if not os.path.exists(ckpt_path):
    #         ckpt_path = os.path.join(model_dir, ckpt_path)
    #     assert os.path.exists(ckpt_path), 'Checkpoint does not exist'
    #     agent = torch.load(ckpt_path)
    #     print('Loaded checkpoint:', ckpt_path)
    #     if args.start_steps == 0:
    #         args.start_steps = int(re.findall(r'\d+', os.path.basename(ckpt_path))[0])
    #     print('Start step count:', args.start_steps)
    #     print('Start episode count:', args.start_episodes)

    last_run_checkpoint = os.path.join(rerun_dir, 'last.pt')
    if os.path.exists(last_run_checkpoint):
        agent = torch.load(last_run_checkpoint)
    last_run_info = os.path.join(rerun_dir, 'last_run.json')
    if os.path.exists(last_run_info):
        with open(last_run_info, "r") as json_file:
            loaded_data = json.load(json_file)
            args.start_steps = int(loaded_data['step'])
            args.start_episodes = int(loaded_data['episode'])

    start_step, episode, episode_reward, done = args.start_steps, args.start_episodes, 0, True
    L = Logger(work_dir)
    start_time = time.time()
    for step in range(start_step, args.train_steps + 1):
        if done:
            if step > start_step:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # Evaluate agent periodically
            if step % args.eval_freq == 0:
                print('Evaluating:', work_dir)
                L.log('eval/episode', episode, step)
                evaluate(env, agent, video, args.eval_episodes, L, step)
                if test_env is not None:
                    evaluate(test_env, agent, video, args.eval_episodes, L, step, test_env=True)
                L.dump(step)

                # collect garbage in evaluation
                gc.collect()

            # Save agent periodically
            if step > start_step and step % args.save_freq == 0:
                torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

            # save checkpoint and information of the last done
            torch.save(agent, os.path.join(rerun_dir, 'last.pt'))
            with open(os.path.join(rerun_dir, 'last_run.json'), 'w') as f:
                info = {
                    'step': step,
                    'episode': episode
                }
                json.dump(info, f, indent=4, separators=(',', ': '))

        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if step >= (args.init_steps + start_step):
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs

        episode_step += 1

    print('Completed training for', work_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
