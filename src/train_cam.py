import torch
import os
import numpy as np
import gym
import utils
import time
from arguments_cam import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
from torchcam.methods import SmoothGradCAMpp, CAM
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms.functional import resize, normalize, to_pil_image


def attach_heatmap(model, extractor, obs, args):
    assert isinstance(obs, utils.LazyFrames), 'Heatmap: obs are not Lazyframes'
    obs = np.array(obs)
    frames = None
    if args.cam_attach_mode == 1:
        frames = torch.empty(0).cuda()
        for i in range(args.frame_stack):
            subobs = torch.tensor(obs[i*3:i*3+3]).cuda()
            input_tensor = normalize(resize(subobs, [224, 224]) / 255.,
                               [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
            out = model(input_tensor)
            heatmap = extractor(out[0].argmax().item(), out)[0]
            heatmap = (resize(heatmap, [args.image_crop_size, args.image_crop_size]) * 255)
            slip = torch.cat((subobs.unsqueeze(0), heatmap.unsqueeze(0)), dim=1)
            frames = torch.cat((frames, slip))

    return utils.LazyFrames(frames.tolist())


def evaluate(env, agent, video, num_episodes, L, step, cam_model, cam_extractor, args, test_env=False):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        obs = attach_heatmap(cam_model, cam_extractor, obs, args)
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            obs = attach_heatmap(cam_model, cam_extractor, obs, args)
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
        mode='train'
    )
    test_env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed + 42,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.eval_mode,
        intensity=args.distracting_cs_intensity
    ) if args.eval_mode is not None else None

    # Create working directory
    cam_dir = args.cam_model + '_' + args.cam_extractor + '_mode' + str(args.cam_attach_mode)
    if args.cam_layer is not None:
        cam_dir += args.cam_layer
    work_dir = os.path.join(args.log_dir, args.domain_name + '_' + args.task_name, args.algorithm, cam_dir, str(args.seed))
    print('Working directory:', work_dir)
    assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    utils.write_info(args, os.path.join(work_dir, 'info.log'))

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    obs_shape = env.observation_space.shape
    if args.cam_attach_mode == 1:
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
    if args.cam_attach_mode == 1:
        cropped_obs_shape = (4 * args.frame_stack, args.image_crop_size, args.image_crop_size)
    print('Observations:', env.observation_space.shape)
    print('Cropped observations:', cropped_obs_shape)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=args
    )

    # Prepare CAM
    if args.cam_model == 'resnet18':
        cam_model = resnet18(weights=ResNet18_Weights.DEFAULT).cuda().eval()
    if args.cam_extractor == 'SmoothGradCAMpp':
        cam_extractor = SmoothGradCAMpp(cam_model, args.cam_layer)
    elif args.cam_extractor == 'CAM':
        cam_extractor = CAM(cam_model, args.cam_layer)

    start_step, episode, episode_reward, done = 0, 0, 0, True
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
                evaluate(env, agent, video, args.eval_episodes, L, step, cam_model, cam_extractor, args)
                if test_env is not None:
                    evaluate(test_env, agent, video, args.eval_episodes, L, step, cam_model, cam_extractor, args, test_env=True)
                L.dump(step)

            # Save agent periodically
            if step > start_step and step % args.save_freq == 0:
                torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            obs = attach_heatmap(cam_model, cam_extractor, obs, args)
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log('train/episode', episode, step)

        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        next_obs = attach_heatmap(cam_model, cam_extractor, next_obs, args)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs

        episode_step += 1

    print('Completed training for', work_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
