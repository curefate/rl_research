# SACAM
Here are some gifs showing how agent moves in `/picture`.
***
# Contents
- [Overview](#Overview)
  - [CAM](#CAM)
  - [DMCGB](#DMCGB)
- [Setup](#Setup)
- [Command](#Command)
- [ModelDesign](#ModelDesign)
- [Discussion](#Discussion)
  - [Result](#Result)
  - [Limitation](#Limitation)
  - [Fulture](#Fulture)
***
## Overview
`SACAM` is a project which aim to improve the robustness of vision-based reinforcement learning. It's based on [SAC](https://arxiv.org/abs/1801.01290)(Soft Actor-Critic with maximum entropy) algorithm, which a algorithm with actor-ctiric structure and usually used as baseline. In normal case, the observation(input to agent's nerual network) of reinforcement learning are attributes that represents the state of the environment, for example in `Cartpole`, one of the most classic reinforcement learning environments, the observations are the position, velocity of carts, and angle, angular velocity of pole, agent will use these to output a best action it thinks in every step. But in actual cases, it is difficult to obtain observations similar to above example. While in humans case, the only thing we can get is what we see, it's an image sequence and we can reason about and understand the current state of the environment from it, that's what vision-based reinforcement learning want to achieve, i.e. use images as obversation for agent. 

However, directly changing observation to image usually leads to a significant performance degradation and is easily affected by irrelevant factors, unstable. The same example of Cartpole, to prevent pole from falling, we should focus on the moving trail of both cart and pole, everything else is irrelevant. But the thing is that if we change the color of cart, pole, even background, the agent cannot make correct decisions. There are many researchs about how to make vision-based reinforcement learning more robust, and the project here is aims to how to teach agent which part in observation image should be concentrated. We achieve this by [CAM](https://arxiv.org/abs/1512.04150)(class activation mapping). By adding the activation map of the observation extracted from the classifier pre-trained on Imagenet as an additional channel to the observation, the network can learn to focus on key parts to improve robustness.
### CAM
The implementation of CAM in this project comes from [here](https://github.com/frgfm/torch-cam).
### DMCGB
The enviroments of this project is from DMCGB(DMControl Generalization Benchmark), to make it work for this project, we modified their code, and the modified file will be mark with a `_cam` at last of file name.
## Setup
1.Python==3.9

2.Install Mujoco_200 following [this](https://gist.github.com/ellisbrown/47bfd3e524aed11216cd3c0a0872a654)

3.[Pytorch](https://pytorch.org/get-started/locally/)

4.Run /setup/install_envs.sh

5.Uninstall gym, downgrade setuptools, install gym0.19.0, the file has been uploaded in SLACK

6.Install other dependencies, you can know these just run the code and concern the log
## Command
You can get all commands from /src/arguments_cam.py

Commonly used: Python /src/train_cam.py --seed=114514 --save_video

The default algorithm is SAC, evaluate mode is color_hard
## ModelDesign
## Discussion
### Result
### Limitation
### Fulture


