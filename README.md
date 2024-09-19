# SACAM
Here are some gifs showing how agent moves in `/picture`.
***
## Contents
- [Overview](#Overview)
 - [CAM](#CAM)
 - [DMCGB](#DMCGB)
- [Setup](#Setup)
- [ModelDesign](#ModelDesign)
- [Discussion](#Discussion)
 - [Result](#Result)
 - [Limitation](#Limitation)
 - [Fulture](#Fulture)
***
## Overview
### CAM
### DMCGB
## Setup
## ModelDesign
## Discussion
### Result
### Limitation
### Fulture






**SETUP:**

1.Python==3.9

2.Install Mujoco_200 following [this](https://gist.github.com/ellisbrown/47bfd3e524aed11216cd3c0a0872a654)

3.[Pytorch](https://pytorch.org/get-started/locally/)

4.Run /setup/install_envs.sh

5.Uninstall gym, downgrade setuptools, install gym0.19.0, the file has been uploaded in SLACK

6.Install other dependencies, you can know these just run the code and concern the log



**COMMAND:**

You can get all commands from /src/arguments_cam.py

Commonly used: Python /src/train_cam.py --seed=114514 --save_video

The default algorithm is SAC, evaluate mode is color_hard

