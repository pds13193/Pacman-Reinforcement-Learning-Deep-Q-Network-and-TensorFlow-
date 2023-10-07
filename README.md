# Pacman Reinforcement Learning with TensorFlow and DQN

## Overview

This repository contains the implementation of the classic Pacman game using the Deep Q-Network (DQN) algorithm in conjunction with TensorFlow. The goal of this project was to create an intelligent agent capable of autonomously playing Pacman by leveraging deep reinforcement learning techniques.

## Key Features

- Reinforcement learning with DQN
- TensorFlow-based neural network architecture
- Autonomous Pacman gameplay

## Project Structure

- `deepQNetwork.py`: Implementation of the DQN algorithm.
- `dqnPacmanAgents.py`: Integration of DQN with Pacman game.

## Getting Started

To execute the code, use the below command:

python3 pacman.py -p <agent name> -n <total number of episodes> -x <number of training episodes> -l smallGrid

Ex: python3 pacman.py -p PacmanDQN -n 2 -x 1 -l smallGrid
