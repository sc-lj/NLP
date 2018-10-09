# coding:utf-8

from gym import spaces
import gym
from random import random
import numpy as np
import math

class ConvS2SEnv(gym.Env):
    PENALTY = 1  # 0.999756079
    def __init__(self,vocab_size):
        self.vocab_size=vocab_size


        self.action_space=spaces.Discrete(vocab_size)
        self.observation_space=spaces.Box()


    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        return







