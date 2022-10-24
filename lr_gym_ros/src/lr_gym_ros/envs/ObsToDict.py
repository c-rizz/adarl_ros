#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

import gym
import lr_gym_ros.utils

from lr_gym_ros.envs.LrWrapper import LrWrapper

class ObsToDict(LrWrapper):

    def __init__(self,
                 env : lr_gym_ros.envs.BaseEnv,
                 key : str = "obs"):
        super().__init__(env=env)
        self._key = key
        self.observation_space = gym.spaces.Dict({key: env.observation_space})
        self.action_space = env.action_space
        self.metadata = env.metadata

    def getObservation(self, state):
        obs = self.env.getObservation(state)
        return {self._key : obs}