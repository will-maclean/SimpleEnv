from typing import Optional, List

import gym
import numpy as np


class Node:
    def __init__(self, 
               idx: int,
               nodes: Optional[List['Node']]=None, 
               rewards: Optional[List['Node']]=None,
               terminal=False):
        
        if nodes is not None or rewards is not None:
            assert len(nodes) == len(rewards)
        
        self.id = idx
        self.nodes = nodes if nodes is not None else []
        self.rewards = rewards if rewards is not None else []

        self.terminal = terminal

    def add_connection(self, node, reward):
        self.nodes.append(node)
        self.rewards.append(reward)


class SimpleEnv(gym.Env):
    def __init__(self, random_prob=0.1):

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(5)
        self.random_prob = random_prob
        
        # create nodes
        self.nodes = [Node(i) for i in range(5)]
        
        # set final node(s)
        self.nodes[4].terminal = True

        # set initial node(s)
        self.initial_nodes = [self.nodes[0], self.nodes[1]]

        # create connections

        self.nodes[0].add_connection(self.nodes[1], 0)
        self.nodes[0].add_connection(self.nodes[2], -0.1)

        self.nodes[1].add_connection(self.nodes[2], -0.1)
        self.nodes[1].add_connection(self.nodes[3], -0.1)

        self.nodes[2].add_connection(self.nodes[2], -0.01)
        self.nodes[2].add_connection(self.nodes[4], 0)

        self.nodes[3].add_connection(self.nodes[4], 1)
        self.nodes[3].add_connection(self.nodes[0], 0)

        self.current_node = None
        self.needs_reset = True

        # Make sure our action space is set up properly
        self._check_mdp()

    def _check_mdp(self):
        n_actions = self.action_space.n

        for node in self.nodes:
            if not node.terminal:
                assert n_actions == len(node.nodes)

    def step(self, action):
        if self.needs_reset:
            raise ValueError("You need to rest the environment first!")

        if np.random.random() < self.random_prob:
            # random action
            action = np.random.randint(2)
        
        next_node = self.current_node.nodes[action]
        reward = self.current_node.rewards[action]
        done = next_node.terminal

        self.current_node = next_node

        if done:
            self.needs_reset = True

        return self._obs(), reward, done, {}


    def reset(self):
        init_node = np.random.randint(len(self.initial_nodes))
        self.current_node = self.initial_nodes[init_node]
        self.needs_reset = False
        return self._obs()
    
    def _obs(self):
        vec = np.zeros(self.observation_space.n)
        vec[self.current_node.id] = 1

        return vec
