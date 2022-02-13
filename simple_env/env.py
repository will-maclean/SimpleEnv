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
        n_obs = 6

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=np.zeros(n_obs), high=np.ones(n_obs), dtype=np.float32)
        self.random_prob = random_prob
        
        # create nodes
        self.nodes = [Node(i) for i in range(6)]
        
        # set final node(s)
        self.nodes[4].terminal = True
        self.nodes[5].terminal = True

        # set initial node(s)
        self.initial_nodes = [self.nodes[0], self.nodes[1]]

        # create connections

        self.nodes[0].add_connection(self.nodes[0], -0.1)
        self.nodes[0].add_connection(self.nodes[2], -0.2)
        self.nodes[0].add_connection(self.nodes[1], 0)

        self.nodes[1].add_connection(self.nodes[2], -0.2)
        self.nodes[1].add_connection(self.nodes[3], -0.1)
        self.nodes[1].add_connection(self.nodes[5], -1)

        self.nodes[2].add_connection(self.nodes[2], -0.1)
        self.nodes[2].add_connection(self.nodes[4], 1)
        self.nodes[2].add_connection(self.nodes[3], 0)

        self.nodes[3].add_connection(self.nodes[4], 1)
        self.nodes[3].add_connection(self.nodes[5], -1)
        self.nodes[3].add_connection(self.nodes[0], -0.1)

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

            # Define action space
            n_actions = self.action_space.n
            action_space = list(range(n_actions)).remove(action)

            # Select random action
            action = np.random.randint(n_actions)

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
        vec = np.zeros(self.observation_space.shape[0])
        vec[self.current_node.id] = 1

        return vec

    def __generate_mdp_transitions_list(self):
        """
        Private method to generate list that represents state-action transitions
        :return: List of transitions
        """
        transitions = []

        for node in self.nodes:
            current_state_transitions = []
            for connected_node in node.nodes:
                current_state_transitions.append(connected_node.id)

            transitions.append(current_state_transitions)

        return transitions

    def get_mdp_transitions(self):
        """
        Getter method to get list of state-action transitions
        :return: List of transitions
        """
        return self.__generate_mdp_transitions_list()

    def __generate_mdp_rewards_list(self):
        """
        Private Method to generate list that represents the reward function
        :return: List of rewards [S, S']
        """
        rewards = []

        for node in self.nodes:
            rewards.append(node.rewards)

        return rewards

    def get_mdp_rewards(self):
        """
        Getter Method to get list of rewards
        :return: List of rewards [S, S']
        """
        return self.__generate_mdp_rewards_list()

    def get_terminal_states(self):
        """
        Getter Method to get a list of the terminal states
        :return: List of terminal states
        """
        terminal_states = [node.id for node in self.nodes if node.terminal]

        return terminal_states
