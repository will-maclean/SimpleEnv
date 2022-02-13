import gym
from operator import add, mul
from simple_env import SimpleEnv


class ValueFunction():
    def __init__(self):
        self._value_table = []
        self._transitions = []
        self._env_transition_probability = 0
        self._env_rewards = []
        self._env_terminal_states = []
        self.gamma = 0
        self.smallest_value_change = 0

    def get_value_of_state(self, state_element):

        return self._value_table[state_element]

    def set_value_of_state(self, state_element, new_value):

        self._value_table[state_element] = new_value

    def _get_next_states_values(self, state_element, multiply_discount=False):
        state_space = self._transitions[state_element]

        discount = 1

        if multiply_discount:
            discount = self.gamma

        next_state_values = [self.gamma * self.get_value_of_state(next_state_element) for next_state_element in state_space]

        return next_state_values

    def _get_next_states_rewards(self, state_element):
        num_next_states = len(self._transitions[state_element])

        next_state_rewards = [
            self._env_rewards[state_element][next_state_index] for next_state_index in range(num_next_states)
        ]

        return next_state_rewards

    def get_next_action(self, state_element):

        value_space = self._get_next_states_values(state_element)
        max_value = max(value_space)

        return value_space.index(max_value)

    def compute_value(self, state_element):
        q_values = []

        # Compute expected rewards + discounted value function
        next_states_values = list(
            map(add, self._get_next_states_rewards(state_element), self._get_next_states_values(state_element))
        )

        # For all actions within the action space from a given state
        for action in range(len(self._transitions[state_element])):
            # Compute q value of this state-action pair, by adding the expected reward of a state-action pair and the
            # discount factor multiplied by the sum of all possible next states transition probabilities times their
            # state value

            transition_probabilities = []
            for prob_index in range(len(next_states_values)):
                if action == prob_index:
                    transition_probabilities.append(self._env_transition_probability)
                else:
                    transition_probabilities.append(1 - self._env_transition_probability)

            q_value = sum(list(map(mul, transition_probabilities, next_states_values)))

            q_values.append(q_value)

        return max(q_values)

    def train(self, env_random_probability, largest_value_change, gamma):
        """
        Perform Value Iteration on the simple MDP environment
        :param env_random_probability: desired probability of transitioning to an unexpected state.
        :param largest_value_change:
        :param gamma:
        :return:
        """

        self._env_transition_probability = 1 - env_random_probability
        self.largest_value_change = largest_value_change
        self.gamma = gamma
        delta = 0

        env = SimpleEnv(env_random_probability)
        state_space = range(len(env.reset()))

        self._value_table = [0 for _ in state_space]
        self._env_rewards = env.get_mdp_rewards()
        self._transitions = env.get_mdp_transitions()
        self._env_terminal_states = env.get_terminal_states()

        iteration = 0

        while True:
            delta = 0

            for state_index in state_space:
                if state_index not in self._env_terminal_states:
                    current_value = self.get_value_of_state(state_index)

                    new_value = self.compute_value(state_index)

                    self._value_table[state_index] = new_value

                    delta = max(delta, abs(current_value - new_value))

            if iteration % 10 == 0:
                print("Iteration {}: Complete".format(iteration))
                print("Value Function is given below:")
                print(self._value_table)

            if delta < self.largest_value_change:
                print("Iteration {}: Complete".format(iteration))
                print("Value Function is given below:")
                print(self._value_table)
                break

            iteration += 1


if __name__ == "__main__":

    value_iteration = ValueFunction()

    value_iteration.train(
        env_random_probability=0.1,
        largest_value_change=1e-9,
        gamma=0.99
    )