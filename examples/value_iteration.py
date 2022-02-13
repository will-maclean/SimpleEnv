import gym
from operator import add, mul
from simple_env import SimpleEnv


class ValueFunction():
    def __init__(self):
        """
        Value Function Class to execute a vanilla Value Iteration Algorithm as described by
        Richard S. Sutton and Andrew G. Barto in their book "Reinforcement Learning: An Introduction".
        http://incompleteideas.net/book/ebook/node44.html
        """

        # Initialise value table, environment dynamics, and learning requirements as variables
        self._value_table = []
        self._transitions = []
        self._env_transition_probability = 0
        self._env_rewards = []
        self._env_terminal_states = []
        self.gamma = 0
        self.smallest_value_change = 0

    def get_value_of_state(self, state_index):
        """
        Getter to get value of state from Value Function
        :param state_index: index of target state
        :return: value of state as given by the value function
        """

        return self._value_table[state_index]

    def set_value_of_state(self, state_index, new_value):
        """
        Setter to set value of state for the Value Function
        :param state_index: index of target state
        :param new_value: new value of V(s)
        """

        self._value_table[state_index] = new_value

    def __get_next_states_values(self, state_index, multiply_discount=False):
        """
        Private Method to get next state values
        :param state_index: index of target state
        :param multiply_discount: Declare whether we are multiplying our V(s) by our discount factor gamma
        :return: Values V(s') for all next states from the current state
        """

        # Get state-space
        state_space = self._transitions[state_index]

        # Check if we are discounting
        discount = 1
        if multiply_discount:
            discount = self.gamma

        # Calculate next state values
        next_state_values = [self.gamma * self.get_value_of_state(action_index) for action_index in state_space]

        return next_state_values

    def __get_next_states_rewards(self, state_index):
        """
        Privare method to get the expected rewards of each of the next states given (state, action) pairs
        :param state_index: index of target state
        :return: Expected next state rewards
        """

        # Get number of viable next states
        num_next_states = len(self._transitions[state_index])

        # Get a list of all the next states rewards
        next_state_rewards = [
            self._env_rewards[state_index][next_state_index] for next_state_index in range(num_next_states)
        ]

        return next_state_rewards

    def get_next_action(self, state_index):
        """
        Method to get next action if one was to use Value Iteration in a Planning and Control Setting
        :param state_index: index of target state
        :return: Next action given current Value Function
        """

        # Get next states values
        value_space = self.__get_next_states_values(state_index)

        # Get the max value
        max_value = max(value_space)

        # Return action by selecting index at which max value is stored in the value space
        return value_space.index(max_value)

    def compute_value(self, state_index):
        """
        Method computes the new Value V(s)
        :param state_index: index of target state
        :return: New V(s)
        """
        q_values = []

        # Compute expected rewards + discounted value function
        next_states_values = list(
            map(add, self.__get_next_states_rewards(state_index), self.__get_next_states_values(state_index, True))
        )

        # For all actions within the action space from a given state
        for action in range(len(self._transitions[state_index])):
            # Need to create a list of the transition probabilities for every given action
            transition_probabilities = []
            for prob_index in range(len(next_states_values)):
                if action == prob_index:
                    transition_probabilities.append(self._env_transition_probability)
                else:
                    transition_probabilities.append(1 - self._env_transition_probability)

            # Compute q value of this state-action pair, by adding the expected reward of a state-action pair and the
            # discount factor multiplied by the sum of all possible next states transition probabilities times their
            # state value
            q_value = sum(list(map(mul, transition_probabilities, next_states_values)))
            q_values.append(q_value)

        # Return the new value which is the max Q-Value
        return max(q_values)

    def log_value_function(self, iteration):
        """
        Method to log the current state of the value function
        :param iteration: Iteration of the training loop
        """
        print("Iteration {}: Complete".format(iteration))
        print("Value Function is given below:")
        print(self._value_table)


    def train(self, env_random_probability, largest_value_change, gamma):
        """
        Perform Value Iteration on the simple MDP environment
        :param env_random_probability: desired probability of transitioning to an unexpected state.
        :param largest_value_change: Largest value at which the change can occur for training to stop
        :param gamma: Discount Factor
        """

        # Environment Dynamics and Necessary Training Setup
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

        # Training Loop
        while True:
            # Set the Delta to zero
            delta = 0

            # Loop through every state in the state-space
            for state_index in state_space:
                # Determine if the current state is a terminal state
                if state_index not in self._env_terminal_states:

                    # Get current value v
                    current_value = self.get_value_of_state(state_index)

                    # Compute V(s)
                    new_value = self.compute_value(state_index)

                    # Set new value for V(s)
                    self._value_table[state_index] = new_value

                    # Determine Delta
                    delta = max(delta, abs(current_value - new_value))

            # Check to see if it is time to log results
            if iteration % 5 == 0:
                self.log_value_function(iteration)

            # Check to see if training is complete
            if delta < self.largest_value_change:
                self.log_value_function(iteration)
                break

            # Increment Iteration
            iteration += 1


if __name__ == "__main__":

    # Instantiate ValueFunction() for Value Iteration Algorithm
    value_iteration = ValueFunction()

    # Perform experiment with Value Iteration on the SimpleEnv MDP
    value_iteration.train(
        env_random_probability=0.1,
        largest_value_change=1e-9,
        gamma=0.99
    )