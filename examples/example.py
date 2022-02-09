import gym

if __name__ == "__main__":
    env = gym.make("SimpleEnv/SimpleEnv-v0")

    state = env.reset()

    print(state)

    action = 0

    next_state, reward, done, info = env.step(action)

    print(next_state)
