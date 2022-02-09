# SimpleEnv
A simple MDP Gym enironment

State space: One hot vector, length = 5

Action space: Discrete(2)

## Installation
```
git clone https://github.com/will-maclean/SimpleEnv.git
cd SimpleEnv
pip install -e .
```

## Usage

```python
import gym

env = gym.make("SimpleEnv/SimpleEnv-v0")

state = env.reset()
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)
```
