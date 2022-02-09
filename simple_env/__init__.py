from gym.envs import register
from .env import SimpleEnv


def register_envs():
    register(
        id="SimpleEnv-v0".format(name),
        entry_point="simple_env:SimpleEnv",
    )
