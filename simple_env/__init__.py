from gym.envs import register
from .env import SimpleEnv


def register_envs():
    print("registering envs...")
    register(
        id="SimpleEnv-v0",
        entry_point="simple_env:SimpleEnv",
    )
