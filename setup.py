from setuptools import setup

packages = ['simple_env']
install_requires = [
    'gym',
]

entry_points = {
    'gym.envs': ['SimpleEnv=simple_env:register_envs']
}

setup(
    name='simple_env',
    version='0.1',
    description='A simple MDP Gym environment',
    url='https://github.com/will-maclean/SimpleEnv',
    author='Will Maclean',
    author_email='macleanwill1@gmail.com',
    license='GPL',
    packages=packages,
    entry_points=entry_points,
    install_requires=install_requires,
)
