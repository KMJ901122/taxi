import os
import gym
from config import Config
from model import DQN
from agent import QAgent

env = gym.make("Taxi-v3")
config = "config_pytorch.yaml"
for i in range(10):
    agent = QAgent(env=env, seed=i, config=config, model_class=DQN)
    agent.compile()
    agent.fit()

# env = gym.make("Taxi-v3").env
# env.seed(100)
# config = "config_pytorch_high_eps.yaml"
# agent = QAgent(env=env, config=config, model_class=DQN)
# agent.compile()
# agent.fit()