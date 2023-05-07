import pickle
import matplotlib.pyplot as plt
import numpy as np


load_path = "/home/mj/Projects/taxi/rewards.pkl"

def plot_graph(load_path):    

    with open(load_path, 'rb') as f:
        r = pickle.load(f)

    r = np.array(r)

    print(r.shape)

    plt.plot(range(len(r)), r)
    plt.show()

def plot_graph_two(step_path, reward_path):
    with open(reward_path, 'rb') as f:
        r = pickle.load(f)

    with open(step_path, 'rb') as f:
        t = pickle.load(f)

    r = np.array(r)
    t = np.array(t)
    

    plt.plot(range(len(r)), r)
    plt.show()

step_path = "/home/mj/Projects/taxi/pytorch/mean_steps.pkl"
reward_path = "/home/mj/Projects/taxi/pytorch/mean_rewards.pkl"

plot_graph_two(step_path, reward_path)