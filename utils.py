import os
import torch
import random
import numpy as np
from matplotlib import pyplot as plt


def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def device(force_cpu=True):
    return "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"


def create_trends_plot(data):
    demand = []
    grid_demand = []
    price = []
    soc = []
    action = []
    actual_action = []
    reward = []

    for data_point in data[:168]:
        demand.append(data_point["demand"])
        grid_demand.append(data_point["grid_demand"])
        price.append(data_point["price"])
        soc.append(data_point["soc"])
        action.append((data_point["action"]))
        actual_action.append(data_point["actual_action"])
        reward.append(data_point["reward"])

    x = np.arange(0, 168)

    fig, [ax0, ax1, ax2, ax3, ax4, ax5, ax6] = plt.subplots(nrows=7, ncols=1, figsize=(8, 21), tight_layout=True, sharex=True)

    ax0.plot(x, demand)
    ax0.set_title("Total Demand")
    ax0.set_ylabel("kW")
    ax1.plot(x, grid_demand)
    ax1.set_title("Grid Demand")
    ax1.set_ylabel("kW")
    ax2.plot(x, price)
    ax2.set_title("Price")
    ax2.set_ylabel("EUR per kWh")
    ax3.plot(x, soc)
    ax3.set_title("State of Charge")
    ax3.set_ylabel("Percent")
    ax4.plot(x, action)
    ax4.set_title("Action")
    ax4.set_ylabel("Percent charge/discharge")
    ax5.plot(x, actual_action)
    ax5.set_title("Actual Action")
    ax5.set_ylabel("Percent charge/discharge")
    ax6.plot(x, reward)
    ax6.set_title("Reward")
    ax6.set_ylabel("Amount")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(dir_path + "/plots/trends.jpg", dpi=200, bbox_inches='tight')
    plt.show()

