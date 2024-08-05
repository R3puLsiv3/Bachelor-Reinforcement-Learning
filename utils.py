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


def create_trends_plot(data, method="uni_dqn"):
    pv_gen = []
    grid_demand = []
    price = []
    soc = []
    action = []
    actual_action = []
    reward = []

    hours_per_week = 168

    for data_point in data[:hours_per_week]:
        pv_gen.append(data_point["pv_gen"])
        grid_demand.append(data_point["grid_demand"])
        price.append(data_point["price"])
        soc.append(data_point["soc"])
        action.append((data_point["action"]))
        actual_action.append(data_point["actual_action"])
        reward.append(data_point["reward"])

    x = np.arange(0, hours_per_week)

    fig, [ax0, ax1, ax2, ax3, ax4, ax5, ax6] = plt.subplots(nrows=7, ncols=1, figsize=(8, 21), tight_layout=True, sharex=True)

    ax0.plot(x, pv_gen)
    ax0.set_title("PV Generation")
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
    match method:
        case "uni_dqn":
            plt.savefig(dir_path + "/plots/trends_uni_dqn.pdf", dpi=200, bbox_inches='tight')
        case "uni_ddqn":
            plt.savefig(dir_path + "/plots/trends_uni_ddqn.pdf", dpi=200, bbox_inches='tight')
        case "prio_dqn":
            plt.savefig(dir_path + "/plots/trends_prio_dqn.pdf", dpi=200, bbox_inches='tight')
        case "prio_ddqn":
            plt.savefig(dir_path + "/plots/trends_prio_ddqn.pdf", dpi=200, bbox_inches='tight')
    plt.show()


def create_learning_plot(test_every, mode, mean_reward=None, std_reward=None,
                            mean_reward_double=None, std_reward_double=None,
                            mean_priority_reward=None, std_priority_reward=None,
                            mean_priority_reward_double=None, std_priority_reward_double=None):
    if not (mean_reward is None or std_reward is None):
        steps = np.arange(mean_reward.shape[0]) * test_every
        plt.plot(steps, mean_reward, label="Uniform")
        plt.fill_between(steps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.4)
    if not (mean_reward_double is None or std_reward_double is None):
        steps = np.arange(mean_reward_double.shape[0]) * test_every
        plt.plot(steps, mean_reward_double, label="Uniform DDQN")
        plt.fill_between(steps, mean_reward_double - std_reward_double, mean_reward_double + std_reward_double, alpha=0.4)
    if not (mean_priority_reward is None or std_priority_reward is None):
        steps = np.arange(mean_priority_reward.shape[0]) * test_every
        plt.plot(steps, mean_priority_reward, label="Prioritized")
        plt.fill_between(steps, mean_priority_reward - std_priority_reward, mean_priority_reward + std_priority_reward, alpha=0.4)
    if not (mean_priority_reward_double is None or std_priority_reward_double is None):
        steps = np.arange(mean_priority_reward_double.shape[0]) * test_every
        plt.plot(steps, mean_priority_reward_double, label="Prioritized DDQN")
        plt.fill_between(steps, mean_priority_reward_double - std_priority_reward_double, mean_priority_reward_double + std_priority_reward_double, alpha=0.4)

    plt.legend()
    plt.title("Learning Progress")
    plt.xlabel("Steps")
    plt.ylabel("Reward")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if mode == "train":
        plt.savefig(dir_path + "/plots/training_progress.pdf", dpi=200, bbox_inches='tight')
    if mode == "test":
        plt.savefig(dir_path + "/plots/testing_progress.pdf", dpi=200, bbox_inches='tight')
    plt.show()


def create_cost_plot(rewards, baseline_rewards, method="uni_dqn"):
    x = np.arange(0, len(rewards))
    y1 = np.cumsum(np.abs(rewards))
    y2 = np.cumsum(np.abs(baseline_rewards))

    plt.plot(x, y1, label="Test")
    plt.fill_between(x, y1)
    plt.plot(x, y2, label="Baseline")
    plt.fill_between(x, y2, y1)

    plt.legend()
    plt.title("Cost Comparison")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Cost")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    match method:
        case "uni_dqn":
            plt.savefig(dir_path + "/plots/cost_comparison_uni_dqn.pdf", dpi=200, bbox_inches='tight')
        case "uni_ddqn":
            plt.savefig(dir_path + "/plots/cost_comparison_uni_ddqn.pdf", dpi=200, bbox_inches='tight')
        case "prio_dqn":
            plt.savefig(dir_path + "/plots/cost_comparison_prio_dqn.pdf", dpi=200, bbox_inches='tight')
        case "prio_ddqn":
            plt.savefig(dir_path + "/plots/cost_comparison_prio_ddqn.pdf", dpi=200, bbox_inches='tight')
    plt.show()
