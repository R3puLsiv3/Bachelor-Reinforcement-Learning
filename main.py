import torch
import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from memory import UniformMemory, PrioritizedExperienceReplayMemory
from dqn import DQN


def main():
    n_seeds = 10

    # Training parameters
    env_name = "CartPole-v0"
    timesteps = 50_000
    batch_size = 64
    test_every = 5000
    eps_max = 0.5
    eps_min = 0.05

    # Model parameters
    model_state_size = 4
    model_action_size = 2
    gamma = 0.99
    tau = 0.01
    lr = 1e-4

    # Memory parameters
    memory_state_size = 4
    memory_action_size = 1
    memory_size = 50_000
    eps = 1e-2
    alpha = 0.7
    beta = 0.4

    agent = Agent(env_name, timesteps, batch_size, test_every, eps_max, eps_min)

    # Training on uniform memory
    torch.manual_seed(0)
    mean_rewards = []
    for seed in range(n_seeds):
        memory = UniformMemory(memory_state_size, memory_action_size, memory_size)
        model = DQN(model_state_size, model_action_size, gamma, tau, lr)
        seed_reward, seed_std = agent.train(seed=seed, model=model, memory=memory)
        mean_rewards.append(seed_reward)
    mean_rewards = np.array(mean_rewards)
    mean_reward, std_reward = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)

    # Training on prioritized memory
    torch.manual_seed(0)
    mean_rewards = []
    for seed in range(n_seeds):
        memory = PrioritizedExperienceReplayMemory(memory_state_size, memory_action_size, memory_size, eps, alpha, beta)
        model = DQN(model_state_size, model_action_size, gamma, tau, lr)
        seed_reward, seed_std = agent.train(seed=seed, model=model, memory=memory)
        mean_rewards.append(seed_reward)
    mean_rewards = np.array(mean_rewards)
    mean_priority_reward, std_priority_reward = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)

    steps = np.arange(mean_reward.shape[0]) * test_every

    plt.plot(steps, mean_reward, label="Uniform")
    plt.fill_between(steps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.4)
    plt.plot(steps, mean_priority_reward, label="Prioritized")
    plt.fill_between(steps, mean_priority_reward - std_priority_reward, mean_priority_reward + std_priority_reward,
                     alpha=0.4)

    plt.legend()
    plt.title(env_name)
    plt.xlabel("Transitions")
    plt.ylabel("Reward")
    plt.savefig(f"{env_name}.jpg", dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    main()
