import torch
import matplotlib.pyplot as plt
import numpy as np

from agent import DQNAgent, BaseAgent, TestAgent, PPOAgent
from memory import UniformMemory, PrioritizedExperienceReplayMemory
from model import DQN, ActorCritic
from utils import create_trends_plot


def main():
    n_seeds = 1

    # Training parameters
    env_name = "environment:house_base"
    timesteps = 20_000
    batch_size = 64
    test_every = 1_000
    eps_max = 0.5
    eps_min = 0.01

    # Model parameters
    model_state_size = 3
    model_action_size = 21
    gamma = 0.99
    tau = 0.0001
    lr = 0.00005

    # Memory parameters
    memory_state_size = 3
    memory_action_size = 1
    memory_size = 50_000
    eps = 1e-2
    alpha = 0.7
    beta = 0.4

    agent = BaseAgent(env_name)
    rewards = agent.test(0)
    print(f"Baseline reward: {sum(rewards)}")

    agent = DQNAgent(env_name, timesteps, batch_size, test_every, eps_max, eps_min)

    # Training on uniform memory
    torch.manual_seed(0)
    mean_rewards = []
    for seed in range(n_seeds):
        memory = UniformMemory(memory_state_size, memory_action_size, memory_size)
        model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=False)
        seed_reward, seed_std = agent.train(seed=seed, model=model, memory=memory)
        mean_rewards.append(seed_reward)
    mean_rewards = np.array(mean_rewards)
    mean_reward, std_reward = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)

    # Load best performing network
    best_model = torch.load("agent.pkl")
    best_model.eval()
    model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=False)
    model.model = best_model

    # Run on test environment and visualize results
    env_name = "environment:house_base_test"
    agent = TestAgent(env_name)
    data = agent.test(model)
    create_trends_plot(data)


    # # Training on uniform memory with double DQN
    # torch.manual_seed(0)
    # mean_rewards = []
    # for seed in range(n_seeds):
    #     memory = UniformMemory(memory_state_size, memory_action_size, memory_size)
    #     model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=True)
    #     seed_reward, seed_std = agent.train(seed=seed, model=model, memory=memory)
    #     mean_rewards.append(seed_reward)
    # mean_rewards = np.array(mean_rewards)
    # mean_reward_double, std_reward_double = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)

    # # Training on prioritized memory
    # torch.manual_seed(0)
    # mean_rewards = []
    # for seed in range(n_seeds):
    #     memory = PrioritizedExperienceReplayMemory(memory_state_size, memory_action_size, memory_size, eps, alpha, beta)
    #     model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=False)
    #     seed_reward, seed_std = agent.train(seed=seed, model=model, memory=memory)
    #     mean_rewards.append(seed_reward)
    # mean_rewards = np.array(mean_rewards)
    # mean_priority_reward, std_priority_reward = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)
    #
    # # Training on prioritized memory with double DQN
    # torch.manual_seed(0)
    # mean_rewards = []
    # for seed in range(n_seeds):
    #     memory = PrioritizedExperienceReplayMemory(memory_state_size, memory_action_size, memory_size, eps, alpha, beta)
    #     model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=True)
    #     seed_reward, seed_std = agent.train(seed=seed, model=model, memory=memory)
    #     mean_rewards.append(seed_reward)
    # mean_rewards = np.array(mean_rewards)
    # mean_priority_reward_double, std_priority_reward_double = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)

    steps = np.arange(mean_reward.shape[0]) * test_every

    plt.plot(steps, mean_reward, label="Uniform")
    plt.fill_between(steps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.4)
    # plt.plot(steps, mean_reward_double, label="Uniform DDQN")
    # plt.fill_between(steps, mean_reward_double - std_reward_double, mean_reward_double + std_reward_double, alpha=0.4)
    # plt.plot(steps, mean_priority_reward, label="Prioritized")
    # plt.fill_between(steps, mean_priority_reward - std_priority_reward, mean_priority_reward + std_priority_reward, alpha=0.4)
    # plt.plot(steps, mean_priority_reward_double, label="Prioritized DDQN")
    # plt.fill_between(steps, mean_priority_reward_double - std_priority_reward_double, mean_priority_reward_double + std_priority_reward_double, alpha=0.4)

    plt.legend()
    plt.title("Learning Rate in Base Environment")
    plt.xlabel("Transitions")
    plt.ylabel("Reward")

    plt.savefig(f"{env_name}.jpg", dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
