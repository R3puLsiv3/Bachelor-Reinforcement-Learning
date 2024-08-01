import torch
import numpy as np

from agent import DQNAgent, BaseAgent, TestAgent, PPOAgent
from memory import UniformMemory, PrioritizedExperienceReplayMemory
from model import DQN, ActorCritic
from utils import create_trends_plot, create_learning_plot, create_cost_plot


def main():
    n_seeds = 10

    # Training parameters
    env_name = "environment:house_base"
    timesteps = 15_000
    batch_size = 64
    test_every = 1_000
    eps_decay = 0.9994

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
    alpha = 0.6
    beta = 0.4
    anneal = 1.000061

    agent = BaseAgent(env_name)
    rewards = agent.test(0)
    print(f"Baseline reward: {sum(rewards)}")

    agent = DQNAgent(env_name, timesteps, batch_size, test_every, eps_decay)

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
    create_trends_plot(data, "uni_dqn")

    # Compare rewards
    rewards = [item["reward"] for item in data]
    print(sum(rewards))
    agent = BaseAgent(env_name)
    baseline_rewards = agent.test(0)
    print(sum(baseline_rewards))
    create_cost_plot(rewards, baseline_rewards, "uni_dqn")

    env_name = "environment:house_base"
    agent = DQNAgent(env_name, timesteps, batch_size, test_every, eps_decay)

    # Training on uniform memory with double DQN
    torch.manual_seed(0)
    mean_rewards = []
    for seed in range(n_seeds):
        memory = UniformMemory(memory_state_size, memory_action_size, memory_size)
        model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=True)
        seed_reward, seed_std = agent.train(seed=seed, model=model, memory=memory)
        mean_rewards.append(seed_reward)
    mean_rewards = np.array(mean_rewards)
    mean_reward_double, std_reward_double = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)

    # Load best performing network
    best_model = torch.load("agent.pkl")
    best_model.eval()
    model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=True)
    model.model = best_model

    # Run on test environment and visualize results
    env_name = "environment:house_base_test"
    agent = TestAgent(env_name)
    data = agent.test(model)
    create_trends_plot(data, "uni_ddqn")

    # Compare rewards
    rewards = [item["reward"] for item in data]
    print(sum(rewards))
    agent = BaseAgent(env_name)
    baseline_rewards = agent.test(0)
    print(sum(baseline_rewards))
    create_cost_plot(rewards, baseline_rewards, "uni_ddqn")

    env_name = "environment:house_base"
    agent = DQNAgent(env_name, timesteps, batch_size, test_every, eps_decay)

    # Training on prioritized memory
    torch.manual_seed(0)
    mean_rewards = []
    for seed in range(n_seeds):
        memory = PrioritizedExperienceReplayMemory(memory_state_size, memory_action_size, memory_size, eps, alpha, beta, anneal)
        model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=False)
        seed_reward, seed_std = agent.train(seed=seed, model=model, memory=memory)
        mean_rewards.append(seed_reward)
    mean_rewards = np.array(mean_rewards)
    mean_priority_reward, std_priority_reward = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)

    # Load best performing network
    best_model = torch.load("agent.pkl")
    best_model.eval()
    model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=False)
    model.model = best_model

    # Run on test environment and visualize results
    env_name = "environment:house_base_test"
    agent = TestAgent(env_name)
    data = agent.test(model)
    create_trends_plot(data, "prio_dqn")

    # Compare rewards
    rewards = [item["reward"] for item in data]
    print(sum(rewards))
    agent = BaseAgent(env_name)
    baseline_rewards = agent.test(0)
    print(sum(baseline_rewards))
    create_cost_plot(rewards, baseline_rewards, "prio_dqn")

    env_name = "environment:house_base"
    agent = DQNAgent(env_name, timesteps, batch_size, test_every, eps_decay)

    # Training on prioritized memory with double DQN
    torch.manual_seed(0)
    mean_rewards = []
    for seed in range(n_seeds):
        memory = PrioritizedExperienceReplayMemory(memory_state_size, memory_action_size, memory_size, eps, alpha, beta, anneal)
        model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=True)
        seed_reward, seed_std = agent.train(seed=seed, model=model, memory=memory)
        mean_rewards.append(seed_reward)
    mean_rewards = np.array(mean_rewards)
    mean_priority_reward_double, std_priority_reward_double = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)

    # Load best performing network
    best_model = torch.load("agent.pkl")
    best_model.eval()
    model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=True)
    model.model = best_model

    # Run on test environment and visualize results
    env_name = "environment:house_base_test"
    agent = TestAgent(env_name)
    data = agent.test(model)
    create_trends_plot(data, "prio_ddqn")

    # Compare rewards
    rewards = [item["reward"] for item in data]
    print(sum(rewards))
    agent = BaseAgent(env_name)
    baseline_rewards = agent.test(0)
    print(sum(baseline_rewards))
    create_cost_plot(rewards, baseline_rewards, "prio_ddqn")

    create_learning_plot(test_every, mean_reward=mean_reward, std_reward=std_reward,
                            mean_reward_double=mean_reward_double, std_reward_double=std_reward_double,
                            mean_priority_reward=mean_priority_reward, std_priority_reward=std_priority_reward,
                            mean_priority_reward_double=mean_priority_reward_double, std_priority_reward_double=std_priority_reward_double)


if __name__ == "__main__":
    main()
