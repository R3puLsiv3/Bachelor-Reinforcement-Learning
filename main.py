import torch
import numpy as np
import os

from agent import DQNAgent, BaseAgent, TestAgent, QLearningAgent
from memory import UniformMemory, PrioritizedExperienceReplayMemory
from model import DQN, QTable
from utils import create_trends_plot, create_learning_plot, create_cost_plot


def main():
    n_seeds = 10

    # Training parameters
    env_name = "environment:house_base"
    test_env_name = "environment:house_base_test"
    timesteps = 20_000
    batch_size = 64
    test_every = 500
    eps_decay = 0.9997

    # Model parameters
    model_state_size = 3
    model_action_size = 21
    gamma = 0.99
    tau = 0.0001
    lr = 0.00001

    # Memory parameters
    memory_state_size = 3
    memory_action_size = 1
    memory_size = 100_000
    eps = 1e-2
    alpha = 0.6
    beta = 0.4
    anneal = 1.0000456

    # Q-Learning
    q_table = QTable(model_action_size)
    agent = QLearningAgent(env_name, 50_000_000, 500_000, 0.9999999, 0.0001, 0.99)
    agent.train(q_table, seed=0)

    agent = BaseAgent(env_name)
    rewards = agent.test(0)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(f"Baseline reward: {sum(rewards)}")

    agent = DQNAgent(env_name, timesteps, batch_size, test_every, eps_decay)
    test_agent = TestAgent(test_env_name)
    base_agent = BaseAgent(test_env_name)
    baseline_rewards = base_agent.test(0)

    # Training on uniform memory
    torch.manual_seed(0)
    mean_rewards = []
    mean_test_rewards = []
    rewards = []
    overall_best_reward = -np.inf
    for seed in range(n_seeds):
        memory = UniformMemory(memory_state_size, memory_action_size, memory_size)
        model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=False)
        seed_reward, seed_test_reward = agent.train(seed=seed, model=model, memory=memory)
        mean_rewards.append(seed_reward)
        mean_test_rewards.append(seed_test_reward)

        # Load best performing network
        best_model = torch.load(dir_path + "/models/current_best_model.pkl")
        best_model.eval()
        model.model = best_model

        # Record rewards and reset agent
        data = test_agent.test(model)
        reward = [item["reward"] for item in data]
        rewards.append(reward)
        total_reward = sum(reward)
        if total_reward > overall_best_reward:
            overall_best_reward = total_reward
            model.save("overall_best_model_dqn")
        agent.best_reward = -np.inf

    mean_rewards = np.array(mean_rewards)
    mean_test_rewards = np.array(mean_test_rewards)
    mean_reward, std_reward = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)
    mean_test_reward, std_test_reward = mean_test_rewards.mean(axis=0), mean_test_rewards.std(axis=0)

    # Visualize cost function
    rewards = np.array(rewards)
    rewards = rewards.mean(axis=0)
    create_cost_plot(rewards, baseline_rewards, "uni_dqn")

    # Load overall best performing network
    best_model = torch.load(dir_path + "/models/overall_best_model_dqn.pkl")
    best_model.eval()
    model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=False)
    model.model = best_model

    # Run on test environment and visualize results
    data = test_agent.test(model)
    create_trends_plot(data, "uni_dqn")
    reward = [item["reward"] for item in data]
    print(sum(reward))

    # Training on uniform memory with double DQN
    torch.manual_seed(0)
    mean_rewards = []
    mean_test_rewards = []
    rewards = []
    overall_best_reward = -np.inf
    for seed in range(n_seeds):
        memory = UniformMemory(memory_state_size, memory_action_size, memory_size)
        model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=True)
        seed_reward, seed_test_reward = agent.train(seed=seed, model=model, memory=memory)
        mean_rewards.append(seed_reward)
        mean_test_rewards.append(seed_test_reward)

        # Load best performing network
        best_model = torch.load(dir_path + "/models/current_best_model.pkl")
        best_model.eval()
        model.model = best_model

        # Record rewards and reset agent
        data = test_agent.test(model)
        reward = [item["reward"] for item in data]
        rewards.append(reward)
        total_reward = sum(reward)
        if total_reward > overall_best_reward:
            overall_best_reward = total_reward
            model.save("overall_best_model_ddqn")
        agent.best_reward = -np.inf

    mean_rewards = np.array(mean_rewards)
    mean_test_rewards = np.array(mean_test_rewards)
    mean_reward_double, std_reward_double = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)
    mean_test_reward_double, std_test_reward_double = mean_test_rewards.mean(axis=0), mean_test_rewards.std(axis=0)

    # Visualize cost function
    rewards = np.array(rewards)
    rewards = rewards.mean(axis=0)
    create_cost_plot(rewards, baseline_rewards, "uni_ddqn")

    # Load overall best performing network
    best_model = torch.load(dir_path + "/models/overall_best_model_ddqn.pkl")
    best_model.eval()
    model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=True)
    model.model = best_model

    # Run on test environment and visualize results
    data = test_agent.test(model)
    create_trends_plot(data, "uni_ddqn")
    reward = [item["reward"] for item in data]
    print(sum(reward))


    # Training on prioritized memory
    torch.manual_seed(0)
    mean_rewards = []
    mean_test_rewards = []
    rewards = []
    overall_best_reward = -np.inf
    for seed in range(n_seeds):
        memory = PrioritizedExperienceReplayMemory(memory_state_size, memory_action_size, memory_size, eps, alpha, beta, anneal)
        model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=False)
        seed_reward, seed_test_reward = agent.train(seed=seed, model=model, memory=memory)
        mean_rewards.append(seed_reward)
        mean_test_rewards.append(seed_test_reward)

        # Load best performing network
        best_model = torch.load(dir_path + "/models/current_best_model.pkl")
        best_model.eval()
        model.model = best_model

        # Record rewards and reset agent
        data = test_agent.test(model)
        reward = [item["reward"] for item in data]
        rewards.append(reward)
        total_reward = sum(reward)
        if total_reward > overall_best_reward:
            overall_best_reward = total_reward
            model.save("overall_best_model_dqn_prio")
        agent.best_reward = -np.inf

    mean_rewards = np.array(mean_rewards)
    mean_test_rewards = np.array(mean_test_rewards)
    mean_priority_reward, std_priority_reward = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)
    mean_test_priority_reward, std_test_priority_reward = mean_test_rewards.mean(axis=0), mean_test_rewards.std(axis=0)

    # Visualize cost function
    rewards = np.array(rewards)
    rewards = rewards.mean(axis=0)
    create_cost_plot(rewards, baseline_rewards, "prio_dqn")

    # Load overall best performing network
    best_model = torch.load(dir_path + "/models/overall_best_model_dqn_prio.pkl")
    best_model.eval()
    model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=False)
    model.model = best_model

    # Run on test environment and visualize results
    data = test_agent.test(model)
    create_trends_plot(data, "prio_dqn")
    reward = [item["reward"] for item in data]
    print(sum(reward))

    # Training on prioritized memory with double DQN
    torch.manual_seed(0)
    mean_rewards = []
    mean_test_rewards = []
    rewards = []
    overall_best_reward = -np.inf
    for seed in range(n_seeds):
        memory = PrioritizedExperienceReplayMemory(memory_state_size, memory_action_size, memory_size, eps, alpha, beta, anneal)
        model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=True)
        seed_reward, seed_test_reward = agent.train(seed=seed, model=model, memory=memory)
        mean_rewards.append(seed_reward)
        mean_test_rewards.append(seed_test_reward)

        # Load best performing network
        best_model = torch.load(dir_path + "/models/current_best_model.pkl")
        best_model.eval()
        model.model = best_model

        # Record rewards and reset agent
        data = test_agent.test(model)
        reward = [item["reward"] for item in data]
        rewards.append(reward)
        total_reward = sum(reward)
        if total_reward > overall_best_reward:
            overall_best_reward = total_reward
            model.save("overall_best_model_ddqn_prio")
        agent.best_reward = -np.inf

    mean_rewards = np.array(mean_rewards)
    mean_test_rewards = np.array(mean_test_rewards)
    mean_priority_reward_double, std_priority_reward_double = mean_rewards.mean(axis=0), mean_rewards.std(axis=0)
    mean_test_priority_reward_double, std_test_priority_reward_double = mean_test_rewards.mean(axis=0), mean_test_rewards.std(axis=0)

    # Visualize cost function
    rewards = np.array(rewards)
    rewards = rewards.mean(axis=0)
    create_cost_plot(rewards, baseline_rewards, "prio_ddqn")

    # Load overall best performing network
    best_model = torch.load(dir_path + "/models/overall_best_model_ddqn_prio.pkl")
    best_model.eval()
    model = DQN(model_state_size, model_action_size, gamma, tau, lr, double_dqn=True)
    model.model = best_model

    # Run on test environment and visualize results
    data = test_agent.test(model)
    create_trends_plot(data, "prio_ddqn")
    reward = [item["reward"] for item in data]
    print(sum(reward))

    create_learning_plot(test_every, "train", mean_reward=mean_reward, std_reward=std_reward,
                            mean_reward_double=mean_reward_double, std_reward_double=std_reward_double,
                            mean_priority_reward=mean_priority_reward, std_priority_reward=std_priority_reward,
                            mean_priority_reward_double=mean_priority_reward_double, std_priority_reward_double=std_priority_reward_double)

    create_learning_plot(test_every, "test", mean_reward=mean_test_reward, std_reward=std_test_reward,
                            mean_reward_double=mean_test_reward_double, std_reward_double=std_test_reward_double,
                            mean_priority_reward=mean_test_priority_reward, std_priority_reward=std_test_priority_reward,
                            mean_priority_reward_double=mean_test_priority_reward_double, std_priority_reward_double=std_test_priority_reward_double)


if __name__ == "__main__":
    main()
