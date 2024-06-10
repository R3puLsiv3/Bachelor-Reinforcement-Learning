import gymnasium as gym
import random
import numpy as np

from utils import device, set_seed
from memory import UniformMemory, PrioritizedExperienceReplayMemory


def evaluate_policy(env_name, agent, episodes=5, seed=0):
    env = gym.make(env_name)
    set_seed(env, seed=seed)

    returns = []
    for ep in range(episodes):
        done, total_reward = False, 0
        state, _ = env.reset(seed=seed + ep)

        while not done:
            state, reward, terminated, truncated, _ = env.step(agent.act(state))
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)


class Agent:
    def __init__(self, env_name, timesteps=50_000, batch_size=64, test_every=5000, eps_max=0.5, eps_min=0.05):
        self.env_name = env_name
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.test_every = test_every
        self.eps_max = eps_max
        self.eps_min = eps_min

    def train(self, model, memory, seed=0):
        print(f"Training on: {self.env_name}, Device: {device()}, Seed: {seed}")

        env = gym.make(self.env_name)

        rewards_total, stds_total = [], []
        loss_count, total_loss = 0, 0

        episodes = 0
        best_reward = -np.inf

        done = False
        state, _ = env.reset(seed=seed)

        for step in range(1, self.timesteps + 1):
            if done:
                done = False
                state, _ = env.reset(seed=seed)
                episodes += 1

            eps = self.eps_max - (self.eps_max - self.eps_min) * step / self.timesteps

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = model.act(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.add((state, action, reward, next_state, int(done)))

            state = next_state

            if step > self.batch_size:
                if isinstance(memory, UniformMemory):
                    batch = memory.sample(self.batch_size)
                    loss, td_error = model.update(batch)
                elif isinstance(memory, PrioritizedExperienceReplayMemory):
                    batch, weights, tree_idxs = memory.sample(self.batch_size)
                    loss, td_error = model.update(batch, weights=weights)

                    memory.update_priorities(tree_idxs, td_error.numpy())
                else:
                    raise RuntimeError("Unknown memory")

                total_loss += loss
                loss_count += 1

                if step % self.test_every == 0:
                    mean, std = evaluate_policy(self.env_name, model, episodes=10, seed=seed)

                    print(
                        f"Episode: {episodes}, Step: {step}, Reward mean: {mean:.2f}, Reward std: {std:.2f}, Loss: {total_loss / loss_count:.4f}, Eps: {eps}")

                    if mean > best_reward:
                        best_reward = mean
                        model.save()

                    rewards_total.append(mean)
                    stds_total.append(std)

        return np.array(rewards_total), np.array(stds_total)
