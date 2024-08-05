import gymnasium as gym
import random
import numpy as np
import torch

from utils import device, set_seed
from memory import UniformMemory, PrioritizedExperienceReplayMemory
from multiprocessing_env import SubprocVecEnv


def evaluate_policy(env_name, model, episodes=5, seed=0):
    env = gym.make(env_name)
    set_seed(env, seed=seed)

    returns = []
    for ep in range(episodes):
        done, total_reward = False, 0
        state, _ = env.reset(seed=seed + ep)

        while not done:
            state, reward, terminated, truncated, _ = env.step(model.act(state))
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns)


class DQNAgent:
    def __init__(self, env_name, timesteps=50_000, batch_size=64, test_every=5000, eps_decay=0.999):
        self.env_name = env_name
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.test_every = test_every
        self.eps_decay = eps_decay
        self.best_reward = -np.inf

    def train(self, model, memory, seed=0):
        print(f"Training on: {self.env_name}, Device: {device()}, Seed: {seed}")

        env = gym.make(self.env_name)

        rewards_total, rewards_test_total = [], []
        loss_count, total_loss = 0, 0

        episodes = 0
        eps = 1

        done = False
        state, _ = env.reset(seed=seed)

        for step in range(1, self.timesteps + 1):
            if done:
                done = False
                episodes += 1
                state, _ = env.reset(seed=seed+episodes)

            eps *= self.eps_decay

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
                    mean = evaluate_policy(self.env_name, model, episodes=1, seed=seed)

                    if mean > self.best_reward:
                        self.best_reward = mean
                        model.save("current_best_model")

                    rewards_total.append(mean)
                    mean_test = evaluate_policy("environment:house_base_test", model, episodes=1, seed=seed)
                    rewards_test_total.append(mean_test)

                    print(f"Episode: {episodes}, Step: {step}, Reward mean: {mean:.2f}, Reward test mean: {mean_test:.2f}, Loss: {total_loss / loss_count:.4f}, Eps: {eps}")

        return np.array(rewards_total), np.array(rewards_test_total)


class BaseAgent:
    def __init__(self, env_name):
        self.env_name = env_name

    def test(self, seed=0):

        env = gym.make(self.env_name)

        rewards = []

        done = False
        state, _ = env.reset(seed=seed)

        while not done:
            if done:
                break

            # Do not charge/discharge
            action = 10

            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rewards.append(reward)

        return rewards


class TestAgent:
    def __init__(self, env_name):
        self.env_name = env_name

    def test(self, model, seed=0):

        env = gym.make(self.env_name)

        data = []

        done = False
        state, _ = env.reset(seed=seed)

        while not done:
            if done:
                break

            # Choose action from best model
            action = model.act(state)

            # Use empty info for additional information
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            data_point = {"pv_gen": info["pv_gen"], "grid_demand": info["grid_demand"], "price": state[1], "soc": state[2], "action": info["action"], "reward": reward, "actual_action": info["actual_action"]}
            data.append(data_point)

            state = next_state

        return data


def evaluate_policy_q_learning(env_name, q_table, episodes=5, seed=0):
    env = gym.make(env_name)
    set_seed(env, seed=seed)

    returns = []
    for ep in range(episodes):
        done, total_reward = False, 0
        state, _ = env.reset(seed=seed + ep)
        state = q_table.preprocess_state(state)

        while not done:
            state, reward, terminated, truncated, _ = env.step(np.argmax(q_table.q_table[state]))
            state = q_table.preprocess_state(state)
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns)


class QLearningAgent:
    def __init__(self, env_name, timesteps, test_every, eps_decay, alpha, alpha_decay, gamma):
        self.env_name = env_name
        self.timesteps = timesteps
        self.test_every = test_every
        self.eps_decay = eps_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma

    def train(self, q_table, seed):
        env = gym.make(self.env_name)

        rewards_total, rewards_test_total = [], []

        episodes = 0
        eps = 1

        done = False
        state, _ = env.reset(seed=seed)
        state = q_table.preprocess_state(state)

        for step in range(1, self.timesteps + 1):
            if done:
                done = False
                episodes += 1
                state, _ = env.reset(seed=seed + episodes)
                state = q_table.preprocess_state(state)

            eps *= self.eps_decay

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table.q_table[state])


            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = q_table.preprocess_state(next_state)

            q_table.q_table[state][action] += self.alpha * (reward + self.gamma * max(q_table.q_table[next_state]) - q_table.q_table[state][action])

            self.alpha *= 0.99999
            state = next_state

            if step % self.test_every == 0:
                mean = evaluate_policy_q_learning(self.env_name, q_table, episodes=1, seed=seed)
                rewards_total.append(mean)
                mean_test = evaluate_policy_q_learning("environment:house_base_test", q_table, episodes=1, seed=seed)
                rewards_test_total.append(mean_test)

                print(
                    f"Episode: {episodes}, Step: {step}, Reward mean: {mean:.2f}, Reward test mean: {mean_test:.2f}, Eps: {eps}")

        return np.array(rewards_total), np.array(rewards_test_total)


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                       rand_ids, :]


def ppo_update(model, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs,
                                                                         returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()


class PPOAgent:
    def __init__(self, env_name, timesteps=50_000, batch_size=5, test_every=5000, num_steps=20, ppo_epochs=4):
        self.env_name = env_name
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.test_every = test_every
        self.num_steps = num_steps
        self.ppo_epochs = ppo_epochs

    def train(self, model, seed=0):
        print(f"Training on: {self.env_name}, Device: {device()}, Seed: {seed}")

        num_envs = 16

        def make_env():
            def _thunk():
                _env = gym.make(self.env_name)
                return _env
            return _thunk

        envs = [make_env() for i in range(num_envs)]
        envs = SubprocVecEnv(envs)

        state = envs.reset()

        rewards_total, stds_total = [], []
        step = 0

        while step < self.timesteps:
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(self.num_steps):
                state = torch.FloatTensor(state).to(device())
                dist, value = model(state)

                action = dist.sample()
                next_state, reward, done, _ = envs.step(action.cpu().numpy())

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device()))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device()))

                states.append(state)
                actions.append(action)

                state = next_state
                step += 1

                if step % self.test_every == 0:
                    mean, std = evaluate_policy(self.env_name, model, episodes=10, seed=seed)

                    print(f"Step: {step}, Reward mean: {mean:.2f}, Reward std: {std:.2f}")

                    rewards_total.append(mean)
                    stds_total.append(std)

            next_state = torch.FloatTensor(next_state).to(device())
            _, next_value = model(next_state)
            returns = compute_gae(next_value, rewards, masks, values)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values

            ppo_update(model, self.ppo_epochs, self.batch_size, states, actions, log_probs, returns, advantage)

        return np.array(rewards_total), np.array(stds_total)