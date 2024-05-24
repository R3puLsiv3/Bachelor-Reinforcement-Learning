import gymnasium as gym

from agent import Agent


def main():
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    batch_size = 64
    memory_size = 20_000
    epsilon = 1
    min_epsilon = 0.01
    epsilon_decay = 0.999
    discount = 0.97
    target_update = 2_000

    agent = Agent(state_size, action_size, batch_size, memory_size, epsilon, min_epsilon, epsilon_decay, discount,
                  target_update)

    num_episodes = 5_000

    for episode in range(num_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        sum_reward = 0
        while not (terminated or truncated):
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            sum_reward += reward

            agent.memory.add(state, action, reward, next_state, terminated or truncated)
            state = next_state

            agent.train()

        print(f"Episode {episode}: Success = {terminated}, Reward = {sum_reward}")

        agent.decay_epsilon()


if __name__ == "__main__":
    main()
