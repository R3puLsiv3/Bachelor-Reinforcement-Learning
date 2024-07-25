import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from utils import device
from torch.distributions import Normal


class DQN:
    def __init__(self, state_size, action_size, gamma, tau, lr, double_dqn):
        self.model = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        ).to(device())
        self.target_model = deepcopy(self.model).to(device())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.double_dqn = double_dqn

    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def act(self, state):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float).to(device())
            action = torch.argmax(self.model(state)).cpu().numpy().item()
        return action

    def update(self, batch, weights=None):
        state, action, reward, next_state, done = batch

        Q = self.model(state)[torch.arange(len(action)), action.to(torch.long).flatten()]
        if self.double_dqn:
            next_q_values = self.model(next_state)
            next_q_state_values = self.target_model(next_state)

            Q_next = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        else:
            Q_next = self.target_model(next_state).max(dim=1).values

        Q_target = reward + self.gamma * (1 - done) * Q_next

        assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

        if weights is None:
            weights = torch.ones_like(Q)

        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target) ** 2 * weights)

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.model.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        with torch.no_grad():
            self.soft_update(self.target_model, self.model)

        return loss.item(), td_error

    def save(self):
        torch.save(self.model, "agent.pkl")


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, lr, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        ).to(device())

        self.actor = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        ).to(device())

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value
