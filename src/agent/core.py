import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = (observation_space.shape[0] - 1) * observation_space.shape[1]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
        self.vc = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)
        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class SAC_Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(SAC_Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs

    def evaluate(self, state, action_mask=None):
        """
        评估状态并返回动作
        Args:
            state: 状态输入
            action_mask: 动作掩码，用于屏蔽非法动作
        """
        action_probs = self.forward(state)
        
        # 如果提供了动作掩码，则应用掩码
        if action_mask is not None:
            if action_mask.shape != action_probs.shape:
                print(f"Mask shape: {action_mask.shape}, Probs shape: {action_probs.shape}")
                raise ValueError("Action mask shape does not match action probabilities shape.")
            
            # 应用掩码并重新归一化
            masked_probs = action_probs * action_mask
            # 确保每个样本的和不为0
            sum_probs = masked_probs.sum(dim=-1, keepdim=True)
            valid_mask = (sum_probs > 0).float()
            
            # 对有效的样本进行归一化，无效样本保持均匀分布
            action_probs = torch.where(
                valid_mask == 1,
                masked_probs / (sum_probs + 1e-10),
                torch.ones_like(masked_probs) / masked_probs.size(-1)
            )

        # 数值稳定性处理
        action_probs = torch.clamp(action_probs, min=1e-10, max=1.0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        # 检查 action_probs 的有效性
        if (action_probs < 0).any():
            print("Warning: Negative probabilities detected")
            action_probs = F.relu(action_probs)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            
        if torch.isnan(action_probs).any():
            print("Warning: NaN values detected in probabilities")
            action_probs = torch.where(torch.isnan(action_probs), 
                                     torch.ones_like(action_probs) / action_probs.size(-1),
                                     action_probs)
            
        # 使用更宽松的容差检查
        if not torch.allclose(action_probs.sum(dim=-1), torch.ones_like(action_probs.sum(dim=-1)), rtol=1e-4, atol=1e-4):
            print(f"Sum of probabilities: {action_probs.sum(dim=-1)}")
            print(f"Action probabilities: {action_probs}")
            raise ValueError("Action probabilities do not sum to 1.")

        # 创建分布并采样
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # 计算对数概率
        log_action_probabilities = torch.log(action_probs + 1e-10)
        
        return action.detach().cpu(), action_probs, log_action_probabilities

    def step(self, state, action_mask=None):
        """
        执行一步动作
        Args:
            state: 状态输入
            action_mask: 动作掩码
        """
        not_use1, not_use2, not_use3 = 0, 0, 0
        action_probs = self.forward(state)
        
        # 如果提供了动作掩码，则应用掩码
        if action_mask is not None:
            if action_mask.shape != action_probs.shape:
                raise ValueError("Action mask shape does not match action probabilities shape.")
            action_probs = action_probs * action_mask
            action_probs = action_probs / (action_probs.sum() + 1e-10)

        dist = Categorical(action_probs)
        action = dist.sample()
        return action.detach().cpu(), not_use1, not_use2, not_use3

class SAC_Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_size (int): Number of nodes in the network layers
        """
        super(SAC_Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


