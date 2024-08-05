import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple

action_size = 5

def update(
        value_net, value_optim, observations: List[np.ndarray], 
        actions: List[int], rewards: List[float], gamma: float
        ) -> Dict[str, float]:
        # Initialise loss and returns
        v_loss = 0
        p_loss = 0
        traj_length = len(observations)

        # # Normalise rewards
        # rewards = np.array(rewards)
        # reward_mean = np.mean(rewards)
        # reward_std = np.std(rewards) + 1e-10  # Add a small value to avoid division by zero
        # normalised_rewards = (rewards - reward_mean) / reward_std

        # Flatten each observation and create one-hot encodings for actions
        flattened_observations = [obs.flatten() for obs in observations]
        one_hot_actions = [np.eye(action_size, dtype=int)[int(action)] for action in actions]
        
        # Concatenate flattened observations with one-hot actions
        state_action_pairs = [np.concatenate((obs, act)) for obs, act in zip(flattened_observations, one_hot_actions)]
        
        # Convert state-action pairs to a tensor
        state_action_tensor = torch.tensor(state_action_pairs, dtype=torch.float32)

        # Compute baseline values using the current value network
        baseline_values = value_net(state_action_tensor).squeeze()

        # obs_tensor = torch.tensor(flattened_observations, dtype=torch.float32)
        # # Compute action probabilities using the current policy
        # eps = 1e-10
        # action_probabilities = policy_net(obs_tensor) + eps
        
        # # Precompute returns G for every timestep
        # G = [ 0 for n in range(traj_length) ]
        # G[-1] = normalised_rewards[-1]
        # for t in range(traj_length - 2, -1, -1):
        #     G[t] = normalised_rewards[t] + gamma * G[t + 1]

        G = torch.tensor(rewards, dtype=torch.float32)
        v_loss = F.mse_loss(baseline_values, G)
        # advantage = G - baseline_values
        # p_loss = torch.mean(-advantage * torch.log(action_probabilities[torch.arange(traj_length), actions]))

        # # Backpropogate and perform optimisation step for the policy
        # policy_optim.zero_grad()
        # p_loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)       # Gradient clipping
        # policy_optim.step()
        
        # Backpropogate and perform optimisation step for the value function
        value_optim.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)       # Gradient clipping
        value_optim.step()

        return {"v_loss": float(v_loss)}