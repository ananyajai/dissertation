import numpy as np 
from tqdm import tqdm
from BSE import market_session
from matplotlib import pyplot as plt
from typing import List, Dict, DefaultDict, Tuple
from epsilon_scheduling import epsilon_decay
from load_episode import load_episode_data

import torch
import torch.nn.functional as F

action_size = 3


def evaluate(market_params: tuple, value_net, file='testing_data.csv') -> Tuple[float, float]:
    total_return = 0.0
    obs_list, action_list, reward_list = load_episode_data(file)

    # Compute the return using the value_net
    returns = []
    with torch.no_grad():  # No gradient calculation needed during evaluation
        for obs, action, reward in zip(obs_list, action_list, reward_list):
            # Flatten observation and create one-hot encoding for action
            flattened_obs = obs.flatten()
            one_hot_action = np.eye(action_size, dtype=int)[int(action)]
            
            # Concatenate flattened observation with one-hot action
            state_action_pair = np.concatenate((flattened_obs, one_hot_action))
            
            # Convert state-action pair to a tensor
            obs_tensor = torch.tensor(state_action_pair, dtype=torch.float32).unsqueeze(0)

            value = value_net(obs_tensor).item()
            returns.append(value)
            total_return += reward

    # Calculate mean return
    mean_return = total_return / len(reward_list)
    
    # Calculate value loss (MSE)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float32)
    value_loss = F.mse_loss(returns_tensor, rewards_tensor)
    total_value_loss = value_loss.item()

    return mean_return, total_value_loss


# def evaluate(episodes: int, market_params: tuple, value_net, file) -> float:
#     total_return = 0.0

#     updated_market_params = list(market_params)    
#     updated_market_params[3]['sellers'][1][2]['value_func'] = value_net
#     updated_market_params[3]['sellers'][1][2]['epsilon'] = 0.0         # No exploring

#     for _ in range(episodes):
#         balance = 0.0
#         market_session(*updated_market_params)

#         # Read the episode file
#         with open(file, 'r') as f:
#             reader = csv.reader(f)
#             next(reader)  # Skip the header
#             for row in reader:
#                 reward = float(row[2])
#                 balance += reward

#     # Profit made by the RL agent at the end of the trading window
#         total_return += balance
#         mean_return = total_return / episodes

#     return mean_return
