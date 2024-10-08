import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple

action_size = 50

def update(
    value_net: torch.nn.Module, 
    value_optim: torch.optim.Optimizer, 
    observations: List[np.ndarray], 
    actions: List[int], 
    returns: List[float]
) -> Dict[str, float]:
        """
        Performs a training update on the value network using the given observations, actions, and rewards.

        This function calculates the value loss based on the difference between the predicted values 
        from the value network and the observed returns (rewards). It then backpropagates this loss 
        and updates the network's parameters using the provided optimizer.

        Args:
        value_net (torch.nn.Module): The neural network representing the value function.
        
        value_optim (torch.optim.Optimizer): The optimizer used 
                for updating the parameters of the value network.
        
        observations (List[np.ndarray]): A list of observations (states) encountered during the episode.
        
        actions (List[int]): A list of actions taken by the agent corresponding to the observations.
        
        returns (List[float]): A list of returns received by the agent after taking the actions.

        Returns:
                Dict[str, float]: A dictionary containing the value lossafter the update step.
        """
        # Initialise loss and returns
        v_loss = 0

        # Flatten each observation and create one-hot encodings for actions
        flattened_observations = [obs.flatten() for obs in observations]
        one_hot_actions = [np.eye(action_size, dtype=int)[int(action)] for action in actions]
        
        # Concatenate flattened observations with one-hot actions
        state_action_pairs = [
                np.concatenate((obs, act)) 
                for obs, act in zip(flattened_observations, one_hot_actions)
        ]
        
        # Convert state-action pairs to a tensor
        state_action_tensor = torch.tensor(state_action_pairs, dtype=torch.float32)

        # Compute baseline values using the current value network
        baseline_values = value_net(state_action_tensor).squeeze()

        G = torch.tensor(returns, dtype=torch.float32)
        v_loss = F.mse_loss(baseline_values, G)
        
        # Backpropogate and perform optimisation step for the value function
        value_optim.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)       # Gradient clipping
        value_optim.step()

        return {"v_loss": float(v_loss)}

