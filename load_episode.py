import csv
import numpy as np 
from tqdm import tqdm
from BSE import market_session
from matplotlib import pyplot as plt
from typing import List, Dict, DefaultDict, Tuple
from q_table_data import load_q_table, dump_q_table
from epsilon_scheduling import epsilon_decay

import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F


def load_episode_data(file: str) -> Tuple[List[np.ndarray], List[float], List[float]]:
    obs_list, action_list, reward_list = [], [], []

    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            obs_str = row[0].strip()
            obs_str = obs_str.replace('"', '')                   # Remove quotes
            obs_str = obs_str.replace('[', '').replace(']', '')  # Remove brackets

            # Convert the string to a numpy array and reshape
            obs_array = np.fromstring(obs_str, sep=' ').reshape((2, 10, 2))

            obs_list.append(normalise_lob(obs_array))

            # Convert the Action and Reward to floats
            action_list.append(float(row[1]))
            reward_list.append(float(row[2]))

    return obs_list, action_list, reward_list


def normalise_lob(lob: np.ndarray) -> np.ndarray:
    """
    Normalises the LOB data using Min-Max normalisation.

    Args:
        lob (np.ndarray): The LOB data to normalize with shape (2, max_length, 2).

    Returns:
        np.ndarray: The normalized LOB data.
    """
    # Separate prices and quantities
    prices = lob[:, :, 0]
    quantities = lob[:, :, 1]

    # Normalize prices
    price_min = prices.min()
    price_max = prices.max()
    normalised_prices = (prices - price_min) / (price_max - price_min + 1e-10)  # Add small value to avoid division by zero

    # Normalize quantities
    quantity_min = quantities.min()
    quantity_max = quantities.max()
    normalised_quantities = (quantities - quantity_min) / (quantity_max - quantity_min + 1e-10)  # Add small value to avoid division by zero

    # Combine normalized prices and quantities back into LOB format
    normalised_lob = np.stack([normalised_prices, normalised_quantities], axis=-1)

    return normalised_lob