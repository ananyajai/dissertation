import numpy as np

def flatten(lst):
    """Flatten a list of lists into a single list."""
    return [item for episode in lst for item in episode]


def calculate_global_stats(lst):
    """
    Calculate global mean and standard deviation 
    for a list of lists (vectors).
    """
    flat_list = flatten(lst)
    
    mean = np.mean(flat_list, axis=0)
    std = np.std(flat_list, axis=0) + 1e-10

    return mean, std


def normalise_returns(returns_list, norm_params=None):
    """
    Normalise returns globally across all episodes.

    Args:
        returns_list (List[List[float]]): A list of episodes, each containing a list of returns.
        norm_params (Tuple[float, float], optional): The global mean and standard deviation. Defaults to None.

    Returns:
        List[List[float]]: The globally normalised returns.
    """
    if norm_params is None:
        mean, std = calculate_global_stats(returns_list)
    else:
        mean, std = norm_params
    
    normalised_returns = []
    for returns in returns_list:
        normalised_episode = np.array([(G - mean) / std for G in returns])
        normalised_returns.append(normalised_episode)
    
    return normalised_returns, (mean, std)


def normalise_obs(obs_list, norm_params=None):
    """
    Normalise observations globally across all episodes.

    Args:
        obs_list (List[List[List[float]]]): A list of episodes, each containing a list of states.
        norm_params (Tuple[np.ndarray, np.ndarray], optional): The global mean and standard deviation for each dimension. Defaults to None.

    Returns:
        List[List[List[float]]]: The globally normalised observations.
    """
    if norm_params is None:
        mean, std = calculate_global_stats(obs_list)
    else:
        mean, std = norm_params

    normalised_obs = []
    for episode in obs_list:
        normalised_episode = [(np.array(state) - mean) / std for state in episode]
        normalised_obs.append(normalised_episode)

    return normalised_obs, (mean, std)

