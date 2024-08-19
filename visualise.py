import numpy as np
import matplotlib.pyplot as plt

def visualise_data(obs, actions, rewards):
    """
    Visualises the distribution of features in the observations, 
    actions, and rewards from the training data.

    Args:
        obs (List[List[np.ndarray]]): A list of episodes, where 
            each episode is a list of observations.
        
        actions (List[List[int]]): A list of episodes, where each 
            episode is a list of actions taken during the episode.
        
        rewards (List[List[float]]): A list of episodes, where each 
            episode is a list of rewards received during the episode.

    Returns: 
        None
    """

    # Flatten the data from all episodes into single lists
    flat_obs = np.concatenate([np.array(episode) for episode in obs], axis=0)
    flat_actions = np.concatenate([np.array(episode) for episode in actions], axis=0)
    flat_rewards = np.concatenate([np.array(episode) for episode in rewards], axis=0)

    # Determine the number of features and unique actions
    n_features = flat_obs.shape[1]
    n_actions = len(np.unique(flat_actions))

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
    axes = axes.flatten()

    # Feature labels (you can adjust these as per your specific features)
    feature_labels = [
        "Time", "Order", "N Buyers", "N Sellers", 
        "Best Bid", "Best Ask", "Worst Bid", "Worst Ask", 
        "Avg Bid", "Avg Ask", "Var Bid", "Var Ask",
        "Avg Trade Price", "Trade Price MA"
    ]

    # Plot each feature of the observations
    for i in range(n_features):
        axes[i].hist(flat_obs[:, i], bins=20, alpha=0.75, color='#085ea8', edgecolor='#03045e')
        axes[i].set_title(feature_labels[i])
        axes[i].set_ylabel("Frequency")

    # Plot actions
    axes[n_features].hist(flat_actions, bins=np.arange(n_actions + 1) - 0.5, rwidth=0.8, color='#085ea8', edgecolor='#03045e')
    axes[n_features].set_title("Actions")
    axes[n_features].set_xticks(range(n_actions))
    axes[n_features].set_ylabel("Frequency")

    # Plot histogram for rewards
    axes[n_features + 1].hist(flat_rewards, bins=20, alpha=0.75, color='#085ea8', edgecolor='#03045e')
    axes[n_features + 1].set_title("Rewards")
    axes[n_features + 1].set_ylabel("Frequency")

    # Hide any unused subplots
    for j in range(n_features + 2, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.savefig("data_visuals.png")
    plt.show()

