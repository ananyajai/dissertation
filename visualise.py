import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def visualise_data(obs, actions, rewards, total_eps):
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

    # Determine the number of features in observations
    n_features = flat_obs.shape[1]
    n_actions = len(np.unique(flat_actions))

    # Create a figure for observation features with subplots
    fig_obs, axes_obs = plt.subplots(nrows=3, ncols=5, figsize=(20, 12))
    axes_obs = axes_obs.flatten()

    # Feature labels (you can adjust these as per your specific features)
    feature_labels = [
        "Time", "Order Price", "Number of Buyers", "Number of Sellers", 
        "Best Bid", "Best Ask", "Worst Bid", "Worst Ask", 
        "Average Bid", "Average Ask", "Variance of Bids", "Variance of Asks",
        "Average Trade Price", "Trade Price MA", "Countdown"
    ]

    # Plot each feature of the observations
    for i in range(n_features):
        axes_obs[i].hist(flat_obs[:, i], bins=20, alpha=0.6, color='#085ea8', edgecolor='#03045e')
        axes_obs[i].set_title(feature_labels[i], fontsize=18)
        axes_obs[i].set_ylabel("Frequency", fontsize=16)
        axes_obs[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    # Hide any unused subplots if there are fewer features than subplots
    for j in range(n_features, len(axes_obs)):
        fig_obs.delaxes(axes_obs[j])

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"observations_visuals_{total_eps}.png")
    # plt.show()

    # Create a separate figure for actions and rewards
    fig_act_rew, axes_act_rew = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot actions
    axes_act_rew[0].hist(flat_actions, bins=np.arange(n_actions + 1) - 0.5, rwidth=0.8, alpha=0.6, color='#d43d51')
    axes_act_rew[0].set_xlabel("Actions", fontsize=18)
    axes_act_rew[0].set_ylabel("Frequency", fontsize=16)
    axes_act_rew[0].yaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot rewards
    axes_act_rew[1].hist(flat_rewards, bins=20, alpha=0.75, color='#03045e', edgecolor='#03045e')
    axes_act_rew[1].set_xlabel("Rewards", fontsize=18)
    axes_act_rew[1].set_ylabel("Frequency", fontsize=16)
    axes_act_rew[1].yaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"actions_rewards_visuals_{total_eps}.png")
    # plt.show()


# a = [11.083964632599976, 11.387095250199984, 12.163231292999976, 12.374571425999987, 12.125556467399988, 11.46521309799989]
# b = [11.214669392599978, 11.332176876199998, 11.453262590199975, 11.816668027799992, 12.230663947599979, 12.221529250599989]
# c = [10.680813612599986, 11.631020412399979, 12.012621769999981, 11.529902046799979, 11.88506258779999, 12.030308843799988]
# d = [10.944563269799996, 11.559146940000003, 11.619519732599992, 11.664451699999972, 11.772623130599966, 11.586545579599981]
# e = [11.14029660419998, 11.4089863956, 11.951420416399971, 11.613594556399978, 12.08305170220001, 12.212604085000004]
# f = [11.096938779199988, 12.333707483999985, 12.008661225999973, 11.939201366199965, 11.927960544799992, 11.715423131999998]

# gamma = [0, 0.2, 0.4, 0.6, 0.8, 1]
# colours = ['#03045e','#085ea8', '#7e95c5', '#eeadad', '#df676e', '#d43d51']

# fig, ax = plt.subplots(2, 3, figsize=(20, 15))
# ax = ax.flatten()

# for i, data in enumerate([a, b, c, d, e, f]):
#     ax[i].plot(data, color=colours[i])
#     ax[i].set_title(f"γ={gamma[i]:.1f}", fontsize=18)
#     ax[i].set_xlabel('Iterations', fontsize=16)
#     ax[i].set_ylabel('Average Profit', fontsize=16)

# fig.tight_layout()
# plt.show()