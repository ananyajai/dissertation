import csv
import numpy as np
from tqdm import tqdm
from send2trash import send2trash
from BSE import market_session
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, DefaultDict, Tuple
from epsilon_scheduling import linear_epsilon_decay
from early_stopping import EarlyStopping
from update import update
from normalise import normalise_returns, normalise_obs
from visualise import visualise_data
from matplotlib.ticker import MaxNLocator

from neural_network import Network
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F

CONFIG = {
    "total_eps": 50,
    "eval_freq": 1,
    "train_data_eps": 50,
    "val_data_eps": 40,
    "eval_data_eps": 30,
    "gamma": 0.8,
    "epsilon": 1.0,
    "batch_size": 32
}

colours = ['#03045e','#085ea8', '#7e95c5', '#eeadad', '#df676e', '#d43d51']
five_colours = ['#03045e','#085ea8', '#7e95c5', '#eeadad', '#d43d51']
mb = '#085ea8'
mp = '#d43d51'

# Define the value function neural network
state_size = 15
action_size = 50
value_net = Network(dims=(state_size+action_size, 32, 32, 32, 1), output_activation=None)
value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)

# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 60.0

# range1 = (50, 100)
# range2 = (100, 150)
# supply_schedule = [{'from': start_time, 'to': 20.0, 'ranges': [range1], 'stepmode': 'fixed'},
#                    {'from': 20.0, 'to': 40.0, 'ranges': [range2], 'stepmode': 'fixed'},
#                    {'from': 40.0, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (50, 150)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

demand_schedule = supply_schedule

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 60

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-fixed'}

sellers_spec = [('GVWY', 9), ('DRL', 1, {'epsilon': 1.0})]
buyers_spec = [('GVWY', 10)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

dump_flags = {'dump_strats': False, 'dump_lobs': False, 'dump_avgbals': True, 'dump_tape': False, 'dump_blotters': False}
verbose = False

market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose)

def generate_data(
        total_eps: int, market_params: tuple, eps_file: str, 
        norm_params:tuple=None, value_net=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple]:
    """
    Generates data by running market session total_eps times.

    Args:
        total_eps (int): Total number of times to run market session.
        gamma (float): Discounting factor for calculating returns
        market_params (tuple): Parameters for running market session.
        eps_file (str): File path where the output of each market session is stored.

    Returns:
        Tuple: Normalised observations, actions, and returns (G).
    """
    obs_list, action_list, rewards_list = [], [], []

    # Remove previous data files if they exist
    try:
        send2trash([eps_file])
    except:
        pass 

    updated_market_params = list(market_params)   
    updated_market_params[3]['sellers'][1][2]['norm_params'] = norm_params

    if value_net is not None:
        updated_market_params[3]['sellers'][1][2]['value_func'] = value_net 

    for i in range(total_eps):
        market_session(*updated_market_params)
        eps_obs, eps_actions, eps_rewards = load_episode_data(eps_file)

        if len(eps_obs) == 0:
            continue
        
        obs_list.append(eps_obs)
        action_list.append(eps_actions)
        rewards_list.append(eps_rewards)

    return obs_list, action_list, rewards_list


def calculate_returns(
        rewards: List[List[float]], gamma: float, norm_params=None
    ) -> List[float]:
    """
    Calculate the returns (discounted cumulative rewards) for each episode.

    Args:
        rewards (List[List[float]]): A list of episodes, where each episode is a list of rewards.
        gamma (float): The discount factor, between 0 and 1, that represents the value of future rewards.
        norm_params (Optional[Tuple[float, float]]): Normalisation parameters (mean and std) to normalise the returns.
                                                    If None, the function will compute and use them.

    Returns:
        Tuple[List[List[float]], Tuple[float, float]]: 
            - A list of episodes, where each episode is a list of normalised returns.
            - A tuple containing the mean and standard deviation used for normalisation.
    """
    G_list = []
    for eps_rewards in rewards:
        if len(eps_rewards) > 0:
            # Precompute returns G for every timestep
            G = [ 0 for n in range(len(eps_rewards)) ]
            G[-1] = eps_rewards[-1]
            for t in range(len(eps_rewards) - 2, -1, -1):
                G[t] = eps_rewards[t] + gamma * G[t + 1]

            G_list.append(G)
        
        else:
            G_list.append([])

    # Normalise returns
    normalised_G, G_norm_params = normalise_returns(G_list, norm_params)

    return normalised_G, G_norm_params


def load_episode_data(file: str) -> Tuple[List, List, List]:
    obs_list, action_list, reward_list = [], [], []
    
    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            obs_str = row[0].strip('()').split(", ")
            obs_list.append(np.array([float(x.strip("'")) for x in obs_str]))        # Convert the string values to floats
            action_list.append((float(row[1])))
            reward_list.append(float(row[2]))

    return obs_list, action_list, reward_list


def train(
        train_obs, train_actions, train_G,
        val_obs, val_actions, val_G,
        test_obs, test_actions, test_G,
        epochs: int, eval_freq: int,
        value_net, value_optim
    ) -> DefaultDict:
    """
    Trains the value network using provided training data and evaluates 
    the model on validation and test sets at specified intervals.
    """

    # Dictionary to store training statistics
    stats = defaultdict(list)
    valid_loss_list = []
    test_loss_list = []
    
    for iteration in range(1, epochs + 1):
        ep_value_loss = []
        # Iterate over each episode
        for i in range(0, len(train_obs)):

            try:
                update_results = update(
                    value_net, value_optim, train_obs[i], train_actions[i], train_G[i]
                )

                for key, value in update_results.items():
                    ep_value_loss.append(value)

            except Exception as e:
                pass

        # Aggregate v_loss for the episode
        avg_v_loss = np.mean(ep_value_loss)
        stats['v_loss'].append(avg_v_loss)
        
        # Evaluate the policy at specified intervals
        if iteration % eval_freq == 0:
            torch.save(value_net.state_dict(), 'value_net_checkpoint.pt')
            value_net.load_state_dict(torch.load('value_net_checkpoint.pt'))
            value_net.eval()

            validation_loss = evaluate(
                val_obs, val_actions, val_G, value_net=value_net
            )
            print(f"VALIDATION: EPOCH {iteration} - VALUE LOSS {validation_loss}")
            valid_loss_list.append(validation_loss)

            early_stop = EarlyStopping()
            if early_stop.should_stop(validation_loss, value_net):
                print(f"Early stopping at epoch {iteration} with validation loss {validation_loss}")
                break

            testing_loss = evaluate(
                test_obs, test_actions, test_G, value_net=value_net
            )
            print(f"TESTING: EPOCH {iteration} - VALUE LOSS {testing_loss}")
            test_loss_list.append(testing_loss)

    return stats, valid_loss_list, test_loss_list, value_net


def evaluate(
        obs_list, action_list, reward_list, value_net
    ) -> float:
    """
    Evaluates the performance of a trained value network on a set of episodes.

    Args:
        obs_list (list): A list of lists containing observations for each episode.
        action_list (list): A list of lists containing actions for each episode.
        reward_list (list): A list of lists containing rewards for each episode.
        value_net (torch.nn.Module): The trained value network model.

    Returns:
        float: The average value loss (mean squared error) across all episodes.
    """

    total_loss = 0.0
    num_episodes = len(obs_list)

    with torch.no_grad():  # No gradient calculation needed during evaluation
        for obs_episode, action_episode, reward_episode in zip(obs_list, action_list, reward_list):
            returns = []

            for obs, action in zip(obs_episode, action_episode):
                # Flatten observation and create one-hot encoding for action
                flattened_obs = obs.flatten()
                one_hot_action = np.eye(action_size, dtype=int)[int(action)]
                
                # Concatenate flattened observation with one-hot action
                state_action_pair = np.concatenate((flattened_obs, one_hot_action))
                
                # Convert state-action pair to a tensor
                obs_tensor = torch.tensor(state_action_pair, dtype=torch.float32).unsqueeze(0)

                value = value_net(obs_tensor).item()
                returns.append(value)

            # Calculate value loss (MSE) for this episode
            returns_tensor = torch.tensor(returns, dtype=torch.float32)
            rewards_tensor = torch.tensor(reward_episode, dtype=torch.float32)
            value_loss = F.mse_loss(returns_tensor, rewards_tensor)
            total_loss += value_loss.item()

    # Return average value loss over all episodes
    average_loss = total_loss / num_episodes

    return average_loss


def eval_mean_returns(
        num_trials, value_net, market_params, 
        norm_params:tuple=(0, 1), 
        model_path:str = 'value_net_checkpoint.pt'
    ):
    # value_net.load_state_dict(torch.load(model_path))
    value_net.eval()

    rl_return = 0.0
    gvwy_return = 0.0
    prev_rl_return = 0.0
    zic_return = 0.0
    check_interval = 100

    updated_market_params = list(market_params)    
    updated_market_params[3]['sellers'][1][2]['value_func'] = value_net
    updated_market_params[3]['sellers'][1][2]['epsilon'] = 0.0         # No exploring
    updated_market_params[3]['sellers'][1][2]['norm_params'] = norm_params

    # for trial in range(num_trials):
    for trial in range(1, num_trials + 1):
        market_session(*updated_market_params)

        with open('session_1_avg_balance.csv', 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip().split(',')
            gvwy_value = float(last_line[11])  # Assuming the value is in the 8th column (index 7)
            gvwy_return += gvwy_value

            rl_value = float(last_line[7])
            rl_return += rl_value

            # zic_return += float(last_line[15])
        
        # Check every `check_interval` trials to see if the mean has stabilized
        if trial % check_interval == 0:
            current_mean_rl_return = rl_return / trial
            # Check if the change in mean over the last `check_interval` trials is below the threshold
            if abs(current_mean_rl_return - prev_rl_return) < 1e-9:
                print(f"Converged at trial {trial}, mean RL Return: {current_mean_rl_return}")
                break
            prev_rl_return = current_mean_rl_return
        
        # # Output progress every 10 trials
        # if trial % 10 == 0:
        #     print(f"Trial {trial}, mean RL Return: {rl_return/trial}, previous RL Return: {prev_rl_return}")

    mean_rl_return = rl_return / trial
    mean_gvwy_return = gvwy_return / trial
    # mean_zic_return = zic_return / trial

    return mean_rl_return, mean_gvwy_return


# Generate training data and normalization parameters
train_obs, train_actions, train_rewards = generate_data(
    total_eps=CONFIG['train_data_eps'], 
    market_params=market_params, 
    eps_file='episode_seller.csv'
)

# Generate validation data using training normalization parameters
val_obs, val_actions, val_rewards = generate_data(
    total_eps=CONFIG['val_data_eps'], 
    market_params=market_params, 
    eps_file='episode_seller.csv'
)

# Generate test data using training normalization parameters
test_obs, test_actions, test_rewards = generate_data(
    total_eps=CONFIG['eval_data_eps'], 
    market_params=market_params, 
    eps_file='episode_seller.csv'
)

# Normalise training observations
train_obs, obs_norm_params = normalise_obs(train_obs)

# Normalise validation observations
val_obs, _ = normalise_obs(val_obs, obs_norm_params)

# Normalise testing observations
test_obs, _ = normalise_obs(test_obs, obs_norm_params)

# Calculate returns
train_G, G_norm_params = calculate_returns(train_rewards, gamma=0.4)
val_G, _ = calculate_returns(val_rewards, gamma=0.4, norm_params=G_norm_params)
test_G, _ = calculate_returns(test_rewards, gamma=0.4, norm_params=G_norm_params)

# Train the value function
stats, valid_loss_list, test_loss_list, value_net = train(
        train_obs, train_actions, train_G,
        val_obs, val_actions, val_G,
        test_obs, test_actions, test_G,
        epochs=CONFIG['total_eps'],
        eval_freq=CONFIG["eval_freq"],
        value_net=value_net, value_optim=value_optim
)

value_loss = stats['v_loss']
plt.plot(value_loss, mb, linewidth=1.0, label='Training Loss')
plt.plot(valid_loss_list, mp, linewidth=1.0, label='Validation Loss')
plt.title(f"Value Loss")
plt.xlabel("Epoch")
plt.legend()
# plt.savefig("training_valid_loss.png")
# plt.close()
plt.show()

# # x_ticks = np.arange(CONFIG['eval_freq'], CONFIG['total_eps'] + 1, CONFIG['eval_freq'])
# # plt.plot(x_ticks, valid_loss_list, linewidth=1.0)
# # plt.title(f"Value Loss - Validation Data")
# # plt.xlabel("Epoch")
# # # plt.savefig("validation_loss.png")
# # # plt.close()
# # plt.show()

# plt.plot(test_loss_list, linewidth=1.0)
# plt.title(f"Value Loss - Testing Data")
# plt.xlabel("Epoch")
# # plt.savefig("testing_loss.png")
# # plt.close()
# plt.show()

# # plt.plot(x_ticks, mean_return_list, linewidth=1.0, label='RL')
# # plt.plot(x_ticks, gvwy_returns_list, linewidth=1.0, label='Giveaway')
# # plt.title(f"Mean Returns")
# # plt.xlabel("Epoch")
# # plt.legend()
# # # plt.savefig("mean_returns.png")
# # # plt.close()
# # plt.show()


# gamma_list = np.linspace(0, 1, 11)

# # Set up the subplot grid
# fig_training, axs_training = plt.subplots(3, 4, figsize=(20, 15))
# fig_testing, axs_testing = plt.subplots(3, 4, figsize=(20, 15))
# # fig_validation, axs_validation = plt.subplots(3, 4, figsize=(20, 15))
# # fig_returns, axs_returns = plt.subplots(3, 4, figsize=(20, 15))

# # Flatten the axes arrays for easy indexing
# axs_training = axs_training.flatten()
# axs_testing = axs_testing.flatten()
# # axs_validation = axs_validation.flatten()
# # axs_returns = axs_returns.flatten()

# # Remove the last subplot (12th) if not needed
# fig_training.delaxes(axs_training[-1])
# fig_testing.delaxes(axs_testing[-1])
# # fig_validation.delaxes(axs_validation[-1])
# # fig_returns.delaxes(axs_returns[-1])

# # Start training
# for i, gamma in enumerate(gamma_list):
#     # Reinitialise the neural network and optimizer for each gamma value
#     value_net = Network(dims=(state_size + action_size, 32, 32, 32, 1), output_activation=None)
#     value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)

#     # Calculate returns
#     train_G, G_norm_params = calculate_returns(train_rewards, gamma=gamma)
#     val_G, _ = calculate_returns(val_rewards, gamma=gamma, norm_params=G_norm_params)
#     test_G, _ = calculate_returns(test_rewards, gamma=gamma, norm_params=G_norm_params)

#     # Train the value function
#     stats, valid_loss_list, test_loss_list, value_net = train(
#             train_obs, train_actions, train_G,
#             val_obs, val_actions, val_G,
#             test_obs, test_actions, test_G,
#             epochs=CONFIG['total_eps'],
#             eval_freq=CONFIG["eval_freq"],
#             value_net=value_net, value_optim=value_optim
#         )

#     value_loss = stats['v_loss']

#     # # Plot mean return
#     # axs_returns[i].plot(mean_return_list, 'c', linewidth=1.0)
#     # axs_returns[i].set_title(f"Mean Return, γ={gamma:.1f}")
#     # axs_returns[i].set_xlabel("Iteration")

#     # Plot training loss
#     axs_training[i].plot(value_loss, mb, linewidth=1.0, label='Training Loss')
#     axs_training[i].plot(valid_loss_list, mp, linewidth=1.0, label='Validation Loss')
#     axs_training[i].set_title(f"γ={gamma:.1f}", fontsize=18)
#     axs_training[i].set_xlabel("Epochs", fontsize=16)
#     axs_training[i].set_ylabel("Loss", fontsize=16)
#     axs_training[i].legend()

#     # Plot testing loss
#     x_ticks = np.arange(CONFIG['eval_freq'], CONFIG['total_eps'] + 1, CONFIG['eval_freq'])
#     axs_testing[i].plot(x_ticks, test_loss_list, '#03045e')
#     axs_testing[i].set_title(f"γ={gamma:.1f}", fontsize=18)
#     axs_testing[i].set_xlabel("Epochs", fontsize=16)
#     axs_testing[i].set_ylabel("Loss", fontsize=16)

#     # # Plot validation loss
#     # axs_validation[i].plot(x_ticks, valid_loss_list, 'g')
#     # axs_validation[i].set_title(f"Validation Loss, γ={gamma:.1f}")
#     # axs_validation[i].set_xlabel("Epoch")
#     # axs_validation[i].set_ylabel("Loss")

# # Adjust layout
# fig_training.tight_layout()
# # fig_returns.tight_layout()
# fig_testing.tight_layout()
# # fig_validation.tight_layout()

# # # Save figures
# fig_training.savefig("train_valid_loss_tradwinds.png")
# # # fig_returns.savefig("mean_return_gammas.png")
# fig_testing.savefig("testing_loss_tradwinds.png")
# # # fig_validation.savefig("validation_loss_gammas.png")

# # plt.show()

