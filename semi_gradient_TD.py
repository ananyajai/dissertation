import csv
import numpy as np
from tqdm import tqdm
from send2trash import send2trash
from BSE import market_session
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, DefaultDict, Tuple
from epsilon_scheduling import epsilon_decay
from early_stopping import EarlyStopping
from update import update
from metrics_reinforce import eval_mean_returns, load_episode_data

from neural_network import Network
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F


CONFIG = {
    "total_eps": 50,
    "eval_freq": 1,
    "train_data_eps": 3500,
    "val_data_eps": 1000,
    "eval_data_eps": 500,
    "gamma": 0.8,
    "epsilon": 1.0,
    "batch_size": 32
}
# Define the value function neural network
state_size = 14
action_size = 20
value_net = Network(dims=(state_size, 32, 32, 1), output_activation=None)
value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)

# colours = ['#085ea8', '#5379b7', '#7e95c5', '#a5b3d4', '#cbd1e2', 
#            '#f1cfce', '#eeadad', '#e88b8d', '#df676e', '#d43d51']

# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 60.0

# range1 = (50, 100)
# range2 = (100, 150)
# supply_schedule = [{'from': start_time, 'to': 20.0, 'ranges': [range1], 'stepmode': 'fixed'},
#                    {'from': 20.0, 'to': 40.0, 'ranges': [range2], 'stepmode': 'fixed'},
#                    {'from': 40.0, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range1 = (50, 150)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]
# range2 = (50, 150)
# demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]
demand_schedule = supply_schedule

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 60

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-fixed'}
# 'max_order_price': supply_schedule[0]['ranges'][0][1]
sellers_spec = [('GVWY', 4), ('REINFORCE', 1, {'epsilon': 1.0, 'max_order_price': supply_schedule[0]['ranges'][0][1]})]
buyers_spec = [('GVWY', 5)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

dump_flags = {'dump_strats': False, 'dump_lobs': False, 'dump_avgbals': True, 'dump_tape': False, 'dump_blotters': False}
verbose = False


def generate_data(total_eps: int, market_params: tuple, eps_file: str) -> np.ndarray:
    """
    Generates testing data by running market session total_eps times.

    Args:
        total_eps (int): Total number of times to run market session.
        gamma (float): Discounting factor for calculating returns
        market_params (tuple): Parameters for running market session.
        eps_file (str): File path where the output of each market session is stored.

    Returns:
        np.ndarray: Array of data points collected from all episodes.
    """
    obs_list, action_list, reward_list, next_obs_list, next_act_list = [], [], [], [], []

    # Remove previous data files if they exist
    try:
        send2trash([eps_file])
    except:
        pass 
    
    for i in range(total_eps):
        market_session(*market_params)
        eps_obs, eps_actions, eps_rewards = load_episode_data(eps_file)
        
        for t in range(len(eps_obs) - 1):
            obs_list.append(eps_obs[t])
            action_list.append(eps_actions[t])
            reward_list.append(eps_rewards[t])
            next_obs_list.append(eps_obs[t + 1])
            next_act_list.append(eps_actions[t + 1])

    obs_array = np.array(obs_list)
    action_array = np.array(action_list)
    reward_array = np.array(reward_list)
    next_obs_array = np.array(next_obs_list)
    next_act_array = np.array(next_act_list)

    # Normalise observations and rewards
    obs_array = (obs_array - np.mean(obs_array, axis=0)) / (np.std(obs_array, axis=0) + + 1e-10)
    next_obs_array = (next_obs_array - np.mean(next_obs_array, axis=0)) / (np.std(next_obs_array, axis=0) + 1e-10)
    reward_array = (reward_array - np.mean(reward_array)) / (np.std(reward_array) + 1e-10)

    return obs_array, action_array, reward_array, next_obs_array, next_act_array


def update(
    value_net: nn.Module, 
    value_optim: torch.optim.Optimizer, 
    batch_obs: np.ndarray, 
    batch_actions: np.ndarray, 
    batch_rewards: np.ndarray, 
    batch_next_obs: np.ndarray, 
    gamma: float
) -> dict:
    """
    Performs a semi-gradient TD(0) update on the value network for a batch of data.

    Args:
        value_net (nn.Module): The value function approximator (neural network).
        value_optim (torch.optim.Optimizer): Optimizer for the value network.
        batch_obs (np.ndarray): Batch of observations (states).
        batch_actions (np.ndarray): Batch of actions taken (not used directly in this update).
        batch_rewards (np.ndarray): Batch of rewards received.
        batch_next_obs (np.ndarray): Batch of next observations (states).
        gamma (float): Discount factor.

    Returns:
        float: The average TD loss for the batch.
    """
    value_net.train()

    # Convert numpy arrays to PyTorch tensors
    states = torch.FloatTensor(batch_obs)
    next_states = torch.FloatTensor(batch_next_obs)
    rewards = torch.FloatTensor(batch_rewards).unsqueeze(1)  # Add an extra dimension for correct tensor shape

    # Predict V(s_t) using the current value network
    values = value_net(states)

    # Predict V(s_{t+1}) using the current value network
    next_values = value_net(next_states).detach()

    # Calculate the TD target
    targets = rewards + gamma * next_values

    # Calculate the TD error
    loss = F.mse_loss(values, targets)

    # Perform gradient descent step
    value_optim.zero_grad()
    loss.backward()
    value_optim.step()

    return loss.item()


def train(
        train_obs, train_actions, train_rewards, train_next_obs, train_next_act,
        val_obs, val_actions, val_rewards, val_next_obs, val_next_act,
        test_obs, test_actions, test_rewards, test_next_obs, test_next_act,
        epochs: int, eval_freq: int, gamma: float,
        value_net, value_optim, batch_size: int=32,
    ) -> DefaultDict:

    # Dictionary to store training statistics
    stats = defaultdict(list)
    valid_loss_list = []
    test_loss_list = []
    
    num_batches = len(train_obs) // batch_size

    for iteration in range(1, epochs + 1):
        ep_value_loss = []

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size

            batch_obs = train_obs[batch_start:batch_end]
            batch_actions = train_actions[batch_start:batch_end]
            batch_rewards = train_rewards[batch_start:batch_end]
            batch_next_obs = train_next_obs[batch_start:batch_end]
            batch_next_act = train_next_act[batch_start:batch_end]

            loss = update(value_net, value_optim, batch_obs, batch_actions, batch_rewards, batch_next_obs, gamma)
            ep_value_loss.append(loss)

        # Average loss for the epoch
        avg_loss = np.mean(ep_value_loss)
        stats['v_loss'].append(avg_loss)

        # Evaluate the policy at specified intervals
        if iteration % eval_freq == 0:
            torch.save(value_net.state_dict(), 'value_net_checkpoint.pt')
            value_net.load_state_dict(torch.load('value_net_checkpoint.pt'))
            value_net.eval()

            validation_loss = evaluate(
                val_obs, val_actions, val_rewards, val_next_obs, val_next_act, 
                value_net=value_net, gamma=gamma
            )
            # print(f"VALIDATION: EPOCH {iteration} - VALUE LOSS {validation_loss}")
            valid_loss_list.append(validation_loss)

            early_stop = EarlyStopping()
            if early_stop.should_stop(validation_loss, value_net):
                print(f"Early stopping at epoch {iteration} with validation loss {validation_loss}")
                break

            testing_loss = evaluate(
                test_obs, test_actions, test_rewards, test_next_obs, test_next_act, 
                value_net=value_net, gamma=gamma
            )

            # tqdm.write(f"TESTING: EPOCH {iteration} - VALUE LOSS {testing_loss}")
            test_loss_list.append(testing_loss)

    return stats, valid_loss_list, test_loss_list, value_net


def evaluate(
    obs: np.ndarray, 
    actions: np.ndarray, 
    rewards: np.ndarray, 
    next_obs: np.ndarray, 
    next_actions: np.ndarray, 
    value_net: nn.Module, 
    gamma: float
) -> float:
    """
    Evaluates the value network on a given dataset by computing the average TD(0) loss.

    Args:
        obs (np.ndarray): Array of observations (states).
        actions (np.ndarray): Array of actions taken.
        rewards (np.ndarray): Array of rewards received.
        next_obs (np.ndarray): Array of next observations (next states).
        next_actions (np.ndarray): Array of next actions (not used directly in this function).
        value_net (nn.Module): The value function approximator (neural network).
        gamma (float): Discount factor for future rewards.

    Returns:
        float: The average TD(0) loss over the dataset.
    """
    value_net.eval()  # Set the network to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for t in range(len(obs)):
            state = torch.FloatTensor(obs[t])
            next_state = torch.FloatTensor(next_obs[t])
            reward = torch.FloatTensor([rewards[t]])

            # Predict V(s_t) using the current value network
            value = value_net(state)

            # Predict V(s_{t+1}) using the current value network
            next_value = value_net(next_state)

            # Calculate the TD target
            target = reward + gamma * next_value

            # Calculate the TD error
            loss = F.mse_loss(value, target)

            total_loss += loss.item()

    average_loss = total_loss / len(obs)
    return average_loss

# Generate training data
train_obs, train_actions, train_rewards, train_next_obs, train_next_act = generate_data(CONFIG['train_data_eps'],                                    
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv' 
              )

# Generate validation data
val_obs, val_actions, val_rewards, val_next_obs, val_next_act = generate_data(CONFIG['val_data_eps'],                                                 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv' 
              )

# Generate testing data
test_obs, test_actions, test_rewards, test_next_obs, test_next_act = generate_data(CONFIG['eval_data_eps'], 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv'
              )

# # Train the value function
# stats, valid_loss_list, test_loss_list, value_net = train(
#         train_obs, train_actions, train_rewards, train_next_obs, train_next_act,
#         val_obs, val_actions, val_rewards, val_next_obs, val_next_act,
#         test_obs, test_actions, test_rewards, test_next_obs, test_next_act,
#         epochs=CONFIG['total_eps'],
#         eval_freq=CONFIG["eval_freq"],
#         gamma=0.2, value_net=value_net, value_optim=value_optim,
#         batch_size=CONFIG["batch_size"]
#     )

# value_loss = stats['v_loss']
# plt.plot(value_loss, 'c', linewidth=1.0, label='Training Loss')
# plt.plot(valid_loss_list, 'g', linewidth=1.0, label='Validation Loss')
# plt.title(f"Value Loss")
# plt.xlabel("Epoch")
# plt.legend()
# # plt.savefig("training_valid_loss.png")
# # plt.close()
# plt.show()

# mean_returns_list = []
# gvwy_returns_list = []

# for iter in range(1, 6):
#     print(f"GPI - {iter}")

#     # Policy evaluation
#     stats, valid_loss_list, test_loss_list, value_net = train(
#         train_obs, train_actions, train_rewards, train_next_obs, train_next_act,
#         val_obs, val_actions, val_rewards, val_next_obs, val_next_act,
#         test_obs, test_actions, test_rewards, test_next_obs, test_next_act,
#         epochs=CONFIG['total_eps'],
#         eval_freq=CONFIG["eval_freq"],
#         gamma=0.2, value_net=value_net, value_optim=value_optim,
#         batch_size=CONFIG["batch_size"]
#     )

#     # Policy improvement
#     mean_rl_return, mean_gvwy_return = eval_mean_returns(
#                 num_trials=100, value_net=value_net, 
#                 market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose)
#             )
    
#     print(f"EVALUATION: ITERATION {iter} - MEAN RETURN {mean_rl_return}")
#     mean_returns_list.append(mean_rl_return)
#     gvwy_returns_list.append(mean_gvwy_return)


# # Plotting
# plt.plot(mean_returns_list, 'c', label='RL')
# plt.plot(gvwy_returns_list, 'g', label='GVWY')
# plt.legend()
# plt.xlabel('Iterations')
# plt.ylabel('Mean Returns')
# plt.title('Policy Improvement')
# # plt.savefig('policy_improvement.png')
# plt.show()


gamma_list = np.linspace(0, 1, 11)

# Set up the subplot grid
fig_training, axs_training = plt.subplots(3, 4, figsize=(20, 15))
fig_testing, axs_testing = plt.subplots(3, 4, figsize=(20, 15))
# fig_validation, axs_validation = plt.subplots(3, 4, figsize=(20, 15))
# fig_returns, axs_returns = plt.subplots(3, 4, figsize=(20, 15))

# Flatten the axes arrays for easy indexing
axs_training = axs_training.flatten()
axs_testing = axs_testing.flatten()
# axs_validation = axs_validation.flatten()
# axs_returns = axs_returns.flatten()

# Remove the last subplot (12th) if not needed
fig_training.delaxes(axs_training[-1])
fig_testing.delaxes(axs_testing[-1])
# fig_validation.delaxes(axs_validation[-1])
# fig_returns.delaxes(axs_returns[-1])

# Start training
for i, gamma in enumerate(gamma_list):
    # Reinitialise the neural network and optimizer for each gamma value
    value_net = Network(dims=(state_size, 32, 32, 1), output_activation=None)
    value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)

    stats, valid_loss_list, test_loss_list, value_net = train(
        train_obs, train_actions, train_rewards, train_next_obs, train_next_act,
        val_obs, val_actions, val_rewards, val_next_obs, val_next_act,
        test_obs, test_actions, test_rewards, test_next_obs, test_next_act,
        epochs=CONFIG['total_eps'],
        eval_freq=CONFIG["eval_freq"],
        gamma=gamma, value_net=value_net, value_optim=value_optim,
        batch_size=CONFIG["batch_size"]
    )

    value_loss = stats['v_loss']

    # # Plot mean return
    # axs_returns[i].plot(mean_return_list, 'c', linewidth=1.0)
    # axs_returns[i].set_title(f"Mean Return, γ={gamma:.1f}")
    # axs_returns[i].set_xlabel("Iteration")

    # Plot training loss
    axs_training[i].plot(value_loss, 'c', linewidth=1.0, label='Training Loss')
    axs_training[i].plot(valid_loss_list, 'g', linewidth=1.0, label='Validation Loss')
    axs_training[i].set_title(f"Value Loss, γ={gamma:.1f}")
    axs_training[i].set_xlabel("Iteration")
    axs_training[i].set_ylabel("Loss")
    axs_training[i].legend()

    # Plot testing loss
    x_ticks = np.arange(CONFIG['eval_freq'], CONFIG['total_eps'] + 1, CONFIG['eval_freq'])
    axs_testing[i].plot(x_ticks, test_loss_list, 'c')
    axs_testing[i].set_title(f"Testing Loss, γ={gamma:.1f}")
    axs_testing[i].set_xlabel("Iteration")
    axs_testing[i].set_ylabel("Loss")

    # # Plot validation loss
    # axs_validation[i].plot(x_ticks, valid_loss_list, 'g')
    # axs_validation[i].set_title(f"Validation Loss, γ={gamma:.1f}")
    # axs_validation[i].set_xlabel("Epoch")
    # axs_validation[i].set_ylabel("Loss")

# Adjust layout
fig_training.tight_layout()
# fig_returns.tight_layout()
fig_testing.tight_layout()
# fig_validation.tight_layout()

# # Save figures
fig_training.savefig("TD_valid_loss_gammas.png")
# # fig_returns.savefig("mean_return_gammas.png")
fig_testing.savefig("TD_test_loss_gammas.png")
# # fig_validation.savefig("validation_loss_gammas.png")

# plt.show()

