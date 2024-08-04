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

from neural_network import Network
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F


CONFIG = {
    "total_eps": 50,
    "eval_freq": 1,
    "train_data_eps": 1600,
    "eval_data_eps": 200,
    "val_data_eps": 200,
    "gamma": 0.3,
    "epsilon": 1.0,
    "batch_size": 32
}

# Define the value function neural network
state_size = 12
action_size = 5
value_net = Network(dims=(state_size+action_size, 32, 32, 1), output_activation=None)
value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)

# policy_net = Network(
#     dims=(state_size, 32, 32, action_size), output_activation=nn.Softmax(dim=-1)
#     )
# policy_optim = Adam(policy_net.parameters(), lr=1e-3, eps=1e-3)


# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 60.0

sellers_spec = [('GVWY', 4), ('REINFORCE', 1, {'epsilon': 1.0, 'value_func': value_net})]
buyers_spec = [('GVWY', 5)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

range1 = (50, 150)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (50, 150)
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 60

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-fixed'}

dump_flags = {'dump_strats': False, 'dump_lobs': False, 'dump_avgbals': False, 'dump_tape': False, 'dump_blotters': False}
verbose = False


def generate_data(total_eps: int, market_params: tuple, eps_file: str, output_file: str):
    """
    Generates testing data by running market session eval_eps 
    times and saving the trajectory data to a csv file.

    Args:
        eval_eps (int): Total number of times to run market session.
        market_params (tuple): Parameters for running market session.
        file (str): File path where the trajectory from each market session is stored.
    """
    obs_list = []
    action_list = []
    reward_list = []

    try:
        send2trash([eps_file, output_file])
    except:
        pass 

    for _ in range(total_eps):
        market_session(*market_params)
        eps_obs, eps_action, eps_reward = load_episode_data(eps_file)
        for obs, action, reward in zip(eps_obs, eps_action, eps_reward):
            if not np.all(obs == 0):  # Check if the observation is not a zero tensor
                obs_list.append(obs)
                action_list.append(action)
                reward_list.append(reward)

    return np.array(obs_list), np.array(action_list), np.array(reward_list)


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
        train_obs, train_actions, train_rewards,
        val_obs, val_actions, val_rewards,
        test_obs, test_actions, test_rewards,
        total_eps: int, eval_freq: int, market_params: tuple, 
        gamma: float, batch_size: int=32
    ) -> DefaultDict:
    # Dictionary to store training statistics
    stats = defaultdict(list)
    mean_returns_list = []
    valid_loss_list = []
    test_loss_list = []

    # obs_list, action_list, reward_list = load_episode_data('training_data.csv')
    for episode in range(1, total_eps + 1):
        ep_value_loss = []
        ep_policy_loss = []

        try:
            # Process data in batches
            for i in range(0, len(train_obs), batch_size):
                obs_batch = train_obs[i:i + batch_size]
                action_batch = train_actions[i:i + batch_size]
                reward_batch = train_rewards[i:i + batch_size]

                update_results, G = update(value_net, value_optim, obs_batch, action_batch, reward_batch, gamma=gamma)

                for key, value in update_results.items():
                    ep_value_loss.append(value)
                # ep_policy_loss.append(update_results['p_loss'])
                # ep_value_loss.append(update_results['v_loss'])

        except Exception as e:
            pass

        # Aggregate v_loss for the episode
        avg_v_loss = np.mean(ep_value_loss)
        stats['v_loss'].append(avg_v_loss)

        # Evaluate the policy at specified intervals
        if episode % eval_freq == 0:
            torch.save(value_net.state_dict(), 'value_net_checkpoint.pt')

            mean_return = eval_mean_returns(
                num_trials=10, value_net=value_net, market_params=market_params
            )
            print(f"EVALUATION: EPOCH {episode} - MEAN RETURN {mean_return}")
            mean_returns_list.append(mean_return)

            validation_loss = evaluate(
                val_obs, val_actions, val_rewards,
                market_params=market_params, value_net=value_net
            )
            print(f"VALIDATION: EPOCH {episode} - VALUE LOSS {validation_loss}")
            valid_loss_list.append(validation_loss)

            early_stop = EarlyStopping()
            if early_stop.should_stop(validation_loss, value_net):
                print(f"Early stopping at epoch {episode} with validation loss {validation_loss}")
                break

            testing_loss = evaluate(
                test_obs, test_actions, test_rewards,
                market_params=market_params, value_net=value_net
            )
            tqdm.write(f"TESTING: EPOCH {episode} - VALUE LOSS {testing_loss}")
            test_loss_list.append(testing_loss)

    return stats, mean_returns_list, valid_loss_list, test_loss_list, G


def evaluate(
        obs_list, action_list, reward_list, 
        market_params: tuple, value_net
    ) -> Tuple[float, float]:

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
    
    # Calculate value loss (MSE)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float32)
    value_loss = F.mse_loss(returns_tensor, rewards_tensor)
    total_value_loss = value_loss.item()

    return total_value_loss


def eval_mean_returns(num_trials, value_net, market_params, model_path:str = 'value_net_checkpoint.pt'):
    value_net.load_state_dict(torch.load(model_path))
    value_net.eval()

    total_return = 0.0

    updated_market_params = list(market_params)    
    updated_market_params[3]['sellers'][1][2]['value_func'] = value_net
    updated_market_params[3]['sellers'][1][2]['epsilon'] = 0.0         # No exploring

    for _ in range(num_trials):
        balance = 0.0
        market_session(*updated_market_params)

        # Read the episode file
        with open('episode_seller.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                reward = float(row[2])
                balance += reward

        # Profit made by the RL agent at the end of the trading window
        total_return += balance

    mean_return = total_return / num_trials

    return mean_return


# Generate training data
train_obs, train_actions, train_rewards = generate_data(CONFIG['train_data_eps'], 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv', 
              output_file='training_data.csv'
              )

# Generate validation data
val_obs, val_actions, val_rewards = generate_data(CONFIG['val_data_eps'], 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv', 
              output_file='validation_data.csv'
              )

# Generate testing data
test_obs, test_actions, test_rewards = generate_data(CONFIG['eval_data_eps'], 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv', 
              output_file='testing_data.csv'
              )

stats, mean_return_list, valid_loss_list, test_loss_list, G = train(
        train_obs, train_actions, train_rewards,
        val_obs, val_actions, val_rewards,
        test_obs, test_actions, test_rewards,
        total_eps=CONFIG['total_eps'],
        eval_freq=CONFIG["eval_freq"],
        market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose),
        gamma=0.2,
        batch_size=CONFIG["batch_size"]
    )

# customer_orders = [ train_obs[i][1] for i in range(len(train_obs)) ]
# print(train_obs, customer_orders)
# plt.plot(customer_orders, G)
# plt.xlabel('Customer Order')
# plt.ylabel('Returns (G value)')
# plt.show()

value_loss = stats['v_loss']
plt.plot(value_loss, 'c', linewidth=1.0, label='Training Loss')
plt.plot(valid_loss_list, 'g', linewidth=1.0, label='Validation Loss')
plt.title(f"Value Loss")
plt.xlabel("Epoch")
plt.legend()
# plt.savefig("training_valid_loss.png")
# plt.close()
plt.show()

# x_ticks = np.arange(CONFIG['eval_freq'], CONFIG['total_eps'] + 1, CONFIG['eval_freq'])
# plt.plot(x_ticks, valid_loss_list, linewidth=1.0)
# plt.title(f"Value Loss - Validation Data")
# plt.xlabel("Epoch")
# # plt.savefig("validation_loss.png")
# # plt.close()
# plt.show()

# plt.plot(x_ticks, test_loss_list, linewidth=1.0)
# plt.title(f"Value Loss - Testing Data")
# plt.xlabel("Epoch")
# # plt.savefig("testing_loss.png")
# # plt.close()
# plt.show()

# plt.plot(x_ticks, mean_return_list, linewidth=1.0)
# plt.title(f"Mean Return")
# plt.xlabel("Epoch")
# plt.savefig("mean_returns.png")
# plt.close()
# plt.show()


gamma_list = np.linspace(0, 1, 3)

# Set up the subplot grid
fig_training, axs_training = plt.subplots(3, 4, figsize=(20, 15))
# fig_testing, axs_testing = plt.subplots(3, 4, figsize=(20, 15))
# fig_validation, axs_validation = plt.subplots(3, 4, figsize=(20, 15))
fig_returns, axs_returns = plt.subplots(3, 4, figsize=(20, 15))

# Flatten the axes arrays for easy indexing
axs_training = axs_training.flatten()
# axs_testing = axs_testing.flatten()
# axs_validation = axs_validation.flatten()
axs_returns = axs_returns.flatten()

# Remove the last subplot (12th) if not needed
fig_training.delaxes(axs_training[-1])
# fig_testing.delaxes(axs_testing[-1])
# fig_validation.delaxes(axs_validation[-1])
fig_returns.delaxes(axs_returns[-1])

# Start training
for i, gamma in enumerate(gamma_list):
    stats, mean_return_list, valid_loss_list, test_loss_list, G = train(
        train_obs, train_actions, train_rewards,
        val_obs, val_actions, val_rewards,
        test_obs, test_actions, test_rewards,
        total_eps=CONFIG['total_eps'],
        eval_freq=CONFIG["eval_freq"],
        market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose),
        gamma=gamma,
        batch_size=CONFIG["batch_size"]
    )

    value_loss = stats['v_loss']

    # Plot mean return
    axs_returns[i].plot(mean_return_list, 'c', linewidth=1.0)
    axs_returns[i].set_title(f"Mean Return, γ={gamma:.1f}")
    axs_returns[i].set_xlabel("Iteration")

    # Plot training loss
    axs_training[i].plot(value_loss, 'c', linewidth=1.0, label='Training Loss')
    axs_training[i].plot(valid_loss_list, 'g', linewidth=1.0, label='Validation Loss')
    axs_training[i].set_title(f"Value Loss, γ={gamma:.1f}")
    axs_training[i].set_xlabel("Iteration")
    axs_training[i].set_ylabel("Loss")

#     # Plot testing loss
#     x_ticks = np.arange(CONFIG['eval_freq'], CONFIG['total_eps'] + 1, CONFIG['eval_freq'])
#     axs_testing[i].plot(x_ticks, test_loss_list, 'c')
#     axs_testing[i].set_title(f"Testing Loss, γ={gamma:.1f}")
#     axs_testing[i].set_xlabel("Epoch")
#     axs_testing[i].set_ylabel("Loss")

#     # Plot validation loss
#     axs_validation[i].plot(x_ticks, valid_loss_list, 'g')
#     axs_validation[i].set_title(f"Validation Loss, γ={gamma:.1f}")
#     axs_validation[i].set_xlabel("Epoch")
#     axs_validation[i].set_ylabel("Loss")

# # Adjust layout
fig_training.tight_layout()
fig_returns.tight_layout()
# fig_testing.tight_layout()
# fig_validation.tight_layout()

# # Save figures
fig_training.savefig("train_valid_loss_gammas.png")
fig_returns.savefig("mean_return_gammas.png")
# fig_testing.savefig("testing_loss_gammas.png")
# fig_validation.savefig("validation_loss_gammas.png")

# plt.show()

