import random
from send2trash import send2trash
import csv
import numpy as np 
import ast
from tqdm import tqdm
from BSE import market_session
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from epsilon_scheduling import epsilon_decay
from evaluate import evaluate
from load_episode import load_episode_data
from update import update
from early_stopping import EarlyStopping

from neural_network import Network
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F


CONFIG = {
    "total_eps": 100,
    "eval_freq": 10,
    "train_data_eps": 70,
    "eval_data_eps": 10,
    "val_data_eps": 10,
    "gamma": 1.0,
    "epsilon": 1.0,
    "batch_size": 32
}

# Define the value function neural network
state_size = 12
action_size = 3
value_net = Network(dims=(state_size+action_size, 32, 32, 1), output_activation=None)
value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)


# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 60.0

sellers_spec = [('GVWY', 9), ('REINFORCE', 1, {'epsilon': 1.0, 'value_func': value_net})]
buyers_spec = [('GVWY', 10)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

range1 = (50, 150)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (50, 150)
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 60

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-poisson'}

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
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Observations', 'Actions', 'Rewards'])
        for obs, action, reward in zip(obs_list, action_list, reward_list):
            writer.writerow([obs, action, reward])

    print(f"Data has been generated and saved to {output_file}")


def train(total_eps: int, eval_freq: int, market_params: tuple, gamma: float, batch_size: int=32) -> DefaultDict:
    # Dictionary to store training statistics
    stats = defaultdict(list)
    mean_return_list = []
    valid_loss_list = []
    test_loss_list = []

    obs_list, action_list, reward_list = load_episode_data('training_data.csv')
    for episode in range(1, total_eps + 1):
        ep_value_loss = []

        try:
            # Process data in batches
            for i in range(0, len(obs_list), batch_size):
                obs_batch = obs_list[i:i + batch_size]
                action_batch = action_list[i:i + batch_size]
                reward_batch = reward_list[i:i + batch_size]

                update_results = update(obs_batch, action_batch, reward_batch, gamma=gamma)

                for key, value in update_results.items():
                    ep_value_loss.append(value)

        except Exception as e:
            pass

        # Aggregate v_loss for the episode
        avg_v_loss = np.mean(ep_value_loss)
        stats['v_loss'].append(avg_v_loss)

        # Evaluate the policy at specified intervals
        if episode % eval_freq == 0:
            val_mean_return, val_value_loss = evaluate(
                market_params=market_params, value_net=value_net, file='validation_data.csv'
            )
            print(f"VALIDATION: EPOCH {episode} - MEAN RETURN {val_mean_return}, VALUE LOSS {val_value_loss}")
            valid_loss_list.append(val_value_loss)

            early_stop = EarlyStopping()
            if early_stop.should_stop(val_value_loss, value_net):
                print(f"Early stopping at epoch {episode} with validation loss {val_value_loss}")
                break

            mean_return_seller, value_loss = evaluate(
                market_params=market_params, value_net=value_net, file='testing_data.csv'
            )
            tqdm.write(f"TESTING: EPOCH {episode} - MEAN RETURN {mean_return_seller}, VALUE LOSS {value_loss}")
            mean_return_list.append(mean_return_seller)
            test_loss_list.append(value_loss)

    return stats, mean_return_list, valid_loss_list, test_loss_list


# Generate training data
generate_data(CONFIG['train_data_eps'], 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv', 
              output_file='training_data.csv'
              )

# Generate validation data
generate_data(CONFIG['val_data_eps'], 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv', 
              output_file='validation_data.csv'
              )

# Generate testing data
generate_data(CONFIG['eval_data_eps'], 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv', 
              output_file='testing_data.csv'
              )


stats, mean_return_list, valid_loss_list, test_loss_list = train(
        total_eps=CONFIG['total_eps'],
        eval_freq=CONFIG["eval_freq"],
        market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose),
        gamma=0.9,
        batch_size=CONFIG["batch_size"]
    )

value_loss = stats['v_loss']
plt.plot(value_loss, linewidth=1.0)
plt.title(f"Value Loss - Training Data")
plt.xlabel("Epoch")
# plt.savefig(f"training_loss_g{gamma}.png")
# plt.close()
plt.show()

gamma_list = np.linspace(0, 1, 11)

# # Set up the subplot grid
# fig_training, axs_training = plt.subplots(3, 4, figsize=(20, 15))
# fig_testing, axs_testing = plt.subplots(3, 4, figsize=(20, 15))
# fig_validation, axs_validation = plt.subplots(3, 4, figsize=(20, 15))

# # Flatten the axes arrays for easy indexing
# axs_training = axs_training.flatten()
# axs_testing = axs_testing.flatten()
# axs_validation = axs_validation.flatten()

# # Remove the last subplot (12th) if not needed
# fig_training.delaxes(axs_training[-1])
# fig_testing.delaxes(axs_testing[-1])
# fig_validation.delaxes(axs_validation[-1])

# # Start training
# for i, gamma in enumerate(gamma_list):
#     stats, mean_return_list, valid_loss_list, test_loss_list = train(
#         total_eps=CONFIG['total_eps'],
#         eval_freq=CONFIG["eval_freq"],
#         market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose),
#         gamma=gamma,
#         batch_size=CONFIG["batch_size"]
#     )

#     value_loss = stats['v_loss']

#     # Plot training loss
#     axs_training[i].plot(value_loss, linewidth=1.0)
#     axs_training[i].set_title(f"Training Loss, γ={gamma:.1f}")
#     axs_training[i].set_xlabel("Epoch")
#     axs_training[i].set_ylabel("Loss")

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
# fig_training.tight_layout()
# fig_testing.tight_layout()
# fig_validation.tight_layout()

# # Save figures
# fig_training.savefig("training_loss_subplots.png")
# fig_testing.savefig("testing_loss_subplots.png")
# fig_validation.savefig("validation_loss_subplots.png")

# # plt.show()
