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
from metrics_reinforce import train, generate_data, eval_mean_returns, calculate_returns
from normalise import normalise_obs

from neural_network import Network
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F


CONFIG = {
    "num_epochs": 20,
    "eval_freq": 1,
    "train_data_eps": 2100,
    "val_data_eps": 600,
    "eval_data_eps": 300,
    "policy_improv": 5,
    "epsilon": 1.0,
    "batch_size": 64
}

colours = ['#03045e','#085ea8', '#7e95c5', '#eeadad', '#df676e', '#d43d51']
five_colours = ['#03045e','#085ea8', '#7e95c5', '#eeadad', '#d43d51']
mb = '#085ea8'
mp = '#eeadad'

# Define the value function neural network
state_size = 15
action_size = 50

# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 60.0

range1 = (50, 150)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (50, 150)
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 60

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-fixed'}
# 'max_order_price': supply_schedule[0]['ranges'][0][1]
sellers_spec = [('GVWY', 9), ('REINFORCE', 1, {'epsilon': 0.97, 'max_order_price': supply_schedule[0]['ranges'][0][1]})]
buyers_spec = [('GVWY', 10)]


trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

dump_flags = {'dump_strats': False, 'dump_lobs': False, 'dump_avgbals': True, 'dump_tape': False, 'dump_blotters': False}
verbose = False

market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose)

gamma_list = np.linspace(0, 1, 6)

# Set up the subplot grid
fig, axes = plt.subplots(2, 3, figsize=(20, 15))

# Flatten the axes arrays for easy indexing
axes = axes.flatten()

# Start training
for i, gamma in enumerate(gamma_list):

    rl_returns_list = []

    value_net = Network(dims=(state_size+action_size, 32, 32, 32, 1), output_activation=None)
    value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)

    for iter in range(0, CONFIG['policy_improv']+1):
        print(f"GPI - {iter}")

        # Generate training data and normalisation parameters
        train_obs, train_actions, train_rewards = generate_data(
            total_eps=CONFIG['train_data_eps'], 
            market_params=market_params, 
            eps_file='episode_seller.csv'
        )

        # Generate validation data using training normalisation parameters
        val_obs, val_actions, val_rewards = generate_data(
            total_eps=CONFIG['val_data_eps'], 
            market_params=market_params, 
            eps_file='episode_seller.csv'
        )

        # Generate test data using training normaliation parameters
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
        train_G, G_norm_params = calculate_returns(train_rewards, gamma=gamma)
        val_G, _ = calculate_returns(val_rewards, gamma=gamma, norm_params=G_norm_params)
        test_G, _ = calculate_returns(test_rewards, gamma=gamma, norm_params=G_norm_params)

        # Policy improvement
        mean_rl_return, mean_gvwy_return = eval_mean_returns(
                    num_trials=5000, value_net=value_net, 
                    market_params=market_params,
                    norm_params=obs_norm_params
                )
        
        print(f"EVALUATION: ITERATION {iter} - RL RETURN {mean_rl_return}, GVWY RETURN {mean_gvwy_return}")
        rl_returns_list.append(mean_rl_return)

        # Policy evaluation
        stats, valid_loss_list, test_loss_list, value_net = train(
            train_obs, train_actions, train_G,
            val_obs, val_actions, val_G,
            test_obs, test_actions, test_G,
            epochs=CONFIG['num_epochs'],
            eval_freq=CONFIG["eval_freq"],
            value_net=value_net, value_optim=value_optim,
            batch_size=CONFIG["batch_size"]
        )

        # Update epsilon value using linear decay
        epsilon = linear_epsilon_decay(iter, CONFIG['policy_improv'])
        market_params = list(market_params)    
        market_params[3]['sellers'][1][2]['epsilon'] = epsilon
        market_params[3]['sellers'][1][2]['value_func'] = value_net
        market_params = tuple(market_params)

        axes[i].plot(rl_returns_list, mb)
        axes[i].set_title(f"γ={gamma:.1f}", fontsize=18)
        axes[i].set_xlabel("Iterations", fontsize=16)
        axes[i].set_ylabel("Average Profit", fontsize=16)


fig.tight_layout()
fig.savefig('gpi_gammas.png')
plt.show()

