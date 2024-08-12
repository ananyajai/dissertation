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
from metrics_reinforce import train, generate_data, eval_mean_returns

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
    "batch_size": 32
}

# Define the value function neural network
state_size = 14
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
sellers_spec = [('GVWY', 4), ('REINFORCE', 1, {'epsilon': 0.97, 'max_order_price': supply_schedule[0]['ranges'][0][1]})]
buyers_spec = [('GVWY', 5)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

dump_flags = {'dump_strats': False, 'dump_lobs': False, 'dump_avgbals': True, 'dump_tape': False, 'dump_blotters': False}
verbose = False

market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose)

mean_returns_list = []
gvwy_returns_list = []

value_net = Network(dims=(state_size+action_size, 32, 32, 1), output_activation=None)
value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)

for iter in range(1, CONFIG['policy_improv']+1):

    # Generate training data
    train_obs, train_actions, train_rewards = generate_data(CONFIG['train_data_eps'],                                    
                market_params=market_params, 
                eps_file='episode_seller.csv' 
                )

    # Generate validation data
    val_obs, val_actions, val_rewards = generate_data(CONFIG['val_data_eps'],                                                 
                market_params=market_params, 
                eps_file='episode_seller.csv' 
                )

    # Generate testing data
    test_obs, test_actions, test_rewards = generate_data(CONFIG['eval_data_eps'], 
                market_params=market_params, 
                eps_file='episode_seller.csv'
                )

    print(f"GPI - {iter}")

    # Policy evaluation
    stats, valid_loss_list, test_loss_list, value_net = train(
        train_obs, train_actions, train_rewards,
        val_obs, val_actions, val_rewards,
        test_obs, test_actions, test_rewards,
        epochs=CONFIG['num_epochs'],
        eval_freq=CONFIG["eval_freq"],
        gamma=0.4, value_net=value_net, value_optim=value_optim,
        batch_size=CONFIG["batch_size"]
    )

    # Policy improvement
    mean_rl_return, mean_gvwy_return = eval_mean_returns(
                num_trials=200, value_net=value_net, 
                market_params=market_params
            )
    
    print(f"EVALUATION: ITERATION {iter} - MEAN RETURN {mean_rl_return}")
    mean_returns_list.append(mean_rl_return)
    gvwy_returns_list.append(mean_gvwy_return)

    # Update epsilon value using linear decay
    epsilon = linear_epsilon_decay(iter, CONFIG['policy_improv'])
    market_params = list(market_params)    
    market_params[3]['sellers'][1][2]['epsilon'] = epsilon
    market_params[3]['sellers'][1][2]['value_func'] = value_net
    market_params = tuple(market_params)

# Plotting
plt.plot(mean_returns_list, 'c', label='RL')
plt.plot(gvwy_returns_list, 'g', label='GVWY')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Mean Returns')
# plt.title('Policy Improvement')
plt.savefig('policy_improvement.png')
plt.show()

# value_loss = stats['v_loss']
# plt.plot(value_loss, '#085ea8', linewidth=1.0, label='Training Loss')
# plt.plot(valid_loss_list, '#d43d51', linewidth=1.0, label='Validation Loss')
# plt.title(f"Value Loss")
# plt.xlabel("Epoch")
# plt.legend()
# # plt.savefig('value_loss.png')
# plt.show()
